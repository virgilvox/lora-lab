// Custom WGSL Kernels for LoRA Lab Training
// INT4 Matrix Multiplication and Fused Optimizer Kernels

// ============================================================================
// INT4 Matrix Multiplication Kernel
// ============================================================================

struct MatMulParams {
    M: u32,           // Rows of A (and C)
    N: u32,           // Cols of B (and C)  
    K: u32,           // Cols of A / Rows of B
    alpha: f32,       // Scaling factor
    beta: f32,        // Bias scaling
}

@group(0) @binding(0) var<uniform> params: MatMulParams;
@group(0) @binding(1) var<storage, read> matrixA: array<u32>;    // INT4 packed
@group(0) @binding(2) var<storage, read> matrixB: array<u32>;    // INT4 packed  
@group(0) @binding(3) var<storage, read_write> matrixC: array<f32>; // FP32 output

// Workgroup size optimized for most GPUs
const TILE_SIZE: u32 = 16u;
const WORKGROUP_SIZE: u32 = 256u;

// Shared memory for tile computation
var<workgroup> tileA: array<array<f32, TILE_SIZE>, TILE_SIZE>;
var<workgroup> tileB: array<array<f32, TILE_SIZE>, TILE_SIZE>;

@compute @workgroup_size(16, 16, 1)
fn int4_matmul_main(@builtin(global_invocation_id) global_id: vec3<u32>,
                    @builtin(local_invocation_id) local_id: vec3<u32>,
                    @builtin(workgroup_id) workgroup_id: vec3<u32>) {
    
    let row = global_id.y;
    let col = global_id.x;
    
    // Bounds check
    if (row >= params.M || col >= params.N) {
        return;
    }
    
    var sum: f32 = 0.0;
    
    // Tile-based computation for cache efficiency
    let numTiles = (params.K + TILE_SIZE - 1u) / TILE_SIZE;
    
    for (var t: u32 = 0u; t < numTiles; t++) {
        // Load tile A into shared memory
        let tileRow = local_id.y;
        let tileCol = local_id.x;
        let globalRow = workgroup_id.y * TILE_SIZE + tileRow;
        let globalColA = t * TILE_SIZE + tileCol;
        
        if (globalRow < params.M && globalColA < params.K) {
            tileA[tileRow][tileCol] = unpack_int4_to_float_A(globalRow * params.K + globalColA);
        } else {
            tileA[tileRow][tileCol] = 0.0;
        }
        
        // Load tile B into shared memory  
        let globalRowB = t * TILE_SIZE + tileRow;
        let globalColB = workgroup_id.x * TILE_SIZE + tileCol;
        
        if (globalRowB < params.K && globalColB < params.N) {
            tileB[tileRow][tileCol] = unpack_int4_to_float_B(globalRowB * params.N + globalColB);
        } else {
            tileB[tileRow][tileCol] = 0.0;
        }
        
        // Synchronize workgroup
        workgroupBarrier();
        
        // Compute partial dot product
        for (var k: u32 = 0u; k < TILE_SIZE; k++) {
            sum += tileA[local_id.y][k] * tileB[k][local_id.x];
        }
        
        // Synchronize before next tile
        workgroupBarrier();
    }
    
    // Write result with scaling
    let index = row * params.N + col;
    matrixC[index] = params.alpha * sum + params.beta * matrixC[index];
}

// Helper function to unpack INT4 values to float from matrixA
fn unpack_int4_to_float_A(index: u32) -> f32 {
    let packedIndex = index / 8u;  // 8 INT4 values per u32
    let elementIndex = index % 8u;
    
    let packed = matrixA[packedIndex];
    let shift = elementIndex * 4u;
    let mask = 0xFu;
    
    let int4Value = (packed >> shift) & mask;
    
    // Convert to signed INT4 (-8 to 7)
    var signedValue: i32;
    if (int4Value >= 8u) {
        signedValue = i32(int4Value) - 16;
    } else {
        signedValue = i32(int4Value);
    }
    
    // Scale to appropriate float range
    return f32(signedValue) / 8.0;
}

// Helper function to unpack INT4 values to float from matrixB
fn unpack_int4_to_float_B(index: u32) -> f32 {
    let packedIndex = index / 8u;  // 8 INT4 values per u32
    let elementIndex = index % 8u;
    
    let packed = matrixB[packedIndex];
    let shift = elementIndex * 4u;
    let mask = 0xFu;
    
    let int4Value = (packed >> shift) & mask;
    
    // Convert to signed INT4 (-8 to 7)
    var signedValue: i32;
    if (int4Value >= 8u) {
        signedValue = i32(int4Value) - 16;
    } else {
        signedValue = i32(int4Value);
    }
    
    // Scale to appropriate float range
    return f32(signedValue) / 8.0;
}

// ============================================================================
// LoRA Adapter Forward Pass Kernel (Optimized with Tiling)
// ============================================================================

struct LoRAParams {
    inputDim: u32,
    outputDim: u32, 
    rank: u32,
    alpha: f32,
    scaling: f32,
}

@group(1) @binding(0) var<uniform> loraParams: LoRAParams;
@group(1) @binding(1) var<storage, read> input: array<f32>;
@group(1) @binding(2) var<storage, read> lora_matrixA: array<f32>;  // LoRA A matrix (inputDim x rank)
@group(1) @binding(3) var<storage, read> lora_matrixB: array<f32>;  // LoRA B matrix (rank x outputDim)
@group(1) @binding(4) var<storage, read_write> output: array<f32>; // The model's original output tensor

// Intermediate result of input * A, stored in a separate buffer for efficiency.
// This buffer is written to in the first pass and read from in the second pass.
@group(1) @binding(5) var<storage, read_write> intermediateResult: array<f32>; // Size: rank

// --- Kernel 1: First half of LoRA forward pass (input * A) ---

const TILE_DIM_A = 16u;
var<workgroup> tileA_shared: array<array<f32, TILE_DIM_A>, TILE_DIM_A>;

@compute @workgroup_size(16, 16, 1)
fn lora_forward_A_main(@builtin(global_invocation_id) global_id: vec3<u32>,
                       @builtin(local_invocation_id) local_id: vec3<u32>) {
    
    let r = global_id.x; // Corresponds to rank dimension
    let i_tile = local_id.y;
    let num_tiles = (loraParams.inputDim + TILE_DIM_A - 1u) / TILE_DIM_A;

    if (r >= loraParams.rank) {
        return;
    }

    var sum: f32 = 0.0;
    for (var t: u32 = 0u; t < num_tiles; t = t + 1u) {
        let i_global = t * TILE_DIM_A + i_tile;
        if (i_global < loraParams.inputDim) {
            tileA_shared[i_tile][local_id.x] = input[i_global] * lora_matrixA[i_global * loraParams.rank + r];
        } else {
            tileA_shared[i_tile][local_id.x] = 0.0;
        }

        workgroupBarrier(); // Ensure all threads have loaded their data into shared memory

        // Reduction within the workgroup
        for (var j: u32 = 1u; j < TILE_DIM_A; j = j * 2u) {
            if (i_tile % (j * 2u) == 0u && (i_tile + j) < TILE_DIM_A) {
                tileA_shared[i_tile][local_id.x] += tileA_shared[i_tile + j][local_id.x];
            }
            workgroupBarrier();
        }

        if (i_tile == 0u) {
            sum += tileA_shared[0u][local_id.x];
        }
    }

    // Only one thread per 'r' writes the final result
    if (local_id.y == 0u) {
        intermediateResult[r] = sum;
    }
}

// --- Kernel 2: Second half of LoRA forward pass (intermediate * B) ---

@compute @workgroup_size(256, 1, 1)
fn lora_forward_B_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let outputIdx = global_id.x;
    
    if (outputIdx >= loraParams.outputDim) {
        return;
    }
    
    var result: f32 = 0.0;
    for (var r: u32 = 0u; r < loraParams.rank; r = r + 1u) {
        result += intermediateResult[r] * lora_matrixB[r * loraParams.outputDim + outputIdx];
    }
    
    // Add scaled result to existing output
    // LoRA formula: h' = h + (B * A * x) * (alpha / rank)
    let scaling = loraParams.alpha / f32(loraParams.rank);
    output[outputIdx] += result * scaling;
}

// ============================================================================
// LoRA Adapter Backward Pass Kernels
// ============================================================================

// --- Kernel 3: Backward pass for LoRA B matrix gradient (dL/dB) ---
@group(2) @binding(0) var<uniform> loraParams_b: LoRAParams;
@group(2) @binding(1) var<storage, read> outputGradient: array<f32>;      // dL/dh'
@group(2) @binding(2) var<storage, read> intermediateResult_b: array<f32>; // Result of (A * x)
@group(2) @binding(3) var<storage, read_write> gradientB: array<f32>;      // dL/dB

@compute @workgroup_size(16, 16, 1)
fn lora_backward_B_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let r = global_id.x; // Rank dimension
    let o = global_id.y; // Output dimension
    
    if (r >= loraParams_b.rank || o >= loraParams_b.outputDim) {
        return;
    }
    
    // dL/dB = intermediate^T * dL/dh'
    // The gradient for a single element B[r, o] is intermediate[r] * outputGradient[o]
    let scaling = loraParams_b.alpha / f32(loraParams_b.rank);
    gradientB[r * loraParams_b.outputDim + o] = intermediateResult_b[r] * outputGradient[o] * scaling;
}


// --- Kernel 4: Backward pass for LoRA A matrix gradient (dL/dA) ---
@group(3) @binding(0) var<uniform> loraParams_a: LoRAParams;
@group(3) @binding(1) var<storage, read> outputGradient_a: array<f32>; // dL/dh'
@group(3) @binding(2) var<storage, read> matrixB_a: array<f32>;        // LoRA B matrix
@group(3) @binding(3) var<storage, read> input_a: array<f32>;          // Original input x
@group(3) @binding(4) var<storage, read_write> gradientA: array<f32>;   // dL/dA

@compute @workgroup_size(16, 16, 1)
fn lora_backward_A_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x; // Input dimension
    let r = global_id.y; // Rank dimension

    if (i >= loraParams_a.inputDim || r >= loraParams_a.rank) {
        return;
    }

    // dL/dA = (dL/dh' * B)^T * x
    // First, compute the intermediate gradient: (dL/dh' * B)
    var intermediate_grad: f32 = 0.0;
    for (var o: u32 = 0u; o < loraParams_a.outputDim; o = o + 1u) {
        intermediate_grad += outputGradient_a[o] * matrixB_a[r * loraParams_a.outputDim + o];
    }
    
    // Then, multiply by the corresponding input value
    let scaling = loraParams_a.alpha / f32(loraParams_a.rank);
    gradientA[i * loraParams_a.rank + r] = intermediate_grad * input_a[i] * scaling;
}

// ============================================================================
// Fused Adam Optimizer Kernel (8-bit)
// ============================================================================

struct AdamParams {
    learningRate: f32,
    beta1: f32,
    beta2: f32,
    epsilon: f32,
    weightDecay: f32,
    step: u32,
}

@group(2) @binding(0) var<uniform> adamParams: AdamParams;
@group(2) @binding(1) var<storage, read> gradients: array<f32>;
@group(2) @binding(2) var<storage, read_write> weights: array<f32>;
@group(2) @binding(3) var<storage, read_write> momentum: array<u32>;    // 8-bit packed momentum
@group(2) @binding(4) var<storage, read_write> velocity: array<u32>;    // 8-bit packed velocity
@group(2) @binding(5) var<storage, read> momentum_scale: array<f32>; // Scaling factor for momentum
@group(2) @binding(6) var<storage, read> velocity_scale: array<f32>; // Scaling factor for velocity

@compute @workgroup_size(256, 1, 1)
fn adam_optimizer_8bit_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let numParams = arrayLength(&weights);
    
    if (idx >= numParams) {
        return;
    }
    
    let grad = gradients[idx];
    let weight = weights[idx];
    
    // Dequantize momentum and velocity
    let m_dq = dequantize_u8_momentum(idx, momentum_scale[0]);
    let v_dq = dequantize_u8_velocity(idx, velocity_scale[0]);

    // Add weight decay to gradient
    let gradWithDecay = grad + adamParams.weightDecay * weight;
    
    // Update biased first moment estimate
    let m_t = adamParams.beta1 * m_dq + (1.0 - adamParams.beta1) * gradWithDecay;
    
    // Update biased second raw moment estimate
    let v_t = adamParams.beta2 * v_dq + (1.0 - adamParams.beta2) * gradWithDecay * gradWithDecay;
    
    // Compute bias correction
    let stepFloat = f32(adamParams.step);
    let beta1Correction = 1.0 - pow(adamParams.beta1, stepFloat);
    let beta2Correction = 1.0 - pow(adamParams.beta2, stepFloat);
    
    // Bias-corrected estimates
    let mHat = m_t / beta1Correction;
    let vHat = v_t / beta2Correction;
    
    // Update weights
    weights[idx] -= adamParams.learningRate * mHat / (sqrt(vHat) + adamParams.epsilon);

    // Quantize and store new momentum and velocity
    quantize_u8_momentum(idx, m_t, momentum_scale[0]);
    quantize_u8_velocity(idx, v_t, velocity_scale[0]);
}

// Helper functions for 8-bit quantization/dequantization
fn dequantize_u8_momentum(index: u32, scale: f32) -> f32 {
    let packed_idx = index / 4u;
    let shift = (index % 4u) * 8u;
    let packed_val = (momentum[packed_idx] >> shift) & 0xFFu;
    return (f32(packed_val) - 128.0) * scale;
}

fn dequantize_u8_velocity(index: u32, scale: f32) -> f32 {
    let packed_idx = index / 4u;
    let shift = (index % 4u) * 8u;
    let packed_val = (velocity[packed_idx] >> shift) & 0xFFu;
    return (f32(packed_val) - 128.0) * scale;
}

fn quantize_u8_momentum(index: u32, value: f32, scale: f32) {
    let packed_idx = index / 4u;
    let shift = (index % 4u) * 8u;
    let quantized_val = u32(clamp(round(value / scale) + 128.0, 0.0, 255.0));
    
    // Atomic operation to prevent race conditions on the same u32 word
    let mask = ~(0xFFu << shift);
    atomicAnd(&momentum[packed_idx], mask); // Clear the 8 bits for the current index
    atomicOr(&momentum[packed_idx], quantized_val << shift); // Set the new 8-bit value
}

fn quantize_u8_velocity(index: u32, value: f32, scale: f32) {
    let packed_idx = index / 4u;
    let shift = (index % 4u) * 8u;
    let quantized_val = u32(clamp(round(value / scale) + 128.0, 0.0, 255.0));
    
    // Atomic operation to prevent race conditions on the same u32 word
    let mask = ~(0xFFu << shift);
    atomicAnd(&velocity[packed_idx], mask); // Clear the 8 bits for the current index
    atomicOr(&velocity[packed_idx], quantized_val << shift); // Set the new 8-bit value
}

// ============================================================================
// INT4 Quantization Kernel
// ============================================================================

struct QuantParams {
    numElements: u32,
    scale: f32,
    zeroPoint: i32,
}

@group(3) @binding(0) var<uniform> quantParams: QuantParams;
@group(3) @binding(1) var<storage, read> floatWeights: array<f32>;
@group(3) @binding(2) var<storage, read_write> quantizedWeights: array<u32>;

@compute @workgroup_size(256, 1, 1) 
fn quantize_int4_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let startIdx = global_id.x * 8u;  // 8 INT4 values per thread
    
    if (startIdx >= quantParams.numElements) {
        return;
    }
    
    var packed: u32 = 0u;
    
    // Pack 8 INT4 values into one u32
    for (var i: u32 = 0u; i < 8u; i++) {
        let idx = startIdx + i;
        if (idx < quantParams.numElements) {
            // Quantize float to INT4
            let quantized = clamp(
                i32(round(floatWeights[idx] / quantParams.scale)) + quantParams.zeroPoint,
                -8, 
                7
            );
            
            // Convert to unsigned for packing
            let unsigned = u32(quantized + 8);
            
            // Pack into the u32
            packed |= (unsigned & 0xFu) << (i * 4u);
        }
    }
    
    quantizedWeights[global_id.x] = packed;
}

// ============================================================================
// Gradient Accumulation Kernel  
// ============================================================================

struct GradAccumParams {
    numGradients: u32,
    scaleFactor: f32,
}

@group(4) @binding(0) var<uniform> gradAccumParams: GradAccumParams;
@group(4) @binding(1) var<storage, read> newGradients: array<f32>;
@group(4) @binding(2) var<storage, read_write> accumulatedGradients: array<f32>;

@compute @workgroup_size(256, 1, 1)
fn gradient_accumulation_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    
    if (idx >= gradAccumParams.numGradients) {
        return;
    }
    
    // Accumulate gradients with scaling
    accumulatedGradients[idx] += gradAccumParams.scaleFactor * newGradients[idx];
}

// ============================================================================
// Memory Copy Kernel (for efficient GPU-GPU transfers)
// ============================================================================

@group(5) @binding(0) var<storage, read> source: array<f32>;
@group(5) @binding(1) var<storage, read_write> destination: array<f32>;

@compute @workgroup_size(256, 1, 1)
fn memory_copy_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let numElements = arrayLength(&source);
    
    if (idx < numElements) {
        destination[idx] = source[idx];
    }
}