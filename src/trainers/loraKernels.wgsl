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
            tileA[tileRow][tileCol] = unpack_int4_to_float(matrixA, globalRow * params.K + globalColA);
        } else {
            tileA[tileRow][tileCol] = 0.0;
        }
        
        // Load tile B into shared memory  
        let globalRowB = t * TILE_SIZE + tileRow;
        let globalColB = workgroup_id.x * TILE_SIZE + tileCol;
        
        if (globalRowB < params.K && globalColB < params.N) {
            tileB[tileRow][tileCol] = unpack_int4_to_float(matrixB, globalRowB * params.N + globalColB);
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

// Helper function to unpack INT4 values to float
fn unpack_int4_to_float(packedData: array<u32>, index: u32) -> f32 {
    let packedIndex = index / 8u;  // 8 INT4 values per u32
    let elementIndex = index % 8u;
    
    let packed = packedData[packedIndex];
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
// LoRA Adapter Forward Pass Kernel
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
@group(1) @binding(2) var<storage, read> matrixA: array<f32>;  // LoRA A matrix
@group(1) @binding(3) var<storage, read> matrixB: array<f32>;  // LoRA B matrix
@group(1) @binding(4) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(256, 1, 1)
fn lora_forward_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let outputIdx = global_id.x;
    
    if (outputIdx >= loraParams.outputDim) {
        return;
    }
    
    // First: input * A -> intermediate (rank dimensions)
    var intermediate: array<f32, 64>; // Max rank of 64
    
    for (var r: u32 = 0u; r < loraParams.rank; r++) {
        var sum: f32 = 0.0;
        for (var i: u32 = 0u; i < loraParams.inputDim; i++) {
            sum += input[i] * matrixA[i * loraParams.rank + r];
        }
        intermediate[r] = sum;
    }
    
    // Second: intermediate * B -> output  
    var result: f32 = 0.0;
    for (var r: u32 = 0u; r < loraParams.rank; r++) {
        result += intermediate[r] * matrixB[r * loraParams.outputDim + outputIdx];
    }
    
    // Add scaled result to existing output
    output[outputIdx] += loraParams.scaling * result;
}

// ============================================================================
// Fused Adam Optimizer Kernel
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
@group(2) @binding(3) var<storage, read_write> momentum: array<f32>;    // m_t
@group(2) @binding(4) var<storage, read_write> velocity: array<f32>;    // v_t

@compute @workgroup_size(256, 1, 1)
fn adam_optimizer_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let numParams = arrayLength(&weights);
    
    if (idx >= numParams) {
        return;
    }
    
    let grad = gradients[idx];
    let weight = weights[idx];
    
    // Add weight decay to gradient
    let gradWithDecay = grad + adamParams.weightDecay * weight;
    
    // Update biased first moment estimate
    momentum[idx] = adamParams.beta1 * momentum[idx] + (1.0 - adamParams.beta1) * gradWithDecay;
    
    // Update biased second raw moment estimate
    velocity[idx] = adamParams.beta2 * velocity[idx] + (1.0 - adamParams.beta2) * gradWithDecay * gradWithDecay;
    
    // Compute bias correction
    let stepFloat = f32(adamParams.step);
    let beta1Correction = 1.0 - pow(adamParams.beta1, stepFloat);
    let beta2Correction = 1.0 - pow(adamParams.beta2, stepFloat);
    
    // Bias-corrected estimates
    let mHat = momentum[idx] / beta1Correction;
    let vHat = velocity[idx] / beta2Correction;
    
    // Update weights
    weights[idx] -= adamParams.learningRate * mHat / (sqrt(vHat) + adamParams.epsilon);
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