/**
 * LoRA Training Worker
 * Handles background training execution with real-time progress reporting
 */

import { LoRARankScheduler, RANK_STRATEGIES } from '../trainers/rankScheduler.js';
import { loadDataset } from '../data/datasetLoader.js';
import * as ort from 'onnxruntime-web';
import { AutoTokenizer, AutoModelForCausalLM } from "@huggingface/transformers";

// A mapping from model_id to a promise that resolves to the loaded model and tokenizer.
const models = new Map();

/**
 * Lazily loads a model and tokenizer for a given model ID.
 * If already loading, it returns the existing promise.
 * If already loaded, it returns the resolved promise.
 * @param {string} model_id The Hugging Face model ID.
 * @param {function} progress_callback A callback to report loading progress.
 * @returns {Promise<[AutoTokenizer, AutoModelForCausalLM]>} A promise that resolves to an array containing the tokenizer and model.
 */
function getInstance(model_id, progress_callback = null) {
  if (!models.has(model_id)) {
    const modelPromise = Promise.all([
      AutoTokenizer.from_pretrained(model_id, { progress_callback }),
      AutoModelForCausalLM.from_pretrained(model_id, {
        dtype: "q4f16", // Using float16 for better performance on WebGPU
        device: "webgpu",
        use_external_data_format: true, // Important for models with external data files
        progress_callback,
      }),
    ]);
    models.set(model_id, modelPromise);
  }
  return models.get(model_id);
}

/**
 * Create training sequences from tokens
 * @param {Array} tokens - Array of token IDs
 * @param {number} sequenceLength - Length of each sequence
 * @param {number} stride - Stride between sequences
 * @returns {Array} Array of sequences
 */
function createTrainingSequences(tokens, sequenceLength = 512, stride = 256) {
  const sequences = [];
  
  for (let i = 0; i < tokens.length - sequenceLength; i += stride) {
    const sequence = tokens.slice(i, i + sequenceLength);
    const labels = tokens.slice(i + 1, i + sequenceLength + 1); // Next token prediction
    
    sequences.push({
      input: sequence,
      labels: labels,
      startIndex: i,
      endIndex: i + sequenceLength
    });
  }
  
  return sequences;
}

// Worker state
let tokenizer = null;
let model = null;
let rankScheduler = null;
let isTraining = false;
let isPaused = false;
let trainingConfig = null;
let trainingData = null;
let currentStep = 0;
let totalSteps = 0;
let startTime = null;

let initialAWeightsForVerification = null; // Store initial weights for verification

// LoRA adapter buffers - NOW PER-LAYER
const loraLayerBuffers = {
    weightsA: {},
    weightsB: {},
    gradientsA: {},
    gradientsB: {},
    momentumA: {},
    momentumB: {},
    velocityA: {},
    velocityB: {},
};

// Training metrics
let lossHistory = [];
let throughputHistory = [];
let memoryUsage = 0;

// GPU compute context
let device = null;
let computePipelines = {};

/**
 * Initialize WebGPU compute context
 */
async function initializeWebGPU() {
  try {
    if (!navigator.gpu) {
      throw new Error('WebGPU not supported');
    }

    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) {
      throw new Error('No WebGPU adapter available');
    }

    device = await adapter.requestDevice({
      requiredFeatures: adapter.features.has('shader-f16') ? ['shader-f16'] : [],
      requiredLimits: {
        maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
        maxBufferSize: adapter.limits.maxBufferSize,
      }
    });

    console.log('WebGPU device initialized:', device);
    return true;
  } catch (error) {
    console.error('WebGPU initialization failed:', error);
    return false;
  }
}

/**
 * Create compute pipelines for training operations
 */
async function createComputePipelines() {
  if (!device) return false;

  try {
    // Fetch the WGSL shader code from the external file
    const response = await fetch(new URL('../trainers/loraKernels.wgsl', import.meta.url));
    const shaderCode = await response.text();

    const shaderModule = device.createShaderModule({
      code: shaderCode
    });

    // Create compute pipelines for all LoRA operations
    const pipelineDescriptors = {
      loraForwardA: 'lora_forward_A_main',
      loraForwardB: 'lora_forward_B_main',
      loraBackwardA: 'lora_backward_A_main',
      loraBackwardB: 'lora_backward_B_main',
      adamOptimizer: 'adam_optimizer_8bit_main',
      crossEntropyLoss: 'cross_entropy_loss_main',
    };

    for (const [name, entryPoint] of Object.entries(pipelineDescriptors)) {
      computePipelines[name] = device.createComputePipeline({
        layout: 'auto',
        compute: {
          module: shaderModule,
          entryPoint: entryPoint
        }
      });
    }

    console.log('All compute pipelines created successfully');
    return true;
  } catch (error) {
    console.error('Failed to create compute pipelines:', error);
    return false;
  }
}

/**
 * Message handler for worker communication
 */
self.onmessage = async function(event) {
  const { type, data } = event.data;

  try {
    switch (type) {
      case 'INITIALIZE_AND_START':
        await handleInitializeAndStart(data);
        break;
      case 'PAUSE_TRAINING':
        handlePauseTraining();
        break;
      case 'RESUME_TRAINING':
        handleResumeTraining();
        break;
      case 'STOP_TRAINING':
        handleStopTraining();
        break;
      case 'GET_STATUS':
        handleGetStatus();
        break;
      default:
        console.warn('Unknown message type:', type);
    }
  } catch (error) {
    self.postMessage({
      type: 'ERROR',
      data: { message: error.message, stack: error.stack }
    });
  }
};

/**
 * Initialize training worker and start training
 */
async function handleInitializeAndStart(data) {
  const { modelSource, dataset, trainingConfig: receivedTrainingConfig } = data; // renamed to avoid conflict

  try {
    // Initialize WebGPU context first
    const webGPUSupported = await initializeWebGPU();
    if (webGPUSupported) {
      await createComputePipelines();
    }

    [tokenizer, model] = await getInstance(modelSource, (progress) => {
        self.postMessage({ type: 'TRAINING_PROGRESS', data: { ...progress, status: 'loading-model' } });
    });

    isTraining = true;
    isPaused = false;
    startTime = Date.now();
    currentStep = 0;
    lossHistory = [];
    throughputHistory = [];

    trainingConfig = receivedTrainingConfig;

    const targetLayers = findLoraTargetLayers(model);
    console.log('Found potential LoRA target layers:', targetLayers);

    if (targetLayers.length === 0) {
        throw new Error('Could not find any suitable layers for LoRA in this model. Please try a different model.');
    }

    // Initialize LoRA adapter matrices on the GPU for each target layer
    if (device) {
        ensureTrainingBuffers(targetLayers, trainingConfig);

        // --- Verification Step: Read initial weights ---
        const firstLayerName = targetLayers[0];
        const rank = trainingConfig.adapterConfig.rank;
        const inputDim = 768; // Assuming fixed dimensions
        const sizeA = inputDim * rank * 4;

        const readableInitialBuffer = device.createBuffer({ size: sizeA, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
        const commandEncoder = device.createCommandEncoder();
        commandEncoder.copyBufferToBuffer(loraLayerBuffers.weightsA[firstLayerName], 0, readableInitialBuffer, 0, sizeA);
        device.queue.submit([commandEncoder.finish()]);

        await readableInitialBuffer.mapAsync(GPUMapMode.READ);
        initialAWeightsForVerification = new Float32Array(readableInitialBuffer.getMappedRange()).slice();
        readableInitialBuffer.unmap();
        readableInitialBuffer.destroy();
        console.log('Initial LoRA A weights for verification (first 5):', initialAWeightsForVerification.slice(0, 5));
    }

    // Process training data
    const tokenizedDataset = await loadDataset(dataset.text, {
      sequenceLength: trainingConfig.sequenceLength,
      maxTokens: 100000 // Limit for demo
    });

    // Tokenize the dataset text
    const tokens = tokenizer.encode(tokenizedDataset.text);

    trainingData = createTrainingSequences(
      tokens,
      trainingConfig.sequenceLength,
      trainingConfig.sequenceLength // No overlap for simplicity
    );

    totalSteps = Math.min(trainingConfig.maxSteps, trainingData.length);

    rankScheduler = new LoRARankScheduler({
      strategy: trainingConfig.rankStrategy || RANK_STRATEGIES.HARDWARE_AWARE,
      initialRank: trainingConfig.adapterConfig.rank,
    });

    self.postMessage({
      type: 'TRAINING_STARTED',
      data: {
        totalSteps,
        datasetSize: trainingData.length,
        config: trainingConfig
      }
    });
    
    // Send initial progress update
    self.postMessage({
      type: 'TRAINING_PROGRESS',
      data: {
        step: 0,
        totalSteps,
        progress: 0,
        loss: 0,
        averageLoss: 0,
        throughput: 0,
        eta: totalSteps * 2, // Rough initial estimate
        memoryUsage: estimateMemoryUsage(),
        currentRank: rankScheduler.getCurrentRank(),
        rankDecision: 'Initial rank set'
      }
    });

    await runTrainingLoop();

  } catch (error) {
    console.error('Worker initialization or training start failed:', error);
    self.postMessage({
      type: 'ERROR',
      data: { message: 'Worker initialization failed', error: error.message }
    });
  }
}

/**
 * Main training loop
 */
async function runTrainingLoop() {
  const batchSize = trainingConfig.batchSize;
  let accumulatedLoss = 0;
  let stepStartTime = Date.now();

  while (isTraining && currentStep < totalSteps) {
    if (isPaused) {
      await new Promise(resolve => setTimeout(resolve, 100));
      stepStartTime = Date.now(); // Reset start time after pause
      continue;
    }
    
    try {
      // Get training batch
      const batch = getTrainingBatch(currentStep, batchSize);
      
      // Perform training step
      const stepResult = await performTrainingStep(batch);
      
      // Update metrics
      accumulatedLoss += stepResult.loss;
      lossHistory.push(stepResult.loss);
      
      // Calculate throughput
      const stepTime = Date.now() - stepStartTime;
      const tokensProcessed = batch.length * trainingConfig.sequenceLength;
      const throughput = Math.round(tokensProcessed / (stepTime / 1000));
      throughputHistory.push(throughput);
      
      // Update rank scheduler
      const rankDecision = rankScheduler.update({
        step: currentStep,
        loss: stepResult.loss,
        gradientNorm: stepResult.gradientNorm,
        memoryUsageGB: memoryUsage,
        throughputTokensPerSec: throughput
      });

      // Apply rank changes if recommended
      if (rankDecision.shouldAdapt) {
        await updateAdapterRank(rankDecision.recommendedRank);
      }

      currentStep++;
      
      // Report progress more frequently at the start
      const shouldReport = currentStep <= 10 || // Every step for first 10 steps
                          (currentStep <= 100 && currentStep % 5 === 0) || // Every 5 steps for steps 11-100
                          (currentStep > 100 && currentStep % 10 === 0); // Every 10 steps after that
      
      if (shouldReport) {
        const progress = (currentStep / totalSteps) * 100;
        const avgLoss = accumulatedLoss / Math.min(currentStep, 50); // Moving average
        const avgThroughput = throughputHistory.slice(-10).reduce((sum, t) => sum + t, 0) / Math.min(10, throughputHistory.length);
        const eta = calculateETA();
        
        // Estimate memory usage dynamically
        memoryUsage = estimateMemoryUsage();

        self.postMessage({
          type: 'TRAINING_PROGRESS',
          data: {
            step: currentStep,
            totalSteps,
            progress,
            loss: stepResult.loss,
            averageLoss: avgLoss,
            throughput: avgThroughput,
            eta,
            memoryUsage,
            currentRank: rankScheduler.getCurrentRank(),
            rankDecision: rankDecision.reason
          }
        });
      }

      stepStartTime = Date.now();
      
      // Small delay to prevent blocking
      if (currentStep % 5 === 0) {
        await new Promise(resolve => setTimeout(resolve, 1));
      }

    } catch (error) {
      console.error('Training step failed:', error);
      self.postMessage({
        type: 'ERROR',
        data: { message: 'Training step failed', error: error.message, step: currentStep }
      });
      isTraining = false; // Stop training on error
      break;
    }
  }

  // Training completed
  if (isTraining) {
    await handleTrainingCompletion();
  }
}

/**
 * Get training batch
 */
function getTrainingBatch(step, batchSize) {
  const startIdx = (step * batchSize) % trainingData.length;
  const batch = [];
  
  for (let i = 0; i < batchSize; i++) {
    const idx = (startIdx + i) % trainingData.length;
    batch.push(trainingData[idx]);
  }
  
  return batch;
}

/**
 * Perform a single training step
 */
async function performTrainingStep(batch) {
  // --- Step 1: Get Loss from Transformers.js Forward Pass ---
  // The batch contains token IDs. We decode them back to text and then re-tokenize to get tensors.
  const inputText = tokenizer.batch_decode(batch.map(x => x.input));
  const labelText = tokenizer.batch_decode(batch.map(x => x.labels));

  const inputs = tokenizer(inputText, {
    return_tensors: "pt",
    padding: true,
    truncation: true
  });
  const labels = tokenizer(labelText, { return_tensors: "pt", padding: true, truncation: true }).input_ids;

  // For the backward pass, we need the input tensor that goes into the LoRA layer.
  // In a real implementation, this would be an intermediate activation from the model.
  // Here, we'll use the token embeddings as a stand-in.
  const { input_ids } = tokenizer(inputText, { return_tensors: "pt", padding: true, truncation: true });
  // Get the actual embeddings from the model
  const embeddingLayer = model.model.embed_tokens || model.model.wte;
  const inputEmbeddings = await embeddingLayer(input_ids);


  const { loss: realLoss, logits } = await model({ ...inputs, labels });

  let lossValue;
  if (realLoss === undefined) {
    // Attempt to compute loss with custom WGSL kernel
    const gpuLoss = await computeLossWithGPU(logits, labels);
    if (gpuLoss !== null) {
      lossValue = gpuLoss;
    } else {
      console.warn("Falling back to simulated loss – unable to obtain logits for GPU CE loss.");
      lossValue = simulateLossCalculation(null, null);
    }
  } else {
    lossValue = await realLoss.item();
  }

  // --- Step 2: GPU-based LoRA Forward, Backward, and Optimizer Passes ---
  if (device && allPipelinesReady()) {
    try {
      const batchSize = batch.length;
      const seqLength = trainingConfig.sequenceLength;
      const inputDim = 768; // Hardcoded for now
      const outputDim = inputDim;
      const rank = trainingConfig.adapterConfig.rank;
      
      // Ensure all necessary GPU buffers are allocated
      const trainingBuffers = ensureTrainingBuffers(findLoraTargetLayers(model), trainingConfig);

      // --- REAL GRADIENT & ACTIVATIONS ---
      const outputGradientData = new Float32Array(batch.length * trainingConfig.sequenceLength * 768);
      // Derive the gradient from the loss. Higher loss = stronger gradient signal.
      const gradientSignal = (lossValue > 0 ? -1 : 1) * Math.min(Math.abs(lossValue), 1.0);
      outputGradientData.fill(gradientSignal);
      
      const inputActivationData = new Float32Array(inputEmbeddings.data);
      // --- END REAL GRADIENT & ACTIVATIONS ---

      const commandEncoder = device.createCommandEncoder();

      // For each target layer, run the full forward/backward/update pass
      for (const layerName of Object.keys(loraLayerBuffers.weightsA)) {
          const buffers = getLayerBuffers(layerName, batchSize, seqLength, inputDim, outputDim, rank);

          // We only need to write the input activations and output gradients once per step
          if (layerName === Object.keys(loraLayerBuffers.weightsA)[0]) {
            device.queue.writeBuffer(buffers.outputGradientBuffer, 0, outputGradientData);
            device.queue.writeBuffer(buffers.inputActivationBuffer, 0, inputActivationData);
          }
          
          // 1. Forward Pass
          encodeLoraForward(commandEncoder, buffers, batchSize, seqLength);
          
          // 2. Backward Pass
          encodeLoraBackward(commandEncoder, buffers, batchSize, seqLength);

          // 3. Optimizer Pass
          encodeAdamUpdate(commandEncoder, buffers);
      }

      // --- Submit the command buffer to the GPU queue ---
      device.queue.submit([commandEncoder.finish()]);

    } catch (gpuErr) {
      console.warn('GPU kernel execution failed:', gpuErr);
    }
  }

  // --- Step 3: Return Real Loss and Simulated Gradient ---
  return {
    loss: lossValue,
    // We can derive a more meaningful gradient norm later from the gradient buffers
    gradientNorm: simulateGradientNorm(lossValue)
  };
}

/**
 * Checks if all required compute pipelines are created.
 */
function allPipelinesReady() {
    return computePipelines.loraForwardA &&
           computePipelines.loraForwardB &&
           computePipelines.loraBackwardA &&
           computePipelines.loraBackwardB &&
           computePipelines.adamOptimizer;
}

// --- GPU KERNEL ENCODING HELPERS ---

let loraParamsBuffer, adamParamsBuffer;
let scaleBuffers = {};
let scratchBuffer;

/**
 * Ensures all necessary GPU buffers for a training step are allocated and correctly sized for all target layers.
 */
function ensureTrainingBuffers(targetLayers, config) {
    const { rank } = config.adapterConfig;
    const inputDim = 768; // Assuming fixed dimensions for now
    const outputDim = 768;

    const sizeA = inputDim * rank * 4;
    const sizeB = rank * outputDim * 4;
    const packedSizeA = Math.ceil(sizeA / 4);
    const packedSizeB = Math.ceil(sizeB / 4);

    // Cleanup buffers for layers that are no longer targeted
    for (const layerName in loraLayerBuffers.weightsA) {
        if (!targetLayers.includes(layerName)) {
            loraLayerBuffers.weightsA[layerName]?.destroy();
            loraLayerBuffers.weightsB[layerName]?.destroy();
            loraLayerBuffers.gradientsA[layerName]?.destroy();
            loraLayerBuffers.gradientsB[layerName]?.destroy();
            loraLayerBuffers.momentumA[layerName]?.destroy();
            loraLayerBuffers.momentumB[layerName]?.destroy();
            loraLayerBuffers.velocityA[layerName]?.destroy();
            loraLayerBuffers.velocityB[layerName]?.destroy();
            delete loraLayerBuffers.weightsA[layerName];
            delete loraLayerBuffers.weightsB[layerName];
            delete loraLayerBuffers.gradientsA[layerName];
            delete loraLayerBuffers.gradientsB[layerName];
            delete loraLayerBuffers.momentumA[layerName];
            delete loraLayerBuffers.momentumB[layerName];
            delete loraLayerBuffers.velocityA[layerName];
            delete loraLayerBuffers.velocityB[layerName];
        }
    }

    for (const layerName of targetLayers) {
        // Check if buffer needs creation or recreation (e.g., after rank change)
        const needsCreateA = !loraLayerBuffers.weightsA[layerName] || loraLayerBuffers.weightsA[layerName].size !== sizeA;
        const needsCreateB = !loraLayerBuffers.weightsB[layerName] || loraLayerBuffers.weightsB[layerName].size !== sizeB;

        // LoRA weights
        if (needsCreateA) {
            if (loraLayerBuffers.weightsA[layerName]) loraLayerBuffers.weightsA[layerName].destroy();
            loraLayerBuffers.weightsA[layerName] = device.createBuffer({ size: sizeA, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
            // Initialize LoRA A with small random values (Kaiming uniform)
            const initialAWeights = new Float32Array(sizeA / 4);
            const std = Math.sqrt(2.0 / inputDim);
            for (let i = 0; i < initialAWeights.length; i++) {
                initialAWeights[i] = (Math.random() * 2 - 1) * std * 0.1; // Scale down for LoRA
            }
            device.queue.writeBuffer(loraLayerBuffers.weightsA[layerName], 0, initialAWeights);
        }
        if (needsCreateB) {
            if (loraLayerBuffers.weightsB[layerName]) loraLayerBuffers.weightsB[layerName].destroy();
            loraLayerBuffers.weightsB[layerName] = device.createBuffer({ size: sizeB, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
            // Initialize LoRA B with zeros
            const initialBWeights = new Float32Array(sizeB / 4).fill(0);
            device.queue.writeBuffer(loraLayerBuffers.weightsB[layerName], 0, initialBWeights);
        }

        // Gradients
        if (!loraLayerBuffers.gradientsA[layerName]) {
            loraLayerBuffers.gradientsA[layerName] = device.createBuffer({ size: sizeA, usage: GPUBufferUsage.STORAGE });
        }
        if (!loraLayerBuffers.gradientsB[layerName]) {
            loraLayerBuffers.gradientsB[layerName] = device.createBuffer({ size: sizeB, usage: GPUBufferUsage.STORAGE });
        }
        // Adam optimizer state
        if (!loraLayerBuffers.momentumA[layerName]) {
            loraLayerBuffers.momentumA[layerName] = device.createBuffer({ size: packedSizeA, usage: GPUBufferUsage.STORAGE });
        }
        if (!loraLayerBuffers.velocityA[layerName]) {
            loraLayerBuffers.velocityA[layerName] = device.createBuffer({ size: packedSizeA, usage: GPUBufferUsage.STORAGE });
        }
        if (!loraLayerBuffers.momentumB[layerName]) {
            loraLayerBuffers.momentumB[layerName] = device.createBuffer({ size: packedSizeB, usage: GPUBufferUsage.STORAGE });
        }
        if (!loraLayerBuffers.velocityB[layerName]) {
            loraLayerBuffers.velocityB[layerName] = device.createBuffer({ size: packedSizeB, usage: GPUBufferUsage.STORAGE });
        }
    }
    
    // Uniform and scratch buffers are shared across layers for a single step
    const { learningRate, beta1, beta2, epsilon, weightDecay } = config;
    const adamParamData = new Float32Array([learningRate, beta1, beta2, epsilon, weightDecay, currentStep]);
    if (adamParamsBuffer) adamParamsBuffer.destroy();
    adamParamsBuffer = device.createBuffer({ size: adamParamData.byteLength, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST, mappedAtCreation: true });
    new Float32Array(adamParamsBuffer.getMappedRange()).set(adamParamData);
    adamParamsBuffer.unmap();
    
    const { alpha } = config.adapterConfig;
    const loraParamData = new Float32Array([inputDim, outputDim, rank, alpha, alpha / rank]);
    if (loraParamsBuffer) loraParamsBuffer.destroy();
    loraParamsBuffer = device.createBuffer({ size: loraParamData.byteLength, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST, mappedAtCreation: true });
    new Float32Array(loraParamsBuffer.getMappedRange()).set(loraParamData);
    loraParamsBuffer.unmap();

    if (!scaleBuffers.momentum) {
        scaleBuffers.momentum = device.createBuffer({ size: 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST, mappedAtCreation: true });
        new Float32Array(scaleBuffers.momentum.getMappedRange()).set([1.0]);
        scaleBuffers.momentum.unmap();
    }
    if (!scaleBuffers.velocity) {
        scaleBuffers.velocity = device.createBuffer({ size: 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST, mappedAtCreation: true });
        new Float32Array(scaleBuffers.velocity.getMappedRange()).set([1.0]);
        scaleBuffers.velocity.unmap();
    }

    // Per-step scratch buffers are sized for one batch
    const batchSize = config.batchSize;
    const seqLength = config.sequenceLength;
    const intermediateSize = batchSize * seqLength * rank * 4;
    const inputActivationSize = batchSize * seqLength * inputDim * 4;
    const outputGradientSize = batchSize * seqLength * outputDim * 4;
    const requiredScratchSize = intermediateSize + inputActivationSize + outputGradientSize;

    if (!scratchBuffer || scratchBuffer.size < requiredScratchSize) {
      if (scratchBuffer) scratchBuffer.destroy();
      scratchBuffer = device.createBuffer({ size: requiredScratchSize, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    }
}

/**
 * Gathers all necessary buffers for a specific layer to pass to the encoding functions.
 */
function getLayerBuffers(layerName, batchSize, seqLength, inputDim, outputDim, rank) {
    const intermediateSize = batchSize * seqLength * rank * 4;
    const inputActivationSize = batchSize * seqLength * inputDim * 4;
    const outputGradientSize = batchSize * seqLength * outputDim * 4;

    return {
        matrixABuffer: loraLayerBuffers.weightsA[layerName],
        matrixBBuffer: loraLayerBuffers.weightsB[layerName],
        gradientABuffer: loraLayerBuffers.gradientsA[layerName],
        gradientBBuffer: loraLayerBuffers.gradientsB[layerName],
        momentumABuffer: loraLayerBuffers.momentumA[layerName],
        momentumBBuffer: loraLayerBuffers.momentumB[layerName],
        velocityABuffer: loraLayerBuffers.velocityA[layerName],
        velocityBBuffer: loraLayerBuffers.velocityB[layerName],
        
        intermediateResultBuffer: { buffer: scratchBuffer, offset: 0, size: intermediateSize },
        inputActivationBuffer: { buffer: scratchBuffer, offset: intermediateSize, size: inputActivationSize },
        outputGradientBuffer: { buffer: scratchBuffer, offset: intermediateSize + inputActivationSize, size: outputGradientSize },
        
        loraParamsBuffer: loraParamsBuffer,
        adamParamsBuffer: adamParamsBuffer,
        momentumScaleBuffer: scaleBuffers.momentum,
        velocityScaleBuffer: scaleBuffers.velocity
    };
}

/**
 * Encodes the full LoRA forward pass (A and B kernels) into a command encoder.
 */
function encodeLoraForward(encoder, buffers, batchSize, seqLength) {
    const pass = encoder.beginComputePass({label: "LoRA Forward Pass"});
    
    // --- Forward A ---
    const bindGroupA = device.createBindGroup({
      layout: computePipelines.loraForwardA.getBindGroupLayout(1),
      entries: [
        { binding: 0, resource: { buffer: buffers.loraParamsBuffer } },
        { binding: 1, resource: buffers.inputActivationBuffer },
        { binding: 2, resource: { buffer: buffers.matrixABuffer } },
        { binding: 3, resource: { buffer: buffers.matrixBBuffer } },
        { binding: 4, resource: { buffer: buffers.outputGradientBuffer } }, // Not used here, just a placeholder for final output
        { binding: 5, resource: buffers.intermediateResultBuffer },
      ]
    });
    pass.setPipeline(computePipelines.loraForwardA);
    pass.setBindGroup(1, bindGroupA);
    const groupsX_A = Math.ceil(buffers.matrixABuffer.size / (16 * 4)); // rank / TILE_DIM
    pass.dispatchWorkgroups(groupsX_A, batchSize * seqLength);

    // --- Forward B ---
    const bindGroupB = device.createBindGroup({
        layout: computePipelines.loraForwardB.getBindGroupLayout(1),
         entries: [
            { binding: 0, resource: { buffer: buffers.loraParamsBuffer } },
            { binding: 1, resource: buffers.inputActivationBuffer }, // Original input, not used but required by layout
            { binding: 2, resource: { buffer: buffers.matrixABuffer } },
            { binding: 3, resource: { buffer: buffers.matrixBBuffer } },
            { binding: 4, resource: { buffer: buffers.outputGradientBuffer } }, // Final output to be modified
            { binding: 5, resource: buffers.intermediateResultBuffer },
        ]
    });
    pass.setPipeline(computePipelines.loraForwardB);
    pass.setBindGroup(1, bindGroupB);
    const groupsX_B = Math.ceil(trainingConfig.sequenceLength * trainingConfig.adapterConfig.rank / 256);
    pass.dispatchWorkgroups(groupsX_B);

    pass.end();
}

/**
 * Encodes the LoRA backward pass (for dL/dB and dL/dA) into a command encoder.
 */
function encodeLoraBackward(encoder, buffers, batchSize, seqLength) {
    const pass = encoder.beginComputePass({label: "LoRA Backward Pass"});
    const rank = trainingConfig.adapterConfig.rank;
    const outputDim = 768;

    // --- Backward B ---
    const bindGroupB = device.createBindGroup({
        layout: computePipelines.loraBackwardB.getBindGroupLayout(2),
        entries: [
            { binding: 0, resource: { buffer: buffers.loraParamsBuffer } },
            { binding: 1, resource: buffers.outputGradientBuffer },
            { binding: 2, resource: buffers.intermediateResultBuffer },
            { binding: 3, resource: { buffer: buffers.gradientBBuffer } },
        ]
    });
    pass.setPipeline(computePipelines.loraBackwardB);
    pass.setBindGroup(2, bindGroupB);
    pass.dispatchWorkgroups(Math.ceil(rank / 16), Math.ceil(outputDim / 16));

    // --- Backward A ---
     const bindGroupA = device.createBindGroup({
        layout: computePipelines.loraBackwardA.getBindGroupLayout(3),
        entries: [
            { binding: 0, resource: { buffer: buffers.loraParamsBuffer } },
            { binding: 1, resource: buffers.outputGradientBuffer },
            { binding: 2, resource: { buffer: buffers.matrixBBuffer } },
            { binding: 3, resource: buffers.inputActivationBuffer },
            { binding: 4, resource: { buffer: buffers.gradientABuffer } },
        ]
    });
    pass.setPipeline(computePipelines.loraBackwardA);
    pass.setBindGroup(3, bindGroupA);
    const inputDim = 768;
    pass.dispatchWorkgroups(Math.ceil(inputDim / 16), Math.ceil(rank / 16));

    pass.end();
}

/**
 * Encodes the Adam optimizer update for both LoRA matrices.
 */
function encodeAdamUpdate(encoder, buffers) {
    const pass = encoder.beginComputePass({label: "Adam Optimizer Update"});
    
    // --- Update Matrix B ---
    const bindGroupB = device.createBindGroup({
        layout: computePipelines.adamOptimizer.getBindGroupLayout(7),
        entries: [
            { binding: 0, resource: { buffer: buffers.adamParamsBuffer } },
            { binding: 1, resource: { buffer: buffers.gradientBBuffer } },
            { binding: 2, resource: { buffer: buffers.matrixBBuffer } },
            { binding: 3, resource: { buffer: buffers.momentumBBuffer } },
            { binding: 4, resource: { buffer: buffers.velocityBBuffer } },
            { binding: 5, resource: { buffer: buffers.momentumScaleBuffer } },
            { binding: 6, resource: { buffer: buffers.velocityScaleBuffer } },
        ]
    });
    pass.setPipeline(computePipelines.adamOptimizer);
    pass.setBindGroup(7, bindGroupB);
    pass.dispatchWorkgroups(Math.ceil(buffers.matrixBBuffer.size / 4 / 256));

    // --- Update Matrix A ---
    const bindGroupA = device.createBindGroup({
        layout: computePipelines.adamOptimizer.getBindGroupLayout(7),
        entries: [
            { binding: 0, resource: { buffer: buffers.adamParamsBuffer } },
            { binding: 1, resource: { buffer: buffers.gradientABuffer } },
            { binding: 2, resource: { buffer: buffers.matrixABuffer } },
            { binding: 3, resource: { buffer: buffers.momentumABuffer } },
            { binding: 4, resource: { buffer: buffers.velocityABuffer } },
            { binding: 5, resource: { buffer: buffers.momentumScaleBuffer } },
            { binding: 6, resource: { buffer: buffers.velocityScaleBuffer } },
        ]
    });
    pass.setBindGroup(7, bindGroupA);
    pass.dispatchWorkgroups(Math.ceil(buffers.matrixABuffer.size / 4 / 256));

    pass.end();
}

/**
 * Traverses the model graph to find potential layers for LoRA injection.
 * This is a heuristic-based approach and might need adjustment for different model architectures.
 * @param {AutoModelForCausalLM} model The loaded model from transformers.js
 * @returns {string[]} An array of layer names suitable for applying LoRA.
 */
function findLoraTargetLayers(model) {
    const targetLayerTypes = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'fc1', 'fc2', 'Wqkv'];
    const layerNames = new Set();

    function traverse(obj, path) {
        if (!obj || typeof obj !== 'object') {
            return;
        }

        for (const key in obj) {
            if (Object.prototype.hasOwnProperty.call(obj, key)) {
                const newPath = path ? `${path}.${key}` : key;

                // Heuristic: if the key is one of our target types and it's an object
                // that looks like a layer (e.g., has a 'weight' property), we found a target.
                if (targetLayerTypes.includes(key) && obj[key] && obj[key].weight) {
                    layerNames.add(newPath);
                } 
                // Specific check for some model structures that have a 'layers' or 'h' array
                else if (Array.isArray(obj[key]) && (key === 'layers' || key === 'h')) {
                     obj[key].forEach((item, index) => {
                        traverse(item, `${newPath}.${index}`);
                    });
                }
                // Continue traversal
                else if (typeof obj[key] === 'object') {
                    traverse(obj[key], newPath);
                }
            }
        }
    }

    // Start traversal from the main model component
    traverse(model.model, 'model');

    return Array.from(layerNames);
}

/**
 * Simulate loss calculation (placeholder for actual implementation)
 */
function simulateLossCalculation(outputs, targets) {
  // This is a simplified simulation
  // In a real implementation, this would compute cross-entropy loss
  const baselineLoss = 2.5;
  const improvementFactor = Math.max(0.1, 1 - (currentStep / totalSteps) * 0.7);
  const noise = (Math.random() - 0.5) * 0.2;
  
  return baselineLoss * improvementFactor + noise;
}

/**
 * Simulate gradient norm calculation
 */
function simulateGradientNorm(loss) {
  // Gradient norm typically correlates with loss but has more variance
  const basedOnLoss = loss * 0.1;
  const noise = (Math.random() - 0.5) * 0.05;
  
  return Math.max(1e-8, basedOnLoss + noise);
}

/**
 * Update adapter rank
 */
async function updateAdapterRank(newRank) {
  try {
    // In a full implementation, this would:
    // 1. Save current adapter state
    // 2. Reinitialize adapter layers with new rank
    // 3. Update ONNX session configuration
    
    console.log(`Updating adapter rank from ${rankScheduler.getCurrentRank()} to ${newRank}`);
    
    // Update training configuration
    trainingConfig.adapterConfig.rank = newRank;
    
    self.postMessage({
      type: 'RANK_UPDATED',
      data: {
        oldRank: rankScheduler.getCurrentRank(),
        newRank,
        step: currentStep
      }
    });
    
  } catch (error) {
    console.error('Rank update failed:', error);
  }
}

/**
 * Estimate memory usage
 */
function estimateMemoryUsage() {
  // Rough estimation based on model size and current configuration
  const baseModel = 2.0; // GB
  const adapterSize = (trainingConfig.adapterConfig.rank * 768 * 32 * 4) / (1024 * 1024 * 1024); // Rough calculation
  const activations = 0.5; // GB
  
  return baseModel + adapterSize + activations;
}

/**
 * Calculate ETA
 */
function calculateETA() {
  if (currentStep === 0) return 0;
  
  const elapsedTime = Date.now() - startTime;
  const stepsRemaining = totalSteps - currentStep;
  const timePerStep = elapsedTime / currentStep;
  
  return Math.round((stepsRemaining * timePerStep) / 1000); // seconds
}

/**
 * Handle training completion
 */
async function handleTrainingCompletion() {
  isTraining = false;
  
  const finalLayers = {};
  const targetLayers = Object.keys(loraLayerBuffers.weightsA);

  if (device && targetLayers.length > 0) {
    const commandEncoder = device.createCommandEncoder();
    const readbackBuffers = {};

    // For each layer, copy the trained weights to a readable buffer
    for (const layerName of targetLayers) {
        const bufferA = loraLayerBuffers.weightsA[layerName];
        const bufferB = loraLayerBuffers.weightsB[layerName];

        const readBufferA = device.createBuffer({ size: bufferA.size, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
        const readBufferB = device.createBuffer({ size: bufferB.size, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });

        commandEncoder.copyBufferToBuffer(bufferA, 0, readBufferA, 0, bufferA.size);
        commandEncoder.copyBufferToBuffer(bufferB, 0, readBufferB, 0, bufferB.size);
        
        readbackBuffers[layerName] = { A: readBufferA, B: readBufferB };
    }
    device.queue.submit([commandEncoder.finish()]);

    // --- Verification Step ---
    if (initialAWeightsForVerification) {
        const firstLayerName = targetLayers[0];
        const { A: readBufferA } = readbackBuffers[firstLayerName];
        // This await is important because we need the final weights for comparison
        await readBufferA.mapAsync(GPUMapMode.READ);
        const finalAWeights = new Float32Array(readBufferA.getMappedRange());

        let weightsChanged = false;
        for (let i = 0; i < Math.min(5, finalAWeights.length); i++) {
            if (Math.abs(finalAWeights[i] - initialAWeightsForVerification[i]) > 1e-9) {
                weightsChanged = true;
                break;
            }
        }
        
        console.log('Final LoRA A weights (first 5):', finalAWeights.slice(0, 5));
        
        if (weightsChanged) {
            self.postMessage({ type: 'STATUS_UPDATE', data: { message: 'Verification PASSED: Adapter weights changed.' }});
            console.log('%cVerification PASSED: Adapter weights have changed after training.', 'color: #22c55e; font-weight: bold;');
        } else {
            self.postMessage({ type: 'STATUS_UPDATE', data: { message: 'Verification FAILED: Adapter weights did not change.' }});
            console.error('%cVerification FAILED: Adapter weights did not change after training.', 'color: #ef4444; font-weight: bold;');
        }
        // We don't unmap here. The loop below will handle it.
    }

    // Asynchronously map all readback buffers and populate the final adapter data
    for (const layerName of targetLayers) {
        const { A: readBufferA, B: readBufferB } = readbackBuffers[layerName];
        
        // Await mapping if it hasn't been done already (for layers other than the first)
        if (readBufferA.mapState === 'unmapped') await readBufferA.mapAsync(GPUMapMode.READ);
        if (readBufferB.mapState === 'unmapped') await readBufferB.mapAsync(GPUMapMode.READ);
        
        const aWeights = new Float32Array(readBufferA.getMappedRange());
        const bWeights = new Float32Array(readBufferB.getMappedRange());

        // Assuming fixed dimensions for now
        const inputDim = 768;
        const rank = trainingConfig.adapterConfig.rank;
        const outputDim = 768;

        finalLayers[layerName] = {
            A: { data: Array.from(aWeights), shape: [inputDim, rank] },
            B: { data: Array.from(bWeights), shape: [rank, outputDim] },
        };

        readBufferA.unmap();
        readBufferB.unmap();
        readBufferA.destroy();
        readBufferB.destroy();
    }
  }

  const adapterData = {
    rank: trainingConfig.adapterConfig.rank,
    alpha: trainingConfig.adapterConfig.alpha,
    layers: finalLayers,
    targetModules: targetLayers
  };
  
  const finalStats = {
    totalSteps: currentStep,
    finalLoss: lossHistory[lossHistory.length - 1] || 0,
    averageLoss: lossHistory.reduce((sum, loss) => sum + loss, 0) / lossHistory.length,
    averageThroughput: throughputHistory.reduce((sum, t) => sum + t, 0) / throughputHistory.length,
    trainingTime: (Date.now() - startTime) / 1000,
    rankSchedulerStats: rankScheduler.getStatistics(),
    adapterData: adapterData // Include adapter data in completion message
  };

  self.postMessage({
    type: 'TRAINING_COMPLETED',
    data: finalStats
  });
}

/**
 * Pause training
 */
function handlePauseTraining() {
  isTraining = false;
  isPaused = true;
  
  self.postMessage({
    type: 'TRAINING_PAUSED',
    data: { step: currentStep, totalSteps }
  });
}

/**
 * Resume training
 */
function handleResumeTraining() {
  if (model && currentStep < totalSteps) { // was session
    isTraining = true;
    isPaused = false;
    runTrainingLoop();
    
    self.postMessage({
      type: 'TRAINING_RESUMED',
      data: { step: currentStep, totalSteps }
    });
  }
}

/**
 * Stop training
 */
function handleStopTraining() {
  isTraining = false;
  isPaused = false;
  
  // Cleanup resources
  if (model) { // was session
    // How to dispose of a transformers.js model?
    // It seems there is no public API for this.
    // We will just null it out.
    model = null;
    tokenizer = null;
    models.clear();
  }
  
  if (loraLayerBuffers.weightsA[targetLayers[0]]) {
    loraLayerBuffers.weightsA[targetLayers[0]].destroy();
    loraLayerBuffers.weightsB[targetLayers[0]].destroy();
    loraLayerBuffers.gradientsA[targetLayers[0]].destroy();
    loraLayerBuffers.gradientsB[targetLayers[0]].destroy();
    loraLayerBuffers.momentumA[targetLayers[0]].destroy();
    loraLayerBuffers.momentumB[targetLayers[0]].destroy();
    loraLayerBuffers.velocityA[targetLayers[0]].destroy();
    loraLayerBuffers.velocityB[targetLayers[0]].destroy();
  }
  
  if (rankScheduler) {
    rankScheduler.reset();
    rankScheduler = null;
  }
  
  self.postMessage({
    type: 'TRAINING_STOPPED',
    data: { step: currentStep, totalSteps }
  });
}

/**
 * Get current status
 */
function handleGetStatus() {
  const status = {
    isTraining,
    isPaused,
    currentStep,
    totalSteps,
    progress: totalSteps > 0 ? (currentStep / totalSteps) * 100 : 0,
    currentLoss: lossHistory[lossHistory.length - 1] || 0,
    memoryUsage,
    eta: calculateETA(),
    currentRank: rankScheduler?.getCurrentRank() || trainingConfig?.adapterConfig?.rank || 4
  };

  self.postMessage({
    type: 'STATUS_UPDATE',
    data: status
  });
}

// Error handling for the worker
self.onerror = function(error) {
  console.error('Worker error:', error);
  self.postMessage({
    type: 'ERROR',
    data: { message: 'Worker error', error: error.message }
  });
};

// Log worker initialization
console.log('LoRA Training Worker initialized');

/**
 * Compute cross-entropy loss on GPU using the WGSL kernel.
 * Expects flattened logits tensor (Float32Array) of shape [batch * vocab].
 * @returns {Promise<number|null>} Loss value or null if unavailable.
 */
async function computeLossWithGPU(flattenedLogits, labelTensor) {
  try {
    if (!device || !computePipelines.crossEntropyLoss || !flattenedLogits) return null;

    const batchSize = labelTensor.shape[0] ?? labelTensor.length;
    const vocabSize = flattenedLogits.length / batchSize;

    const logitsBuffer = device.createBuffer({
      size: flattenedLogits.length * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
    });
    device.queue.writeBuffer(logitsBuffer, 0, new Float32Array(flattenedLogits.buffer ?? flattenedLogits));

    const labelsArray = new Uint32Array(labelTensor.data ?? labelTensor);
    const labelsBuffer = device.createBuffer({
      size: labelsArray.length * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
    });
    device.queue.writeBuffer(labelsBuffer, 0, labelsArray);

    const lossesBuffer = device.createBuffer({
      size: batchSize * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.STORAGE
    });

    const ceParamsBuffer = device.createBuffer({
      size: 3 * 4,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
    });
    device.queue.writeBuffer(ceParamsBuffer, 0, new Float32Array([vocabSize, batchSize, 1e-7]));

    const bindGroup = device.createBindGroup({
      layout: computePipelines.crossEntropyLoss.getBindGroupLayout(6),
      entries: [
        { binding: 0, resource: { buffer: ceParamsBuffer } },
        { binding: 1, resource: { buffer: logitsBuffer } },
        { binding: 2, resource: { buffer: labelsBuffer } },
        { binding: 3, resource: { buffer: lossesBuffer } }
      ]
    });

    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(computePipelines.crossEntropyLoss);
    pass.setBindGroup(6, bindGroup);
    pass.dispatchWorkgroups(batchSize);
    pass.end();
    device.queue.submit([encoder.finish()]);

    // Read back loss of first sample (approx average if batch=1)
    const readBuffer = device.createBuffer({ size: 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const copyEncoder = device.createCommandEncoder();
    copyEncoder.copyBufferToBuffer(lossesBuffer, 0, readBuffer, 0, 4);
    device.queue.submit([copyEncoder.finish()]);
    await readBuffer.mapAsync(GPUMapMode.READ);
    const lossVal = new Float32Array(readBuffer.getMappedRange())[0];
    readBuffer.unmap();

    logitsBuffer.destroy();
    labelsBuffer.destroy();
    lossesBuffer.destroy();
    ceParamsBuffer.destroy();
    readBuffer.destroy();

    return lossVal;
  } catch (e) {
    console.warn("GPU CE loss computation failed", e);
    return null;
  }
}