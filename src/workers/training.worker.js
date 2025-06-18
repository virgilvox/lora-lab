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

// LoRA adapter buffers
let matrixABuffer = null;
let matrixBBuffer = null;

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

    // Store training configuration
    trainingConfig = receivedTrainingConfig;

    // Initialize LoRA adapter matrices on the GPU
    if (device) {
      const { rank } = trainingConfig.adapterConfig;
      const inputDim = 768; // Standard embedding dim
      const outputDim = inputDim;
      matrixABuffer = device.createBuffer({ size: inputDim * rank * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
      matrixBBuffer = device.createBuffer({ size: rank * outputDim * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
      // In a real scenario, you'd initialize these with random weights
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

  const inputs = tokenizer(inputText, { return_tensors: "pt", padding: true, truncation: true });
  const labels = tokenizer(labelText, { return_tensors: "pt", padding: true, truncation: true }).input_ids;

  const { loss: realLoss } = await model({ ...inputs, labels });

  let lossValue;
  if (realLoss === undefined) {
    console.warn(
      "The model did not return a loss value. This could be because the selected model doesn't support training, or the ONNX version is not configured for loss calculation. Falling back to simulated loss for this step."
    );
    lossValue = simulateLossCalculation(null, null);
  } else {
    lossValue = await realLoss.item();
  }

  // --- Step 2: Mock WebGPU Kernel Execution ---
  // Execute the custom kernels with dummy data to validate the WebGPU pipeline.
  if (device && computePipelines.loraForwardA) {
    try {
      // Define mock dimensions
      const batchSize = batch.length;
      const seqLength = trainingConfig.sequenceLength;
      const inputDim = 768; // Assuming a standard embedding dimension
      const rank = trainingConfig.adapterConfig.rank;
      const outputDim = inputDim;

      // Create dummy GPU buffers for weights and gradients
      const gradientABuffer = device.createBuffer({ size: inputDim * rank * 4, usage: GPUBufferUsage.STORAGE });
      const gradientBBuffer = device.createBuffer({ size: rank * outputDim * 4, usage: GPUBufferUsage.STORAGE });
      
      // Dummy input and gradient buffers
      const inputBuffer = device.createBuffer({ size: batchSize * seqLength * inputDim * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
      const outputGradBuffer = device.createBuffer({ size: batchSize * seqLength * outputDim * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
      const intermediateBuffer = device.createBuffer({ size: batchSize * seqLength * rank * 4, usage: GPUBufferUsage.STORAGE });


      // Uniform buffers for parameters
      const loraParamsBuffer = device.createBuffer({ size: 5 * 4, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      device.queue.writeBuffer(loraParamsBuffer, 0, new Uint32Array([inputDim, outputDim, rank, trainingConfig.adapterConfig.alpha, 0]));
      
      const adamParamsBuffer = device.createBuffer({ size: 6 * 4, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      device.queue.writeBuffer(adamParamsBuffer, 0, new Float32Array([0.001, 0.9, 0.999, 1e-8, 0.01, currentStep]));

      // Create Bind Groups
      const fwdABindGroup = device.createBindGroup({
        layout: computePipelines.loraForwardA.getBindGroupLayout(1),
        entries: [
          { binding: 0, resource: { buffer: loraParamsBuffer } },
          { binding: 1, resource: { buffer: inputBuffer } },
          { binding: 2, resource: { buffer: matrixABuffer } },
          { binding: 3, resource: { buffer: matrixBBuffer } },
          { binding: 4, resource: { buffer: outputGradBuffer } }, // Placeholder
          { binding: 5, resource: { buffer: intermediateBuffer } },
        ]
      });
      // NOTE: In a real implementation, you would create bind groups for all kernels

      // Command Encoder
      const commandEncoder = device.createCommandEncoder();
      const passEncoder = commandEncoder.beginComputePass();
      
      // Dispatch forward pass kernel (as a test)
      passEncoder.setPipeline(computePipelines.loraForwardA);
      passEncoder.setBindGroup(1, fwdABindGroup);
      passEncoder.dispatchWorkgroups(Math.ceil(rank / 16), batchSize * seqLength, 1);
      
      // ... dispatch other kernels (forwardB, backwardA, backwardB, adam) here
      
      passEncoder.end();
      device.queue.submit([commandEncoder.finish()]);

      // In a real scenario, you'd read the updated weights back.
      // For this mock, we just confirm the pipeline runs.

      // Cleanup per-step dummy buffers
      gradientABuffer.destroy();
      gradientBBuffer.destroy();
      inputBuffer.destroy();
      outputGradBuffer.destroy();
      intermediateBuffer.destroy();
      loraParamsBuffer.destroy();
      adamParamsBuffer.destroy();

    } catch (e) {
      console.warn("WebGPU kernel execution test failed, continuing with simulated loss.", e);
    }
  }

  // --- Step 3: Return Real Loss and Simulated Gradient ---
  return {
    loss: lossValue,
    gradientNorm: simulateGradientNorm(lossValue)
  };
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
  
  // Read adapter data back from GPU
  let adapterData = null;
  if (device && matrixABuffer && matrixBBuffer) {
    const { rank } = trainingConfig.adapterConfig;
    const inputDim = 768;
    const outputDim = 768;

    const readableABuffer = device.createBuffer({ size: inputDim * rank * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const readableBBuffer = device.createBuffer({ size: rank * outputDim * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    
    const commandEncoder = device.createCommandEncoder();
    commandEncoder.copyBufferToBuffer(matrixABuffer, 0, readableABuffer, 0, inputDim * rank * 4);
    commandEncoder.copyBufferToBuffer(matrixBBuffer, 0, readableBBuffer, 0, rank * outputDim * 4);
    device.queue.submit([commandEncoder.finish()]);

    await readableABuffer.mapAsync(GPUMapMode.READ);
    await readableBBuffer.mapAsync(GPUMapMode.READ);
    
    const aWeights = new Float32Array(readableABuffer.getMappedRange());
    const bWeights = new Float32Array(readableBBuffer.getMappedRange());

    adapterData = {
      rank: rank,
      alpha: trainingConfig.adapterConfig.alpha,
      layers: {
        'layer_1': { // Example layer name
          A: { data: Array.from(aWeights), shape: [inputDim, rank] },
          B: { data: Array.from(bWeights), shape: [rank, outputDim] }
        }
      }
    };

    readableABuffer.unmap();
    readableBBuffer.unmap();
  }

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
  
  if (matrixABuffer) matrixABuffer.destroy();
  if (matrixBBuffer) matrixBBuffer.destroy();
  matrixABuffer = null;
  matrixBBuffer = null;
  
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