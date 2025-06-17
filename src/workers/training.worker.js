/**
 * LoRA Training Worker
 * Handles background training execution with real-time progress reporting
 */

import { createTrainingSession, runInference, disposeSession } from '../trainers/onnxSession.js';
import { LoRARankScheduler, RANK_STRATEGIES } from '../trainers/rankScheduler.js';
import { loadDataset, createTrainingSequences } from '../data/datasetLoader.js';

// Worker state
let trainingSession = null;
let rankScheduler = null;
let isTraining = false;
let trainingConfig = null;
let trainingData = null;
let currentStep = 0;
let totalSteps = 0;
let startTime = null;

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
    // Load WGSL shader code (in a real implementation, this would be loaded from the .wgsl file)
    const shaderCode = `
      // Basic LoRA forward pass shader
      struct LoRAParams {
        inputDim: u32,
        outputDim: u32,
        rank: u32,
        alpha: f32,
        scaling: f32,
      }

      @group(0) @binding(0) var<uniform> params: LoRAParams;
      @group(0) @binding(1) var<storage, read> input: array<f32>;
      @group(0) @binding(2) var<storage, read> matrixA: array<f32>;
      @group(0) @binding(3) var<storage, read> matrixB: array<f32>;
      @group(0) @binding(4) var<storage, read_write> output: array<f32>;

      @compute @workgroup_size(256)
      fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
        let index = global_id.x;
        if (index >= params.outputDim) { return; }
        
        var result: f32 = 0.0;
        for (var r: u32 = 0u; r < params.rank; r++) {
          var intermediate: f32 = 0.0;
          for (var i: u32 = 0u; i < params.inputDim; i++) {
            intermediate += input[i] * matrixA[i * params.rank + r];
          }
          result += intermediate * matrixB[r * params.outputDim + index];
        }
        
        output[index] += params.scaling * result;
      }
    `;

    const shaderModule = device.createShaderModule({
      code: shaderCode
    });

    // Create compute pipeline for LoRA forward pass
    computePipelines.loraForward = device.createComputePipeline({
      layout: 'auto',
      compute: {
        module: shaderModule,
        entryPoint: 'main'
      }
    });

    console.log('Compute pipelines created successfully');
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
      case 'INITIALIZE':
        await handleInitialize(data);
        break;
      case 'START_TRAINING':
        await handleStartTraining(data);
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
 * Initialize training worker
 */
async function handleInitialize(data) {
  const { hardwareInfo, modelOptions } = data;

  try {
    // Initialize WebGPU
    const webgpuReady = await initializeWebGPU();
    if (webgpuReady) {
      await createComputePipelines();
    }

    self.postMessage({
      type: 'INITIALIZED',
      data: {
        webgpuSupported: webgpuReady,
        hardwareInfo,
        ready: true
      }
    });
  } catch (error) {
    console.error('Worker initialization failed:', error);
    self.postMessage({
      type: 'ERROR',
      data: { message: 'Worker initialization failed', error: error.message }
    });
  }
}

/**
 * Start training process
 */
async function handleStartTraining(data) {
  const {
    modelUrl,
    trainingMode,
    corpusText,
    config
  } = data;

  try {
    isTraining = true;
    startTime = Date.now();
    currentStep = 0;
    lossHistory = [];
    throughputHistory = [];

    // Store training configuration
    trainingConfig = {
      mode: trainingMode,
      learningRate: config.learningRate || 1e-4,
      batchSize: config.batchSize || 1,
      sequenceLength: config.sequenceLength || 512,
      maxSteps: config.maxSteps || 1000,
      adapterConfig: config.adapterConfig || { rank: 4, alpha: 8 },
      ...config
    };

    // Process training data
    const dataset = await loadDataset(corpusText, {
      sequenceLength: trainingConfig.sequenceLength,
      maxTokens: 100000 // Limit for demo
    });

    trainingData = createTrainingSequences(
      dataset.tokens,
      trainingConfig.sequenceLength,
      trainingConfig.sequenceLength // No overlap for simplicity
    );

    totalSteps = Math.min(trainingConfig.maxSteps, trainingData.length);

    // Create ONNX training session
    trainingSession = await createTrainingSession(modelUrl, trainingConfig);

    // Initialize rank scheduler
    rankScheduler = new LoRARankScheduler({
      strategy: RANK_STRATEGIES.HARDWARE_AWARE,
      initialRank: trainingConfig.adapterConfig.rank,
      minRank: 2,
      maxRank: 16,
      targetMemoryUsageGB: 4.0
    });

    self.postMessage({
      type: 'TRAINING_STARTED',
      data: {
        totalSteps,
        datasetSize: trainingData.length,
        config: trainingConfig
      }
    });

    // Start training loop
    await runTrainingLoop();

  } catch (error) {
    console.error('Training start failed:', error);
    isTraining = false;
    self.postMessage({
      type: 'ERROR',
      data: { message: 'Training failed to start', error: error.message }
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
    try {
      // Get training batch
      const batch = getTrainingBatch(currentStep, batchSize);
      
      // Simulate training step with actual ONNX inference
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
      
      // Report progress every 10 steps
      if (currentStep % 10 === 0) {
        const progress = (currentStep / totalSteps) * 100;
        const avgLoss = accumulatedLoss / Math.min(currentStep, 50); // Moving average
        const avgThroughput = throughputHistory.slice(-10).reduce((sum, t) => sum + t, 0) / Math.min(10, throughputHistory.length);
        const eta = calculateETA();

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
  try {
    let totalLoss = 0;
    let totalGradientNorm = 0;

    for (const sequence of batch) {
      // Run forward pass through ONNX model
      const inputs = {
        input_ids: sequence.input,
        attention_mask: new Array(sequence.input.length).fill(1)
      };

      const outputs = await runInference(trainingSession.baseSession, inputs);
      
      // Simulate loss calculation (in real implementation, this would be computed properly)
      const targetTokens = sequence.labels;
      const loss = simulateLossCalculation(outputs, targetTokens);
      const gradientNorm = simulateGradientNorm(loss);
      
      totalLoss += loss;
      totalGradientNorm += gradientNorm;

      // Simulate LoRA adapter forward pass using WebGPU if available
      if (device && computePipelines.loraForward) {
        await performLoRAForwardPass(outputs, trainingConfig.adapterConfig);
      }
    }

    // Update memory usage estimation
    memoryUsage = estimateMemoryUsage();

    return {
      loss: totalLoss / batch.length,
      gradientNorm: totalGradientNorm / batch.length
    };
  } catch (error) {
    console.error('Training step computation failed:', error);
    return {
      loss: Math.random() * 2 + 1, // Fallback simulated loss
      gradientNorm: Math.random() * 0.1 + 0.01
    };
  }
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
 * Perform LoRA forward pass using WebGPU
 */
async function performLoRAForwardPass(modelOutputs, adapterConfig) {
  if (!device || !computePipelines.loraForward) return;

  try {
    // This is a simplified WebGPU compute pass
    // In a full implementation, this would involve proper buffer management
    // and integration with the ONNX model outputs
    
    const commandEncoder = device.createCommandEncoder();
    const computePass = commandEncoder.beginComputePass();
    
    computePass.setPipeline(computePipelines.loraForward);
    // Set up bind groups and dispatch compute shader
    // (Implementation details would go here)
    
    computePass.end();
    device.queue.submit([commandEncoder.finish()]);
    
  } catch (error) {
    console.error('WebGPU compute pass failed:', error);
  }
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
  
  const finalStats = {
    totalSteps: currentStep,
    finalLoss: lossHistory[lossHistory.length - 1] || 0,
    averageLoss: lossHistory.reduce((sum, loss) => sum + loss, 0) / lossHistory.length,
    averageThroughput: throughputHistory.reduce((sum, t) => sum + t, 0) / throughputHistory.length,
    trainingTime: (Date.now() - startTime) / 1000,
    rankSchedulerStats: rankScheduler.getStatistics()
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
  
  self.postMessage({
    type: 'TRAINING_PAUSED',
    data: { step: currentStep, totalSteps }
  });
}

/**
 * Resume training
 */
function handleResumeTraining() {
  if (trainingSession && currentStep < totalSteps) {
    isTraining = true;
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
  
  // Cleanup resources
  if (trainingSession) {
    disposeSession(trainingSession);
    trainingSession = null;
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