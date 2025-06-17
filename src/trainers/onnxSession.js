/**
 * ONNX Runtime Web Session Management for LoRA Lab
 * Handles model loading, WebGPU configuration, and training sessions
 */

import * as ort from 'onnxruntime-web';

// Configure ONNX Runtime for WebGPU
ort.env.wasm.simd = true;
ort.env.wasm.numThreads = navigator.hardwareConcurrency || 4;

// WebGPU execution provider options
const webgpuOptions = {
  deviceType: 'gpu',
  powerPreference: 'high-performance',
};

/**
 * Initialize ONNX Runtime with WebGPU support
 * @returns {Promise<boolean>} Success status
 */
export async function initializeONNX() {
  try {
    console.log('Initializing ONNX Runtime with WebGPU...');
    
    // Check WebGPU support
    if (!navigator.gpu) {
      console.warn('WebGPU not supported, falling back to WASM');
      return false;
    }

    // Test WebGPU adapter
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) {
      console.warn('No WebGPU adapter available, falling back to WASM');
      return false;
    }

    console.log('WebGPU adapter found:', adapter);
    return true;
  } catch (error) {
    console.error('ONNX Runtime initialization failed:', error);
    return false;
  }
}

/**
 * Load a model from URL or local file
 * @param {string} modelUrl - URL or path to the ONNX model
 * @param {Object} options - Loading options
 * @returns {Promise<Object>} Model information and session
 */
export async function loadModel(modelUrl, options = {}) {
  const {
    useWebGPU = true,
    enableProfiling = false,
    optimizeModel = true
  } = options;

  try {
    console.log('Loading model from:', modelUrl);

    // Determine execution providers
    const executionProviders = [];
    
    if (useWebGPU && await initializeONNX()) {
      executionProviders.push(['webgpu', webgpuOptions]);
    }
    
    // Always add WASM as fallback
    executionProviders.push(['wasm']);

    // Session options
    const sessionOptions = {
      executionProviders,
      enableProfiling,
      graphOptimizationLevel: optimizeModel ? 'all' : 'disabled',
      executionMode: 'sequential',
    };

    // Load the model
    const session = await ort.InferenceSession.create(modelUrl, sessionOptions);
    
    // Get model metadata
    const modelInfo = {
      inputNames: session.inputNames,
      outputNames: session.outputNames,
      executionProviders: session.executionProviders,
      modelUrl,
      sessionOptions
    };

    console.log('Model loaded successfully:', modelInfo);
    
    return {
      session,
      modelInfo,
      isWebGPU: session.executionProviders.includes('webgpu')
    };

  } catch (error) {
    console.error('Model loading failed:', error);
    throw new Error(`Failed to load model: ${error.message}`);
  }
}

/**
 * Create a training session (preparation for future training implementation)
 * @param {string} modelUrl - Training model URL
 * @param {Object} trainingConfig - Training configuration
 * @returns {Promise<Object>} Training session info
 */
export async function createTrainingSession(modelUrl, trainingConfig) {
  const {
    mode = 'adapter',
    learningRate = 1e-4,
    batchSize = 1,
    sequenceLength = 512,
    adapterConfig = { rank: 4, alpha: 8 }
  } = trainingConfig;

  try {
    console.log('Creating training session for mode:', mode);

    // Load the base model for inference
    const { session, modelInfo, isWebGPU } = await loadModel(modelUrl, {
      useWebGPU: true,
      enableProfiling: true,
      optimizeModel: true
    });

    // Training session configuration
    const trainingSession = {
      baseSession: session,
      modelInfo,
      isWebGPU,
      mode,
      config: {
        learningRate,
        batchSize,
        sequenceLength,
        adapterConfig
      },
      status: 'initialized',
      createdAt: Date.now()
    };

    // Initialize adapter layers if in adapter mode
    if (mode === 'adapter') {
      trainingSession.adapterLayers = await initializeAdapterLayers(
        modelInfo,
        adapterConfig
      );
    }

    console.log('Training session created:', trainingSession);
    return trainingSession;

  } catch (error) {
    console.error('Training session creation failed:', error);
    throw new Error(`Failed to create training session: ${error.message}`);
  }
}

/**
 * Initialize LoRA adapter layers
 * @param {Object} modelInfo - Model information
 * @param {Object} adapterConfig - Adapter configuration
 * @returns {Promise<Object>} Adapter layer configuration
 */
async function initializeAdapterLayers(modelInfo, adapterConfig) {
  const { rank, alpha } = adapterConfig;
  
  // This is a simplified implementation
  // In a full implementation, this would analyze the model architecture
  // and create appropriate LoRA matrices for each layer
  
  const adapterLayers = {
    rank,
    alpha,
    scaling: alpha / rank,
    layers: {},
    totalParams: 0
  };

  // Simulate adapter layer initialization
  // In reality, this would inspect the model graph and identify linear layers
  const estimatedLayers = 32; // Typical for small language models
  
  for (let i = 0; i < estimatedLayers; i++) {
    const layerName = `layer_${i}`;
    const hiddenSize = 768; // Typical hidden size
    
    adapterLayers.layers[layerName] = {
      A: { shape: [hiddenSize, rank], initialized: false },
      B: { shape: [rank, hiddenSize], initialized: false },
      paramCount: hiddenSize * rank * 2
    };
    
    adapterLayers.totalParams += hiddenSize * rank * 2;
  }

  console.log(`Initialized ${estimatedLayers} adapter layers with ${adapterLayers.totalParams} parameters`);
  return adapterLayers;
}

/**
 * Run inference on a model session
 * @param {Object} session - ONNX session
 * @param {Object} inputs - Input tensors
 * @returns {Promise<Object>} Output tensors
 */
export async function runInference(session, inputs) {
  try {
    const feeds = {};
    
    // Convert inputs to ONNX tensors
    for (const [name, data] of Object.entries(inputs)) {
      if (Array.isArray(data)) {
        feeds[name] = new ort.Tensor('int64', new BigInt64Array(data.map(x => BigInt(x))), [data.length]);
      } else if (data instanceof Float32Array) {
        feeds[name] = new ort.Tensor('float32', data, [data.length]);
      } else {
        feeds[name] = data; // Assume it's already a tensor
      }
    }

    // Run inference
    const results = await session.run(feeds);
    
    return results;
  } catch (error) {
    console.error('Inference failed:', error);
    throw new Error(`Inference failed: ${error.message}`);
  }
}

/**
 * Estimate model memory usage
 * @param {Object} modelInfo - Model information
 * @param {Object} config - Training configuration
 * @returns {Object} Memory estimates
 */
export function estimateMemoryUsage(modelInfo, config) {
  const { batchSize = 1, sequenceLength = 512 } = config;
  
  // Rough estimates based on typical model sizes
  const baseModelMemoryMB = 2000; // ~2GB for typical small models
  const activationMemoryMB = (batchSize * sequenceLength * 768 * 4) / (1024 * 1024); // Float32
  const adapterMemoryMB = 50; // Typical adapter size
  
  const totalMemoryMB = baseModelMemoryMB + activationMemoryMB + adapterMemoryMB;
  
  return {
    baseModel: baseModelMemoryMB,
    activations: activationMemoryMB,
    adapters: adapterMemoryMB,
    total: totalMemoryMB,
    totalGB: totalMemoryMB / 1024
  };
}

/**
 * Dispose of ONNX session and free memory
 * @param {Object} sessionData - Session data to dispose
 */
export async function disposeSession(sessionData) {
  try {
    if (sessionData.session) {
      await sessionData.session.release();
    }
    if (sessionData.baseSession) {
      await sessionData.baseSession.release();
    }
    console.log('Session disposed successfully');
  } catch (error) {
    console.error('Session disposal failed:', error);
  }
}

/**
 * Get supported execution providers
 * @returns {Promise<Array>} List of available execution providers
 */
export async function getSupportedProviders() {
  const providers = [];
  
  // Check WebGPU support
  if (navigator.gpu) {
    try {
      const adapter = await navigator.gpu.requestAdapter();
      if (adapter) {
        providers.push('webgpu');
      }
    } catch (error) {
      console.warn('WebGPU check failed:', error);
    }
  }
  
  // WASM is always available
  providers.push('wasm');
  
  return providers;
}

// Export default object for easier imports
export default {
  initializeONNX,
  loadModel,
  createTrainingSession,
  runInference,
  estimateMemoryUsage,
  disposeSession,
  getSupportedProviders
};