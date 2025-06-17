/**
 * ONNX Runtime Web Session Management for LoRA Lab
 * Handles WebGPU configuration and training session initialization
 */

// NOTE: The direct import of 'onnxruntime-web' has been removed to avoid
// conflicts with Transformers.js's internal ONNX Runtime management.
// The training engine will need to be adapted to work with this change,
// potentially by receiving the 'ort' object from a single, centralized
// initialization point.

// Default WebGPU execution provider options
export const webgpuOptions = {
  deviceType: 'gpu',
  powerPreference: 'high-performance',
};

/**
 * Create a new ONNX Inference Session for training or inference.
 * This function assumes initializeONNX has been called.
 * @param {ArrayBuffer} modelData - The ONNX model data as an ArrayBuffer.
 * @param {Object} options - Session configuration options.
 * @returns {Promise<ort.InferenceSession>} The created inference session.
 */
export async function createONNXSession(modelData, options = {}) {
  const {
    enableProfiling = false,
    optimizeModel = true,
    executionMode = 'sequential'
  } = options;

  try {
    const sessionOptions = {
      enableProfiling,
      graphOptimizationLevel: optimizeModel ? 'all' : 'disabled',
      executionMode,
    };
    
    const session = await ort.InferenceSession.create(modelData, sessionOptions);
    console.log('ONNX Inference Session created successfully.');
    return session;

  } catch (error) {
    console.error('ONNX Session creation failed:', error);
    throw new Error(`Failed to create ONNX session: ${error.message}`);
  }
}

/**
 * Get a list of supported execution providers on the current device.
 * @returns {Promise<string[]>} A list of available provider names.
 */
export async function getSupportedProviders() {
  const providers = [];
  if (navigator.gpu) {
    try {
      if (await navigator.gpu.requestAdapter()) {
        providers.push('webgpu');
      }
    } catch (e) {
      console.warn('WebGPU check failed:', e);
    }
  }
  providers.push('wasm');
  return providers;
}

/**
 * Run inference on an ONNX session
 * @param {ort.InferenceSession} session - The ONNX session
 * @param {Object} inputs - Input tensors
 * @returns {Promise<Object>} Output tensors
 */
export async function runInference(session, inputs) {
  try {
    return await session.run(inputs);
  } catch (error) {
    console.error('ONNX inference failed:', error);
    throw new Error(`Failed to run inference: ${error.message}`);
  }
}

/**
 * Dispose of an ONNX session
 * @param {ort.InferenceSession} session - The ONNX session to dispose
 */
export async function disposeSession(session) {
  if (session) {
    try {
      await session.release();
      console.log('ONNX session disposed');
    } catch (error) {
      console.error('Failed to dispose ONNX session:', error);
    }
  }
}