/**
 * Model Manager for LoRA Lab
 * 
 * This refactored version acts as a controller that communicates with a 
 * dedicated web worker for model loading and text generation. This prevents
 * the main UI thread from being blocked by heavy computations.
 */

// A map to hold the state of each model being managed.
const modelStates = new Map();
// The single generation worker instance.
let worker = null;

// Callbacks for handling messages from the worker
const onMessageCallbacks = new Map();

/**
 * Initializes the generation worker and sets up the message listener.
 * This is called automatically on the first interaction.
 */
function initializeWorker() {
  if (worker) return;

  worker = new Worker(new URL('../workers/generation.worker.js', import.meta.url), {
    type: 'module'
  });

  worker.addEventListener('message', (e) => {
    const { status, model_id } = e.data;
    if (!model_id) return;
    
    const state = modelStates.get(model_id);
    if (!state) return;

    // Handle progress, readiness, and errors during loading
    if (status === 'loading' || status === 'progress' || status === 'download') {
      state.status = 'loading';
      state.progress = e.data;
      if (state.onProgress) state.onProgress(e.data);
    } else if (status === 'ready') {
      state.status = 'ready';
      state.isLoaded = true;
      if(state.resolve) state.resolve(state);
    } else if (status === 'error') {
      state.status = 'error';
      state.error = e.data.error;
      if(state.reject) state.reject(new Error(e.data.error));
    }

    // Handle generation updates
    const generationCallback = onMessageCallbacks.get(model_id);
    if (generationCallback) {
      generationCallback(e.data);
    }
  });
}

/**
 * Model Manager Class
 * Manages model states and communicates with the generation worker.
 */
export class ModelManager {
  constructor() {
    initializeWorker();
  }

  /**
   * Requests the worker to load a model.
   * @param {string} modelId - Model ID on Hugging Face.
   * @param {Object} options - Loading options, including onProgress callback.
   * @returns {Promise<Object>} A promise that resolves with model info when loaded.
   */
  async loadModel(modelId, options = {}) {
    if (modelStates.has(modelId) && modelStates.get(modelId).status === 'ready') {
      console.log('Model already loaded:', modelId);
      return this.getModelInfo(modelId);
    }

    if (modelStates.has(modelId) && modelStates.get(modelId).status === 'loading') {
       console.log('Model already loading, waiting...:', modelId);
       return modelStates.get(modelId).promise;
    }

    const promise = new Promise((resolve, reject) => {
      modelStates.set(modelId, {
        modelId,
        status: 'loading',
        isLoaded: false,
        onProgress: options.onProgress,
        resolve,
        reject
      });
    });
    
    modelStates.get(modelId).promise = promise;

    worker.postMessage({ type: 'load', data: { model_id: modelId } });
    
    return promise;
  }

  /**
   * Generates text using a loaded model via the worker.
   * @param {string} modelId - Model ID.
   * @param {string} prompt - Input prompt.
   * @param {Object} options - Generation options, including onToken callback for streaming.
   * @returns {Promise<string>} A promise that resolves with the final generated text.
   */
  async generate(modelId, prompt, options = {}) {
     const { onToken } = options;
     const messages = [{ role: 'user', content: prompt }];
     
     return new Promise((resolve, reject) => {
        const callback = (message) => {
            if (message.status === 'update' && onToken) {
                onToken(message.output);
            } else if (message.status === 'complete') {
                onMessageCallbacks.delete(modelId);
                resolve(message.output);
            } else if (message.status === 'error') {
                onMessageCallbacks.delete(modelId);
                reject(new Error(message.error));
            }
        };
        onMessageCallbacks.set(modelId, callback);

        worker.postMessage({ type: 'generate', data: { model_id: modelId, messages } });
     });
  }

  interrupt(modelId) {
      worker.postMessage({ type: 'interrupt', data: { model_id: modelId } });
  }

  getModelInfo(modelId) {
    return this.isModelLoaded(modelId) ? { ...modelStates.get(modelId) } : null;
  }

  isModelLoaded(modelId) {
    return modelStates.has(modelId) && modelStates.get(modelId).isLoaded;
  }

  listLoadedModels() {
    return Array.from(modelStates.values())
      .filter(s => s.isLoaded)
      .map(s => this.getModelInfo(s.modelId));
  }

  async unloadModel(modelId) {
      // In this worker-based architecture, we might not need to explicitly unload models
      // as the worker manages memory. We can just remove it from our state.
      if (modelStates.has(modelId)) {
          modelStates.delete(modelId);
          console.log('Model state removed:', modelId);
      }
  }

  async clearAll() {
    modelStates.clear();
    // We could also terminate and re-initialize the worker if needed.
    if(worker) {
        worker.terminate();
        initializeWorker();
    }
    console.log('All model states cleared and worker reset.');
  }
}

/**
 * Recommended models for use with Transformers.js
 * These are models that have been tested and work well in the browser
 */
export const RecommendedModels = {
  // Phi-3.5 Mini Instruct
  'phi3.5-mini-instruct': {
    modelId: 'onnx-community/Phi-3.5-mini-instruct-onnx-web',
    name: 'Phi-3.5 Mini Instruct',
    description: 'A powerful, lightweight model by Microsoft, ready for on-device deployment.',
    size: '~2.5GB',
    quantization: 'q4f16',
    supportsChat: true,
    requiresWebGPU: true
  },

  // Phi-3 models (Microsoft)
  'phi3-mini': {
    modelId: 'Xenova/Phi-3-mini-4k-instruct',
    name: 'Phi-3 Mini 4K',
    description: 'Compact but capable model from Microsoft',
    size: '~2.5GB',
    quantization: 'q4',
    supportsChat: true
  },
  
  // TinyLlama models
  'tinyllama': {
    modelId: 'Xenova/TinyLlama-1.1B-Chat-v1.0',
    name: 'TinyLlama 1.1B',
    description: 'Small but efficient chat model',
    size: '~600MB',
    quantization: 'q4',
    supportsChat: true
  },

  // Qwen models
  'qwen-0.5b': {
    modelId: 'Xenova/Qwen1.5-0.5B-Chat',
    name: 'Qwen 1.5 0.5B',
    description: 'Efficient small chat model',
    size: '~350MB',
    quantization: 'q4',
    supportsChat: true
  },
};

// Create singleton instance
export const modelManager = new ModelManager();

export default modelManager;