/**
 * Model Manager for LoRA Lab
 * Handles ONNX model loading, validation, and management
 */

import { InferenceSession, Tensor } from 'onnxruntime-web';

/**
 * Model Manager Class
 * Manages model loading, validation, and lifecycle
 */
export class ModelManager {
  constructor() {
    this.loadedModels = new Map();
    this.modelInfo = new Map();
    this.loadingPromises = new Map();
    this.maxCacheSize = 3; // Maximum number of models to keep in memory
    
    // Initialize ONNX Runtime Web
    this.initializeONNXRuntime();
  }
  
  /**
   * Initialize ONNX Runtime with optimal settings
   */
  async initializeONNXRuntime() {
    try {
      // Configure ONNX Runtime for WebGPU if available
      const providers = [];
      
      // Check for WebGPU support
      if (navigator.gpu) {
        try {
          const adapter = await navigator.gpu.requestAdapter();
          if (adapter) {
            providers.push('webgpu');
            console.log('WebGPU provider enabled for ONNX Runtime');
          }
        } catch (error) {
          console.warn('WebGPU not available:', error);
        }
      }
      
      // Fallback to WebAssembly
      providers.push('wasm');
      
      // Set execution providers
      InferenceSession.create.executionProviders = providers;
      
      console.log('ONNX Runtime initialized with providers:', providers);
      
    } catch (error) {
      console.error('ONNX Runtime initialization failed:', error);
      throw new Error(`Failed to initialize ONNX Runtime: ${error.message}`);
    }
  }
  
  /**
   * Load model from URL or File
   * @param {string|File|ArrayBuffer} modelSource - Model source
   * @param {Object} options - Loading options
   * @returns {Promise<Object>} Model info and session
   */
  async loadModel(modelSource, options = {}) {
    const {
      modelName = 'model',
      enableProfiling = false,
      executionMode = 'sequential',
      graphOptimizationLevel = 'basic'
    } = options;
    
    // Check if model is already loaded
    if (this.loadedModels.has(modelName)) {
      console.log('Model already loaded:', modelName);
      return this.getModelInfo(modelName);
    }
    
    // Check if model is currently loading
    if (this.loadingPromises.has(modelName)) {
      console.log('Model already loading, waiting...:', modelName);
      return await this.loadingPromises.get(modelName);
    }
    
    // Start loading process
    const loadingPromise = this._loadModelInternal(modelSource, {
      modelName,
      enableProfiling,
      executionMode,
      graphOptimizationLevel,
      externalFileToURI: options.externalFileToURI || undefined,
      hfAccessToken: options.hfAccessToken ||
                      (typeof window !== 'undefined' && (
                        window.HF_ACCESS_TOKEN ||
                        localStorage.getItem('HF_ACCESS_TOKEN')
                      ))
    });
    
    this.loadingPromises.set(modelName, loadingPromise);
    
    try {
      const result = await loadingPromise;
      this.loadingPromises.delete(modelName);
      return result;
    } catch (error) {
      this.loadingPromises.delete(modelName);
      throw error;
    }
  }
  
  /**
   * Internal model loading implementation
   */
  async _loadModelInternal(modelSource, options) {
    const { modelName, enableProfiling, executionMode, graphOptimizationLevel, hfAccessToken } = options;
    
    try {
      console.log('Loading ONNX model:', modelName);
      
      // -------------------------------------------------------------------
      // Step 1: Prepare optional Hugging Face access token if provided
      // -------------------------------------------------------------------
      const hfToken = hfAccessToken ||
                      (typeof window !== 'undefined' && (
                        window.HF_ACCESS_TOKEN ||
                        localStorage.getItem('HF_ACCESS_TOKEN')
                      ));

      const fetchHeaders = hfToken ? { 'Authorization': `Bearer ${hfToken}` } : undefined;

      // Convert source to ArrayBuffer if needed
      let modelData;
      // This needs to be mutable as it may be overridden for URL sources
      let externalFileToURI = options.externalFileToURI;

      if (typeof modelSource === 'string') {
        // URL – attempt download (optionally with Authorization header)

        const tryFetch = async (withAuth = false) => {
          const init = withAuth && fetchHeaders ? { headers: fetchHeaders } : undefined;
          const resp = await fetch(modelSource, init);
          return resp;
        };

        let response = await tryFetch(!!hfToken);

        // If unauthorised and we haven't tried auth header yet, retry with token (if present)
        if (response.status === 401 && !hfToken) {
          throw new Error('Received 401 Unauthorized while downloading model. The model repository may require accepting a license or using a HuggingFace access token. Please set window.HF_ACCESS_TOKEN or localStorage["HF_ACCESS_TOKEN"].');
        }

        if (response.status === 401 && hfToken) {
          // Token invalid or lacks permission
          throw new Error('Provided HuggingFace access token was rejected (401 Unauthorized). Make sure the token has permission for this repository.');
        }

        if (!response.ok) {
          throw new Error(`Failed to fetch model: ${response.status} ${response.statusText}`);
        }

        modelData = await response.arrayBuffer();
        
        // Create a custom fetcher for external data that includes auth headers
        const modelPath = new URL(modelSource);
        const basePath = modelPath.href.substring(0, modelPath.href.lastIndexOf('/') + 1);
        
        externalFileToURI = async (file) => {
          const externalDataUrl = new URL(file, basePath).toString();
          console.log(`Fetching authenticated external ONNX data: ${externalDataUrl}`);
          
          const externalResponse = await fetch(externalDataUrl, {
            headers: fetchHeaders
          });
          
          if (!externalResponse.ok) {
            throw new Error(`Failed to fetch external ONNX data "${file}": ${externalResponse.status} ${externalResponse.statusText}`);
          }
          
          return externalResponse.arrayBuffer();
        };

      } else if (modelSource instanceof File) {
        // File object
        modelData = await modelSource.arrayBuffer();
      } else if (modelSource instanceof ArrayBuffer) {
        // ArrayBuffer
        modelData = modelSource;
      } else {
        throw new Error('Unsupported model source type');
      }
      
      // Validate model data
      await this._validateONNXModel(modelData);
      
      // Configure session options
      const sessionOptions = {
        executionProviders: ['webgpu', 'wasm'],
        enableProfiling,
        executionMode,
        graphOptimizationLevel,
        externalFileToURI: externalFileToURI || undefined
      };
      
      // Create inference session
      const session = await InferenceSession.create(modelData, sessionOptions);
      
      // Extract model information
      const modelInfo = await this._extractModelInfo(session, modelData);
      
      // Store in cache
      this._addToCache(modelName, session, modelInfo);
      
      console.log('Model loaded successfully:', modelName, modelInfo);
      
      return {
        name: modelName,
        session,
        ...modelInfo
      };
      
    } catch (error) {
      console.error('Model loading failed:', error);
      throw new Error(`Failed to load model ${modelName}: ${error.message}`);
    }
  }
  
  /**
   * Validate ONNX model format
   */
  async _validateONNXModel(modelData) {
    // Basic validation - check for ONNX magic bytes
    const view = new Uint8Array(modelData.slice(0, 8));
    
    // ONNX models typically start with protobuf magic or specific patterns
    // This is a basic check - full validation would require protobuf parsing
    if (modelData.byteLength < 100) {
      throw new Error('Model file too small to be valid ONNX model');
    }
    
    // Check for common ONNX file patterns
    const hasValidHeader = (
      view[0] === 0x08 || // Protobuf varint
      (view[0] === 0x50 && view[1] === 0x4B) || // ZIP/ONNX package
      (view[0] === 0x1A && view[1] >= 0x01) // Protobuf message
    );
    
    if (!hasValidHeader) {
      throw new Error('Invalid ONNX model format');
    }
  }
  
  /**
   * Extract model information from session
   */
  async _extractModelInfo(session, modelData) {
    try {
      const inputs = [];
      const outputs = [];
      
      // Extract input information safely
      for (const name of session.inputNames) {
        const metadata = session.inputMetadata[name];
        if (metadata) {
          inputs.push({
            name,
            type: metadata.type,
            dims: metadata.dims,
            shape: metadata.dims
          });
        } else {
          inputs.push({ name, type: 'unknown', dims: [], shape: [] });
        }
      }
      
      // Extract output information safely
      for (const name of session.outputNames) {
        const metadata = session.outputMetadata[name];
        if (metadata) {
          outputs.push({
            name,
            type: metadata.type,
            dims: metadata.dims,
            shape: metadata.dims
          });
        } else {
          outputs.push({ name, type: 'unknown', dims: [], shape: [] });
        }
      }
      
      // Calculate model size
      const sizeInBytes = modelData.byteLength;
      const sizeInMB = (sizeInBytes / (1024 * 1024)).toFixed(2);
      
      // Estimate parameter count (rough approximation)
      const estimatedParams = Math.floor(sizeInBytes / 4); // Assuming FP32
      
      // Detect model type based on structure
      const modelType = this._detectModelType(inputs, outputs);
      
      return {
        inputs,
        outputs,
        sizeInBytes,
        sizeInMB,
        estimatedParams,
        modelType,
        loadedAt: new Date().toISOString()
      };
      
    } catch (error) {
      console.error('Model info extraction failed:', error);
      return {
        inputs: [],
        outputs: [],
        sizeInBytes: modelData.byteLength,
        sizeInMB: (modelData.byteLength / (1024 * 1024)).toFixed(2),
        estimatedParams: 0,
        modelType: 'unknown',
        loadedAt: new Date().toISOString(),
        error: error.message
      };
    }
  }
  
  /**
   * Detect model type based on input/output structure
   */
  _detectModelType(inputs, outputs) {
    // Language model patterns
    if (inputs.some(i => i.name.includes('input_ids')) && 
        outputs.some(o => o.name.includes('logits'))) {
      return 'language_model';
    }
    
    // Embedding model patterns
    if (outputs.some(o => o.name.includes('embeddings') || o.name.includes('pooler'))) {
      return 'embedding_model';
    }
    
    // Classifier patterns
    if (outputs.length === 1 && outputs[0].dims && outputs[0].dims.length === 2) {
      return 'classifier';
    }
    
    return 'unknown';
  }
  
  /**
   * Add model to cache with size management
   */
  _addToCache(modelName, session, modelInfo) {
    // Check cache size and evict if needed
    if (this.loadedModels.size >= this.maxCacheSize) {
      const oldestModel = this.loadedModels.keys().next().value;
      this.unloadModel(oldestModel);
    }
    
    this.loadedModels.set(modelName, session);
    this.modelInfo.set(modelName, modelInfo);
  }
  
  /**
   * Get model information
   * @param {string} modelName - Model name
   * @returns {Object} Model information
   */
  getModelInfo(modelName) {
    const session = this.loadedModels.get(modelName);
    const info = this.modelInfo.get(modelName);
    
    if (!session || !info) {
      throw new Error(`Model not loaded: ${modelName}`);
    }
    
    return {
      name: modelName,
      session,
      ...info,
      isLoaded: true
    };
  }
  
  /**
   * Check if model is loaded
   * @param {string} modelName - Model name
   * @returns {boolean} Is loaded
   */
  isModelLoaded(modelName) {
    return this.loadedModels.has(modelName);
  }
  
  /**
   * Unload model from memory
   * @param {string} modelName - Model name
   */
  async unloadModel(modelName) {
    try {
      const session = this.loadedModels.get(modelName);
      if (session) {
        await session.release();
        this.loadedModels.delete(modelName);
        this.modelInfo.delete(modelName);
        console.log('Model unloaded:', modelName);
      }
    } catch (error) {
      console.error('Model unload failed:', error);
    }
  }
  
  /**
   * List all loaded models
   * @returns {Array} List of model names and info
   */
  listLoadedModels() {
    return Array.from(this.loadedModels.keys()).map(name => ({
      name,
      ...this.modelInfo.get(name)
    }));
  }
  
  /**
   * Run inference on loaded model
   * @param {string} modelName - Model name
   * @param {Object} inputs - Input tensors
   * @returns {Promise<Object>} Output tensors
   */
  async runInference(modelName, inputs) {
    try {
      const session = this.loadedModels.get(modelName);
      if (!session) {
        throw new Error(`Model not loaded: ${modelName}`);
      }
      
      // Convert inputs to ONNX tensors with correct data types
      const onnxInputs = {};
      for (const [name, data] of Object.entries(inputs)) {
        if (data instanceof Tensor) {
          onnxInputs[name] = data;
        } else {
          // Determine tensor type based on input name
          let tensorType = 'float32';
          if (name.includes('input_ids') || name.includes('mask')) {
            tensorType = 'int64';
          }
          
          // Create tensor with the appropriate type
          const tensorData = tensorType === 'int64' ? 
            new BigInt64Array(data.map(BigInt)) : 
            new Float32Array(data);
            
          const dims = data.dims || [1, data.length];

          onnxInputs[name] = new Tensor(tensorType, tensorData, dims);
        }
      }
      
      // Run inference
      const outputs = await session.run(onnxInputs);
      
      return outputs;
      
    } catch (error) {
      console.error('Inference failed:', error);
      throw new Error(`Inference failed for ${modelName}: ${error.message}`);
    }
  }
  
  /**
   * Get memory usage statistics
   * @returns {Object} Memory usage info
   */
  getMemoryUsage() {
    const models = this.listLoadedModels();
    const totalSize = models.reduce((sum, model) => sum + model.sizeInBytes, 0);
    
    return {
      loadedModels: models.length,
      totalSizeBytes: totalSize,
      totalSizeMB: (totalSize / (1024 * 1024)).toFixed(2),
      models: models.map(m => ({
        name: m.name,
        sizeMB: m.sizeInMB,
        type: m.modelType
      }))
    };
  }
  
  /**
   * Clear all loaded models
   */
  async clearAll() {
    const modelNames = Array.from(this.loadedModels.keys());
    for (const name of modelNames) {
      await this.unloadModel(name);
    }
    console.log('All models cleared from memory');
  }
  
  /**
   * Preload common models for the application
   */
  async preloadModels(modelConfigs) {
    const results = [];
    
    for (const config of modelConfigs) {
      try {
        const result = await this.loadModel(config.url || config.file, {
          modelName: config.name,
          ...config.options
        });
        results.push({ success: true, model: result });
      } catch (error) {
        console.error(`Failed to preload model ${config.name}:`, error);
        results.push({ success: false, error: error.message, name: config.name });
      }
    }
    
    return results;
  }
}

/**
 * Create model loading progress tracker
 * @param {Function} onProgress - Progress callback
 * @returns {Object} Progress tracker
 */
export function createLoadingProgress(onProgress) {
  let totalBytes = 0;
  let loadedBytes = 0;
  
  return {
    setTotal: (bytes) => {
      totalBytes = bytes;
      onProgress({ loaded: loadedBytes, total: totalBytes, percentage: 0 });
    },
    
    update: (bytes) => {
      loadedBytes = bytes;
      const percentage = totalBytes > 0 ? (loadedBytes / totalBytes) * 100 : 0;
      onProgress({ loaded: loadedBytes, total: totalBytes, percentage });
    },
    
    complete: () => {
      onProgress({ loaded: totalBytes, total: totalBytes, percentage: 100 });
    }
  };
}

/**
 * Model validation utilities
 */
export const ModelValidator = {
  /**
   * Validate model for LoRA training compatibility
   */
  validateForLoRA: (modelInfo) => {
    const validation = {
      isValid: false,
      errors: [],
      warnings: [],
      recommendations: []
    };
    
    // Check if it's a language model
    if (modelInfo.modelType !== 'language_model') {
      validation.warnings.push('Model may not be a language model - LoRA works best with transformer models');
    }
    
    // Check input structure
    const hasInputIds = modelInfo.inputs.some(input => 
      input.name.includes('input_ids') || input.name.includes('tokens')
    );
    
    if (!hasInputIds) {
      validation.errors.push('Model missing input_ids - required for language model training');
    }
    
    // Check output structure
    const hasLogits = modelInfo.outputs.some(output => 
      output.name.includes('logits') || output.name.includes('predictions')
    );
    
    if (!hasLogits) {
      validation.errors.push('Model missing logits output - required for training');
    }
    
    // Size recommendations
    if (modelInfo.sizeInBytes > 2 * 1024 * 1024 * 1024) { // 2GB
      validation.warnings.push('Large model (>2GB) may cause memory issues in browser');
    }
    
    if (modelInfo.sizeInBytes < 10 * 1024 * 1024) { // 10MB
      validation.warnings.push('Very small model (<10MB) may not be a full language model');
    }
    
    // Parameter count estimation
    if (modelInfo.estimatedParams < 100000) {
      validation.warnings.push('Low parameter count - model may be too simple for effective LoRA training');
    }
    
    validation.isValid = validation.errors.length === 0;
    
    if (validation.isValid) {
      validation.recommendations.push('Model appears compatible with LoRA training');
      
      // Suggest optimal LoRA rank based on model size
      let suggestedRank = 4;
      if (modelInfo.estimatedParams > 1000000) suggestedRank = 8;
      if (modelInfo.estimatedParams > 10000000) suggestedRank = 16;
      if (modelInfo.estimatedParams > 100000000) suggestedRank = 32;
      
      validation.recommendations.push(`Suggested LoRA rank: ${suggestedRank}`);
    }
    
    return validation;
  }
};

/**
 * Default model configurations for common models
 */
export const DefaultModels = {
  'gemma-2b-it': {
    name: 'gemma-2b-it',
    repo_id: 'Xenova/gemma-2b-it',
    url: 'https://huggingface.co/Xenova/gemma-2b-it/resolve/main/onnx/model.onnx?download=true',
    type: 'language_model',
    description: 'Gemma 2B (Web) - Instruction-tuned model from Xenova.',
    size: '1.4GB',
    recommendedRank: 16
  },

  'phi-2': {
    name: 'phi-2',
    repo_id: 'Xenova/phi-2',
    url: 'https://huggingface.co/Xenova/phi-2/resolve/main/onnx/model.onnx?download=true',
    type: 'language_model',
    description: 'Microsoft Phi-2 (Web) - High-quality small language model.',
    size: '1.5GB',
    recommendedRank: 16
  },

  'tinyllama-1.1b': {
    name: 'tinyllama-1.1b',
    repo_id: 'Xenova/TinyLlama-1.1B-Chat-v1.0',
    url: 'https://huggingface.co/Xenova/TinyLlama-1.1B-Chat-v1.0/resolve/main/onnx/model_quantized.onnx?download=true',
    type: 'language_model',
    description: 'TinyLlama 1.1B (Web) - Ultra-compact chat model.',
    size: '0.6GB',
    recommendedRank: 8
  }
};

// Create singleton instance
export const modelManager = new ModelManager();

export default modelManager;