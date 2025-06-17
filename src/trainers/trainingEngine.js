/**
 * Training Engine for LoRA Lab
 * Main orchestrator for LoRA training using ONNX Runtime Web and WebGPU
 */

import { modelManager } from '../utils/modelManager.js';
import { createONNXSession } from './onnxSession.js';
import { RankScheduler } from './rankScheduler.js';
import { exportAdapter, downloadAdapter } from '../utils/safetensorExport.js';

/**
 * Training Engine Class
 * Manages the complete LoRA training pipeline
 */
export class TrainingEngine {
  constructor() {
    this.isTraining = false;
    this.isPaused = false;
    this.currentModel = null;
    this.currentDataset = null;
    this.trainingWorker = null;
    this.onnxSession = null;
    this.rankScheduler = null;
    
    // Training state
    this.trainingState = {
      currentStep: 0,
      totalSteps: 0,
      epoch: 0,
      totalEpochs: 0,
      currentLoss: 0,
      averageLoss: 0,
      learningRate: 0,
      throughput: 0,
      memoryUsage: 0,
      eta: 0,
      startTime: null,
      lastUpdate: null
    };
    
    // LoRA configuration
    this.loraConfig = {
      rank: 4,
      alpha: 8,
      dropout: 0.1,
      targetModules: ['q_proj', 'k_proj', 'v_proj', 'o_proj'],
      scaling: 2.0
    };
    
    // Training configuration
    this.trainingConfig = {
      learningRate: 2e-4,
      batchSize: 1,
      gradientAccumulationSteps: 4,
      maxSteps: 1000,
      warmupSteps: 100,
      saveEvery: 100,
      validationEvery: 50,
      maxGradNorm: 1.0,
      weightDecay: 0.01,
      optimizer: 'adamw'
    };
    
    // Event listeners
    this.eventListeners = new Map();
    
    // Initialize training components
    this.initializeComponents();
  }
  
  /**
   * Initialize training components
   */
  async initializeComponents() {
    try {
      // Initialize rank scheduler
      this.rankScheduler = new RankScheduler();
      
      console.log('Training engine initialized');
      
    } catch (error) {
      console.error('Training engine initialization failed:', error);
      throw error;
    }
  }
  
  /**
   * Load model for training
   * @param {string|File} modelSource - Model source
   * @param {Object} options - Loading options
   */
  async loadModel(modelSource, options = {}) {
    try {
      console.log('Loading model for training...');
      
      // Load model through model manager
      const modelInfo = await modelManager.loadModel(modelSource, {
        modelName: options.modelName || 'training_model',
        ...options
      });
      
      // Validate model for LoRA training
      const validation = await this.validateModelForTraining(modelInfo);
      if (!validation.isValid) {
        throw new Error(`Model validation failed: ${validation.errors.join(', ')}`);
      }
      
      // Create ONNX session for training
      this.onnxSession = await createONNXSession(modelInfo.session, {
        enableTraining: true,
        loraConfig: this.loraConfig
      });
      
      this.currentModel = modelInfo;
      
      // Update LoRA config based on model
      if (validation.recommendations.length > 0) {
        const rankRec = validation.recommendations.find(r => r.includes('rank'));
        if (rankRec) {
          const suggestedRank = parseInt(rankRec.match(/\d+/)[0]);
          this.loraConfig.rank = suggestedRank;
          this.loraConfig.scaling = this.loraConfig.alpha / this.loraConfig.rank;
        }
      }
      
      this.emit('modelLoaded', { model: modelInfo, validation });
      
      console.log('Model loaded for training:', modelInfo.name);
      return modelInfo;
      
    } catch (error) {
      console.error('Model loading failed:', error);
      this.emit('error', { type: 'modelLoad', error });
      throw error;
    }
  }
  
  /**
   * Load dataset for training
   * @param {Array|Object} dataset - Training dataset
   * @param {Object} options - Dataset options
   */
  async loadDataset(dataset, options = {}) {
    try {
      console.log('Loading dataset for training...');
      
      // Validate dataset
      const validation = this.validateDataset(dataset);
      if (!validation.isValid) {
        throw new Error(`Dataset validation failed: ${validation.errors.join(', ')}`);
      }
      
      this.currentDataset = {
        data: dataset,
        size: Array.isArray(dataset) ? dataset.length : Object.keys(dataset).length,
        tokenized: false,
        ...options
      };
      
      // Calculate total training steps
      const stepsPerEpoch = Math.ceil(this.currentDataset.size / this.trainingConfig.batchSize);
      this.trainingState.totalSteps = this.trainingConfig.maxSteps || 
        (stepsPerEpoch * (this.trainingConfig.epochs || 3));
      
      this.emit('datasetLoaded', { 
        dataset: this.currentDataset,
        estimatedSteps: this.trainingState.totalSteps
      });
      
      console.log('Dataset loaded:', {
        size: this.currentDataset.size,
        estimatedSteps: this.trainingState.totalSteps
      });
      
      return this.currentDataset;
      
    } catch (error) {
      console.error('Dataset loading failed:', error);
      this.emit('error', { type: 'datasetLoad', error });
      throw error;
    }
  }
  
  /**
   * Start training process
   * @param {Object} config - Training configuration override
   */
  async startTraining(config = {}) {
    try {
      if (this.isTraining) {
        throw new Error('Training already in progress');
      }
      
      if (!this.currentModel) {
        throw new Error('No model loaded');
      }
      
      if (!this.currentDataset) {
        throw new Error('No dataset loaded');
      }
      
      console.log('Starting LoRA training...');
      
      // Merge configuration
      this.trainingConfig = { ...this.trainingConfig, ...config };
      
      // Initialize training state
      this.trainingState = {
        ...this.trainingState,
        currentStep: 0,
        epoch: 0,
        startTime: Date.now(),
        lastUpdate: Date.now()
      };
      
      // Initialize rank scheduler
      await this.rankScheduler.initialize({
        initialRank: this.loraConfig.rank,
        targetModules: this.loraConfig.targetModules,
        strategy: config.rankStrategy || 'adaptive'
      });
      
      // Create training worker
      await this.createTrainingWorker();
      
      // Start training
      this.isTraining = true;
      this.isPaused = false;
      
      // Send training configuration to worker
      this.trainingWorker.postMessage({
        type: 'START_TRAINING',
        config: {
          lora: this.loraConfig,
          training: this.trainingConfig,
          model: {
            name: this.currentModel.name,
            inputs: this.currentModel.inputs,
            outputs: this.currentModel.outputs
          },
          dataset: {
            size: this.currentDataset.size,
            batchSize: this.trainingConfig.batchSize
          }
        }
      });
      
      this.emit('trainingStarted', {
        config: this.trainingConfig,
        loraConfig: this.loraConfig,
        estimatedSteps: this.trainingState.totalSteps
      });
      
      console.log('Training started with config:', {
        rank: this.loraConfig.rank,
        learningRate: this.trainingConfig.learningRate,
        maxSteps: this.trainingConfig.maxSteps
      });
      
    } catch (error) {
      console.error('Training start failed:', error);
      this.isTraining = false;
      this.emit('error', { type: 'trainingStart', error });
      throw error;
    }
  }
  
  /**
   * Pause training
   */
  async pauseTraining() {
    if (!this.isTraining || this.isPaused) {
      return;
    }
    
    this.isPaused = true;
    
    if (this.trainingWorker) {
      this.trainingWorker.postMessage({ type: 'PAUSE_TRAINING' });
    }
    
    this.emit('trainingPaused', { step: this.trainingState.currentStep });
    console.log('Training paused');
  }
  
  /**
   * Resume training
   */
  async resumeTraining() {
    if (!this.isTraining || !this.isPaused) {
      return;
    }
    
    this.isPaused = false;
    
    if (this.trainingWorker) {
      this.trainingWorker.postMessage({ type: 'RESUME_TRAINING' });
    }
    
    this.emit('trainingResumed', { step: this.trainingState.currentStep });
    console.log('Training resumed');
  }
  
  /**
   * Stop training
   */
  async stopTraining() {
    if (!this.isTraining) {
      return;
    }
    
    console.log('Stopping training...');
    
    this.isTraining = false;
    this.isPaused = false;
    
    if (this.trainingWorker) {
      this.trainingWorker.postMessage({ type: 'STOP_TRAINING' });
      this.trainingWorker.terminate();
      this.trainingWorker = null;
    }
    
    this.emit('trainingStopped', { 
      finalStep: this.trainingState.currentStep,
      finalLoss: this.trainingState.currentLoss
    });
    
    console.log('Training stopped');
  }
  
  /**
   * Create and configure training worker
   */
  async createTrainingWorker() {
    if (this.trainingWorker) {
      this.trainingWorker.terminate();
    }
    
    // Create worker from the training worker file
    this.trainingWorker = new Worker('/src/workers/training.worker.js', { type: 'module' });
    
    // Set up message handling
    this.trainingWorker.onmessage = (event) => {
      this.handleWorkerMessage(event.data);
    };
    
    this.trainingWorker.onerror = (error) => {
      console.error('Training worker error:', error);
      this.emit('error', { type: 'worker', error });
    };
    
    // Initialize worker with ONNX session
    this.trainingWorker.postMessage({
      type: 'INITIALIZE',
      modelInfo: this.currentModel,
      loraConfig: this.loraConfig
    });
  }
  
  /**
   * Handle messages from training worker
   */
  handleWorkerMessage(message) {
    switch (message.type) {
      case 'TRAINING_PROGRESS':
        this.updateTrainingState(message.data);
        break;
        
      case 'TRAINING_COMPLETE':
        this.handleTrainingComplete(message.data);
        break;
        
      case 'TRAINING_ERROR':
        this.handleTrainingError(message.data);
        break;
        
      case 'RANK_UPDATE':
        this.handleRankUpdate(message.data);
        break;
        
      case 'CHECKPOINT_SAVED':
        this.emit('checkpointSaved', message.data);
        break;
        
      default:
        console.log('Unknown worker message:', message);
    }
  }
  
  /**
   * Update training state from worker progress
   */
  updateTrainingState(progressData) {
    const now = Date.now();
    const timeDelta = now - this.trainingState.lastUpdate;
    
    // Update state
    Object.assign(this.trainingState, progressData, {
      lastUpdate: now
    });
    
    // Calculate throughput (steps per second)
    if (timeDelta > 0) {
      const stepsDelta = progressData.currentStep - this.trainingState.currentStep;
      this.trainingState.throughput = (stepsDelta / timeDelta) * 1000;
    }
    
    // Calculate ETA
    if (this.trainingState.throughput > 0) {
      const remainingSteps = this.trainingState.totalSteps - this.trainingState.currentStep;
      this.trainingState.eta = remainingSteps / this.trainingState.throughput;
    }
    
    // Check if rank should be updated
    if (this.rankScheduler && this.trainingState.currentStep % 50 === 0) {
      const shouldUpdate = this.rankScheduler.shouldUpdateRank({
        step: this.trainingState.currentStep,
        loss: this.trainingState.currentLoss,
        averageLoss: this.trainingState.averageLoss
      });
      
      if (shouldUpdate.shouldUpdate) {
        this.updateRank(shouldUpdate.newRank);
      }
    }
    
    this.emit('trainingProgress', this.trainingState);
  }
  
  /**
   * Handle training completion
   */
  async handleTrainingComplete(data) {
    this.isTraining = false;
    this.isPaused = false;
    
    console.log('Training completed successfully');
    
    // Extract final adapter
    const adapterData = data.adapter;
    
    // Export adapter
    try {
      const metadata = {
        modelName: this.currentModel.name,
        trainingSteps: this.trainingState.currentStep,
        finalLoss: this.trainingState.currentLoss,
        rank: this.loraConfig.rank,
        alpha: this.loraConfig.alpha,
        trainingTime: Date.now() - this.trainingState.startTime
      };
      
      this.emit('trainingComplete', {
        adapter: adapterData,
        metrics: this.trainingState,
        metadata
      });
      
    } catch (error) {
      console.error('Adapter export failed:', error);
      this.emit('error', { type: 'export', error });
    }
    
    // Cleanup
    if (this.trainingWorker) {
      this.trainingWorker.terminate();
      this.trainingWorker = null;
    }
  }
  
  /**
   * Handle training error
   */
  handleTrainingError(errorData) {
    console.error('Training error:', errorData);
    
    this.isTraining = false;
    this.isPaused = false;
    
    this.emit('error', { 
      type: 'training', 
      error: errorData.error,
      step: errorData.step
    });
    
    // Cleanup
    if (this.trainingWorker) {
      this.trainingWorker.terminate();
      this.trainingWorker = null;
    }
  }
  
  /**
   * Handle rank update from scheduler
   */
  async handleRankUpdate(data) {
    console.log('Rank updated:', data);
    this.loraConfig.rank = data.newRank;
    this.loraConfig.scaling = this.loraConfig.alpha / this.loraConfig.rank;
    
    this.emit('rankUpdated', data);
  }
  
  /**
   * Update LoRA rank during training
   */
  async updateRank(newRank) {
    if (!this.isTraining || !this.trainingWorker) {
      return;
    }
    
    this.trainingWorker.postMessage({
      type: 'UPDATE_RANK',
      newRank,
      scaling: this.loraConfig.alpha / newRank
    });
  }
  
  /**
   * Save current training checkpoint
   */
  async saveCheckpoint(filename = null) {
    if (!this.isTraining || !this.trainingWorker) {
      throw new Error('No training in progress');
    }
    
    const checkpointName = filename || `checkpoint_step_${this.trainingState.currentStep}.safetensors`;
    
    this.trainingWorker.postMessage({
      type: 'SAVE_CHECKPOINT',
      filename: checkpointName
    });
  }
  
  /**
   * Export trained adapter
   */
  async exportAdapter(filename = 'lora_adapter.safetensors') {
    if (this.isTraining) {
      throw new Error('Cannot export while training is in progress');
    }
    
    // Request current adapter from worker or session
    if (this.trainingWorker) {
      return new Promise((resolve, reject) => {
        const timeout = setTimeout(() => reject(new Error('Export timeout')), 30000);
        
        const handler = (event) => {
          if (event.data.type === 'ADAPTER_EXPORTED') {
            clearTimeout(timeout);
            this.trainingWorker.removeEventListener('message', handler);
            resolve(event.data.adapter);
          }
        };
        
        this.trainingWorker.addEventListener('message', handler);
        this.trainingWorker.postMessage({
          type: 'EXPORT_ADAPTER',
          filename
        });
      });
    }
    
    throw new Error('No training session available for export');
  }
  
  /**
   * Validate model for LoRA training
   */
  async validateModelForTraining(modelInfo) {
    const validation = {
      isValid: false,
      errors: [],
      warnings: [],
      recommendations: []
    };
    
    // Check model type
    if (modelInfo.modelType !== 'language_model') {
      validation.warnings.push('Model type is not language_model - may not be optimal for LoRA');
    }
    
    // Check inputs
    const hasTextInputs = modelInfo.inputs.some(input => 
      input.name.includes('input_ids') || 
      input.name.includes('tokens') ||
      input.name.includes('text')
    );
    
    if (!hasTextInputs) {
      validation.errors.push('Model missing text inputs (input_ids/tokens)');
    }
    
    // Check outputs
    const hasLogits = modelInfo.outputs.some(output => 
      output.name.includes('logits') || 
      output.name.includes('predictions') ||
      output.name.includes('scores')
    );
    
    if (!hasLogits) {
      validation.errors.push('Model missing logits/predictions output');
    }
    
    // Memory check
    const memoryMB = modelInfo.sizeInBytes / (1024 * 1024);
    if (memoryMB > 1000) {
      validation.warnings.push(`Large model (${memoryMB.toFixed(0)}MB) may cause memory issues`);
    }
    
    validation.isValid = validation.errors.length === 0;
    
    // Add recommendations
    if (validation.isValid) {
      validation.recommendations.push('Model is compatible with LoRA training');
      
      // Suggest rank based on model size
      let suggestedRank = 4;
      if (modelInfo.estimatedParams > 1000000) suggestedRank = 8;
      if (modelInfo.estimatedParams > 10000000) suggestedRank = 16;
      if (modelInfo.estimatedParams > 100000000) suggestedRank = 32;
      
      validation.recommendations.push(`Suggested LoRA rank: ${suggestedRank}`);
    }
    
    return validation;
  }
  
  /**
   * Validate dataset for training
   */
  validateDataset(dataset) {
    const validation = {
      isValid: false,
      errors: [],
      warnings: []
    };
    
    if (!dataset) {
      validation.errors.push('Dataset is null or undefined');
      return validation;
    }
    
    if (Array.isArray(dataset)) {
      if (dataset.length === 0) {
        validation.errors.push('Dataset is empty');
        return validation;
      }
      
      if (dataset.length < 10) {
        validation.warnings.push('Very small dataset (<10 examples) may not train effectively');
      }
      
      // Check data format
      const firstItem = dataset[0];
      if (typeof firstItem === 'string') {
        // Text data
        validation.isValid = true;
      } else if (typeof firstItem === 'object' && firstItem.text) {
        // Object with text field
        validation.isValid = true;
      } else {
        validation.errors.push('Dataset items must be strings or objects with text field');
      }
      
    } else if (typeof dataset === 'object') {
      const keys = Object.keys(dataset);
      if (keys.length === 0) {
        validation.errors.push('Dataset object is empty');
        return validation;
      }
      
      validation.isValid = true;
      
    } else {
      validation.errors.push('Dataset must be an array or object');
    }
    
    return validation;
  }
  
  /**
   * Get current training status
   */
  getTrainingStatus() {
    return {
      isTraining: this.isTraining,
      isPaused: this.isPaused,
      hasModel: !!this.currentModel,
      hasDataset: !!this.currentDataset,
      state: this.trainingState,
      config: {
        lora: this.loraConfig,
        training: this.trainingConfig
      }
    };
  }
  
  /**
   * Update training configuration
   */
  updateConfig(updates) {
    if (updates.lora) {
      Object.assign(this.loraConfig, updates.lora);
    }
    
    if (updates.training) {
      Object.assign(this.trainingConfig, updates.training);
    }
    
    this.emit('configUpdated', {
      lora: this.loraConfig,
      training: this.trainingConfig
    });
  }
  
  /**
   * Event handling
   */
  on(event, callback) {
    if (!this.eventListeners.has(event)) {
      this.eventListeners.set(event, []);
    }
    this.eventListeners.get(event).push(callback);
  }
  
  off(event, callback) {
    if (this.eventListeners.has(event)) {
      const callbacks = this.eventListeners.get(event);
      const index = callbacks.indexOf(callback);
      if (index > -1) {
        callbacks.splice(index, 1);
      }
    }
  }
  
  emit(event, data) {
    if (this.eventListeners.has(event)) {
      this.eventListeners.get(event).forEach(callback => {
        try {
          callback(data);
        } catch (error) {
          console.error('Event callback error:', error);
        }
      });
    }
  }
  
  /**
   * Cleanup resources
   */
  async cleanup() {
    await this.stopTraining();
    
    if (this.onnxSession) {
      await this.onnxSession.cleanup();
      this.onnxSession = null;
    }
    
    this.currentModel = null;
    this.currentDataset = null;
    this.eventListeners.clear();
    
    console.log('Training engine cleaned up');
  }
}

// Create singleton instance
export const trainingEngine = new TrainingEngine();

export default trainingEngine;