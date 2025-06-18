/**
 * Training Engine for LoRA Lab
 * Main orchestrator for LoRA training via a dedicated web worker.
 */

// This engine no longer deals with ONNX directly. It only manages state
// and communicates with the training worker.

/**
 * Training Engine Class
 * Manages the complete LoRA training pipeline by controlling the training worker.
 */
export class TrainingEngine {
  constructor() {
    this.isTraining = false;
    this.isPaused = false;
    this.trainingWorker = null;
    
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
    
    // Event listeners
    this.eventListeners = new Map();
  }
  
  /**
   * Start training process
   * @param {Object} config - Contains modelSource, dataset, and trainingConfig
   */
  async startTraining(config = {}) {
    try {
      if (this.isTraining) {
        throw new Error('Training already in progress');
      }
      
      console.log('Starting LoRA training via worker...');
      
      this.isTraining = true;
      this.isPaused = false;
      
      // Create and configure training worker
      await this.createTrainingWorker();
      
      // Send all necessary info to the worker to initialize and start
      // Make a serializable copy of the config, removing non-serializable objects
      const serializableConfig = JSON.parse(JSON.stringify(config));
      
      this.trainingWorker.postMessage({
        type: 'INITIALIZE_AND_START',
        data: serializableConfig
      });
      
      // Emit a simple event with sanitized data
      this.emit('trainingStarted', {
        estimatedSteps: config.dataset?.tokenCount ? Math.floor(config.dataset.tokenCount / (config.trainingConfig?.batchSize || 4)) : 1000,
        mode: config.trainingConfig?.mode || 'adapter',
        batchSize: config.trainingConfig?.batchSize || 4
      });
      
    } catch (error) {
      console.error('Training start failed:', error);
      this.isTraining = false;
      this.emit('error', { type: 'trainingStart', error });
      throw error;
    }
  }
  
  /**
   * Create and configure training worker
   */
  async createTrainingWorker() {
    if (this.trainingWorker) {
      this.trainingWorker.terminate();
    }
    
    this.trainingWorker = new Worker(new URL('../workers/training.worker.js', import.meta.url), { type: 'module' });
    
    this.trainingWorker.onmessage = (event) => {
      this.handleWorkerMessage(event.data);
    };
    
    this.trainingWorker.onerror = (error) => {
      console.error('Training worker error:', error);
      this.emit('error', { type: 'worker', error });
      this.isTraining = false;
    };
  }

  /**
   * Handle messages from training worker
   */
  handleWorkerMessage(message) {
    switch (message.type) {
      case 'TRAINING_PROGRESS':
        this.emit('trainingProgress', message.data);
        break;
      case 'TRAINING_STARTED':
        this.emit('trainingStarted', message.data);
        break;
      case 'TRAINING_COMPLETED':
        this.isTraining = false;
        this.emit('trainingCompleted', message.data);
        break;
      case 'TRAINING_ERROR':
        this.isTraining = false;
        this.emit('error', { type: 'training', error: message.data });
        break;
      case 'TRAINING_PAUSED':
        this.emit('trainingPaused', message.data);
        break;
      case 'TRAINING_RESUMED':
        this.emit('trainingResumed', message.data);
        break;
      case 'TRAINING_STOPPED':
        this.isTraining = false;
        this.emit('trainingStopped', message.data);
        break;
      case 'RANK_UPDATED':
        this.emit('rankUpdated', message.data);
        break;
      case 'STATUS_UPDATE':
        this.emit('statusUpdate', message.data);
        break;
      default:
        // Forward other events if needed
        this.emit(message.type.toLowerCase(), message.data);
    }
  }

  // ... (pauseTraining, resumeTraining, stopTraining methods now just post messages)

  pauseTraining() {
    if (this.trainingWorker) this.trainingWorker.postMessage({ type: 'PAUSE_TRAINING' });
  }

  resumeTraining() {
    if (this.trainingWorker) this.trainingWorker.postMessage({ type: 'RESUME_TRAINING' });
  }

  stopTraining() {
    if (this.trainingWorker) {
      this.trainingWorker.postMessage({ type: 'STOP_TRAINING' });
      this.isTraining = false;
    }
  }
  
  // ... (Event handling methods: on, off, emit)
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

  async cleanup() {
    if (this.trainingWorker) {
      this.trainingWorker.terminate();
      this.trainingWorker = null;
    }
    this.isTraining = false;
    this.eventListeners.clear();
    console.log('Training engine cleaned up');
  }
}

// Create singleton instance
export const trainingEngine = new TrainingEngine();

export default trainingEngine;