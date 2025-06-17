<template>
  <div class="lora-lab-app">
    <!-- Header Bar -->
    <HeaderBar 
      :selectedModel="selectedModel"
      :modelOptions="modelOptions"
      :isTraining="isTraining"
      :canStartTraining="canStartTraining"
      :hasCorpus="!!corpusInfo"
      :hasTrainingPlan="!!selectedTrainingMode"
      :isGPUReady="hardwareInfo.webGPUSupported"
      :trainingProgress="trainingStatus.progress"
      @model-selected="handleModelSelected"
      @load-model-url="handleLoadModelUrl"
      @corpus-requested="showCorpusModal = true"
      @plan-requested="showPlanModal = true"
      @training-requested="handleTrainingRequest"
    />

    <!-- Main Content Area -->
    <div class="main-content">
      <!-- Chat Panel (Left) -->
      <div class="chat-section">
        <ChatPanel 
          :selectedModel="selectedModel"
          :useLoRA="useLoRA"
          :isTraining="isTraining"
          :adapterLoaded="adapterLoaded"
          :messages="chatHistory"
          @toggle-lora="useLoRA = !useLoRA"
          @message-sent="handleChatMessage"
          @chat-cleared="chatHistory = []"
        />
      </div>

      <!-- Training Console (Right) -->
      <div class="console-section">
        <TrainConsole 
          :trainingConfig="trainingConfig"
          :hardwareInfo="hardwareInfo"
          :isTraining="isTraining"
          ref="trainConsole"
          @training-started="handleTrainingStarted"
          @training-paused="handleTrainingPaused"
          @training-resumed="handleTrainingResumed"
          @training-stopped="handleTrainingStopped"
          @training-aborted="handleTrainingAborted"
          @training-completed="handleTrainingCompleted"
        />
      </div>
    </div>

    <!-- Footer Status -->
    <FooterStatus 
      :hardwareInfo="hardwareInfo"
      :trainingStatus="trainingStatus"
      :adapterReady="adapterReady"
      :currentMemoryUsage="currentMemoryUsage"
      @download-adapter="handleDownloadAdapter"
    />

    <!-- Plan Selection Modal -->
    <PlanModal 
      :visible="showPlanModal"
      :hardwareInfo="hardwareInfo"
      :modelInfo="selectedModel"
      :corpusInfo="corpusInfo"
      @close="showPlanModal = false"
      @mode-selected="handleModeSelected"
    />

    <!-- File Upload Modal -->
    <div v-if="showCorpusModal" class="modal-overlay" @click="showCorpusModal = false">
      <div class="corpus-modal" @click.stop>
        <div class="modal-header">
          <h3>Input Training Corpus</h3>
          <button @click="showCorpusModal = false" class="close-btn">×</button>
        </div>
        
        <div class="modal-body">
          <div class="input-methods">
            <!-- File Upload -->
            <div class="input-method">
              <h4>Upload Text File</h4>
              <div class="file-drop-zone" 
                   :class="{ 'dragover': isDragover }"
                   @drop="handleFileDrop"
                   @dragover.prevent="isDragover = true"
                   @dragleave="isDragover = false"
                   @click="$refs.fileInput.click()">
                <div class="drop-content">
                  <div class="drop-icon">📄</div>
                  <div class="drop-text">
                    <div>Drop a .txt file here or click to browse</div>
                    <div class="drop-subtext">Maximum 10MB, UTF-8 encoded</div>
                  </div>
                </div>
                <input 
                  ref="fileInput" 
                  type="file" 
                  accept=".txt" 
                  @change="handleFileSelect" 
                  style="display: none;"
                />
              </div>
            </div>

            <!-- Text Paste -->
            <div class="input-method">
              <h4>Paste Text</h4>
              <textarea 
                v-model="pastedText"
                placeholder="Paste your training text here..."
                class="text-input"
                rows="8"
              ></textarea>
              <div class="text-stats">
                Characters: {{ pastedText.length.toLocaleString() }} | 
                Est. Tokens: {{ Math.floor(pastedText.length / 4).toLocaleString() }}
              </div>
            </div>
          </div>
        </div>

        <div class="modal-footer">
          <button @click="showCorpusModal = false" class="cancel-btn">Cancel</button>
          <button @click="processCorpusInput" class="confirm-btn" 
                  :disabled="!pastedText && !selectedFile">
            Process Corpus
          </button>
        </div>
      </div>
    </div>

    <!-- Loading Overlay -->
    <div v-if="isLoading" class="loading-overlay">
      <div class="loading-content">
        <div class="loading-spinner"></div>
        <div class="loading-text">{{ loadingMessage }}</div>
        <div v-if="loadingProgress > 0" class="loading-progress">
          <div class="progress-bar">
            <div class="progress-fill" :style="{ width: `${loadingProgress}%` }"></div>
          </div>
          <div class="progress-text">{{ Math.floor(loadingProgress) }}%</div>
        </div>
        <div v-if="loadingDetails" class="loading-details">{{ loadingDetails }}</div>
      </div>
    </div>

    <!-- Hugging Face Token Modal -->
    <div v-if="showTokenModal" class="modal-overlay" @click="showTokenModal = false">
      <div class="corpus-modal" @click.stop>
        <div class="modal-header">
          <h3>Hugging Face Access Token Required</h3>
          <button @click="showTokenModal = false" class="close-btn">×</button>
        </div>
        <div class="modal-body">
          <p class="modal-description">This model requires a Hugging Face access token for download. Please create a token with 'read' permissions and paste it below.</p>
          <div class="input-method">
            <h4>Access Token</h4>
            <input 
              v-model="hfToken"
              type="password"
              placeholder="hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
              class="text-input"
            />
            <a href="https://huggingface.co/settings/tokens" target="_blank" rel="noopener noreferrer" class="hf-token-link">Get your token from Hugging Face settings</a>
          </div>
        </div>
        <div class="modal-footer">
          <button @click="showTokenModal = false" class="cancel-btn">Cancel</button>
          <button @click="handleSaveTokenAndRetry" class="confirm-btn" :disabled="!hfToken">
            Save & Retry Download
          </button>
        </div>
      </div>
    </div>
  </div>
</template>

<script>
import HeaderBar from './HeaderBar.vue'
import ChatPanel from './ChatPanel.vue'
import TrainConsole from './TrainConsole.vue'
import FooterStatus from './FooterStatus.vue'
import PlanModal from './PlanModal.vue'
import { detectHardware } from '../utils/hwDetect.js'
import { loadDataset, tokenizeText } from '../data/datasetLoader.js'
import { runInference, loadModel } from '../trainers/onnxSession.js'
import { generateTextFromOutputs } from '../utils/textGeneration.js'
import { modelManager, DefaultModels } from '../utils/modelManager.js'
import { AutoTokenizer } from '@huggingface/transformers'

export default {
  name: 'LoRALabApp',
  components: {
    HeaderBar,
    ChatPanel,
    TrainConsole,
    FooterStatus,
    PlanModal
  },
  data() {
    return {
      // Hardware & System
      hardwareInfo: {},
      isLoading: true,
      loadingMessage: 'Detecting hardware capabilities...',
      loadingProgress: 0,
      loadingDetails: '',
      currentMemoryUsage: 2.3, // GB - will be updated during training

      // Model Management
      selectedModel: null,
      modelSession: null, // Store the ONNX session here
      tokenizer: null, // Store the tokenizer here
      modelOptions: [],

      // Corpus Management
      corpusInfo: null,
      showCorpusModal: false,
      pastedText: '',
      selectedFile: null,
      isDragover: false,

      // Training Configuration
      trainingConfig: {},
      showPlanModal: false,
      selectedTrainingMode: null,

      // Training State
      isTraining: false,
      trainingStatus: {
        mode: null,
        progress: 0,
        eta: null,
        throughput: 0
      },

      // Adapter Management
      useLoRA: false,
      adapterLoaded: false,
      adapterReady: false,

      // Chat State
      chatHistory: [],
      isTyping: false,

      // HF Token Modal
      showTokenModal: false,
      hfToken: '',
      modelToRetry: null,

      // Training Simulation
      trainingInterval: null,
      memoryMonitoringInterval: null
    }
  },
  computed: {
    canStartTraining() {
      return !this.isTraining && 
             this.selectedModel && 
             this.corpusInfo && 
             this.selectedTrainingMode &&
             this.hardwareInfo.webGPUSupported
    }
  },
  async mounted() {
    await this.initializeApp()
    
    // Start memory monitoring
    this.memoryMonitoringInterval = setInterval(() => {
      this.updateMemoryUsage()
    }, 2000) // Update every 2 seconds
  },
  beforeUnmount() {
    if (this.memoryMonitoringInterval) {
      clearInterval(this.memoryMonitoringInterval)
    }
    
    // Clean up models when component is unmounted
    this.cleanupModels();
  },
  methods: {
    // Clean up models to prevent memory leaks
    async cleanupModels() {
      try {
        if (this.selectedModel && this.selectedModel.id) {
          console.log('Unloading model:', this.selectedModel.id);
          await modelManager.unloadModel(this.selectedModel.id);
        }
      } catch (error) {
        console.error('Error cleaning up models:', error);
      }
    },

    async initializeApp() {
      try {
        this.loadingMessage = 'Initializing LoRA Lab';
        this.loadingProgress = 0;
        this.loadingDetails = 'Detecting hardware capabilities...';
        
        // Simulate progress for hardware detection
        let progress = 0;
        const loadingInterval = setInterval(() => {
          progress += 5;
          if (progress <= 100) {
            this.loadingProgress = progress;
          } else {
            clearInterval(loadingInterval);
          }
        }, 100);
        
        // Detect hardware capabilities
        this.hardwareInfo = await detectHardware();
        clearInterval(loadingInterval);
        
        // Log hardware info for debugging
        console.log('Hardware detection results:', this.hardwareInfo);
        console.log('WebGPU supported:', this.hardwareInfo.webGPUSupported);
        console.log('Warnings:', this.hardwareInfo.warnings);
        
        // Initialize model options from modelManager
        this.initializeModelOptions();
        
        // WebGPU initialization
        this.loadingProgress = 30;
        this.loadingDetails = 'Initializing WebGPU environment...';
        await new Promise(resolve => setTimeout(resolve, 800));
        
        // ONNX runtime initialization
        this.loadingProgress = 60;
        this.loadingDetails = 'Setting up ONNX runtime...';
        await new Promise(resolve => setTimeout(resolve, 800));
        
        // Load default model
        this.loadingProgress = 80;
        this.loadingDetails = 'Loading default model...';
        await this.loadDefaultModel();
        
        // Complete initialization
        this.loadingProgress = 100;
        this.loadingDetails = 'Ready!';
        await new Promise(resolve => setTimeout(resolve, 500));
        
        this.isLoading = false;
        this.loadingProgress = 0;
        this.loadingDetails = '';
      } catch (error) {
        console.error('Failed to initialize app:', error);
        this.loadingDetails = `Error: ${error.message}`;
        this.loadingProgress = 0;
        this.isLoading = false;
        // Continue with degraded functionality
      }
    },

    initializeModelOptions() {
      // Use DefaultModels from modelManager as the source of truth
      const models = Object.values(DefaultModels).map(model => ({
        id: model.name,
        name: model.description, // Use the more descriptive name
        size: 'Unknown', // Placeholder, will be determined on load
        url: model.url,
        description: model.description,
        repo_id: model.repo_id
      }));
      
      // Add the custom URL option
      models.push({
        id: 'custom',
        name: 'Custom Model URL',
        size: 'Variable',
        url: '',
        description: 'Load from a custom ONNX model URL.'
      });

      this.modelOptions = models;
    },

    async loadDefaultModel() {
      // Default to gpt2-small if available
      const defaultModel = this.modelOptions.find(m => m.id === 'gpt2-small');
      
      if (defaultModel) {
        this.selectedModel = { ...defaultModel };
      }
    },

    // Model Management
    async handleModelSelected(model) {
      if (model.id === 'custom') {
        this.selectedModel = model;
        this.modelSession = null;
        this.tokenizer = null;
        return;
      }
      
      this.isLoading = true;
      this.loadingMessage = `Loading model: ${model.name}`;
      this.loadingProgress = 0;
      this.loadingDetails = `Preparing to download model (${model.size})`;
      let loadingInterval = null;
      
      try {
        // Unload previous model if one is loaded
        if (this.selectedModel && this.selectedModel.id) {
          this.loadingDetails = `Unloading previous model: ${this.selectedModel.name}...`;
          await modelManager.unloadModel(this.selectedModel.id);
          this.modelSession = null;
          this.tokenizer = null;
          console.log(`Unloaded ${this.selectedModel.id}`);
        }

        // Simulate progress for model loading
        let progress = 0;
        loadingInterval = setInterval(() => {
          progress += 5;
          if (progress <= 95) {
            this.loadingProgress = progress;
            if (progress < 40) {
              this.loadingDetails = `Downloading model...`;
            } else if (progress < 80) {
              this.loadingDetails = `Initializing model session...`;
            } else {
              this.loadingDetails = `Loading tokenizer...`;
            }
          }
        }, 200);

        // Use modelManager to load the model
        this.loadingDetails = `Downloading model...`
        const modelInfo = await modelManager.loadModel(model.url, {
          modelName: model.id,
          enableProfiling: false
        });
        
        // Load tokenizer
        this.loadingDetails = `Loading tokenizer...`
        const hfToken = localStorage.getItem('HF_ACCESS_TOKEN');
        const tokenizerOptions = hfToken ? { token: hfToken } : {};
        this.tokenizer = await AutoTokenizer.from_pretrained(model.repo_id, tokenizerOptions);
        
        // Store the model session for inference
        this.modelSession = modelInfo.session;
        
        // Update UI
        clearInterval(loadingInterval);
        this.selectedModel = { 
          ...model,
          modelInfo,
          isWebGPU: modelInfo.executionProviders?.includes('webgpu') || false
        };
        
        this.loadingProgress = 100;
        this.loadingDetails = `Model loaded successfully!`;
        
        // Show completion for a moment before closing
        setTimeout(() => {
          this.isLoading = false;
          this.loadingProgress = 0;
          this.loadingDetails = '';
        }, 500);
      } catch (error) {
        console.error('Model loading failed:', error);
        if (loadingInterval) {
          clearInterval(loadingInterval);
        }
        
        if (error.message.includes('401 Unauthorized')) {
          this.modelToRetry = model;
          this.showTokenModal = true;
          this.loadingDetails = `Access token required for ${model.name}.`;
          this.isLoading = false; // Stop loading spinner to show modal
          return;
        }
        
        this.loadingDetails = `Error: ${error.message}`;
        
        // Show error for a moment before closing
        setTimeout(() => {
          this.isLoading = false;
          this.loadingProgress = 0;
          this.loadingDetails = '';
        }, 2000);
      }
    },

    async handleLoadModelUrl(url) {
      this.isLoading = true;
      this.loadingMessage = 'Loading custom model';
      this.loadingProgress = 0;
      this.loadingDetails = `Preparing to fetch model from URL`;
      
      try {
        // Progress updates
        let progress = 0;
        const loadingInterval = setInterval(() => {
          progress += Math.random() * 2;
          if (progress <= 95) { // Cap at 95% until actual loading completes
            this.loadingProgress = progress;
            
            // Update loading details with different stages
            if (progress < 30) {
              this.loadingDetails = `Downloading model from ${url.substring(0, 40)}...`;
            } else if (progress < 60) {
              this.loadingDetails = `Processing model files (estimating size)`;
            } else if (progress < 90) {
              this.loadingDetails = `Initializing ONNX runtime for custom model`;
            } else {
              this.loadingDetails = `Finalizing model setup`;
            }
          }
        }, 100);
        
        // Use modelManager to load custom model
        const modelInfo = await modelManager.loadModel(url, {
          modelName: 'custom-model-' + Date.now(),
          enableProfiling: false
        });
        
        // For custom models, we can't automatically know the tokenizer.
        // We'll have to rely on a generic one or ask the user.
        // For now, let's try a generic one.
        this.loadingDetails = `Loading generic tokenizer...`
        this.tokenizer = await AutoTokenizer.from_pretrained('gpt2');
        
        // Store the model session for inference
        this.modelSession = modelInfo.session;
        
        // Update UI
        clearInterval(loadingInterval);
        this.loadingProgress = 100;
        this.loadingMessage = 'Model loaded successfully';
        this.loadingDetails = `Custom model loaded and ready for use`;
        
        // Set model info
        this.selectedModel = {
          id: 'custom-loaded',
          name: 'Custom Model',
          size: `${modelInfo.sizeInMB}MB` || 'Unknown',
          url: url,
          description: 'Custom ONNX model',
          modelInfo,
          isWebGPU: modelInfo.executionProviders?.includes('webgpu') || false
        };
        
        // Wait a moment to show completion
        setTimeout(() => {
          this.isLoading = false;
          this.loadingProgress = 0;
          this.loadingDetails = '';
        }, 500);
      } catch (error) {
        console.error('Failed to load model:', error);

        if (error.message.includes('401 Unauthorized')) {
          this.modelToRetry = { id: 'custom', name: 'Custom Model', url: url };
          this.showTokenModal = true;
          this.loadingDetails = `Access token required for custom model.`;
          this.isLoading = false; // Stop loading spinner to show modal
          return;
        }

        this.loadingProgress = 0;
        this.loadingDetails = `Error: ${error.message}`;
        
        // Show error for a moment before closing
        setTimeout(() => {
          this.isLoading = false;
        }, 2000);
      }
    },

    async handleSaveTokenAndRetry() {
      if (!this.hfToken) return;

      // Save token to localStorage
      localStorage.setItem('HF_ACCESS_TOKEN', this.hfToken);
      console.log('HF Access Token saved to localStorage.');

      // Hide modal
      this.showTokenModal = false;
      const modelToLoad = { ...this.modelToRetry };

      // Clear retry state
      this.hfToken = '';
      this.modelToRetry = null;

      // Retry loading
      if (modelToLoad) {
        if (modelToLoad.id === 'custom') {
          await this.handleLoadModelUrl(modelToLoad.url);
        } else {
          await this.handleModelSelected(modelToLoad);
        }
      }
    },

    // Corpus Management
    handleCorpusUploaded() {
      this.showCorpusModal = true
    },

    handleCorpusPasted() {
      this.showCorpusModal = true
    },

    handleFileDrop(event) {
      event.preventDefault()
      this.isDragover = false
      
      const files = event.dataTransfer.files
      if (files.length > 0) {
        this.selectedFile = files[0]
      }
    },

    handleFileSelect(event) {
      const files = event.target.files
      if (files.length > 0) {
        this.selectedFile = files[0]
      }
    },

    async processCorpusInput() {
      this.isLoading = true
      this.loadingMessage = 'Processing corpus...'
      
      try {
        let text = ''
        
        if (this.selectedFile) {
          text = await this.readFileAsText(this.selectedFile)
        } else if (this.pastedText) {
          text = this.pastedText
        }
        
        const dataset = await loadDataset(text)
        this.corpusInfo = {
          text: text,
          tokenCount: dataset.tokenCount,
          characterCount: text.length,
          estimatedTrainingTime: this.estimateTrainingTime(dataset.tokenCount)
        }
        
        this.showCorpusModal = false
        this.pastedText = ''
        this.selectedFile = null
        this.isLoading = false
        
      } catch (error) {
        console.error('Failed to process corpus:', error)
        this.isLoading = false
        alert('Failed to process corpus')
      }
    },

    readFileAsText(file) {
      return new Promise((resolve, reject) => {
        const reader = new FileReader()
        reader.onload = e => resolve(e.target.result)
        reader.onerror = reject
        reader.readAsText(file)
      })
    },

    estimateTrainingTime(tokenCount) {
      const throughput = this.hardwareInfo.estimatedTFLOPs * 1000 * 0.4 // 40% efficiency
      return Math.ceil(tokenCount / Math.max(throughput, 100)) // seconds
    },

    // Training Management
    handleModeSelected(selection) {
      this.selectedTrainingMode = selection.mode
      this.trainingConfig = {
        mode: selection.mode,
        config: selection.config,
        hardwareInfo: selection.hardwareInfo,
        totalTokens: this.corpusInfo?.tokenCount || 1000000,
        modelInfo: this.selectedModel
      }
      this.showPlanModal = false
    },

    handleTrainingRequest() {
      if (!this.canStartTraining) {
        if (!this.selectedModel) {
          alert('Please select a model first')
        } else if (!this.corpusInfo) {
          alert('Please load a training corpus first')
        } else if (!this.selectedTrainingMode) {
          alert('Please select a training mode first')
        } else if (!this.hardwareInfo.webGPUSupported) {
          alert('WebGPU is required for training')
        }
        return
      }

      // Trigger training in console component
      if (this.$refs.trainConsole) {
        this.$refs.trainConsole.startTraining()
      }
    },

    // Training Event Handlers
    handleTrainingStarted() {
      this.isTraining = true
      this.trainingStatus.mode = this.selectedTrainingMode
      this.trainingStatus.progress = 0
      this.trainingStatus.throughput = 0
      
      // Start training simulation
      this.startTrainingSimulation()
    },

    handleTrainingPaused() {
      // Training paused but still considered "in training"
      this.pauseTrainingSimulation()
    },

    handleTrainingResumed() {
      // Training resumed
      this.resumeTrainingSimulation()
    },

    handleTrainingStopped() {
      this.isTraining = false
      this.adapterReady = true
      this.stopTrainingSimulation()
    },

    handleTrainingAborted() {
      this.isTraining = false
      this.adapterReady = false
      this.stopTrainingSimulation()
    },

    handleTrainingCompleted() {
      this.isTraining = false
      this.adapterReady = true
      this.adapterLoaded = true
      this.stopTrainingSimulation()
    },

    // Training Simulation
    startTrainingSimulation() {
      const totalTokens = this.corpusInfo?.tokenCount || 100000
      const estimatedThroughput = this.hardwareInfo.estimatedTFLOPs * 1000 * 0.4 // 40% efficiency
      const tokensPerSecond = Math.max(estimatedThroughput, 100)
      const totalSeconds = totalTokens / tokensPerSecond
      
      this.trainingStatus.throughput = Math.round(tokensPerSecond)
      this.trainingStatus.eta = totalSeconds
      
      this.trainingInterval = setInterval(() => {
        if (this.trainingStatus.progress < 100) {
          // Simulate non-linear progress (faster at start, slower toward end)
          const progressIncrement = Math.max(0.5, 3 - (this.trainingStatus.progress / 50))
          this.trainingStatus.progress = Math.min(100, this.trainingStatus.progress + progressIncrement)
          
          // Update ETA
          const remainingProgress = 100 - this.trainingStatus.progress
          this.trainingStatus.eta = Math.round((remainingProgress / 100) * totalSeconds)
          
          // Slight throughput variation
          const variation = 0.9 + (Math.random() * 0.2) // ±10% variation
          this.trainingStatus.throughput = Math.round(tokensPerSecond * variation)
        } else {
          this.handleTrainingCompleted()
        }
      }, 1000) // Update every second
    },

    pauseTrainingSimulation() {
      if (this.trainingInterval) {
        clearInterval(this.trainingInterval)
        this.trainingInterval = null
      }
    },

    resumeTrainingSimulation() {
      if (!this.trainingInterval && this.isTraining) {
        this.startTrainingSimulation()
      }
    },

    stopTrainingSimulation() {
      if (this.trainingInterval) {
        clearInterval(this.trainingInterval)
        this.trainingInterval = null
      }
      
      this.trainingStatus.progress = this.trainingStatus.progress >= 100 ? 100 : 0
      this.trainingStatus.eta = null
      this.trainingStatus.throughput = 0
    },

    // Adapter Management
    handleDownloadAdapter() {
      if (!this.adapterReady) {
        alert('No adapter ready for download')
        return
      }

      // Simulate adapter download
      const blob = new Blob(['// LoRA Adapter weights (simulated)'], { type: 'application/octet-stream' })
      const url = URL.createObjectURL(blob)
      const link = document.createElement('a')
      link.href = url
      link.download = `lora-adapter-${this.selectedModel?.id || 'model'}-${Date.now()}.safetensors`
      document.body.appendChild(link)
      link.click()
      document.body.removeChild(link)
      URL.revokeObjectURL(url)
    },

    // Chat Management
    async handleChatMessage(message) {
      this.chatHistory.push({
        role: 'user',
        content: message,
        timestamp: Date.now()
      })

      // Start typing indicator
      this.isTyping = true

      try {
        // Check if model is loaded
        if (!this.selectedModel) {
          throw new Error('No model loaded')
        }
        
        if (!this.modelSession) {
          throw new Error('Model session not initialized')
        }
        if (!this.tokenizer) {
          throw new Error('Tokenizer not initialized')
        }

        // Tokenize the user message
        const { input_ids } = this.tokenizer(message, {
          return_tensor: false,
          add_special_tokens: true,
        });
        
        // Generate text from model outputs
        const generatedText = await generateTextFromOutputs(
          this.modelSession, 
          this.tokenizer, 
          input_ids
        );
        
        // Add response to chat history
        this.chatHistory.push({
          role: 'assistant',
          content: generatedText,
          timestamp: Date.now()
        })
      } catch (error) {
        console.error('Inference failed:', error)
        
        // Add error message to chat
        this.chatHistory.push({
          role: 'assistant',
          content: `Error: ${error.message}. Please try again or load a different model.`,
          timestamp: Date.now(),
          isError: true
        })
      } finally {
        // Stop typing indicator
        this.isTyping = false
      }
    },

    // Memory management - simulate memory usage during training
    updateMemoryUsage() {
      if (this.isTraining) {
        // Simulate increasing memory usage during training
        const baseUsage = 1.8
        const trainingOverhead = this.selectedTrainingMode === 'full' ? 2.5 : 1.2
        const progress = this.trainingStatus.progress / 100
        this.currentMemoryUsage = baseUsage + (trainingOverhead * progress)
      } else {
        // Base memory usage when idle
        this.currentMemoryUsage = 1.8 + Math.random() * 0.5
      }
    }
  }
}
</script>

<style scoped>
.lora-lab-app {
  display: flex;
  flex-direction: column;
  height: 100vh;
  background-color: #0a0a0a;
  color: #fff;
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
}

/* Main Content Layout */
.main-content {
  display: flex;
  flex: 1;
  overflow: hidden;
}

.chat-section {
  flex: 1;
  min-width: 400px;
  border-right: 1px solid #333;
}

.console-section {
  flex: 1;
  min-width: 500px;
}

/* Modal Styles */
.modal-overlay {
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background-color: rgba(0, 0, 0, 0.8);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 2000;
}

.corpus-modal {
  background-color: #1a1a1a;
  border-radius: 12px;
  width: 90%;
  max-width: 700px;
  max-height: 80vh;
  display: flex;
  flex-direction: column;
  border: 1px solid #333;
}

.modal-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 1.5rem 2rem;
  border-bottom: 1px solid #333;
  background-color: #222;
}

.modal-header h3 {
  margin: 0;
  color: #fff;
  font-size: 1.5rem;
}

.close-btn {
  background: none;
  border: none;
  color: #ccc;
  cursor: pointer;
  font-size: 2rem;
  padding: 0;
  width: 2.5rem;
  height: 2.5rem;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  transition: all 0.2s;
}

.close-btn:hover {
  background-color: #444;
  color: #fff;
}

.modal-body {
  padding: 2rem;
  flex: 1;
  overflow-y: auto;
}

.input-methods {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 2rem;
}

.input-method h4 {
  margin: 0 0 1rem 0;
  color: #fff;
  font-size: 1.1rem;
}

.file-drop-zone {
  border: 2px dashed #555;
  border-radius: 8px;
  padding: 2rem;
  text-align: center;
  cursor: pointer;
  transition: all 0.3s;
  background-color: #2a2a2a;
}

.file-drop-zone:hover,
.file-drop-zone.dragover {
  border-color: #10b981;
  background-color: #1a2f26;
}

.drop-content {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 1rem;
}

.drop-icon {
  font-size: 3rem;
}

.drop-text {
  color: #ccc;
}

.drop-subtext {
  font-size: 0.9rem;
  color: #888;
  margin-top: 0.5rem;
}

.text-input {
  width: 100%;
  padding: 1rem;
  background-color: #2a2a2a;
  border: 1px solid #444;
  border-radius: 6px;
  color: #fff;
  font-family: 'Courier New', monospace;
  font-size: 0.9rem;
  line-height: 1.5;
  resize: vertical;
}

.text-input:focus {
  outline: none;
  border-color: #10b981;
}

.text-stats {
  margin-top: 0.5rem;
  font-size: 0.8rem;
  color: #888;
}

.modal-footer {
  padding: 1.5rem 2rem;
  border-top: 1px solid #333;
  display: flex;
  justify-content: space-between;
  background-color: #222;
}

.cancel-btn,
.confirm-btn {
  padding: 0.8rem 1.5rem;
  border: 1px solid #555;
  border-radius: 6px;
  cursor: pointer;
  font-size: 1rem;
  font-weight: 600;
  transition: all 0.2s;
}

.cancel-btn {
  background-color: #374151;
  color: #fff;
  border-color: #6b7280;
}

.cancel-btn:hover {
  background-color: #4b5563;
}

.confirm-btn {
  background-color: #10b981;
  color: #fff;
  border-color: #10b981;
}

.confirm-btn:hover:not(:disabled) {
  background-color: #059669;
}

.confirm-btn:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

/* Loading Overlay */
.loading-overlay {
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background-color: rgba(0, 0, 0, 0.9);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 3000;
}

.loading-content {
  text-align: center;
  color: #fff;
  max-width: 500px;
  width: 90%;
  background-color: rgba(26, 26, 26, 0.95);
  padding: 2rem;
  border-radius: 12px;
  box-shadow: 0 8px 32px rgba(0, 0, 0, 0.5);
  border: 1px solid rgba(255, 255, 255, 0.1);
}

.loading-spinner {
  width: 50px;
  height: 50px;
  border: 3px solid #333;
  border-top: 3px solid #10b981;
  border-radius: 50%;
  animation: spin 1s linear infinite;
  margin: 0 auto 1rem;
}

@keyframes spin {
  0% { transform: rotate(0deg); }
  100% { transform: rotate(360deg); }
}

.loading-text {
  font-size: 1.2rem;
  color: #fff;
  margin-bottom: 1rem;
  font-weight: 600;
}

.loading-progress {
  margin-top: 1rem;
  width: 100%;
}

.progress-bar {
  height: 8px;
  background-color: #333;
  border-radius: 4px;
  overflow: hidden;
  margin-bottom: 0.5rem;
}

.progress-fill {
  height: 100%;
  background: linear-gradient(90deg, #10b981, #3b82f6);
  border-radius: 4px;
  transition: width 0.3s ease-out;
}

.progress-text {
  text-align: right;
  font-size: 0.9rem;
  color: #aaa;
}

.loading-details {
  margin-top: 1rem;
  font-size: 0.9rem;
  color: #ccc;
  padding: 0.5rem;
  background-color: rgba(0, 0, 0, 0.2);
  border-radius: 4px;
}

/* Responsive Design */
@media (max-width: 1024px) {
  .main-content {
    flex-direction: column;
  }
  
  .chat-section {
    border-right: none;
    border-bottom: 1px solid #333;
    min-height: 300px;
  }
  
  .console-section {
    min-height: 400px;
  }
}

@media (max-width: 768px) {
  .input-methods {
    grid-template-columns: 1fr;
  }
  
  .corpus-modal {
    width: 95%;
    margin: 1rem;
  }
  
  .modal-header,
  .modal-body,
  .modal-footer {
    padding: 1rem;
  }
}

.modal-description {
  color: #ccc;
  line-height: 1.5;
  margin-bottom: 1.5rem;
}

.hf-token-link {
  display: block;
  margin-top: 0.8rem;
  font-size: 0.8rem;
  color: #3b82f6;
  text-decoration: none;
  text-align: right;
}

.hf-token-link:hover {
  text-decoration: underline;
}

.corpus-modal .text-input[type="password"] {
  font-family: monospace;
}
</style>