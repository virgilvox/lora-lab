<template>
  <header class="header-bar">
    <div class="header-content">
      <!-- Logo -->
      <div class="logo-section">
        <div class="logo">
          <span class="logo-icon">🧬</span>
          <span class="logo-text">LoRA Lab</span>
        </div>
      </div>

      <!-- Main Controls -->
      <div class="controls-section">
        <!-- Model Selection -->
        <div class="control-group">
          <label class="control-label">Model</label>
          <div class="model-selector">
            <select 
              :value="selectedModel?.id || ''"
              @change="handleModelChange"
              class="model-dropdown"
            >
              <option value="">Select Model...</option>
              <option 
                v-for="model in modelOptions" 
                :key="model.id"
                :value="model.id"
              >
                {{ model.name }} ({{ model.size }})
              </option>
            </select>
            
            <div v-if="selectedModel" class="model-info">
              <span class="model-description">{{ selectedModel.description }}</span>
            </div>
          </div>
        </div>

        <!-- Custom Model URL -->
        <div class="control-group">
          <label class="control-label">Load URL</label>
          <div class="url-input-group">
            <input 
              v-model="customModelUrl"
              type="url"
              placeholder="https://huggingface.co/..."
              class="url-input"
              :disabled="isTraining"
            />
            <button 
              @click="handleLoadUrl"
              class="load-url-btn"
              :disabled="!customModelUrl || isTraining"
            >
              Load
            </button>
          </div>
        </div>

        <!-- Corpus Controls -->
        <div class="control-group">
          <label class="control-label">Corpus</label>
          <div class="corpus-controls">
            <button 
              @click="$emit('corpus-uploaded')"
              class="corpus-btn"
              :disabled="isTraining"
              title="Upload text file or paste content"
            >
              <span class="btn-icon">📁</span>
              Choose Corpus
            </button>
            
            <button 
              @click="$emit('corpus-pasted')"
              class="corpus-btn secondary"
              :disabled="isTraining"
              title="Paste text directly"
            >
              <span class="btn-icon">📝</span>
              Paste Text
            </button>
          </div>
        </div>

        <!-- Training Plan -->
        <div class="control-group">
          <label class="control-label">Plan</label>
          <button 
            @click="$emit('plan-requested')"
            class="plan-btn"
            :disabled="isTraining"
            :class="{ 'has-plan': hasTrainingPlan }"
          >
            <span class="btn-icon">⚙️</span>
            {{ hasTrainingPlan ? 'Update Plan' : 'Select Plan' }}
            <span v-if="hasTrainingPlan" class="plan-indicator">✓</span>
          </button>
        </div>

        <!-- Train Button -->
        <div class="control-group">
          <button 
            @click="handleTrainClick"
            class="train-btn"
            :disabled="!canStartTraining || isTraining"
            :class="{ 
              'ready': canStartTraining && !isTraining,
              'training': isTraining,
              'disabled': !canStartTraining 
            }"
          >
            <span class="btn-icon">
              {{ isTraining ? '⏸' : '▶' }}
            </span>
            {{ isTraining ? 'Training...' : 'Train' }}
            <div v-if="isTraining" class="training-progress"></div>
          </button>
        </div>
      </div>

      <!-- Status Indicators -->
      <div class="status-section">
        <div class="status-indicators">
          <!-- Model Status -->
          <div class="status-item" :class="{ 'active': selectedModel }">
            <div class="status-dot"></div>
            <span class="status-text">Model</span>
          </div>
          
          <!-- Corpus Status -->
          <div class="status-item" :class="{ 'active': hasCorpus }">
            <div class="status-dot"></div>
            <span class="status-text">Corpus</span>
          </div>
          
          <!-- Plan Status -->
          <div class="status-item" :class="{ 'active': hasTrainingPlan }">
            <div class="status-dot"></div>
            <span class="status-text">Plan</span>
          </div>
          
          <!-- GPU Status -->
          <div class="status-item" :class="{ 'active': isGPUReady }">
            <div class="status-dot"></div>
            <span class="status-text">GPU</span>
          </div>
        </div>
      </div>
    </div>

    <!-- Progress Bar (when training) -->
    <div v-if="isTraining" class="training-progress-bar">
      <div class="progress-fill" :style="{ width: trainingProgress + '%' }"></div>
    </div>
  </header>
</template>

<script>
export default {
  name: 'HeaderBar',
  props: {
    selectedModel: {
      type: Object,
      default: null
    },
    modelOptions: {
      type: Array,
      default: () => []
    },
    isTraining: {
      type: Boolean,
      default: false
    },
    canStartTraining: {
      type: Boolean,
      default: false
    },
    hasCorpus: {
      type: Boolean,
      default: false
    },
    hasTrainingPlan: {
      type: Boolean,
      default: false
    },
    isGPUReady: {
      type: Boolean,
      default: false
    },
    trainingProgress: {
      type: Number,
      default: 0
    }
  },
  emits: [
    'model-selected', 
    'load-model-url', 
    'corpus-uploaded', 
    'corpus-pasted', 
    'plan-requested', 
    'training-requested'
  ],
  data() {
    return {
      customModelUrl: ''
    }
  },
  methods: {
    handleModelChange(event) {
      const modelId = event.target.value
      if (!modelId) return
      
      const model = this.modelOptions.find(m => m.id === modelId)
      if (model) {
        this.$emit('model-selected', model)
      }
    },

    handleLoadUrl() {
      if (!this.customModelUrl) return
      
      this.$emit('load-model-url', this.customModelUrl)
      this.customModelUrl = ''
    },

    handleTrainClick() {
      if (this.canStartTraining && !this.isTraining) {
        this.$emit('training-requested')
      }
    }
  }
}
</script>

<style scoped>
.header-bar {
  background-color: #1a1a1a;
  border-bottom: 1px solid #333;
  position: relative;
}

.header-content {
  display: flex;
  align-items: center;
  padding: 1rem 1.5rem;
  gap: 2rem;
}

/* Logo Section */
.logo-section {
  flex-shrink: 0;
}

.logo {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  font-size: 1.5rem;
  font-weight: 700;
  color: #fff;
}

.logo-icon {
  font-size: 2rem;
}

.logo-text {
  background: linear-gradient(45deg, #10b981, #3b82f6);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
}

/* Controls Section */
.controls-section {
  display: flex;
  align-items: center;
  gap: 1.5rem;
  flex: 1;
  overflow-x: auto;
}

.control-group {
  display: flex;
  flex-direction: column;
  gap: 0.3rem;
  min-width: max-content;
}

.control-label {
  font-size: 0.7rem;
  color: #888;
  text-transform: uppercase;
  font-weight: 600;
  letter-spacing: 0.5px;
}

/* Model Selection */
.model-selector {
  position: relative;
}

.model-dropdown {
  padding: 0.6rem 1rem;
  background-color: #2a2a2a;
  border: 1px solid #444;
  border-radius: 6px;
  color: #fff;
  font-size: 0.9rem;
  min-width: 200px;
  cursor: pointer;
}

.model-dropdown:focus {
  outline: none;
  border-color: #10b981;
}

.model-info {
  position: absolute;
  top: 100%;
  left: 0;
  right: 0;
  background-color: #333;
  border: 1px solid #444;
  border-top: none;
  border-radius: 0 0 6px 6px;
  padding: 0.5rem;
  font-size: 0.8rem;
  color: #ccc;
  z-index: 10;
}

.model-description {
  display: block;
}

/* URL Input */
.url-input-group {
  display: flex;
  gap: 0.5rem;
}

.url-input {
  padding: 0.6rem 1rem;
  background-color: #2a2a2a;
  border: 1px solid #444;
  border-radius: 6px;
  color: #fff;
  font-size: 0.9rem;
  width: 220px;
}

.url-input:focus {
  outline: none;
  border-color: #10b981;
}

.load-url-btn {
  padding: 0.6rem 1rem;
  background-color: #374151;
  border: 1px solid #6b7280;
  border-radius: 6px;
  color: #fff;
  cursor: pointer;
  font-size: 0.9rem;
  font-weight: 600;
  transition: all 0.2s;
}

.load-url-btn:hover:not(:disabled) {
  background-color: #4b5563;
}

.load-url-btn:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

/* Corpus Controls */
.corpus-controls {
  display: flex;
  gap: 0.5rem;
}

.corpus-btn {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  padding: 0.6rem 1rem;
  background-color: #374151;
  border: 1px solid #6b7280;
  border-radius: 6px;
  color: #fff;
  cursor: pointer;
  font-size: 0.9rem;
  font-weight: 600;
  transition: all 0.2s;
}

.corpus-btn:hover:not(:disabled) {
  background-color: #4b5563;
}

.corpus-btn.secondary {
  background-color: #2a2a2a;
  border-color: #444;
}

.corpus-btn.secondary:hover:not(:disabled) {
  background-color: #333;
}

.corpus-btn:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.btn-icon {
  font-size: 1rem;
}

/* Plan Button */
.plan-btn {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  padding: 0.6rem 1rem;
  background-color: #374151;
  border: 1px solid #6b7280;
  border-radius: 6px;
  color: #fff;
  cursor: pointer;
  font-size: 0.9rem;
  font-weight: 600;
  transition: all 0.2s;
  position: relative;
}

.plan-btn:hover:not(:disabled) {
  background-color: #4b5563;
}

.plan-btn.has-plan {
  background-color: #065f46;
  border-color: #10b981;
  color: #fff;
}

.plan-btn.has-plan:hover:not(:disabled) {
  background-color: #047857;
}

.plan-indicator {
  margin-left: 0.5rem;
  color: #10b981;
  font-weight: bold;
}

/* Train Button */
.train-btn {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  padding: 0.8rem 1.5rem;
  border: none;
  border-radius: 8px;
  cursor: pointer;
  font-size: 1rem;
  font-weight: 700;
  transition: all 0.2s;
  position: relative;
  overflow: hidden;
  min-width: 120px;
  justify-content: center;
}

.train-btn.disabled {
  background-color: #374151;
  color: #6b7280;
  cursor: not-allowed;
}

.train-btn.ready {
  background: linear-gradient(45deg, #10b981, #059669);
  color: #fff;
  box-shadow: 0 2px 8px rgba(16, 185, 129, 0.3);
}

.train-btn.ready:hover {
  background: linear-gradient(45deg, #059669, #047857);
  box-shadow: 0 4px 12px rgba(16, 185, 129, 0.4);
}

.train-btn.training {
  background: linear-gradient(45deg, #f59e0b, #d97706);
  color: #fff;
}

.training-progress {
  position: absolute;
  bottom: 0;
  left: 0;
  height: 2px;
  background-color: rgba(255, 255, 255, 0.3);
  animation: training-pulse 2s ease-in-out infinite;
}

@keyframes training-pulse {
  0%, 100% {
    width: 0%;
  }
  50% {
    width: 100%;
  }
}

/* Status Section */
.status-section {
  flex-shrink: 0;
}

.status-indicators {
  display: flex;
  gap: 1rem;
}

.status-item {
  display: flex;
  align-items: center;
  gap: 0.3rem;
  font-size: 0.8rem;
  color: #666;
}

.status-item.active {
  color: #10b981;
}

.status-dot {
  width: 6px;
  height: 6px;
  border-radius: 50%;
  background-color: #666;
}

.status-item.active .status-dot {
  background-color: #10b981;
  box-shadow: 0 0 6px rgba(16, 185, 129, 0.5);
}

.status-text {
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.5px;
}

/* Training Progress Bar */
.training-progress-bar {
  position: absolute;
  bottom: 0;
  left: 0;
  right: 0;
  height: 3px;
  background-color: #333;
}

.progress-fill {
  height: 100%;
  background: linear-gradient(90deg, #10b981, #3b82f6);
  transition: width 0.3s ease;
  border-radius: 0 3px 3px 0;
}

/* Responsive Design */
@media (max-width: 1200px) {
  .controls-section {
    gap: 1rem;
  }
  
  .url-input {
    width: 180px;
  }
  
  .model-dropdown {
    min-width: 180px;
  }
}

@media (max-width: 768px) {
  .header-content {
    flex-direction: column;
    gap: 1rem;
    padding: 1rem;
  }
  
  .controls-section {
    flex-wrap: wrap;
    justify-content: center;
  }
  
  .status-section {
    order: -1;
  }
  
  .status-indicators {
    justify-content: center;
  }
  
  .control-group {
    min-width: auto;
  }
  
  .url-input {
    width: 150px;
  }
  
  .model-dropdown {
    min-width: 150px;
  }
}
</style>