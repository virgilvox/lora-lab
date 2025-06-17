<template>
  <header class="header-bar">
    <div class="header-content">
      <!-- Left: Logo -->
      <div class="logo-section">
        <div class="logo">
          <span class="logo-icon">🧬</span>
          <span class="logo-text">LoRA Lab</span>
        </div>
      </div>

      <!-- Center: Controls -->
      <div class="controls-section">
        <div class="control-group">
          <label for="model-select" class="control-label">Model</label>
          <div class="model-input-group">
            <select 
              id="model-select"
              :value="selectedModel?.id || ''"
              @change="handleModelChange"
              class="model-dropdown"
            >
              <option value="" disabled>Select Model...</option>
              <option 
                v-for="model in modelOptions" 
                :key="model.id"
                :value="model.id"
              >
                {{ model.name }}
              </option>
            </select>
            <input 
              v-if="selectedModel?.id === 'custom'"
              v-model="customModelUrl"
              type="url"
              placeholder="Enter model ID (e.g., Xenova/model-name)"
              class="url-input"
              :disabled="isTraining"
            />
            <button 
              v-if="selectedModel?.id === 'custom'"
              @click="handleLoadUrl"
              class="load-url-btn"
              :disabled="!customModelUrl || isTraining"
            >
              Load
            </button>
          </div>
        </div>

        <div class="separator"></div>

        <div class="control-group">
          <label class="control-label">Setup</label>
          <div class="button-group">
            <button 
              @click="$emit('corpus-requested')"
              class="control-btn"
              :disabled="isTraining"
              :class="{ 'active': hasCorpus }"
              title="Upload or paste training data"
            >
              <span class="btn-icon">📄</span>
              Corpus
            </button>
            <button 
              @click="$emit('plan-requested')"
              class="control-btn"
              :disabled="isTraining"
              :class="{ 'active': hasTrainingPlan }"
              title="Configure Training Plan"
            >
              <span class="btn-icon">⚙️</span>
              Plan
            </button>
          </div>
        </div>
        
        <div class="separator"></div>

        <div class="control-group">
           <button 
            @click="handleTrainClick"
            class="train-btn"
            :disabled="!canStartTraining || isTraining"
            :class="{ 
              'ready': canStartTraining && !isTraining,
              'training': isTraining
            }"
          >
            <span class="btn-icon">
              {{ isTraining ? '⏸' : '▶' }}
            </span>
            {{ isTraining ? 'Training...' : 'Train' }}
          </button>
        </div>
      </div>

      <!-- Right: Status -->
      <div class="status-section">
        <div class="status-indicators">
          <div class="status-item" :class="{ 'active': selectedModel }" title="Model Loaded">
            <div class="status-dot"></div>
          </div>
          <div class="status-item" :class="{ 'active': hasCorpus }" title="Corpus Ready">
            <div class="status-dot"></div>
          </div>
          <div class="status-item" :class="{ 'active': hasTrainingPlan }" title="Plan Configured">
            <div class="status-dot"></div>
          </div>
          <div class="status-item" :class="{ 'active': isGPUReady }" title="GPU Ready">
            <div class="status-dot"></div>
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
    'corpus-requested', 
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
  z-index: 100;
}

.header-content {
  display: flex;
  align-items: center;
  padding: 0.75rem 1.5rem;
  gap: 1.5rem;
}

/* Logo Section */
.logo-section {
  flex-shrink: 0;
}

.logo {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  font-size: 1.25rem;
  font-weight: 700;
  color: #fff;
}

.logo-icon {
  font-size: 1.5rem;
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
}

.control-group {
  display: flex;
  align-items: center;
  gap: 0.75rem;
}

.control-label {
  font-size: 0.8rem;
  color: #888;
  font-weight: 500;
}

.separator {
  width: 1px;
  height: 24px;
  background-color: #333;
}

/* Model Selection */
.model-input-group {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  background-color: #2a2a2a;
  border-radius: 6px;
  border: 1px solid #444;
  padding: 0.25rem;
}

.model-dropdown {
  padding: 0.4rem 0.8rem;
  background-color: transparent;
  border: none;
  color: #fff;
  font-size: 0.9rem;
  cursor: pointer;
  -webkit-appearance: none;
  -moz-appearance: none;
  appearance: none;
}

.model-dropdown:focus {
  outline: none;
}

.url-input {
  padding: 0.4rem 0.8rem;
  background-color: #1a1a1a;
  border: 1px solid #555;
  border-radius: 4px;
  color: #fff;
  font-size: 0.9rem;
  width: 200px;
}

.load-url-btn {
  padding: 0.4rem 0.8rem;
  background-color: #3b82f6;
  border: none;
  border-radius: 4px;
  color: #fff;
  cursor: pointer;
  font-size: 0.9rem;
  font-weight: 500;
}

/* Button Group */
.button-group {
  display: flex;
  gap: 0.5rem;
  background-color: #2a2a2a;
  border: 1px solid #444;
  border-radius: 6px;
  padding: 0.25rem;
}

.control-btn {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  padding: 0.4rem 0.8rem;
  background-color: transparent;
  border: 1px solid transparent;
  border-radius: 4px;
  color: #ccc;
  cursor: pointer;
  font-size: 0.9rem;
  font-weight: 500;
  transition: all 0.2s;
}

.control-btn:hover:not(:disabled) {
  background-color: #333;
}

.control-btn.active {
  background-color: #10b981;
  color: #fff;
  border-color: #10b981;
}

/* Train Button */
.train-btn {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  padding: 0.5rem 1.25rem;
  border: none;
  border-radius: 6px;
  cursor: pointer;
  font-size: 0.9rem;
  font-weight: 600;
  transition: all 0.2s;
  background-color: #374151;
  color: #6b7280;
}

.train-btn.ready {
  background: linear-gradient(45deg, #10b981, #059669);
  color: #fff;
  box-shadow: 0 2px 8px rgba(16, 185, 129, 0.3);
}

.train-btn.ready:hover {
  background: linear-gradient(45deg, #059669, #047857);
}

.train-btn.training {
  background: linear-gradient(45deg, #f59e0b, #d97706);
  color: #fff;
}

.train-btn:disabled:not(.ready) {
  cursor: not-allowed;
}

/* Status Section */
.status-section {
  margin-left: auto;
  flex-shrink: 0;
}

.status-indicators {
  display: flex;
  gap: 0.75rem;
  background-color: #2a2a2a;
  padding: 0.4rem 0.8rem;
  border-radius: 6px;
  border: 1px solid #444;
}

.status-item {
  display: flex;
  align-items: center;
}

.status-dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background-color: #444;
  transition: all 0.3s;
}

.status-item.active .status-dot {
  background-color: #10b981;
  box-shadow: 0 0 6px rgba(16, 185, 129, 0.7);
}

/* Training Progress Bar */
.training-progress-bar {
  position: absolute;
  bottom: -1px;
  left: 0;
  right: 0;
  height: 2px;
  background-color: transparent;
}

.progress-fill {
  height: 100%;
  background: linear-gradient(90deg, #10b981, #3b82f6);
  transition: width 0.3s ease;
  border-radius: 0 2px 2px 0;
}

/* Responsive Design */
@media (max-width: 1200px) {
  .header-content {
    flex-wrap: wrap;
    gap: 1rem;
  }
  .controls-section {
    width: 100%;
    order: 2;
    gap: 1rem;
    justify-content: flex-start;
  }
  .status-section {
    order: 1;
    margin-left: 0;
  }
}

@media (max-width: 768px) {
  .logo-text {
    display: none;
  }
  .controls-section {
    flex-wrap: wrap;
  }
  .separator {
    display: none;
  }
}
</style>