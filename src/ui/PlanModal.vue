<template>
  <div v-if="visible" class="modal-overlay" @click="$emit('close')">
    <div class="modal-content" @click.stop>
      <div class="modal-header">
        <h2>Training Plan Selection</h2>
        <button @click="$emit('close')" class="close-btn">×</button>
      </div>

      <div class="modal-body">
        <!-- Hardware Check Results -->
        <div class="hardware-section">
          <h3>Hardware Detection Results</h3>
          <div class="hardware-grid">
            <div class="hardware-item">
              <div class="hardware-label">WebGPU Support:</div>
              <div class="hardware-value" :class="{ 'supported': hardwareInfo.webGPUSupported, 'unsupported': !hardwareInfo.webGPUSupported }">
                {{ hardwareInfo.webGPUSupported ? '✓ Supported' : '✗ Not Available' }}
              </div>
            </div>

            <div class="hardware-item">
              <div class="hardware-label">GPU:</div>
              <div class="hardware-value">
                {{ hardwareInfo.gpu?.name || 'Unknown' }}
                <span v-if="hardwareInfo.gpu?.vendor" class="gpu-vendor">({{ hardwareInfo.gpu.vendor }})</span>
              </div>
            </div>

            <div class="hardware-item">
              <div class="hardware-label">Estimated TFLOPs:</div>
              <div class="hardware-value tflops">
                {{ hardwareInfo.estimatedTFLOPs?.toFixed(1) || '0.0' }} TFLOPs
              </div>
            </div>

            <div class="hardware-item">
              <div class="hardware-label">Device Memory:</div>
              <div class="hardware-value">
                {{ hardwareInfo.memory?.deviceMemoryGB || 4 }} GB
                <span class="memory-available">(~{{ hardwareInfo.memory?.estimatedAvailableMemoryGB?.toFixed(1) || 2.8 }} GB available)</span>
              </div>
            </div>

            <div class="hardware-item">
              <div class="hardware-label">CPU Cores:</div>
              <div class="hardware-value">
                {{ hardwareInfo.memory?.logicalCores || 4 }} cores
                <span class="cpu-arch">({{ hardwareInfo.memory?.cpuArchitecture || 'unknown' }})</span>
              </div>
            </div>
          </div>
        </div>

        <!-- Training Mode Options -->
        <div class="modes-section">
          <h3>Available Training Modes</h3>
          
          <!-- Adapter Mode -->
          <div 
            class="mode-card" 
            :class="{ 
              'selected': selectedMode === 'adapter', 
              'recommended': adapterMode.recommended,
              'disabled': !adapterMode.feasible 
            }"
            @click="selectMode('adapter')"
          >
            <div class="mode-header">
              <div class="mode-title">
                <h4>Adapter Mode (LoRA)</h4>
                <span v-if="adapterMode.recommended" class="badge recommended">Recommended</span>
                <span v-if="!adapterMode.feasible" class="badge disabled">Not Feasible</span>
              </div>
              <div class="mode-radio">
                <input type="radio" :checked="selectedMode === 'adapter'" :disabled="!adapterMode.feasible" />
              </div>
            </div>
            
            <div class="mode-description">
              <p>Train lightweight LoRA adapters (rank 2-4, 1-2 bit) while keeping the base model frozen.</p>
              
              <div class="mode-specs">
                <div class="spec-item">
                  <span class="spec-label">Scope:</span>
                  <span class="spec-value">LoRA rank {{ adapterMode.rank }}, {{ adapterMode.bits }}-bit quantized</span>
                </div>
                <div class="spec-item">
                  <span class="spec-label">Memory Required:</span>
                  <span class="spec-value">{{ adapterMode.memoryFootprint }} GB</span>
                </div>
                <div class="spec-item">
                  <span class="spec-label">Estimated Time:</span>
                  <span class="spec-value">{{ adapterMode.eta }}</span>
                </div>
                <div class="spec-item">
                  <span class="spec-label">Throughput:</span>
                  <span class="spec-value">~{{ adapterMode.throughput }} tokens/sec</span>
                </div>
              </div>

              <div v-if="!adapterMode.feasible" class="mode-warnings">
                <div v-for="warning in adapterMode.warnings" :key="warning" class="warning">
                  ⚠️ {{ warning }}
                </div>
              </div>
            </div>
          </div>

          <!-- Full-Tune Mode -->
          <div 
            class="mode-card" 
            :class="{ 
              'selected': selectedMode === 'full', 
              'recommended': fullMode.recommended,
              'disabled': !fullMode.feasible 
            }"
            @click="selectMode('full')"
          >
            <div class="mode-header">
              <div class="mode-title">
                <h4>Full-Tune Mode</h4>
                <span v-if="fullMode.recommended" class="badge recommended">Recommended</span>
                <span v-if="!fullMode.feasible" class="badge disabled">Not Feasible</span>
                <span v-if="fullMode.feasible && !fullMode.recommended" class="badge warning">Advanced</span>
              </div>
              <div class="mode-radio">
                <input type="radio" :checked="selectedMode === 'full'" :disabled="!fullMode.feasible" />
              </div>
            </div>
            
            <div class="mode-description">
              <p>Update all model weights with full precision or INT4 quantization.</p>
              
              <div class="mode-specs">
                <div class="spec-item">
                  <span class="spec-label">Scope:</span>
                  <span class="spec-value">All weights, {{ fullMode.precision }} precision</span>
                </div>
                <div class="spec-item">
                  <span class="spec-label">Memory Required:</span>
                  <span class="spec-value">{{ fullMode.memoryFootprint }} GB</span>
                </div>
                <div class="spec-item">
                  <span class="spec-label">Estimated Time:</span>
                  <span class="spec-value">{{ fullMode.eta }}</span>
                </div>
                <div class="spec-item">
                  <span class="spec-label">Throughput:</span>
                  <span class="spec-value">~{{ fullMode.throughput }} tokens/sec</span>
                </div>
              </div>

              <div v-if="fullMode.warnings.length > 0" class="mode-warnings">
                <div v-for="warning in fullMode.warnings" :key="warning" class="warning">
                  ⚠️ {{ warning }}
                </div>
              </div>
            </div>
          </div>
        </div>

        <!-- Model Information -->
        <div v-if="modelInfo" class="model-section">
          <h3>Model Information</h3>
          <div class="model-info">
            <div class="model-item">
              <span class="model-label">Selected Model:</span>
              <span class="model-value">{{ modelInfo.name || 'Custom Model' }}</span>
            </div>
            <div class="model-item">
              <span class="model-label">Estimated Size:</span>
              <span class="model-value">{{ modelInfo.size || 'Unknown' }}</span>
            </div>
            <div v-if="corpusInfo" class="model-item">
              <span class="model-label">Corpus Tokens:</span>
              <span class="model-value">~{{ corpusInfo.tokenCount?.toLocaleString() || 'Unknown' }} tokens</span>
            </div>
          </div>
        </div>
      </div>

      <div class="modal-footer">
        <button @click="$emit('close')" class="cancel-btn">Cancel</button>
        <button 
          @click="confirmSelection" 
          class="confirm-btn"
          :disabled="!selectedMode || !isModeValid"
          :class="{ 'ready': selectedMode && isModeValid }"
        >
          Confirm Plan
        </button>
      </div>
    </div>
  </div>
</template>

<script>
export default {
  name: 'PlanModal',
  props: {
    visible: {
      type: Boolean,
      default: false
    },
    hardwareInfo: {
      type: Object,
      default: () => ({})
    },
    modelInfo: {
      type: Object,
      default: null
    },
    corpusInfo: {
      type: Object,
      default: null
    }
  },
  emits: ['close', 'mode-selected'],
  data() {
    return {
      selectedMode: null
    };
  },
  computed: {
    adapterMode() {
      return this.calculateAdapterMode();
    },
    fullMode() {
      return this.calculateFullMode();
    },
    isModeValid() {
      if (this.selectedMode === 'adapter') {
        return this.adapterMode.feasible;
      } else if (this.selectedMode === 'full') {
        return this.fullMode.feasible;
      }
      return false;
    }
  },
  mounted() {
    // Auto-select the recommended mode
    if (this.adapterMode.recommended) {
      this.selectedMode = 'adapter';
    } else if (this.fullMode.recommended) {
      this.selectedMode = 'full';
    }
  },
  methods: {
    calculateAdapterMode() {
      const tflops = this.hardwareInfo.estimatedTFLOPs || 0;
      const availableMemory = this.hardwareInfo.memory?.estimatedAvailableMemoryGB || 2.8;
      const webGPUSupported = this.hardwareInfo.webGPUSupported;
      const tokens = this.corpusInfo?.tokenCount || 100000;

      // Adapter mode calculations
      const rank = tflops >= 5 ? 4 : 2;
      const bits = tflops >= 8 ? 2 : 1;
      const memoryRequired = 2.0 + (tflops * 0.1); // Base model + adapter overhead
      const gflopsPerToken = rank === 4 ? 1.2 : 0.6; // Adapter computational cost
      const throughput = Math.floor((tflops * 1000) / gflopsPerToken * 0.4); // 40% efficiency
      const etaMinutes = Math.ceil(tokens / Math.max(throughput, 1) / 60);

      const warnings = [];
      let feasible = true;
      let recommended = false;

      if (!webGPUSupported) {
        warnings.push('WebGPU not supported - will fallback to CPU (much slower)');
        feasible = false;
      }

      if (memoryRequired > availableMemory) {
        warnings.push(`Insufficient memory: needs ${memoryRequired.toFixed(1)}GB, available ${availableMemory.toFixed(1)}GB`);
        feasible = false;
      }

      if (tflops < 1.0) {
        warnings.push('Low compute capability detected - training may be very slow');
      }

      if (feasible && webGPUSupported && tflops >= 2.0 && memoryRequired <= availableMemory) {
        recommended = true;
      }

      return {
        rank,
        bits,
        memoryFootprint: memoryRequired.toFixed(1),
        throughput: throughput.toLocaleString(),
        eta: etaMinutes < 60 ? `${etaMinutes} minutes` : `${Math.ceil(etaMinutes / 60)} hours`,
        feasible,
        recommended,
        warnings
      };
    },

    calculateFullMode() {
      const tflops = this.hardwareInfo.estimatedTFLOPs || 0;
      const availableMemory = this.hardwareInfo.memory?.estimatedAvailableMemoryGB || 2.8;
      const webGPUSupported = this.hardwareInfo.webGPUSupported;
      const tokens = this.corpusInfo?.tokenCount || 100000;

      // Full-tune mode calculations
      const precision = tflops >= 15 ? 'FP16' : 'INT4';
      const memoryRequired = precision === 'FP16' ? 8.0 : 4.5; // Much higher memory for full training
      const gflopsPerToken = precision === 'FP16' ? 42 : 25; // Full training computational cost
      const throughput = Math.floor((tflops * 1000) / gflopsPerToken * 0.35); // 35% efficiency
      const etaMinutes = Math.ceil(tokens / Math.max(throughput, 1) / 60);

      const warnings = [];
      let feasible = true;
      let recommended = false;

      if (!webGPUSupported) {
        warnings.push('WebGPU not supported - full training requires GPU acceleration');
        feasible = false;
      }

      if (tflops < 8.0) {
        warnings.push('Minimum 8 TFLOPs recommended for full training');
        feasible = false;
      }

      if (memoryRequired > availableMemory) {
        warnings.push(`Insufficient memory: needs ${memoryRequired.toFixed(1)}GB, available ${availableMemory.toFixed(1)}GB`);
        feasible = false;
      }

      if (etaMinutes > 180) { // More than 3 hours
        warnings.push('Training time may exceed 3 hours - consider using Adapter mode');
      }

      // Only recommend full mode for very powerful hardware
      if (feasible && tflops >= 15 && memoryRequired <= availableMemory && etaMinutes <= 120) {
        recommended = true;
      }

      return {
        precision,
        memoryFootprint: memoryRequired.toFixed(1),
        throughput: throughput.toLocaleString(),
        eta: etaMinutes < 60 ? `${etaMinutes} minutes` : `${Math.ceil(etaMinutes / 60)} hours`,
        feasible,
        recommended,
        warnings
      };
    },

    selectMode(mode) {
      if (mode === 'adapter' && !this.adapterMode.feasible) return;
      if (mode === 'full' && !this.fullMode.feasible) return;
      
      this.selectedMode = mode;
    },

    confirmSelection() {
      if (!this.selectedMode || !this.isModeValid) return;

      const modeData = this.selectedMode === 'adapter' ? this.adapterMode : this.fullMode;
      
      this.$emit('mode-selected', {
        mode: this.selectedMode,
        config: modeData,
        hardwareInfo: this.hardwareInfo
      });
    }
  }
};
</script>

<style scoped>
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

.modal-content {
  background-color: #1a1a1a;
  border-radius: 12px;
  width: 90%;
  max-width: 800px;
  max-height: 90vh;
  display: flex;
  flex-direction: column;
  overflow: hidden;
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

.modal-header h2 {
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

/* Hardware Section */
.hardware-section {
  margin-bottom: 2rem;
}

.hardware-section h3 {
  color: #fff;
  margin: 0 0 1rem 0;
  font-size: 1.2rem;
}

.hardware-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
  gap: 1rem;
}

.hardware-item {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 0.8rem;
  background-color: #2a2a2a;
  border-radius: 6px;
  border: 1px solid #444;
}

.hardware-label {
  color: #ccc;
  font-weight: 500;
}

.hardware-value {
  color: #fff;
  font-weight: 600;
}

.hardware-value.supported {
  color: #4ade80;
}

.hardware-value.unsupported {
  color: #ff6b6b;
}

.hardware-value.tflops {
  color: #3b82f6;
}

.gpu-vendor,
.memory-available,
.cpu-arch {
  font-size: 0.9rem;
  color: #888;
  font-weight: normal;
}

/* Modes Section */
.modes-section {
  margin-bottom: 2rem;
}

.modes-section h3 {
  color: #fff;
  margin: 0 0 1rem 0;
  font-size: 1.2rem;
}

.mode-card {
  background-color: #2a2a2a;
  border: 2px solid #444;
  border-radius: 8px;
  margin-bottom: 1rem;
  padding: 1.5rem;
  cursor: pointer;
  transition: all 0.3s;
}

.mode-card:hover:not(.disabled) {
  border-color: #555;
  background-color: #2d2d2d;
}

.mode-card.selected {
  border-color: #10b981;
  background-color: #0f2419;
}

.mode-card.recommended {
  border-color: #3b82f6;
}

.mode-card.disabled {
  opacity: 0.5;
  cursor: not-allowed;
  background-color: #1a1a1a;
}

.mode-header {
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  margin-bottom: 1rem;
}

.mode-title {
  display: flex;
  align-items: center;
  gap: 1rem;
}

.mode-title h4 {
  margin: 0;
  color: #fff;
  font-size: 1.1rem;
}

.badge {
  font-size: 0.7rem;
  padding: 0.3rem 0.6rem;
  border-radius: 12px;
  font-weight: 600;
  text-transform: uppercase;
}

.badge.recommended {
  background-color: #3b82f6;
  color: #fff;
}

.badge.disabled {
  background-color: #6b7280;
  color: #fff;
}

.badge.warning {
  background-color: #f59e0b;
  color: #000;
}

.mode-radio input {
  width: 1.2rem;
  height: 1.2rem;
}

.mode-description p {
  color: #ccc;
  margin: 0 0 1rem 0;
  line-height: 1.5;
}

.mode-specs {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 0.8rem;
}

.spec-item {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 0.5rem 0;
}

.spec-label {
  color: #888;
  font-size: 0.9rem;
}

.spec-value {
  color: #fff;
  font-weight: 500;
  font-size: 0.9rem;
}

.mode-warnings {
  margin-top: 1rem;
  padding-top: 1rem;
  border-top: 1px solid #444;
}

.warning {
  color: #fbbf24;
  font-size: 0.9rem;
  margin-bottom: 0.5rem;
  display: flex;
  align-items: center;
  gap: 0.5rem;
}

/* Model Section */
.model-section {
  margin-bottom: 1rem;
}

.model-section h3 {
  color: #fff;
  margin: 0 0 1rem 0;
  font-size: 1.2rem;
}

.model-info {
  background-color: #2a2a2a;
  border-radius: 6px;
  padding: 1rem;
  border: 1px solid #444;
}

.model-item {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 0.5rem 0;
}

.model-label {
  color: #ccc;
  font-weight: 500;
}

.model-value {
  color: #fff;
  font-weight: 600;
}

/* Modal Footer */
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
  background-color: #6b7280;
  color: #fff;
  border-color: #6b7280;
}

.confirm-btn:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.confirm-btn.ready {
  background-color: #10b981;
  border-color: #10b981;
}

.confirm-btn.ready:hover:not(:disabled) {
  background-color: #059669;
}

/* Responsive */
@media (max-width: 768px) {
  .modal-content {
    width: 95%;
    margin: 1rem;
  }
  
  .modal-header,
  .modal-body,
  .modal-footer {
    padding: 1rem;
  }
  
  .hardware-grid {
    grid-template-columns: 1fr;
  }
  
  .mode-specs {
    grid-template-columns: 1fr;
  }
  
  .modal-footer {
    flex-direction: column;
    gap: 1rem;
  }
}
</style>