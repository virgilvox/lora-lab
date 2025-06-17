<template>
  <footer class="footer-status">
    <div class="footer-content">
      <!-- GPU Status -->
      <div class="status-group">
        <div class="status-item">
          <div class="status-icon">🔧</div>
          <div class="status-details">
            <div class="status-label">GPU</div>
            <div class="status-value">
              {{ hardwareInfo.gpu?.name || 'Unknown' }}
              <span v-if="hardwareInfo.estimatedTFLOPs" class="tflops">
                ({{ hardwareInfo.estimatedTFLOPs.toFixed(1) }} TFLOPs)
              </span>
            </div>
            <div class="status-indicator" :class="gpuStatusClass">
              {{ gpuStatusText }}
            </div>
          </div>
        </div>
      </div>

      <!-- Memory Usage -->
      <div class="status-group">
        <div class="status-item">
          <div class="status-icon">💾</div>
          <div class="status-details">
            <div class="status-label">Memory</div>
            <div class="memory-info">
              <div class="memory-bar-container">
                <div class="memory-bar">
                  <div 
                    class="memory-used" 
                    :style="{ width: memoryPercentage + '%' }"
                    :class="{ 'warning': memoryPercentage > 80, 'critical': memoryPercentage > 90 }"
                  ></div>
                </div>
                <div class="memory-text">
                  {{ currentMemoryUsage.toFixed(1) }} / {{ totalMemory.toFixed(1) }} GB
                  <span class="memory-percentage">({{ memoryPercentage }}%)</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      <!-- Training Status & ETA -->
      <div class="status-group">
        <div class="status-item">
          <div class="status-icon">⏱️</div>
          <div class="status-details">
            <div class="status-label">Training</div>
            <div class="status-value">
              <span class="training-mode">{{ trainingStatus.mode || 'Idle' }}</span>
              <div v-if="trainingStatus.eta" class="eta-info">
                ETA: {{ formatTime(trainingStatus.eta) }}
              </div>
              <div v-if="trainingStatus.throughput > 0" class="throughput-info">
                {{ formatNumber(trainingStatus.throughput) }} tokens/sec
              </div>
            </div>
            <div class="training-progress" v-if="trainingStatus.progress > 0">
              <div class="progress-bar-mini">
                <div 
                  class="progress-fill-mini" 
                  :style="{ width: trainingStatus.progress + '%' }"
                ></div>
              </div>
              <span class="progress-text">{{ trainingStatus.progress }}%</span>
            </div>
          </div>
        </div>
      </div>

      <!-- WebGPU Status -->
      <div class="status-group">
        <div class="status-item">
          <div class="status-icon">🌐</div>
          <div class="status-details">
            <div class="status-label">WebGPU</div>
            <div class="status-value">
              <span class="webgpu-status" :class="{ 'supported': hardwareInfo.webGPUSupported }">
                {{ hardwareInfo.webGPUSupported ? 'Supported' : 'Not Available' }}
              </span>
              <div v-if="hardwareInfo.webGPUSupported" class="webgpu-details">
                {{ hardwareInfo.gpu?.vendor || 'Unknown Vendor' }}
              </div>
            </div>
          </div>
        </div>
      </div>

      <!-- Adapter Actions -->
      <div class="status-group actions-group">
        <div class="adapter-section">
          <div class="adapter-status">
            <div class="status-icon">🧬</div>
            <div class="adapter-info">
              <div class="adapter-label">LoRA Adapter</div>
              <div class="adapter-state">
                <span v-if="!adapterReady" class="no-adapter">Not Ready</span>
                <span v-else class="adapter-ready">Ready for Download</span>
              </div>
            </div>
          </div>
          
          <button 
            @click="handleDownloadAdapter"
            class="download-btn"
            :disabled="!adapterReady"
            :class="{ 'ready': adapterReady, 'pulse': adapterReady }"
          >
            <span class="btn-icon">📥</span>
            Download Adapter
          </button>
        </div>
      </div>
    </div>

    <!-- System Notifications -->
    <div v-if="showNotifications" class="notifications">
      <div 
        v-for="notification in notifications" 
        :key="notification.id"
        class="notification"
        :class="notification.type"
      >
        <div class="notification-icon">
          {{ getNotificationIcon(notification.type) }}
        </div>
        <div class="notification-content">
          <div class="notification-message">{{ notification.message }}</div>
          <div v-if="notification.details" class="notification-details">
            {{ notification.details }}
          </div>
        </div>
        <button @click="dismissNotification(notification.id)" class="dismiss-btn">
          ×
        </button>
      </div>
    </div>
  </footer>
</template>

<script>
export default {
  name: 'FooterStatus',
  props: {
    hardwareInfo: {
      type: Object,
      default: () => ({})
    },
    trainingStatus: {
      type: Object,
      default: () => ({})
    },
    adapterReady: {
      type: Boolean,
      default: false
    },
    currentMemoryUsage: {
      type: Number,
      default: 2.3
    }
  },
  emits: ['download-adapter'],
  data() {
    return {
      notifications: [],
      nextNotificationId: 1,
      showNotifications: true
    }
  },
  computed: {
    totalMemory() {
      return this.hardwareInfo.memory?.deviceMemoryGB || 4
    },

    memoryPercentage() {
      return Math.round((this.currentMemoryUsage / this.totalMemory) * 100)
    },

    gpuStatusClass() {
      if (!this.hardwareInfo.webGPUSupported) return 'error'
      if (this.hardwareInfo.estimatedTFLOPs >= 8) return 'excellent'
      if (this.hardwareInfo.estimatedTFLOPs >= 4) return 'good'
      if (this.hardwareInfo.estimatedTFLOPs >= 1) return 'fair'
      return 'poor'
    },

    gpuStatusText() {
      if (!this.hardwareInfo.webGPUSupported) return 'Unsupported'
      
      const tflops = this.hardwareInfo.estimatedTFLOPs || 0
      if (tflops >= 8) return 'Excellent'
      if (tflops >= 4) return 'Good'
      if (tflops >= 1) return 'Fair'
      return 'Limited'
    }
  },
  mounted() {
    this.checkSystemStatus()
  },
  watch: {
    'hardwareInfo.webGPUSupported': {
      handler(newVal) {
        if (newVal === false) {
          this.addNotification('warning', 'WebGPU not supported', 'Training will be significantly slower on CPU')
        }
      }
    },
    
    memoryPercentage: {
      handler(newVal) {
        if (newVal > 90) {
          this.addNotification('error', 'Memory critically low', 'Training may fail or become unstable')
        } else if (newVal > 80) {
          this.addNotification('warning', 'Memory usage high', 'Consider reducing batch size or model complexity')
        }
      }
    }
  },
  methods: {
    handleDownloadAdapter() {
      if (!this.adapterReady) return
      
      this.$emit('download-adapter')
      this.addNotification('success', 'Adapter downloaded', 'LoRA adapter has been saved to your downloads')
    },

    checkSystemStatus() {
      // Check for potential issues
      if (this.totalMemory < 4) {
        this.addNotification('warning', 'Low system memory', 'Training large models may not be possible')
      }
      
      if (!this.hardwareInfo.webGPUSupported) {
        this.addNotification('error', 'WebGPU unavailable', 'GPU acceleration not available - training will be slow')
      }
    },

    addNotification(type, message, details = null) {
      const notification = {
        id: this.nextNotificationId++,
        type,
        message,
        details,
        timestamp: Date.now()
      }
      
      this.notifications.push(notification)
      
      // Auto-dismiss after 10 seconds for success/info, 15 seconds for warnings
      const dismissDelay = type === 'error' ? 20000 : type === 'warning' ? 15000 : 10000
      setTimeout(() => {
        this.dismissNotification(notification.id)
      }, dismissDelay)
    },

    dismissNotification(id) {
      const index = this.notifications.findIndex(n => n.id === id)
      if (index !== -1) {
        this.notifications.splice(index, 1)
      }
    },

    getNotificationIcon(type) {
      const icons = {
        success: '✅',
        warning: '⚠️',
        error: '❌',
        info: 'ℹ️'
      }
      return icons[type] || 'ℹ️'
    },

    formatTime(seconds) {
      if (seconds === 0 || !seconds) return '--'
      if (seconds < 60) return `${seconds}s`
      if (seconds < 3600) return `${Math.floor(seconds / 60)}m ${seconds % 60}s`
      const hours = Math.floor(seconds / 3600)
      const minutes = Math.floor((seconds % 3600) / 60)
      return `${hours}h ${minutes}m`
    },

    formatNumber(num) {
      if (num >= 1000000) {
        return (num / 1000000).toFixed(1) + 'M'
      }
      if (num >= 1000) {
        return (num / 1000).toFixed(1) + 'K'
      }
      return num.toString()
    }
  }
}
</script>

<style scoped>
.footer-status {
  background-color: #1a1a1a;
  border-top: 1px solid #333;
  position: relative;
}

.footer-content {
  display: flex;
  align-items: center;
  padding: 0.8rem 1.5rem;
  gap: 2rem;
  overflow-x: auto;
}

/* Status Groups */
.status-group {
  display: flex;
  align-items: center;
  gap: 1rem;
  flex-shrink: 0;
}

.status-item {
  display: flex;
  align-items: center;
  gap: 0.8rem;
}

.status-icon {
  font-size: 1.2rem;
  flex-shrink: 0;
}

.status-details {
  display: flex;
  flex-direction: column;
  gap: 0.2rem;
}

.status-label {
  font-size: 0.7rem;
  color: #888;
  text-transform: uppercase;
  font-weight: 600;
  letter-spacing: 0.5px;
}

.status-value {
  font-size: 0.8rem;
  color: #fff;
  font-weight: 500;
  line-height: 1.2;
}

.tflops {
  color: #3b82f6;
  font-weight: 600;
}

/* Status Indicators */
.status-indicator {
  font-size: 0.7rem;
  font-weight: 600;
  text-transform: uppercase;
  padding: 0.1rem 0.4rem;
  border-radius: 8px;
  letter-spacing: 0.3px;
}

.status-indicator.excellent {
  background-color: #065f46;
  color: #10b981;
}

.status-indicator.good {
  background-color: #1e3a8a;
  color: #3b82f6;
}

.status-indicator.fair {
  background-color: #92400e;
  color: #f59e0b;
}

.status-indicator.poor {
  background-color: #7c2d12;
  color: #f97316;
}

.status-indicator.error {
  background-color: #7f1d1d;
  color: #ef4444;
}

/* Memory Bar */
.memory-info {
  min-width: 120px;
}

.memory-bar-container {
  display: flex;
  flex-direction: column;
  gap: 0.3rem;
}

.memory-bar {
  width: 100%;
  height: 6px;
  background-color: #374151;
  border-radius: 3px;
  overflow: hidden;
}

.memory-used {
  height: 100%;
  background-color: #10b981;
  border-radius: 3px;
  transition: all 0.3s ease;
}

.memory-used.warning {
  background-color: #f59e0b;
}

.memory-used.critical {
  background-color: #ef4444;
}

.memory-text {
  font-size: 0.7rem;
  color: #ccc;
}

.memory-percentage {
  color: #888;
}

/* Training Status */
.training-mode {
  color: #10b981;
  font-weight: 600;
}

.eta-info,
.throughput-info {
  font-size: 0.7rem;
  color: #888;
}

.training-progress {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  margin-top: 0.2rem;
}

.progress-bar-mini {
  width: 60px;
  height: 4px;
  background-color: #374151;
  border-radius: 2px;
  overflow: hidden;
}

.progress-fill-mini {
  height: 100%;
  background-color: #10b981;
  border-radius: 2px;
  transition: width 0.3s ease;
}

.progress-text {
  font-size: 0.7rem;
  color: #ccc;
  font-weight: 600;
}

/* WebGPU Status */
.webgpu-status.supported {
  color: #10b981;
  font-weight: 600;
}

.webgpu-details {
  font-size: 0.7rem;
  color: #888;
}

/* Adapter Section */
.actions-group {
  margin-left: auto;
  flex-shrink: 0;
}

.adapter-section {
  display: flex;
  align-items: center;
  gap: 1rem;
}

.adapter-status {
  display: flex;
  align-items: center;
  gap: 0.8rem;
}

.adapter-info {
  display: flex;
  flex-direction: column;
  gap: 0.2rem;
}

.adapter-label {
  font-size: 0.7rem;
  color: #888;
  text-transform: uppercase;
  font-weight: 600;
  letter-spacing: 0.5px;
}

.adapter-state {
  font-size: 0.8rem;
}

.no-adapter {
  color: #666;
}

.adapter-ready {
  color: #10b981;
  font-weight: 600;
}

/* Download Button */
.download-btn {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  padding: 0.6rem 1rem;
  background-color: #374151;
  border: 1px solid #6b7280;
  border-radius: 6px;
  color: #ccc;
  cursor: pointer;
  font-size: 0.8rem;
  font-weight: 600;
  transition: all 0.2s;
}

.download-btn:hover:not(:disabled) {
  background-color: #4b5563;
}

.download-btn.ready {
  background-color: #065f46;
  border-color: #10b981;
  color: #10b981;
}

.download-btn.ready:hover:not(:disabled) {
  background-color: #047857;
  color: #fff;
}

.download-btn.pulse {
  animation: pulse-glow 2s ease-in-out infinite;
}

@keyframes pulse-glow {
  0%, 100% {
    box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.4);
  }
  50% {
    box-shadow: 0 0 0 8px rgba(16, 185, 129, 0);
  }
}

.download-btn:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.btn-icon {
  font-size: 1rem;
}

/* Notifications */
.notifications {
  position: absolute;
  bottom: 100%;
  right: 1rem;
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
  max-width: 400px;
  z-index: 1000;
}

.notification {
  display: flex;
  align-items: flex-start;
  gap: 0.8rem;
  padding: 1rem;
  background-color: #2a2a2a;
  border: 1px solid #444;
  border-radius: 8px;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
  animation: slideUp 0.3s ease-out;
}

@keyframes slideUp {
  from {
    transform: translateY(20px);
    opacity: 0;
  }
  to {
    transform: translateY(0);
    opacity: 1;
  }
}

.notification.success {
  border-color: #10b981;
  background-color: #064e3b;
}

.notification.warning {
  border-color: #f59e0b;
  background-color: #78350f;
}

.notification.error {
  border-color: #ef4444;
  background-color: #7f1d1d;
}

.notification.info {
  border-color: #3b82f6;
  background-color: #1e3a8a;
}

.notification-icon {
  font-size: 1.2rem;
  flex-shrink: 0;
}

.notification-content {
  flex: 1;
  display: flex;
  flex-direction: column;
  gap: 0.3rem;
}

.notification-message {
  font-size: 0.9rem;
  font-weight: 600;
  color: #fff;
}

.notification-details {
  font-size: 0.8rem;
  color: #ccc;
  line-height: 1.3;
}

.dismiss-btn {
  background: none;
  border: none;
  color: #888;
  cursor: pointer;
  font-size: 1.2rem;
  padding: 0;
  width: 1.5rem;
  height: 1.5rem;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  transition: all 0.2s;
  flex-shrink: 0;
}

.dismiss-btn:hover {
  background-color: rgba(255, 255, 255, 0.1);
  color: #fff;
}

/* Responsive Design */
@media (max-width: 1200px) {
  .footer-content {
    gap: 1.5rem;
  }
  
  .adapter-section {
    flex-direction: column;
    gap: 0.5rem;
    align-items: flex-end;
  }
}

@media (max-width: 768px) {
  .footer-content {
    flex-direction: column;
    gap: 1rem;
    padding: 1rem;
  }
  
  .status-group {
    width: 100%;
    justify-content: space-between;
  }
  
  .actions-group {
    margin-left: 0;
    width: 100%;
  }
  
  .adapter-section {
    flex-direction: row;
    justify-content: space-between;
    align-items: center;
  }
  
  .notifications {
    right: 0.5rem;
    left: 0.5rem;
    max-width: none;
  }
}
</style>