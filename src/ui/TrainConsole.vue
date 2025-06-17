<template>
  <div class="train-console">
    <div class="console-header">
      <h3>Training Console</h3>
      <div class="training-status" :class="{ 'active': isTraining, 'paused': isPaused, 'stopped': !isTraining && !isPaused }">
        <div class="status-indicator"></div>
        <span class="status-text">{{ statusText }}</span>
      </div>
    </div>

    <!-- Training Progress Overview -->
    <div class="progress-section">
      <div class="progress-grid">
        <div class="metric-card">
          <div class="metric-label">Tokens Processed</div>
          <div class="metric-value">
            {{ formatNumber(tokensProcessed) }}
            <span class="metric-total">/ {{ formatNumber(totalTokens) }}</span>
          </div>
          <div class="metric-percentage">{{ progressPercentage }}%</div>
        </div>

        <div class="metric-card">
          <div class="metric-label">Throughput</div>
          <div class="metric-value">{{ formatNumber(throughput) }}</div>
          <div class="metric-unit">tokens/sec</div>
        </div>

        <div class="metric-card">
          <div class="metric-label">Current Loss</div>
          <div class="metric-value">{{ currentLoss.toFixed(4) }}</div>
          <div class="metric-trend" :class="lossTrend">
            {{ lossTrend === 'decreasing' ? '↓' : lossTrend === 'increasing' ? '↑' : '→' }}
          </div>
        </div>

        <div class="metric-card">
          <div class="metric-label">ETA</div>
          <div class="metric-value">{{ formatTime(estimatedTimeRemaining) }}</div>
          <div class="metric-unit">{{ etaAccuracy }}</div>
        </div>
      </div>

      <!-- Progress Bar -->
      <div class="progress-bar-container">
        <div class="progress-bar">
          <div 
            class="progress-fill" 
            :style="{ width: progressPercentage + '%' }"
            :class="{ 'pulsing': isTraining }"
          ></div>
        </div>
        <div class="progress-labels">
          <span>{{ formatNumber(tokensProcessed) }} tokens</span>
          <span>{{ progressPercentage }}%</span>
        </div>
      </div>
    </div>

    <!-- Loss Chart Section -->
    <div class="chart-section">
      <div class="chart-header">
        <h4>Training Loss</h4>
        <div class="chart-controls">
          <button 
            class="chart-toggle"
            :class="{ active: showMovingAverage }"
            @click="showMovingAverage = !showMovingAverage"
          >
            Smoothed
          </button>
          <select v-model="chartTimeWindow" class="time-window-select">
            <option value="all">All Time</option>
            <option value="1000">Last 1K steps</option>
            <option value="500">Last 500 steps</option>
            <option value="100">Last 100 steps</option>
          </select>
        </div>
      </div>
      
      <div class="chart-container">
        <LossChart 
          :lossHistory="displayedLossHistory"
          :showMovingAverage="showMovingAverage"
          :isTraining="isTraining"
          :height="200"
        />
      </div>

      <!-- Loss Statistics -->
      <div class="loss-stats">
        <div class="stat-item">
          <span class="stat-label">Initial:</span>
          <span class="stat-value">{{ initialLoss.toFixed(4) }}</span>
        </div>
        <div class="stat-item">
          <span class="stat-label">Best:</span>
          <span class="stat-value">{{ bestLoss.toFixed(4) }}</span>
        </div>
        <div class="stat-item">
          <span class="stat-label">Improvement:</span>
          <span class="stat-value" :class="{ 'positive': lossImprovement > 0 }">
            {{ lossImprovement > 0 ? '-' : '+' }}{{ Math.abs(lossImprovement).toFixed(4) }}
          </span>
        </div>
      </div>
    </div>

    <!-- Training Configuration -->
    <div class="config-section">
      <h4>Configuration</h4>
      <div class="config-grid">
        <div class="config-item">
          <span class="config-label">Mode:</span>
          <span class="config-value">{{ trainingMode }}</span>
        </div>
        <div class="config-item">
          <span class="config-label">Learning Rate:</span>
          <span class="config-value">{{ learningRate }}</span>
        </div>
        <div class="config-item">
          <span class="config-label">Batch Size:</span>
          <span class="config-value">{{ batchSize }}</span>
        </div>
        <div class="config-item">
          <span class="config-label">Rank:</span>
          <span class="config-value">{{ loraRank }}</span>
        </div>
        <div class="config-item">
          <span class="config-label">Steps:</span>
          <span class="config-value">{{ currentStep }} / {{ totalSteps }}</span>
        </div>
        <div class="config-item">
          <span class="config-label">GPU Memory:</span>
          <span class="config-value">{{ memoryUsage.toFixed(1) }} GB</span>
        </div>
      </div>
    </div>

    <!-- Training Controls -->
    <div class="controls-section">
      <button 
        v-if="!isTraining && !trainingCompleted"
        @click="startTraining"
        class="control-btn start-btn"
        :disabled="!canStartTraining"
      >
        <span class="btn-icon">▶</span>
        Start Training
      </button>

      <button 
        v-if="isTraining"
        @click="pauseTraining"
        class="control-btn pause-btn"
      >
        <span class="btn-icon">⏸</span>
        Pause
      </button>

      <button 
        v-if="isPaused"
        @click="resumeTraining"
        class="control-btn resume-btn"
      >
        <span class="btn-icon">▶</span>
        Resume
      </button>

      <button 
        v-if="isTraining || isPaused"
        @click="stopTraining"
        class="control-btn stop-btn"
      >
        <span class="btn-icon">⏹</span>
        Stop
      </button>

      <button 
        v-if="trainingCompleted"
        @click="resetTraining"
        class="control-btn reset-btn"
      >
        <span class="btn-icon">🔄</span>
        Reset
      </button>

      <!-- Emergency Abort -->
      <button 
        v-if="isTraining"
        @click="abortTraining"
        class="control-btn abort-btn"
        title="Emergency stop - may lose progress"
      >
        <span class="btn-icon">🛑</span>
        Abort
      </button>
    </div>

    <!-- Training Log (Collapsible) -->
    <div class="log-section">
      <div class="log-header" @click="showLog = !showLog">
        <h4>Training Log</h4>
        <span class="toggle-arrow" :class="{ 'open': showLog }">▼</span>
      </div>
      
      <div v-show="showLog" class="log-content">
        <div class="log-entries" ref="logContainer">
          <div 
            v-for="(entry, index) in trainingLog" 
            :key="index"
            class="log-entry"
            :class="entry.level"
          >
            <span class="log-time">{{ formatLogTime(entry.timestamp) }}</span>
            <span class="log-message">{{ entry.message }}</span>
          </div>
        </div>
        
        <div class="log-controls">
          <button @click="clearLog" class="log-clear-btn">Clear Log</button>
          <label class="auto-scroll-toggle">
            <input type="checkbox" v-model="autoScrollLog">
            Auto-scroll
          </label>
        </div>
      </div>
    </div>
  </div>
</template>

<script>
import LossChart from './LossChart.vue'

export default {
  name: 'TrainConsole',
  components: {
    LossChart
  },
  props: {
    trainingConfig: {
      type: Object,
      default: () => ({})
    },
    hardwareInfo: {
      type: Object,
      default: () => ({})
    }
  },
  emits: ['training-started', 'training-paused', 'training-resumed', 'training-stopped', 'training-aborted', 'training-completed'],
  data() {
    return {
      // Training State
      isTraining: false,
      isPaused: false,
      trainingCompleted: false,
      canStartTraining: true,

      // Progress Metrics
      tokensProcessed: 0,
      totalTokens: 1000000, // 1M tokens default
      currentStep: 0,
      totalSteps: 1000,
      currentLoss: 3.2451,
      initialLoss: 3.2451,
      bestLoss: 3.2451,
      throughput: 0,
      estimatedTimeRemaining: 0,
      memoryUsage: 2.3,

      // Training Configuration
      trainingMode: 'Adapter (LoRA)',
      learningRate: '3e-4',
      batchSize: 4,
      loraRank: 4,

      // Loss Tracking
      lossHistory: [],
      showMovingAverage: true,
      chartTimeWindow: 'all',

      // Training Log
      trainingLog: [],
      showLog: false,
      autoScrollLog: true,

      // Internal timers
      trainingTimer: null,
      startTime: null,
      pausedTime: 0
    }
  },
  computed: {
    statusText() {
      if (this.trainingCompleted) return 'Completed'
      if (this.isTraining) return 'Training'
      if (this.isPaused) return 'Paused'
      return 'Ready'
    },

    progressPercentage() {
      if (this.totalTokens === 0) return 0
      return Math.min(100, Math.round((this.tokensProcessed / this.totalTokens) * 100))
    },

    lossTrend() {
      if (this.lossHistory.length < 2) return 'stable'
      const recent = this.lossHistory.slice(-5)
      const current = recent[recent.length - 1]
      const previous = recent[0]
      
      if (current < previous - 0.001) return 'decreasing'
      if (current > previous + 0.001) return 'increasing'
      return 'stable'
    },

    lossImprovement() {
      return this.initialLoss - this.currentLoss
    },

    displayedLossHistory() {
      if (this.chartTimeWindow === 'all') {
        return this.lossHistory
      }
      const windowSize = parseInt(this.chartTimeWindow)
      return this.lossHistory.slice(-windowSize)
    },

    etaAccuracy() {
      if (this.tokensProcessed < 1000) return 'estimating...'
      if (this.tokensProcessed < 10000) return '± 50%'
      if (this.tokensProcessed < 50000) return '± 20%'
      return '± 5%'
    }
  },
  mounted() {
    // Initialize with config if provided
    if (this.trainingConfig.totalTokens) {
      this.totalTokens = this.trainingConfig.totalTokens
    }
    if (this.trainingConfig.mode) {
      this.trainingMode = this.trainingConfig.mode === 'adapter' ? 'Adapter (LoRA)' : 'Full-Tune'
    }
    
    this.addLogEntry('info', 'Training console initialized')
  },
  beforeUnmount() {
    if (this.trainingTimer) {
      clearInterval(this.trainingTimer)
    }
  },
  methods: {
    startTraining() {
      this.isTraining = true
      this.isPaused = false
      this.trainingCompleted = false
      this.startTime = Date.now()
      this.pausedTime = 0
      
      this.addLogEntry('info', 'Training started')
      this.startTrainingLoop()
      this.$emit('training-started')
    },

    pauseTraining() {
      this.isTraining = false
      this.isPaused = true
      
      if (this.trainingTimer) {
        clearInterval(this.trainingTimer)
      }
      
      this.addLogEntry('info', 'Training paused')
      this.$emit('training-paused')
    },

    resumeTraining() {
      this.isTraining = true
      this.isPaused = false
      this.pausedTime += Date.now() - this.startTime
      
      this.addLogEntry('info', 'Training resumed')
      this.startTrainingLoop()
      this.$emit('training-resumed')
    },

    stopTraining() {
      this.isTraining = false
      this.isPaused = false
      this.trainingCompleted = true
      
      if (this.trainingTimer) {
        clearInterval(this.trainingTimer)
      }
      
      this.addLogEntry('success', 'Training completed successfully')
      this.$emit('training-stopped')
    },

    abortTraining() {
      this.isTraining = false
      this.isPaused = false
      this.trainingCompleted = false
      
      if (this.trainingTimer) {
        clearInterval(this.trainingTimer)
      }
      
      this.addLogEntry('error', 'Training aborted by user')
      this.$emit('training-aborted')
    },

    resetTraining() {
      // Reset all metrics
      this.tokensProcessed = 0
      this.currentStep = 0
      this.currentLoss = this.initialLoss
      this.bestLoss = this.initialLoss
      this.throughput = 0
      this.estimatedTimeRemaining = 0
      this.lossHistory = []
      this.trainingCompleted = false
      this.startTime = null
      this.pausedTime = 0
      
      this.addLogEntry('info', 'Training reset')
    },

    startTrainingLoop() {
      // Simulate training progress
      this.trainingTimer = setInterval(() => {
        if (!this.isTraining) return

        // Simulate token processing
        const tokensPerStep = Math.floor(Math.random() * 20) + 50 // 50-70 tokens per step
        this.tokensProcessed = Math.min(this.totalTokens, this.tokensProcessed + tokensPerStep)
        this.currentStep++

        // Simulate loss decrease with some noise
        const lossDecrease = 0.001 + Math.random() * 0.002
        const noise = (Math.random() - 0.5) * 0.0005
        this.currentLoss = Math.max(0.1, this.currentLoss - lossDecrease + noise)
        
        if (this.currentLoss < this.bestLoss) {
          this.bestLoss = this.currentLoss
        }

        // Add to loss history
        this.lossHistory.push(this.currentLoss)
        if (this.lossHistory.length > 10000) {
          this.lossHistory = this.lossHistory.slice(-5000) // Keep last 5000 points
        }

        // Calculate throughput
        const elapsedSeconds = (Date.now() - this.startTime - this.pausedTime) / 1000
        this.throughput = elapsedSeconds > 0 ? Math.round(this.tokensProcessed / elapsedSeconds) : 0

        // Calculate ETA
        const remainingTokens = this.totalTokens - this.tokensProcessed
        this.estimatedTimeRemaining = this.throughput > 0 ? Math.round(remainingTokens / this.throughput) : 0

        // Log progress periodically
        if (this.currentStep % 100 === 0) {
          this.addLogEntry('info', `Step ${this.currentStep}: Loss ${this.currentLoss.toFixed(4)}, Throughput ${this.throughput} t/s`)
        }

        // Check if training is complete
        if (this.tokensProcessed >= this.totalTokens || this.currentStep >= this.totalSteps) {
          this.stopTraining()
        }
      }, 100) // Update every 100ms for smooth animation
    },

    addLogEntry(level, message) {
      const entry = {
        timestamp: Date.now(),
        level,
        message
      }
      
      this.trainingLog.push(entry)
      
      // Keep only last 1000 entries
      if (this.trainingLog.length > 1000) {
        this.trainingLog = this.trainingLog.slice(-500)
      }

      // Auto-scroll to bottom
      if (this.autoScrollLog) {
        this.$nextTick(() => {
          const logContainer = this.$refs.logContainer
          if (logContainer) {
            logContainer.scrollTop = logContainer.scrollHeight
          }
        })
      }
    },

    clearLog() {
      this.trainingLog = []
      this.addLogEntry('info', 'Log cleared')
    },

    formatNumber(num) {
      if (num >= 1000000) {
        return (num / 1000000).toFixed(1) + 'M'
      }
      if (num >= 1000) {
        return (num / 1000).toFixed(1) + 'K'
      }
      return num.toString()
    },

    formatTime(seconds) {
      if (seconds === 0) return '--'
      if (seconds < 60) return `${seconds}s`
      if (seconds < 3600) return `${Math.floor(seconds / 60)}m ${seconds % 60}s`
      const hours = Math.floor(seconds / 3600)
      const minutes = Math.floor((seconds % 3600) / 60)
      return `${hours}h ${minutes}m`
    },

    formatLogTime(timestamp) {
      const date = new Date(timestamp)
      return date.toLocaleTimeString()
    }
  }
}
</script>

<style scoped>
.train-console {
  display: flex;
  flex-direction: column;
  height: 100%;
  background-color: #1a1a1a;
  border-radius: 8px;
  overflow-y: auto;
  border: 1px solid #333;
}

/* Header */
.console-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 1rem 1.5rem;
  background-color: #222;
  border-bottom: 1px solid #333;
}

.console-header h3 {
  margin: 0;
  color: #fff;
  font-size: 1.2rem;
}

.training-status {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  font-size: 0.9rem;
  font-weight: 600;
}

.status-indicator {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background-color: #6b7280;
}

.training-status.active .status-indicator {
  background-color: #10b981;
  animation: pulse 2s infinite;
}

.training-status.paused .status-indicator {
  background-color: #f59e0b;
}

.training-status.stopped .status-indicator {
  background-color: #6b7280;
}

@keyframes pulse {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.5; }
}

.status-text {
  color: #ccc;
}

/* Progress Section */
.progress-section {
  padding: 1.5rem;
  border-bottom: 1px solid #333;
}

.progress-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
  gap: 1rem;
  margin-bottom: 1.5rem;
}

.metric-card {
  background-color: #2a2a2a;
  padding: 1rem;
  border-radius: 6px;
  border: 1px solid #444;
}

.metric-label {
  font-size: 0.8rem;
  color: #888;
  margin-bottom: 0.5rem;
  text-transform: uppercase;
  font-weight: 600;
}

.metric-value {
  font-size: 1.4rem;
  font-weight: 700;
  color: #fff;
  margin-bottom: 0.2rem;
}

.metric-total {
  font-size: 1rem;
  color: #888;
  font-weight: normal;
}

.metric-unit,
.metric-percentage {
  font-size: 0.8rem;
  color: #ccc;
}

.metric-trend {
  font-size: 1rem;
  font-weight: bold;
}

.metric-trend.decreasing {
  color: #10b981;
}

.metric-trend.increasing {
  color: #ef4444;
}

.metric-trend.stable {
  color: #6b7280;
}

/* Progress Bar */
.progress-bar-container {
  margin-top: 1rem;
}

.progress-bar {
  width: 100%;
  height: 8px;
  background-color: #444;
  border-radius: 4px;
  overflow: hidden;
}

.progress-fill {
  height: 100%;
  background: linear-gradient(90deg, #10b981, #3b82f6);
  border-radius: 4px;
  transition: width 0.3s ease;
}

.progress-fill.pulsing {
  animation: progress-pulse 2s ease-in-out infinite;
}

@keyframes progress-pulse {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.8; }
}

.progress-labels {
  display: flex;
  justify-content: space-between;
  margin-top: 0.5rem;
  font-size: 0.8rem;
  color: #ccc;
}

/* Chart Section */
.chart-section {
  padding: 1.5rem;
  border-bottom: 1px solid #333;
}

.chart-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 1rem;
}

.chart-header h4 {
  margin: 0;
  color: #fff;
  font-size: 1.1rem;
}

.chart-controls {
  display: flex;
  gap: 1rem;
  align-items: center;
}

.chart-toggle {
  padding: 0.3rem 0.8rem;
  background-color: #374151;
  border: 1px solid #6b7280;
  border-radius: 4px;
  color: #ccc;
  cursor: pointer;
  font-size: 0.8rem;
  transition: all 0.2s;
}

.chart-toggle.active {
  background-color: #10b981;
  border-color: #10b981;
  color: #fff;
}

.time-window-select {
  padding: 0.3rem 0.6rem;
  background-color: #374151;
  border: 1px solid #6b7280;
  border-radius: 4px;
  color: #ccc;
  font-size: 0.8rem;
}

.chart-container {
  background-color: #2a2a2a;
  border-radius: 6px;
  padding: 1rem;
  border: 1px solid #444;
  min-height: 200px;
}

.loss-stats {
  display: flex;
  gap: 2rem;
  margin-top: 1rem;
  font-size: 0.9rem;
}

.stat-item {
  display: flex;
  gap: 0.5rem;
}

.stat-label {
  color: #888;
}

.stat-value {
  color: #fff;
  font-weight: 600;
}

.stat-value.positive {
  color: #10b981;
}

/* Config Section */
.config-section {
  padding: 1.5rem;
  border-bottom: 1px solid #333;
}

.config-section h4 {
  margin: 0 0 1rem 0;
  color: #fff;
  font-size: 1.1rem;
}

.config-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
  gap: 0.8rem;
}

.config-item {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 0.5rem 0;
}

.config-label {
  color: #888;
  font-size: 0.9rem;
}

.config-value {
  color: #fff;
  font-weight: 600;
  font-size: 0.9rem;
}

/* Controls Section */
.controls-section {
  padding: 1.5rem;
  border-bottom: 1px solid #333;
  display: flex;
  gap: 1rem;
  flex-wrap: wrap;
}

.control-btn {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  padding: 0.8rem 1.2rem;
  border: none;
  border-radius: 6px;
  cursor: pointer;
  font-weight: 600;
  font-size: 0.9rem;
  transition: all 0.2s;
}

.start-btn {
  background-color: #10b981;
  color: #fff;
}

.start-btn:hover:not(:disabled) {
  background-color: #059669;
}

.pause-btn {
  background-color: #f59e0b;
  color: #000;
}

.pause-btn:hover {
  background-color: #d97706;
}

.resume-btn {
  background-color: #3b82f6;
  color: #fff;
}

.resume-btn:hover {
  background-color: #2563eb;
}

.stop-btn {
  background-color: #6b7280;
  color: #fff;
}

.stop-btn:hover {
  background-color: #4b5563;
}

.abort-btn {
  background-color: #ef4444;
  color: #fff;
}

.abort-btn:hover {
  background-color: #dc2626;
}

.reset-btn {
  background-color: #8b5cf6;
  color: #fff;
}

.reset-btn:hover {
  background-color: #7c3aed;
}

.control-btn:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.btn-icon {
  font-size: 1rem;
}

/* Log Section */
.log-section {
  flex: 1;
  display: flex;
  flex-direction: column;
  min-height: 200px;
}

.log-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 1rem 1.5rem;
  background-color: #222;
  cursor: pointer;
  user-select: none;
}

.log-header h4 {
  margin: 0;
  color: #fff;
  font-size: 1rem;
}

.toggle-arrow {
  color: #ccc;
  font-size: 0.8rem;
  transition: transform 0.2s;
}

.toggle-arrow.open {
  transform: rotate(180deg);
}

.log-content {
  flex: 1;
  display: flex;
  flex-direction: column;
}

.log-entries {
  flex: 1;
  padding: 1rem 1.5rem;
  overflow-y: auto;
  max-height: 300px;
  font-family: 'Courier New', monospace;
  font-size: 0.8rem;
  line-height: 1.4;
}

.log-entry {
  margin-bottom: 0.5rem;
  display: flex;
  gap: 1rem;
}

.log-time {
  color: #666;
  min-width: 80px;
}

.log-message {
  color: #ccc;
}

.log-entry.info .log-message {
  color: #ccc;
}

.log-entry.success .log-message {
  color: #10b981;
}

.log-entry.error .log-message {
  color: #ef4444;
}

.log-entry.warning .log-message {
  color: #f59e0b;
}

.log-controls {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 0.8rem 1.5rem;
  background-color: #2a2a2a;
  border-top: 1px solid #444;
}

.log-clear-btn {
  padding: 0.4rem 0.8rem;
  background-color: #6b7280;
  border: none;
  border-radius: 4px;
  color: #fff;
  cursor: pointer;
  font-size: 0.8rem;
}

.log-clear-btn:hover {
  background-color: #4b5563;
}

.auto-scroll-toggle {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  color: #ccc;
  font-size: 0.8rem;
  cursor: pointer;
}

/* Responsive */
@media (max-width: 768px) {
  .progress-grid {
    grid-template-columns: repeat(2, 1fr);
  }
  
  .chart-controls {
    flex-direction: column;
    gap: 0.5rem;
  }
  
  .config-grid {
    grid-template-columns: 1fr;
  }
  
  .controls-section {
    flex-direction: column;
  }
  
  .loss-stats {
    flex-direction: column;
    gap: 1rem;
  }
}
</style>