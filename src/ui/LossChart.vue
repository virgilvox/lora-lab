<template>
  <div class="loss-chart" ref="chartContainer">
    <canvas 
      ref="canvas" 
      :width="chartWidth" 
      :height="chartHeight"
      @mousemove="onMouseMove"
      @mouseleave="onMouseLeave"
    ></canvas>
    
    <!-- Tooltip -->
    <div 
      v-if="tooltip.visible" 
      class="chart-tooltip"
      :style="{ left: tooltip.x + 'px', top: tooltip.y + 'px' }"
    >
      <div class="tooltip-content">
        <div class="tooltip-step">Step: {{ tooltip.step }}</div>
        <div class="tooltip-loss">Loss: {{ tooltip.loss }}</div>
      </div>
    </div>

    <!-- Chart controls overlay -->
    <div class="chart-overlay">
      <div v-if="lossHistory.length === 0" class="no-data-message">
        {{ isTraining ? 'Waiting for training data...' : 'No training data available' }}
      </div>
      
      <div v-if="isTraining && lossHistory.length > 0" class="live-indicator">
        <div class="live-dot"></div>
        <span>LIVE</span>
      </div>
    </div>
  </div>
</template>

<script>
export default {
  name: 'LossChart',
  props: {
    lossHistory: {
      type: Array,
      default: () => []
    },
    showMovingAverage: {
      type: Boolean,
      default: true
    },
    isTraining: {
      type: Boolean,
      default: false
    },
    height: {
      type: Number,
      default: 200
    },
    movingAverageWindow: {
      type: Number,
      default: 10
    }
  },
  data() {
    return {
      chartWidth: 400,
      chartHeight: 200,
      tooltip: {
        visible: false,
        x: 0,
        y: 0,
        step: 0,
        loss: 0
      },
      animationFrame: null,
      lastDataLength: 0
    }
  },
  computed: {
    smoothedLossHistory() {
      if (!this.showMovingAverage || this.lossHistory.length < this.movingAverageWindow) {
        return this.lossHistory
      }

      const smoothed = []
      for (let i = 0; i < this.lossHistory.length; i++) {
        const start = Math.max(0, i - Math.floor(this.movingAverageWindow / 2))
        const end = Math.min(this.lossHistory.length, i + Math.ceil(this.movingAverageWindow / 2))
        const window = this.lossHistory.slice(start, end)
        const average = window.reduce((sum, val) => sum + val, 0) / window.length
        smoothed.push(average)
      }
      return smoothed
    },

    chartBounds() {
      if (this.lossHistory.length === 0) {
        return { minLoss: 0, maxLoss: 4, steps: 100 }
      }

      const minLoss = Math.min(...this.lossHistory)
      const maxLoss = Math.max(...this.lossHistory)
      const padding = (maxLoss - minLoss) * 0.1 || 0.1
      
      return {
        minLoss: Math.max(0, minLoss - padding),
        maxLoss: maxLoss + padding,
        steps: this.lossHistory.length
      }
    }
  },
  mounted() {
    this.setupChart()
    this.drawChart()
    window.addEventListener('resize', this.onResize)
  },
  beforeUnmount() {
    window.removeEventListener('resize', this.onResize)
    if (this.animationFrame) {
      cancelAnimationFrame(this.animationFrame)
    }
  },
  watch: {
    lossHistory: {
      handler() {
        // Only trigger redraw if data has actually changed
        if (this.lossHistory.length !== this.lastDataLength) {
          this.lastDataLength = this.lossHistory.length
          this.drawChart()
        }
      },
      deep: true
    },
    showMovingAverage() {
      this.drawChart()
    },
    height() {
      this.setupChart()
      this.drawChart()
    }
  },
  methods: {
    setupChart() {
      if (!this.$refs.chartContainer) return

      const container = this.$refs.chartContainer
      const rect = container.getBoundingClientRect()
      
      this.chartWidth = Math.max(300, rect.width)
      this.chartHeight = this.height
      
      // Ensure canvas is properly sized for high DPI displays
      const canvas = this.$refs.canvas
      const ctx = canvas.getContext('2d')
      const devicePixelRatio = window.devicePixelRatio || 1
      
      canvas.width = this.chartWidth * devicePixelRatio
      canvas.height = this.chartHeight * devicePixelRatio
      canvas.style.width = this.chartWidth + 'px'
      canvas.style.height = this.chartHeight + 'px'
      
      ctx.scale(devicePixelRatio, devicePixelRatio)
    },

    drawChart() {
      if (!this.$refs.canvas) return
      
      const canvas = this.$refs.canvas
      const ctx = canvas.getContext('2d')
      
      // Clear canvas
      ctx.clearRect(0, 0, this.chartWidth, this.chartHeight)
      
      if (this.lossHistory.length === 0) {
        this.drawEmptyState(ctx)
        return
      }

      // Draw grid
      this.drawGrid(ctx)
      
      // Draw axes
      this.drawAxes(ctx)
      
      // Draw raw loss line (always, but lighter if smoothed is shown)
      if (this.showMovingAverage && this.smoothedLossHistory.length > 1) {
        this.drawLossLine(ctx, this.lossHistory, '#444', 1, true) // Lighter raw line
      }
      
      // Draw main line (raw or smoothed)
      const mainData = this.showMovingAverage ? this.smoothedLossHistory : this.lossHistory
      this.drawLossLine(ctx, mainData, '#10b981', 2, false)
      
      // Draw data points on the main line
      this.drawDataPoints(ctx, mainData)
      
      // Draw real-time animation if training
      if (this.isTraining && mainData.length > 1) {
        this.drawTrainingAnimation(ctx, mainData)
      }
    },

    drawEmptyState(ctx) {
      ctx.fillStyle = '#666'
      ctx.font = '14px sans-serif'
      ctx.textAlign = 'center'
      ctx.fillText(
        'No data to display', 
        this.chartWidth / 2, 
        this.chartHeight / 2
      )
    },

    drawGrid(ctx) {
      const padding = 40
      const gridColor = '#333'
      
      ctx.strokeStyle = gridColor
      ctx.lineWidth = 0.5
      ctx.setLineDash([2, 4])
      
      // Horizontal grid lines
      const ySteps = 5
      for (let i = 0; i <= ySteps; i++) {
        const y = padding + (this.chartHeight - 2 * padding) * (i / ySteps)
        ctx.beginPath()
        ctx.moveTo(padding, y)
        ctx.lineTo(this.chartWidth - padding, y)
        ctx.stroke()
      }
      
      // Vertical grid lines
      const xSteps = Math.min(10, Math.floor(this.chartBounds.steps / 50))
      for (let i = 0; i <= xSteps; i++) {
        const x = padding + (this.chartWidth - 2 * padding) * (i / xSteps)
        ctx.beginPath()
        ctx.moveTo(x, padding)
        ctx.lineTo(x, this.chartHeight - padding)
        ctx.stroke()
      }
      
      ctx.setLineDash([])
    },

    drawAxes(ctx) {
      const padding = 40
      const { minLoss, maxLoss, steps } = this.chartBounds
      
      ctx.strokeStyle = '#666'
      ctx.lineWidth = 1
      ctx.fillStyle = '#ccc'
      ctx.font = '10px sans-serif'
      
      // Y-axis labels (loss values)
      ctx.textAlign = 'right'
      ctx.textBaseline = 'middle'
      const ySteps = 5
      for (let i = 0; i <= ySteps; i++) {
        const loss = maxLoss - (maxLoss - minLoss) * (i / ySteps)
        const y = padding + (this.chartHeight - 2 * padding) * (i / ySteps)
        ctx.fillText(loss.toFixed(3), padding - 5, y)
      }
      
      // X-axis labels (step numbers)
      ctx.textAlign = 'center'
      ctx.textBaseline = 'top'
      const xSteps = Math.min(5, Math.floor(steps / 100))
      for (let i = 0; i <= xSteps; i++) {
        const step = Math.floor(steps * (i / xSteps))
        const x = padding + (this.chartWidth - 2 * padding) * (i / xSteps)
        ctx.fillText(step.toString(), x, this.chartHeight - padding + 5)
      }
      
      // Axis labels
      ctx.fillStyle = '#888'
      ctx.font = '12px sans-serif'
      ctx.textAlign = 'center'
      ctx.fillText('Training Steps', this.chartWidth / 2, this.chartHeight - 10)
      
      ctx.save()
      ctx.translate(15, this.chartHeight / 2)
      ctx.rotate(-Math.PI / 2)
      ctx.fillText('Loss', 0, 0)
      ctx.restore()
    },

    drawLossLine(ctx, data, color, lineWidth = 2, isDashed = false) {
      if (data.length < 2) return
      
      const padding = 40
      const { minLoss, maxLoss } = this.chartBounds
      
      ctx.strokeStyle = color
      ctx.lineWidth = lineWidth
      
      if (isDashed) {
        ctx.setLineDash([3, 3])
      }
      
      ctx.beginPath()
      
      for (let i = 0; i < data.length; i++) {
        const x = padding + (this.chartWidth - 2 * padding) * (i / (data.length - 1))
        const y = padding + (this.chartHeight - 2 * padding) * (1 - (data[i] - minLoss) / (maxLoss - minLoss))
        
        if (i === 0) {
          ctx.moveTo(x, y)
        } else {
          ctx.lineTo(x, y)
        }
      }
      
      ctx.stroke()
      ctx.setLineDash([])
    },

    drawDataPoints(ctx, data) {
      if (data.length === 0) return
      
      const padding = 40
      const { minLoss, maxLoss } = this.chartBounds
      
      ctx.fillStyle = '#10b981'
      
      // Only show points if we have few data points or at specific intervals
      const showAllPoints = data.length < 50
      const interval = showAllPoints ? 1 : Math.max(1, Math.floor(data.length / 20))
      
      for (let i = 0; i < data.length; i += interval) {
        const x = padding + (this.chartWidth - 2 * padding) * (i / (data.length - 1))
        const y = padding + (this.chartHeight - 2 * padding) * (1 - (data[i] - minLoss) / (maxLoss - minLoss))
        
        ctx.beginPath()
        ctx.arc(x, y, 2, 0, 2 * Math.PI)
        ctx.fill()
      }
      
      // Always show the last point with a larger circle
      if (data.length > 0) {
        const lastIndex = data.length - 1
        const x = padding + (this.chartWidth - 2 * padding) * (lastIndex / (data.length - 1))
        const y = padding + (this.chartHeight - 2 * padding) * (1 - (data[lastIndex] - minLoss) / (maxLoss - minLoss))
        
        ctx.fillStyle = '#fff'
        ctx.beginPath()
        ctx.arc(x, y, 4, 0, 2 * Math.PI)
        ctx.fill()
        
        ctx.fillStyle = '#10b981'
        ctx.beginPath()
        ctx.arc(x, y, 3, 0, 2 * Math.PI)
        ctx.fill()
      }
    },

    drawTrainingAnimation(ctx, data) {
      if (data.length < 2) return
      
      const padding = 40
      const { minLoss, maxLoss } = this.chartBounds
      const lastIndex = data.length - 1
      
      const x = padding + (this.chartWidth - 2 * padding) * (lastIndex / (data.length - 1))
      const y = padding + (this.chartHeight - 2 * padding) * (1 - (data[lastIndex] - minLoss) / (maxLoss - minLoss))
      
      // Animated pulse around the last point
      const time = Date.now() / 1000
      const pulseRadius = 8 + Math.sin(time * 3) * 3
      
      ctx.strokeStyle = '#10b981'
      ctx.lineWidth = 2
      ctx.globalAlpha = 0.5 + Math.sin(time * 3) * 0.3
      ctx.beginPath()
      ctx.arc(x, y, pulseRadius, 0, 2 * Math.PI)
      ctx.stroke()
      ctx.globalAlpha = 1
      
      // Schedule next frame
      if (this.isTraining) {
        this.animationFrame = requestAnimationFrame(() => this.drawChart())
      }
    },

    onMouseMove(event) {
      if (this.lossHistory.length === 0) return
      
      const rect = this.$refs.canvas.getBoundingClientRect()
      const x = event.clientX - rect.left
      const y = event.clientY - rect.top
      
      const padding = 40
      const dataX = (x - padding) / (this.chartWidth - 2 * padding)
      
      if (dataX >= 0 && dataX <= 1) {
        const dataIndex = Math.round(dataX * (this.lossHistory.length - 1))
        const displayData = this.showMovingAverage ? this.smoothedLossHistory : this.lossHistory
        
        if (dataIndex >= 0 && dataIndex < displayData.length) {
          this.tooltip = {
            visible: true,
            x: Math.min(x + 10, this.chartWidth - 120),
            y: Math.max(y - 10, 0),
            step: dataIndex,
            loss: displayData[dataIndex].toFixed(4)
          }
        }
      }
    },

    onMouseLeave() {
      this.tooltip.visible = false
    },

    onResize() {
      // Debounce resize events
      clearTimeout(this.resizeTimeout)
      this.resizeTimeout = setTimeout(() => {
        this.setupChart()
        this.drawChart()
      }, 100)
    }
  }
}
</script>

<style scoped>
.loss-chart {
  position: relative;
  width: 100%;
  height: 100%;
  background-color: transparent;
}

canvas {
  display: block;
  cursor: crosshair;
}

.chart-overlay {
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  pointer-events: none;
  display: flex;
  align-items: center;
  justify-content: center;
}

.no-data-message {
  color: #666;
  font-size: 1rem;
  text-align: center;
}

.live-indicator {
  position: absolute;
  top: 10px;
  right: 10px;
  display: flex;
  align-items: center;
  gap: 0.5rem;
  background-color: rgba(0, 0, 0, 0.7);
  padding: 0.3rem 0.6rem;
  border-radius: 12px;
  font-size: 0.7rem;
  font-weight: 600;
  color: #10b981;
  text-transform: uppercase;
}

.live-dot {
  width: 6px;
  height: 6px;
  background-color: #10b981;
  border-radius: 50%;
  animation: live-pulse 1.5s ease-in-out infinite;
}

@keyframes live-pulse {
  0%, 100% {
    opacity: 1;
    transform: scale(1);
  }
  50% {
    opacity: 0.5;
    transform: scale(1.2);
  }
}

.chart-tooltip {
  position: absolute;
  background-color: rgba(0, 0, 0, 0.9);
  border: 1px solid #444;
  border-radius: 6px;
  padding: 0.5rem;
  pointer-events: none;
  z-index: 1000;
  font-size: 0.8rem;
  min-width: 100px;
}

.tooltip-content {
  color: #fff;
}

.tooltip-step {
  color: #ccc;
  margin-bottom: 0.2rem;
}

.tooltip-loss {
  color: #10b981;
  font-weight: 600;
}

/* Responsive adjustments */
@media (max-width: 768px) {
  .chart-tooltip {
    font-size: 0.7rem;
    padding: 0.3rem;
    min-width: 80px;
  }
  
  .live-indicator {
    font-size: 0.6rem;
    padding: 0.2rem 0.4rem;
  }
}
</style>