/**
 * LoRA Rank Scheduler for Dynamic Adapter Optimization
 * Automatically adjusts adapter rank based on training metrics and hardware constraints
 */

/**
 * Rank scheduling strategies
 */
export const RANK_STRATEGIES = {
  FIXED: 'fixed',           // Fixed rank throughout training
  PROGRESSIVE: 'progressive', // Start low, increase gradually
  ADAPTIVE: 'adaptive',     // Adjust based on loss and gradient norms
  HARDWARE_AWARE: 'hardware_aware' // Optimize for available compute/memory
};

/**
 * LoRA Rank Scheduler Class
 */
export class LoRARankScheduler {
  constructor(config = {}) {
    this.config = {
      strategy: RANK_STRATEGIES.ADAPTIVE,
      initialRank: 4,
      minRank: 2,
      maxRank: 64,
      targetMemoryUsageGB: 4.0,
      performanceThreshold: 0.95,
      convergenceWindow: 100,
      adaptationRate: 0.1,
      ...config
    };

    this.currentRank = this.config.initialRank;
    this.trainingHistory = [];
    this.performanceHistory = [];
    this.memoryHistory = [];
    this.lastAdaptation = 0;
    this.adaptationCooldown = 50; // Steps between adaptations
  }

  /**
   * Update scheduler with training metrics
   * @param {Object} metrics - Training metrics
   * @returns {Object} Scheduling decision
   */
  update(metrics) {
    const {
      step,
      loss,
      gradientNorm,
      memoryUsageGB,
      throughputTokensPerSec,
      validationLoss = null
    } = metrics;

    // Record metrics
    this.trainingHistory.push({
      step,
      loss,
      gradientNorm,
      memoryUsageGB,
      throughputTokensPerSec,
      validationLoss,
      rank: this.currentRank
    });

    // Limit history size
    if (this.trainingHistory.length > 1000) {
      this.trainingHistory = this.trainingHistory.slice(-500);
    }

    // Make scheduling decision based on strategy
    const decision = this._makeSchedulingDecision(step, metrics);
    
    return {
      currentRank: this.currentRank,
      recommendedRank: decision.recommendedRank,
      shouldAdapt: decision.shouldAdapt,
      reason: decision.reason,
      confidence: decision.confidence,
      metrics: this._computePerformanceMetrics()
    };
  }

  /**
   * Make scheduling decision based on current strategy
   * @param {number} step - Current training step
   * @param {Object} metrics - Current metrics
   * @returns {Object} Scheduling decision
   */
  _makeSchedulingDecision(step, metrics) {
    switch (this.config.strategy) {
      case RANK_STRATEGIES.FIXED:
        return this._fixedStrategy();
      
      case RANK_STRATEGIES.PROGRESSIVE:
        return this._progressiveStrategy(step);
      
      case RANK_STRATEGIES.ADAPTIVE:
        return this._adaptiveStrategy(step, metrics);
      
      case RANK_STRATEGIES.HARDWARE_AWARE:
        return this._hardwareAwareStrategy(step, metrics);
      
      default:
        return this._adaptiveStrategy(step, metrics);
    }
  }

  /**
   * Fixed rank strategy - no changes
   */
  _fixedStrategy() {
    return {
      recommendedRank: this.currentRank,
      shouldAdapt: false,
      reason: 'Fixed rank strategy',
      confidence: 1.0
    };
  }

  /**
   * Progressive rank strategy - gradually increase rank
   */
  _progressiveStrategy(step) {
    const progressRatio = Math.min(step / 1000, 1.0); // Progress over 1000 steps
    const targetRank = Math.round(
      this.config.minRank + 
      (this.config.maxRank - this.config.minRank) * progressRatio
    );

    const shouldAdapt = targetRank !== this.currentRank && 
                       step - this.lastAdaptation >= this.adaptationCooldown;

    if (shouldAdapt) {
      this.currentRank = targetRank;
      this.lastAdaptation = step;
    }

    return {
      recommendedRank: targetRank,
      shouldAdapt,
      reason: `Progressive increase to rank ${targetRank}`,
      confidence: 0.8
    };
  }

  /**
   * Adaptive rank strategy - based on training metrics
   */
  _adaptiveStrategy(step, metrics) {
    if (this.trainingHistory.length < 10) {
      return {
        recommendedRank: this.currentRank,
        shouldAdapt: false,
        reason: 'Insufficient history for adaptation',
        confidence: 0.5
      };
    }

    // Check if we can adapt (cooldown period)
    if (step - this.lastAdaptation < this.adaptationCooldown) {
      return {
        recommendedRank: this.currentRank,
        shouldAdapt: false,
        reason: 'Adaptation cooldown active',
        confidence: 0.3
      };
    }

    const decision = this._analyzePerformanceTrends(metrics);
    
    if (decision.shouldAdapt) {
      this.currentRank = decision.recommendedRank;
      this.lastAdaptation = step;
    }

    return decision;
  }

  /**
   * Hardware-aware strategy - optimize for memory and compute
   */
  _hardwareAwareStrategy(step, metrics) {
    const { memoryUsageGB, throughputTokensPerSec } = metrics;
    
    // Memory pressure check
    const memoryUtilization = memoryUsageGB / this.config.targetMemoryUsageGB;
    
    // Performance check
    const recentHistory = this.trainingHistory.slice(-10);
    const avgThroughput = recentHistory.reduce((sum, h) => sum + h.throughputTokensPerSec, 0) / recentHistory.length;
    
    let recommendedRank = this.currentRank;
    let reason = 'Hardware metrics stable';
    let shouldAdapt = false;

    // Reduce rank if memory pressure is high
    if (memoryUtilization > 0.9 && this.currentRank > this.config.minRank) {
      recommendedRank = Math.max(this.config.minRank, this.currentRank - 2);
      reason = `Reducing rank due to memory pressure (${(memoryUtilization * 100).toFixed(1)}%)`;
      shouldAdapt = true;
    }
    // Increase rank if we have headroom and performance is good
    else if (memoryUtilization < 0.7 && 
             avgThroughput > this.config.performanceThreshold * throughputTokensPerSec &&
             this.currentRank < this.config.maxRank) {
      recommendedRank = Math.min(this.config.maxRank, this.currentRank + 2);
      reason = `Increasing rank with available resources (${(memoryUtilization * 100).toFixed(1)}% memory)`;
      shouldAdapt = true;
    }

    if (shouldAdapt && step - this.lastAdaptation >= this.adaptationCooldown) {
      this.currentRank = recommendedRank;
      this.lastAdaptation = step;
    } else if (shouldAdapt) {
      shouldAdapt = false;
      reason += ' (cooldown active)';
    }

    return {
      recommendedRank,
      shouldAdapt,
      reason,
      confidence: 0.9
    };
  }

  /**
   * Analyze performance trends for adaptive strategy
   */
  _analyzePerformanceTrends(currentMetrics) {
    const recentWindow = Math.min(this.config.convergenceWindow, this.trainingHistory.length);
    const recentHistory = this.trainingHistory.slice(-recentWindow);
    
    if (recentHistory.length < 20) {
      return {
        recommendedRank: this.currentRank,
        shouldAdapt: false,
        reason: 'Insufficient history for trend analysis',
        confidence: 0.4
      };
    }

    // Analyze loss trends
    const lossTrend = this._computeTrend(recentHistory.map(h => h.loss));
    const gradientTrend = this._computeTrend(recentHistory.map(h => h.gradientNorm));
    
    // Check for convergence issues
    const isConverging = lossTrend < -0.001; // Loss is decreasing
    const hasGradientFlow = recentHistory[recentHistory.length - 1].gradientNorm > 1e-8;
    
    // Check for underfitting (rank too low)
    const recentLoss = recentHistory.slice(-5).reduce((sum, h) => sum + h.loss, 0) / 5;
    const earlyLoss = recentHistory.slice(0, 5).reduce((sum, h) => sum + h.loss, 0) / 5;
    const improvement = (earlyLoss - recentLoss) / earlyLoss;
    
    let recommendedRank = this.currentRank;
    let reason = 'Performance metrics stable';
    let shouldAdapt = false;
    let confidence = 0.7;

    // Increase rank if underfitting
    if (!isConverging && hasGradientFlow && improvement < 0.05 && 
        this.currentRank < this.config.maxRank) {
      recommendedRank = Math.min(this.config.maxRank, this.currentRank + 2);
      reason = `Increasing rank - potential underfitting (improvement: ${(improvement * 100).toFixed(2)}%)`;
      shouldAdapt = true;
      confidence = 0.8;
    }
    // Decrease rank if overfitting or gradient issues
    else if ((!hasGradientFlow || (gradientTrend > 0.1 && lossTrend > 0)) && 
             this.currentRank > this.config.minRank) {
      recommendedRank = Math.max(this.config.minRank, this.currentRank - 1);
      reason = gradientTrend > 0.1 ? 'Reducing rank - gradient explosion detected' : 
               'Reducing rank - potential overfitting';
      shouldAdapt = true;
      confidence = 0.9;
    }
    // Fine-tune rank based on gradient norms
    else if (hasGradientFlow && isConverging) {
      const avgGradNorm = recentHistory.slice(-10).reduce((sum, h) => sum + h.gradientNorm, 0) / 10;
      
      if (avgGradNorm < 1e-6 && this.currentRank > this.config.minRank) {
        recommendedRank = Math.max(this.config.minRank, this.currentRank - 1);
        reason = 'Reducing rank - gradients very small';
        shouldAdapt = true;
      } else if (avgGradNorm > 1e-2 && this.currentRank < this.config.maxRank) {
        recommendedRank = Math.min(this.config.maxRank, this.currentRank + 1);
        reason = 'Increasing rank - strong gradients detected';
        shouldAdapt = true;
      }
    }

    return {
      recommendedRank,
      shouldAdapt,
      reason,
      confidence
    };
  }

  /**
   * Compute trend (slope) of a data series
   */
  _computeTrend(data) {
    if (data.length < 2) return 0;
    
    const n = data.length;
    const x = Array.from({length: n}, (_, i) => i);
    const xMean = (n - 1) / 2;
    const yMean = data.reduce((sum, y) => sum + y, 0) / n;
    
    let numerator = 0;
    let denominator = 0;
    
    for (let i = 0; i < n; i++) {
      numerator += (x[i] - xMean) * (data[i] - yMean);
      denominator += (x[i] - xMean) ** 2;
    }
    
    return denominator === 0 ? 0 : numerator / denominator;
  }

  /**
   * Compute performance metrics
   */
  _computePerformanceMetrics() {
    if (this.trainingHistory.length < 10) {
      return {
        convergenceRate: 0,
        stability: 0,
        efficiency: 0
      };
    }

    const recent = this.trainingHistory.slice(-50);
    
    // Convergence rate (loss improvement per step)
    const lossTrend = this._computeTrend(recent.map(h => h.loss));
    const convergenceRate = Math.max(0, -lossTrend); // Higher is better
    
    // Stability (inverse of loss variance)
    const losses = recent.map(h => h.loss);
    const lossVariance = this._computeVariance(losses);
    const stability = 1 / (1 + lossVariance);
    
    // Efficiency (throughput relative to rank)
    const avgThroughput = recent.reduce((sum, h) => sum + h.throughputTokensPerSec, 0) / recent.length;
    const efficiency = avgThroughput / (this.currentRank * this.currentRank); // Quadratic penalty for rank
    
    return {
      convergenceRate,
      stability,
      efficiency,
      avgThroughput,
      currentLoss: recent[recent.length - 1]?.loss || 0
    };
  }

  /**
   * Compute variance of a data series
   */
  _computeVariance(data) {
    if (data.length < 2) return 0;
    
    const mean = data.reduce((sum, x) => sum + x, 0) / data.length;
    const variance = data.reduce((sum, x) => sum + (x - mean) ** 2, 0) / data.length;
    
    return variance;
  }

  /**
   * Get current rank
   */
  getCurrentRank() {
    return this.currentRank;
  }

  /**
   * Force rank change
   */
  setRank(newRank) {
    this.currentRank = Math.max(
      this.config.minRank,
      Math.min(this.config.maxRank, newRank)
    );
  }

  /**
   * Get training statistics
   */
  getStatistics() {
    return {
      currentRank: this.currentRank,
      trainingSteps: this.trainingHistory.length,
      lastAdaptation: this.lastAdaptation,
      adaptationCount: this.trainingHistory.filter((h, i) => 
        i > 0 && h.rank !== this.trainingHistory[i-1].rank
      ).length,
      performanceMetrics: this._computePerformanceMetrics()
    };
  }

  /**
   * Reset scheduler state
   */
  reset() {
    this.currentRank = this.config.initialRank;
    this.trainingHistory = [];
    this.performanceHistory = [];
    this.memoryHistory = [];
    this.lastAdaptation = 0;
  }
}

/**
 * Factory function to create rank scheduler
 */
export function createRankScheduler(strategy, config = {}) {
  return new LoRARankScheduler({
    strategy,
    ...config
  });
}

/**
 * Utility function to estimate optimal initial rank
 */
export function estimateOptimalRank(modelSize, memoryBudgetGB, targetThroughput) {
  // Rough heuristics based on model size and available resources
  let baseRank = 4;
  
  if (modelSize > 7e9) { // 7B+ parameters
    baseRank = 8;
  } else if (modelSize > 3e9) { // 3B+ parameters
    baseRank = 6;
  } else if (modelSize > 1e9) { // 1B+ parameters
    baseRank = 4;
  } else {
    baseRank = 2;
  }
  
  // Adjust for memory budget
  const memoryMultiplier = Math.min(memoryBudgetGB / 4.0, 2.0); // Scale with memory
  baseRank = Math.round(baseRank * memoryMultiplier);
  
  // Adjust for throughput requirements
  if (targetThroughput > 1000) {
    baseRank = Math.max(2, baseRank - 1); // Reduce for high throughput
  }
  
  return Math.max(2, Math.min(64, baseRank));
}

export default {
  LoRARankScheduler,
  RANK_STRATEGIES,
  createRankScheduler,
  estimateOptimalRank
};