/**
 * Dataset Loader and Processing Utility for LoRA Lab
 * Handles text corpus loading and cleaning. Tokenization is handled by the training worker.
 */

/**
 * Load and process a text dataset
 * @param {string} text - Raw text content
 * @param {Object} options - Processing options
 * @returns {Promise<Object>} Processed dataset information
 */
export async function loadDataset(text, options = {}) {
  const {
    cleanText = true
  } = options

  try {
    // Clean and preprocess text
    const processedText = cleanText ? preprocessText(text) : text
    
    // Calculate dataset statistics based on raw text
    const stats = calculateDatasetStats(processedText)
    
    return {
      text: processedText,
      characterCount: processedText.length,
      stats: stats,
      metadata: {
        timestamp: Date.now(),
        options
      }
    }
  } catch (error) {
    console.error('Dataset loading failed:', error)
    throw new Error(`Failed to load dataset: ${error.message}`)
  }
}

/**
 * Preprocess text to clean and normalize it
 * @param {string} text - Raw text
 * @returns {string} Cleaned text
 */
export function preprocessText(text) {
  if (!text || typeof text !== 'string') {
    throw new Error('Invalid text input')
  }

  let processed = text

  // Remove or replace problematic characters
  processed = processed.replace(/\r\n/g, '\n') // Normalize line endings
  processed = processed.replace(/\r/g, '\n')   // Convert remaining \r to \n
  
  // Remove excessive whitespace but preserve paragraph structure
  processed = processed.replace(/[ \t]+/g, ' ')           // Multiple spaces/tabs to single space
  processed = processed.replace(/\n\s*\n\s*\n+/g, '\n\n') // Multiple newlines to double newline
  processed = processed.replace(/^\s+|\s+$/g, '')         // Trim start/end whitespace
  
  // Remove or replace special characters that might cause issues
  processed = processed.replace(/[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]/g, '') // Control characters
  
  // Normalize quotes and other typography
  processed = processed.replace(/[""]/g, '"')   // Smart quotes to regular quotes
  processed = processed.replace(/['']/g, "'")   // Smart apostrophes
  processed = processed.replace(/[–—]/g, '-')   // Em/en dashes to hyphens
  processed = processed.replace(/…/g, '...')    // Ellipsis
  
  return processed
}

/**
 * Calculate dataset statistics
 * @param {string} text - Processed text
 * @returns {Object} Dataset statistics
 */
function calculateDatasetStats(text) {
  // Character frequency analysis
  const charFreq = {}
  for (const char of text) {
    charFreq[char] = (charFreq[char] || 0) + 1
  }
  
  // Word count analysis
  const words = text.match(/\b\w+\b/g) || []
  const uniqueWords = new Set(words.map(w => w.toLowerCase()))
  
  return {
    characterCount: text.length,
    wordCount: words.length,
    uniqueWordCount: uniqueWords.size,
    mostCommonChars: Object.entries(charFreq)
      .sort(([,a], [,b]) => b - a)
      .slice(0, 10)
      .map(([char, count]) => ({ char, count, percentage: (count / text.length * 100).toFixed(2) }))
  }
}

/**
 * Validate text corpus for training
 * @param {string} text - Text to validate
 * @returns {Object} Validation results
 */
export function validateCorpus(text) {
  const validation = {
    isValid: true,
    warnings: [],
    errors: [],
    recommendations: []
  }
  
  if (!text || typeof text !== 'string') {
    validation.isValid = false
    validation.errors.push('Invalid text input')
    return validation
  }
  
  const length = text.length
  const wordCount = (text.match(/\b\w+\b/g) || []).length
  
  // Check minimum length
  if (length < 1000) {
    validation.warnings.push('Text is very short (< 1000 characters) - may not provide enough training data')
  }
  
  // Check maximum length
  if (length > 10000000) { // 10MB
    validation.warnings.push('Text is very long (> 10MB) - consider splitting into smaller chunks')
  }
  
  // Check word count
  if (wordCount < 100) {
    validation.warnings.push('Very few words detected - ensure text is in a supported language')
  }
  
  // Check for repetitive content
  const lines = text.split('\n')
  const uniqueLines = new Set(lines)
  if (uniqueLines.size < lines.length * 0.5) {
    validation.warnings.push('High repetition detected - may lead to overfitting')
  }
  
  // Check character diversity
  const uniqueChars = new Set(text)
  if (uniqueChars.size < 20) {
    validation.warnings.push('Low character diversity - ensure text is not corrupted')
  }
  
  // Recommendations
  if (length < 50000) {
    validation.recommendations.push('Consider using adapter training mode for small datasets')
  }
  
  if (wordCount > 100000) {
    validation.recommendations.push('Large dataset detected - full fine-tuning may be beneficial')
  }
  
  return validation
}

/**
 * Estimate training time based on corpus size and hardware
 * @param {number} tokenCount - Number of tokens
 * @param {Object} hardwareInfo - Hardware information
 * @param {string} mode - Training mode ('adapter' or 'full')
 * @returns {Object} Time estimates
 */
export function estimateTrainingTime(tokenCount, hardwareInfo, mode = 'adapter') {
  const tflops = hardwareInfo.estimatedTFLOPs || 2.0
  const webGPUSupported = hardwareInfo.webGPUSupported
  
  // Rough estimates based on mode and hardware
  let gflopsPerToken
  let efficiency
  
  if (mode === 'adapter') {
    gflopsPerToken = 1.2 // LoRA forward + backward pass
    efficiency = webGPUSupported ? 0.4 : 0.1 // 40% GPU efficiency, 10% CPU
  } else {
    gflopsPerToken = 42 // Full model forward + backward pass
    efficiency = webGPUSupported ? 0.35 : 0.05 // 35% GPU efficiency, 5% CPU
  }
  
  const effectiveTFlops = tflops * efficiency
  const tokensPerSecond = (effectiveTFlops * 1000) / gflopsPerToken
  const totalSeconds = tokenCount / Math.max(tokensPerSecond, 1)
  
  return {
    tokensPerSecond: Math.round(tokensPerSecond),
    totalSeconds: Math.round(totalSeconds),
    totalMinutes: Math.round(totalSeconds / 60),
    totalHours: Math.round(totalSeconds / 3600),
    formattedTime: formatTrainingTime(totalSeconds),
    confidence: webGPUSupported ? 'medium' : 'low'
  }
}

/**
 * Format training time in human-readable format
 * @param {number} seconds - Time in seconds
 * @returns {string} Formatted time string
 */
function formatTrainingTime(seconds) {
  if (seconds < 60) return `${Math.round(seconds)}s`
  if (seconds < 3600) return `${Math.round(seconds / 60)}m`
  const hours = Math.floor(seconds / 3600)
  const minutes = Math.round((seconds % 3600) / 60)
  return `${hours}h ${minutes}m`
}

// Export all functions
export default {
  loadDataset,
  preprocessText,
  validateCorpus,
  estimateTrainingTime
}