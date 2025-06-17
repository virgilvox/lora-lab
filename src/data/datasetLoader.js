/**
 * Dataset Loader and Tokenization Utility for LoRA Lab
 * Handles text corpus loading, processing, and tokenization
 */

/**
 * Load and process a text dataset
 * @param {string} text - Raw text content
 * @param {Object} options - Processing options
 * @returns {Promise<Object>} Processed dataset information
 */
export async function loadDataset(text, options = {}) {
  const {
    maxTokens = 1000000,
    sequenceLength = 512,
    stride = 256,
    includeSpecialTokens = true,
    cleanText = true
  } = options

  try {
    // Clean and preprocess text
    const processedText = cleanText ? preprocessText(text) : text
    
    // Basic tokenization (simplified for demo - in real implementation would use proper tokenizer)
    const tokens = await tokenizeText(processedText, { includeSpecialTokens })
    
    // Create training sequences
    const sequences = createTrainingSequences(tokens, sequenceLength, stride)
    
    // Calculate dataset statistics
    const stats = calculateDatasetStats(processedText, tokens, sequences)
    
    return {
      originalText: text,
      processedText: processedText,
      tokens: tokens.slice(0, maxTokens), // Limit tokens if needed
      sequences: sequences,
      tokenCount: Math.min(tokens.length, maxTokens),
      sequenceCount: sequences.length,
      characterCount: processedText.length,
      averageTokenLength: processedText.length / tokens.length,
      stats: stats,
      metadata: {
        sequenceLength,
        stride,
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
 * Simple tokenization (basic implementation for demo)
 * In production, this would use a proper tokenizer like tiktoken
 * @param {string} text - Text to tokenize
 * @param {Object} options - Tokenization options
 * @returns {Promise<Array>} Array of token IDs
 */
export async function tokenizeText(text, options = {}) {
  const { includeSpecialTokens = true } = options

  // This is a simplified tokenizer for demo purposes
  // In a real implementation, you'd use a proper tokenizer that matches your model
  
  // Basic word + subword tokenization
  const tokens = []
  
  if (includeSpecialTokens) {
    tokens.push(1) // BOS token
  }
  
  // Split text into words and punctuation
  const words = text.match(/\S+|\s+/g) || []
  
  for (const word of words) {
    if (/^\s+$/.test(word)) {
      // Whitespace token
      tokens.push(220) // Space token ID
    } else {
      // Convert word to token IDs (simplified approach)
      const wordTokens = wordToTokens(word)
      tokens.push(...wordTokens)
    }
  }
  
  if (includeSpecialTokens) {
    tokens.push(2) // EOS token
  }
  
  return tokens
}

/**
 * Convert a word to token IDs (simplified implementation)
 * @param {string} word - Word to tokenize
 * @returns {Array} Token IDs
 */
function wordToTokens(word) {
  // This is a very simplified approach
  // Real tokenizers use BPE, SentencePiece, or other algorithms
  
  const tokens = []
  const lowerWord = word.toLowerCase()
  
  // Simple hash-based approach for consistent token assignment
  let hash = 0
  for (let i = 0; i < lowerWord.length; i++) {
    const char = lowerWord.charCodeAt(i)
    hash = ((hash << 5) - hash) + char
    hash = hash & hash // Convert to 32-bit integer
  }
  
  // Map to vocabulary range (3-50000, avoiding special tokens)
  const tokenId = 3 + Math.abs(hash) % 49997
  tokens.push(tokenId)
  
  // For longer words, might split into multiple tokens
  if (word.length > 8) {
    const midpoint = Math.floor(word.length / 2)
    const secondPart = word.slice(midpoint)
    let secondHash = 0
    for (let i = 0; i < secondPart.length; i++) {
      const char = secondPart.charCodeAt(i)
      secondHash = ((secondHash << 5) - secondHash) + char
      secondHash = secondHash & secondHash
    }
    const secondTokenId = 3 + Math.abs(secondHash) % 49997
    tokens.push(secondTokenId)
  }
  
  return tokens
}

/**
 * Create training sequences from tokens
 * @param {Array} tokens - Array of token IDs
 * @param {number} sequenceLength - Length of each sequence
 * @param {number} stride - Stride between sequences
 * @returns {Array} Array of sequences
 */
export function createTrainingSequences(tokens, sequenceLength = 512, stride = 256) {
  const sequences = []
  
  for (let i = 0; i < tokens.length - sequenceLength; i += stride) {
    const sequence = tokens.slice(i, i + sequenceLength)
    const labels = tokens.slice(i + 1, i + sequenceLength + 1) // Next token prediction
    
    sequences.push({
      input: sequence,
      labels: labels,
      startIndex: i,
      endIndex: i + sequenceLength
    })
  }
  
  return sequences
}

/**
 * Calculate dataset statistics
 * @param {string} text - Processed text
 * @param {Array} tokens - Token array
 * @param {Array} sequences - Training sequences
 * @returns {Object} Dataset statistics
 */
function calculateDatasetStats(text, tokens, sequences) {
  // Character frequency analysis
  const charFreq = {}
  for (const char of text) {
    charFreq[char] = (charFreq[char] || 0) + 1
  }
  
  // Word count analysis
  const words = text.match(/\b\w+\b/g) || []
  const uniqueWords = new Set(words.map(w => w.toLowerCase()))
  
  // Token vocabulary analysis
  const uniqueTokens = new Set(tokens)
  
  return {
    characterCount: text.length,
    wordCount: words.length,
    uniqueWordCount: uniqueWords.size,
    tokenCount: tokens.length,
    uniqueTokenCount: uniqueTokens.size,
    sequenceCount: sequences.length,
    averageWordsPerSequence: sequences.length > 0 ? words.length / sequences.length : 0,
    averageTokensPerWord: words.length > 0 ? tokens.length / words.length : 0,
    vocabularySize: uniqueTokens.size,
    compressionRatio: text.length / tokens.length, // Characters per token
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
  tokenizeText,
  createTrainingSequences,
  validateCorpus,
  estimateTrainingTime
}