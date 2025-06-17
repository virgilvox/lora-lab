/**
 * Text Generation Utilities for LoRA Lab
 * Handles text generation from model outputs
 */
import { Tensor } from 'onnxruntime-web';

/**
 * Generate text from model outputs using an autoregressive loop.
 * @param {import('onnxruntime-web').InferenceSession} session - The ONNX session.
 * @param {import('@huggingface/transformers').PreTrainedTokenizer} tokenizer - The tokenizer.
 * @param {number[]} input_ids - The initial token IDs from the user prompt.
 * @param {Object} options - Generation options.
 * @returns {Promise<string>} Generated text.
 */
export async function generateTextFromOutputs(session, tokenizer, input_ids, options = {}) {
  const {
    maxLength = 100,
    temperature = 0.7,
    topK = 40,
    eos_token_id = tokenizer.eos_token_id,
  } = options;

  let generated_ids = [...input_ids];

  try {
    for (let i = 0; i < maxLength; i++) {
      const current_ids_bigint = new BigInt64Array(generated_ids.map(BigInt));
      const inputTensor = new Tensor('int64', current_ids_bigint, [1, generated_ids.length]);
      
      const attention_mask = new BigInt64Array(generated_ids.length).fill(1n);
      const attentionTensor = new Tensor('int64', attention_mask, [1, generated_ids.length]);

      const feeds = {
        input_ids: inputTensor,
        attention_mask: attentionTensor,
      };

      const { logits } = await session.run(feeds);
      
      // Get the logits for the last token
      const nextTokenLogits = logits.data.slice(logits.dims[2] * (logits.dims[1] - 1), logits.dims[2] * logits.dims[1]);
      
      const nextTokenId = sampleNextToken(nextTokenLogits, { temperature, topK });

      if (nextTokenId === eos_token_id) {
        break;
      }
      
      generated_ids.push(nextTokenId);
    }
    
    // Decode the final sequence of tokens
    return tokenizer.decode(generated_ids, { skip_special_tokens: true });

  } catch (error) {
    console.error('Text generation failed:', error);
    return `Error generating text: ${error.message}`;
  }
}

/**
 * Simple token decoding (placeholder implementation) - No longer used
 * @param {Array} tokens - Token IDs
 * @returns {string} Decoded text
 */
export function decodeTokens(tokens) {
  // This is a placeholder for actual token decoding
  // In a real implementation, this would use a proper tokenizer
  return tokens.join(' ');
}

/**
 * Sample next token from logits
 * @param {Float32Array} logits - Logits from model output
 * @param {Object} options - Sampling options
 * @returns {number} Next token ID
 */
export function sampleNextToken(logits, options = {}) {
  const { temperature = 1.0, topK = 40 } = options;
  
  // Apply temperature
  const scaled = new Float32Array(logits.length);
  for (let i = 0; i < logits.length; i++) {
    scaled[i] = logits[i] / temperature;
  }
  
  // Get top-k indices
  const indices = Array.from({ length: logits.length }, (_, i) => i)
    .sort((a, b) => scaled[b] - scaled[a])
    .slice(0, topK);
  
  // Convert to probabilities with softmax
  const probs = softmax(indices.map(i => scaled[i]));
  
  // Sample from distribution
  const rand = Math.random();
  let cumulative = 0;
  for (let i = 0; i < indices.length; i++) {
    cumulative += probs[i];
    if (rand < cumulative) {
      return indices[i];
    }
  }
  
  // Fallback to most likely
  return indices[0];
}

/**
 * Softmax function
 * @param {Array} arr - Input array
 * @returns {Array} Softmax probabilities
 */
function softmax(arr) {
  const max = Math.max(...arr);
  const exp = arr.map(x => Math.exp(x - max));
  const sum = exp.reduce((a, b) => a + b, 0);
  return exp.map(x => x / sum);
}

export default {
  generateTextFromOutputs,
  decodeTokens,
  sampleNextToken
}; 