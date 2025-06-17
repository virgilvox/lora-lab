/**
 * Web Worker for Text Generation using Hugging Face Transformers.js
 *
 * This worker handles the loading of models and tokenizers,
 * and performs text generation in a separate thread to avoid
 * blocking the main UI thread.
 *
 * It is based on the singleton pattern for lazy-loading models
 * and tokenizers, adapted to handle multiple models.
 */

import {
  AutoTokenizer,
  AutoModelForCausalLM,
  TextStreamer,
  InterruptableStoppingCriteria,
} from "@huggingface/transformers";

// A mapping from model_id to a promise that resolves to the loaded model and tokenizer.
const models = new Map();

// Stopping criteria for generation, allowing interruption from the main thread.
const stopping_criteria = new InterruptableStoppingCriteria();

/**
 * Lazily loads a model and tokenizer for a given model ID.
 * If already loading, it returns the existing promise.
 * If already loaded, it returns the resolved promise.
 * @param {string} model_id The Hugging Face model ID.
 * @param {function} progress_callback A callback to report loading progress.
 * @returns {Promise<[AutoTokenizer, AutoModelForCausalLM]>} A promise that resolves to an array containing the tokenizer and model.
 */
function getInstance(model_id, progress_callback = null) {
  if (!models.has(model_id)) {
    const modelPromise = Promise.all([
      AutoTokenizer.from_pretrained(model_id, { progress_callback }),
      AutoModelForCausalLM.from_pretrained(model_id, {
        dtype: "q4f16", // Using float16 for better performance on WebGPU
        device: "webgpu",
        use_external_data_format: true, // Important for models with external data files
        progress_callback,
      }),
    ]);
    models.set(model_id, modelPromise);
  }
  return models.get(model_id);
}

/**
 * Handles the 'load' message from the main thread.
 * It pre-loads the model and warms it up by running a dummy generation.
 * @param {string} model_id The model ID to load.
 */
async function handleLoad(model_id) {
  self.postMessage({ status: "loading", data: "Loading model...", model_id });

  const [tokenizer, model] = await getInstance(model_id, (progress) => {
    self.postMessage({ ...progress, model_id });
  });

  self.postMessage({
    status: "loading",
    data: "Compiling shaders and warming up model...",
    model_id
  });

  // Run model with dummy input to compile shaders and warm up the model
  const inputs = tokenizer("a", { return_tensors: "pt" });
  await model.generate({ ...inputs, max_new_tokens: 1 });

  self.postMessage({ status: "ready", model_id });
}


/**
 * Handles the 'generate' message from the main thread.
 * Generates text based on the provided messages.
 * @param {object} data The data from the main thread, containing model_id and messages.
 */
async function handleGenerate({ model_id, messages }) {
  const [tokenizer, model] = await getInstance(model_id);

  const inputs = tokenizer.apply_chat_template(messages, {
    add_generation_prompt: true,
    return_dict: true,
  });

  let startTime;
  let numTokens = 0;
  const token_callback_function = () => {
    if (numTokens === 0) {
      startTime = performance.now();
    }
    numTokens++;
    const tps = (numTokens / (performance.now() - startTime)) * 1000;
    self.postMessage({ status: "token", tps, numTokens, model_id });
  };

  const streamer = new TextStreamer(tokenizer, {
    skip_prompt: true,
    skip_special_tokens: true,
    callback_function: (output) => {
      self.postMessage({ status: "update", output, model_id });
    },
    token_callback_function,
  });

  self.postMessage({ status: "start", model_id });

  try {
    const { sequences } = await model.generate({
      ...inputs,
      max_new_tokens: 1024,
      do_sample: true,
      top_k: 3,
      temperature: 0.2,
      streamer,
      stopping_criteria,
      return_dict_in_generate: true,
    });

    const decoded = tokenizer.batch_decode(sequences, { skip_special_tokens: true });

    self.postMessage({
      status: "complete",
      output: decoded,
      model_id,
    });

  } catch (e) {
    if (e.name !== 'InterruptException') {
      console.error(e);
      self.postMessage({ status: "error", error: e.toString(), model_id });
    } else {
       self.postMessage({ status: "interrupted", model_id });
    }
  }
}

// Listen for messages from the main thread
self.addEventListener("message", async (e) => {
  const { type, data } = e.data;

  switch (type) {
    case "load":
      handleLoad(data.model_id);
      break;

    case "generate":
      stopping_criteria.reset();
      handleGenerate(data);
      break;

    case "interrupt":
      stopping_criteria.interrupt();
      break;

    case "reset":
      // Note: This pattern doesn't use past_key_values caching yet.
      stopping_criteria.reset();
      break;
  }
}); 