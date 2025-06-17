import { createApp } from 'vue'
import './style.css'
// Test plain JS module import
import testUtils, { add, PI } from './utils/testModule.js'
// Test Vue SFC import
import TestComponent from './ui/TestComponent.vue'
import LoRALabApp from './ui/LoRALabApp.vue'
import { env } from '@huggingface/transformers'

// --- Centralized Transformers.js Configuration ---
// Serve WASM files from the public root directory.
env.backends.onnx.wasm.wasmPaths = '/'
// Use a proxy to avoid CORS issues when downloading models from the Hub.
env.remoteHost = '/huggingface'

// Allow remote models from the Hub.
env.allowRemoteModels = true
// Disable loading from local cache to ensure we use the fetched model.
env.allowLocalModels = false

// Test plain JS module functionality
console.log('Plain JS module test:', {
  greeting: testUtils.greet('LoRA Lab'),
  addition: add(5, 3),
  pi: PI
})

// Create Vue app with test component
const app = createApp(LoRALabApp)
app.mount('#app')
