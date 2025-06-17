import { createApp } from 'vue'
import './style.css'
// Test plain JS module import
import testUtils, { add, PI } from './utils/testModule.js'
// Test Vue SFC import
import TestComponent from './ui/TestComponent.vue'
import LoRALabApp from './ui/LoRALabApp.vue'

// Test plain JS module functionality
console.log('Plain JS module test:', {
  greeting: testUtils.greet('LoRA Lab'),
  addition: add(5, 3),
  pi: PI
})

// Create Vue app with test component
const app = createApp(LoRALabApp)
app.mount('#app')
