import { createApp } from 'vue'
import './style.css'
// Test plain JS module import
import testUtils, { add, PI } from './utils/testModule.js'
// Test Vue SFC import
import TestComponent from './ui/TestComponent.vue'

// Test plain JS module functionality
console.log('Plain JS module test:', {
  greeting: testUtils.greet('LoRA Lab'),
  addition: add(5, 3),
  pi: PI
})

// Create Vue app with test component
const app = createApp({
  components: {
    TestComponent
  },
  template: `
    <div id="app">
      <h1>LoRA Lab - Setup Verification</h1>
      <TestComponent />
      <p>Plain JS modules: ✅ Working</p>
      <p>Vue SFCs: ✅ Working</p>
    </div>
  `
})

app.mount('#app')
