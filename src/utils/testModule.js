// Test module to verify plain JS module imports work correctly

export function add(a, b) {
  return a + b
}

export function multiply(a, b) {
  return a * b
}

export const PI = 3.14159

export default {
  greet(name) {
    return `Hello, ${name}!`
  },
  formatNumber(num) {
    return num.toFixed(2)
  }
} 