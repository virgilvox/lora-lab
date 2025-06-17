<template>
  <div class="test-component">
    <h2>{{ message }}</h2>
    <button @click="updateMessage">Click Me</button>
    <p>Count: {{ count }}</p>
  </div>
</template>

<script>
export default {
  name: 'TestComponent',
  data() {
    return {
      message: 'Vue SFC is working!',
      count: 0
    }
  },
  methods: {
    updateMessage() {
      this.count++
      this.message = `Clicked ${this.count} times!`
    }
  }
}
</script>

<style scoped>
.test-component {
  padding: 20px;
  border: 2px solid #42b983;
  border-radius: 8px;
  text-align: center;
}

button {
  background-color: #42b983;
  color: white;
  border: none;
  padding: 10px 20px;
  border-radius: 4px;
  cursor: pointer;
  font-size: 16px;
}

button:hover {
  background-color: #3aa876;
}
</style> 