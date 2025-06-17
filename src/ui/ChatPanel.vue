<template>
  <div class="chat-panel">
    <!-- Chat Header -->
    <div class="chat-header">
      <div class="chat-title">
        <h3>Chat Interface</h3>
        <div class="model-indicator">
          <span class="model-name">{{ displayModelName }}</span>
          <div v-if="selectedModel" class="model-status">
            <div class="status-dot" :class="{ 'ready': selectedModel }"></div>
            <span>{{ selectedModel ? 'Ready' : 'No Model' }}</span>
          </div>
        </div>
      </div>
      
      <!-- LoRA Toggle -->
      <div class="lora-toggle-section">
        <label class="lora-toggle" :class="{ 'disabled': !adapterLoaded }">
          <input 
            type="checkbox" 
            :checked="useLoRA" 
            @change="$emit('toggle-lora')"
            :disabled="!adapterLoaded || isTraining"
          />
          <span class="toggle-slider"></span>
          <span class="toggle-label">Use LoRA</span>
        </label>
        
        <div v-if="!adapterLoaded" class="lora-status">
          <span class="status-text">{{ isTraining ? 'Training...' : 'No adapter available' }}</span>
        </div>
        <div v-else class="lora-status active">
          <span class="status-text">Adapter ready</span>
        </div>
      </div>
    </div>

    <!-- Chat Messages -->
    <div class="chat-messages" ref="messagesContainer">
      <div v-if="messages.length === 0" class="welcome-message">
        <div class="welcome-content">
          <div class="welcome-icon">🤖</div>
          <h4>Welcome to LoRA Lab!</h4>
          <p>Start a conversation to test your {{ useLoRA ? 'fine-tuned' : 'base' }} model.</p>
          <div class="suggestion-chips">
            <button 
              v-for="suggestion in suggestions" 
              :key="suggestion"
              @click="sendSuggestion(suggestion)"
              class="suggestion-chip"
              :disabled="!selectedModel"
            >
              {{ suggestion }}
            </button>
          </div>
        </div>
      </div>

      <div 
        v-for="(message, index) in messages" 
        :key="index" 
        class="message"
        :class="{ 'user': message.role === 'user', 'assistant': message.role === 'assistant' }"
      >
        <div class="message-avatar">
          <div class="avatar-icon">
            {{ message.role === 'user' ? '👤' : '🤖' }}
          </div>
        </div>
        
        <div class="message-content">
          <div class="message-header">
            <span class="message-role">
              {{ message.role === 'user' ? 'You' : (message.model || 'Assistant') }}
            </span>
            <span class="message-time">{{ formatTime(message.timestamp) }}</span>
          </div>
          
          <div class="message-text">
            {{ message.content }}
          </div>
          
          <!-- Message Actions -->
          <div v-if="message.role === 'assistant'" class="message-actions">
            <button @click="copyMessage(message.content)" class="action-btn" title="Copy">
              📋
            </button>
            <button @click="regenerateResponse(message, index)" class="action-btn" title="Regenerate">
              🔄
            </button>
            <button @click="rateMessage(message, 'good')" class="action-btn" title="Good response">
              👍
            </button>
            <button @click="rateMessage(message, 'bad')" class="action-btn" title="Poor response">
              👎
            </button>
          </div>
        </div>
      </div>

      <!-- Typing Indicator -->
      <div v-if="isTyping" class="message assistant">
        <div class="message-avatar">
          <div class="avatar-icon">🤖</div>
        </div>
        <div class="message-content">
          <div class="typing-indicator">
            <div class="typing-dots">
              <span></span>
              <span></span>
              <span></span>
            </div>
            <span class="typing-text">{{ displayModelName }} is thinking...</span>
          </div>
        </div>
      </div>
    </div>

    <!-- Chat Input -->
    <div class="chat-input-section">
      <div class="input-container">
        <textarea
          v-model="inputMessage"
          @keydown="handleKeyDown"
          @input="handleInput"
          placeholder="Type your message here..."
          class="message-input"
          rows="1"
          :disabled="!selectedModel || isTyping"
        ></textarea>
        
        <button 
          @click="sendMessage"
          class="send-button"
          :disabled="!canSend"
          :class="{ 'ready': canSend }"
        >
          <span class="send-icon">{{ isTyping ? '⏸' : '➤' }}</span>
        </button>
      </div>
      
      <!-- Input Actions -->
      <div class="input-actions">
        <div class="left-actions">
          <span class="char-count">{{ inputMessage.length }}/2000</span>
          <button 
            v-if="inputMessage.length > 0" 
            @click="clearInput" 
            class="clear-btn"
          >
            Clear
          </button>
        </div>
        
        <div class="right-actions">
          <button @click="exportChat" class="export-btn" :disabled="messages.length === 0">
            Export Chat
          </button>
          <button @click="clearChat" class="clear-chat-btn" :disabled="messages.length === 0">
            Clear Chat
          </button>
        </div>
      </div>
    </div>
  </div>
</template>

<script>
export default {
  name: 'ChatPanel',
  props: {
    selectedModel: {
      type: Object,
      default: null
    },
    useLoRA: {
      type: Boolean,
      default: false
    },
    isTraining: {
      type: Boolean,
      default: false
    },
    adapterLoaded: {
      type: Boolean,
      default: false
    },
    messages: {
      type: Array,
      default: () => []
    }
  },
  emits: ['toggle-lora', 'message-sent', 'message-regenerated', 'message-rated'],
  data() {
    return {
      inputMessage: '',
      isTyping: false,
      suggestions: [
        "Hello! How are you?",
        "What can you help me with?",
        "Tell me about yourself",
        "What's the weather like?",
        "Write a short story"
      ]
    }
  },
  computed: {
    displayModelName() {
      if (!this.selectedModel) return 'No Model Selected'
      
      const baseName = this.selectedModel.name || 'Unknown Model'
      return this.useLoRA ? `${baseName} + LoRA` : baseName
    },
    
    canSend() {
      return this.inputMessage.trim().length > 0 && 
             this.selectedModel && 
             !this.isTyping &&
             this.inputMessage.length <= 2000
    }
  },
  mounted() {
    this.adjustTextareaHeight()
  },
  watch: {
    messages: {
      handler() {
        this.$nextTick(() => {
          this.scrollToBottom()
        })
      },
      deep: true
    }
  },
  methods: {
    sendMessage() {
      if (!this.canSend) return
      
      const message = this.inputMessage.trim()
      this.inputMessage = ''
      this.adjustTextareaHeight()
      
      this.$emit('message-sent', message)
    },
    
    sendSuggestion(suggestion) {
      if (!this.selectedModel) return
      
      this.inputMessage = suggestion
      this.sendMessage()
    },
    
    handleKeyDown(event) {
      if (event.key === 'Enter' && !event.shiftKey) {
        event.preventDefault()
        this.sendMessage()
      }
    },
    
    handleInput() {
      this.adjustTextareaHeight()
    },
    
    adjustTextareaHeight() {
      this.$nextTick(() => {
        const textarea = this.$el.querySelector('.message-input')
        if (textarea) {
          textarea.style.height = 'auto'
          textarea.style.height = Math.min(textarea.scrollHeight, 120) + 'px'
        }
      })
    },
    
    clearInput() {
      this.inputMessage = ''
      this.adjustTextareaHeight()
    },
    
    clearChat() {
      if (confirm('Are you sure you want to clear the chat history?')) {
        // Emit event to parent to clear messages
        this.$emit('chat-cleared')
      }
    },
    
    exportChat() {
      if (this.messages.length === 0) return
      
      const chatData = {
        model: this.displayModelName,
        timestamp: new Date().toISOString(),
        messages: this.messages.map(msg => ({
          role: msg.role,
          content: msg.content,
          timestamp: msg.timestamp
        }))
      }
      
      const blob = new Blob([JSON.stringify(chatData, null, 2)], { 
        type: 'application/json' 
      })
      const url = URL.createObjectURL(blob)
      const link = document.createElement('a')
      link.href = url
      link.download = `lora-lab-chat-${Date.now()}.json`
      document.body.appendChild(link)
      link.click()
      document.body.removeChild(link)
      URL.revokeObjectURL(url)
    },
    
    copyMessage(content) {
      navigator.clipboard.writeText(content).then(() => {
        // Could show a toast notification here
        console.log('Message copied to clipboard')
      }).catch(err => {
        console.error('Failed to copy message:', err)
      })
    },
    
    regenerateResponse(message, index) {
      // Find the user message that prompted this response
      const userMessage = this.messages[index - 1]
      if (userMessage && userMessage.role === 'user') {
        this.$emit('message-regenerated', userMessage.content, index)
      }
    },
    
    rateMessage(message, rating) {
      this.$emit('message-rated', { message, rating })
      // Visual feedback could be added here
    },
    
    scrollToBottom() {
      const container = this.$refs.messagesContainer
      if (container) {
        container.scrollTop = container.scrollHeight
      }
    },
    
    formatTime(timestamp) {
      const date = new Date(timestamp)
      return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
    }
  }
}
</script>

<style scoped>
.chat-panel {
  display: flex;
  flex-direction: column;
  height: 100%;
  background-color: #1a1a1a;
  border-radius: 8px 0 0 0;
  overflow: hidden;
}

/* Chat Header */
.chat-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 1rem 1.5rem;
  background-color: #222;
  border-bottom: 1px solid #333;
}

.chat-title {
  display: flex;
  flex-direction: column;
  gap: 0.3rem;
}

.chat-title h3 {
  margin: 0;
  color: #fff;
  font-size: 1.1rem;
}

.model-indicator {
  display: flex;
  align-items: center;
  gap: 1rem;
}

.model-name {
  font-size: 0.9rem;
  color: #ccc;
  font-weight: 600;
}

.model-status {
  display: flex;
  align-items: center;
  gap: 0.3rem;
  font-size: 0.8rem;
  color: #888;
}

.status-dot {
  width: 6px;
  height: 6px;
  border-radius: 50%;
  background-color: #666;
}

.status-dot.ready {
  background-color: #10b981;
  box-shadow: 0 0 6px rgba(16, 185, 129, 0.5);
}

/* LoRA Toggle */
.lora-toggle-section {
  display: flex;
  flex-direction: column;
  align-items: flex-end;
  gap: 0.3rem;
}

.lora-toggle {
  display: flex;
  align-items: center;
  gap: 0.8rem;
  cursor: pointer;
  user-select: none;
}

.lora-toggle.disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.lora-toggle input {
  display: none;
}

.toggle-slider {
  position: relative;
  width: 44px;
  height: 24px;
  background-color: #374151;
  border-radius: 24px;
  transition: all 0.3s;
}

.toggle-slider::before {
  content: '';
  position: absolute;
  top: 2px;
  left: 2px;
  width: 20px;
  height: 20px;
  background-color: #fff;
  border-radius: 50%;
  transition: all 0.3s;
}

.lora-toggle input:checked + .toggle-slider {
  background-color: #10b981;
}

.lora-toggle input:checked + .toggle-slider::before {
  transform: translateX(20px);
}

.toggle-label {
  font-size: 0.9rem;
  font-weight: 600;
  color: #ccc;
}

.lora-status {
  font-size: 0.8rem;
  color: #666;
}

.lora-status.active {
  color: #10b981;
}

/* Chat Messages */
.chat-messages {
  flex: 1;
  overflow-y: auto;
  padding: 1rem;
  display: flex;
  flex-direction: column;
  gap: 1rem;
}

.welcome-message {
  display: flex;
  justify-content: center;
  align-items: center;
  height: 100%;
  min-height: 200px;
}

.welcome-content {
  text-align: center;
  max-width: 400px;
  padding: 2rem;
}

.welcome-icon {
  font-size: 4rem;
  margin-bottom: 1rem;
}

.welcome-content h4 {
  margin: 0 0 0.5rem 0;
  color: #fff;
  font-size: 1.5rem;
}

.welcome-content p {
  margin: 0 0 1.5rem 0;
  color: #ccc;
  line-height: 1.5;
}

.suggestion-chips {
  display: flex;
  flex-wrap: wrap;
  gap: 0.5rem;
  justify-content: center;
}

.suggestion-chip {
  padding: 0.5rem 1rem;
  background-color: #2a2a2a;
  border: 1px solid #444;
  border-radius: 20px;
  color: #ccc;
  cursor: pointer;
  font-size: 0.8rem;
  transition: all 0.2s;
}

.suggestion-chip:hover:not(:disabled) {
  background-color: #333;
  border-color: #10b981;
  color: #10b981;
}

.suggestion-chip:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

/* Message Styles */
.message {
  display: flex;
  gap: 1rem;
  max-width: 100%;
}

.message.user {
  flex-direction: row-reverse;
}

.message-avatar {
  flex-shrink: 0;
  width: 40px;
  height: 40px;
}

.avatar-icon {
  width: 100%;
  height: 100%;
  background-color: #2a2a2a;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 1.2rem;
}

.message.user .avatar-icon {
  background-color: #10b981;
}

.message-content {
  flex: 1;
  min-width: 0;
}

.message.user .message-content {
  text-align: right;
}

.message-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 0.5rem;
  font-size: 0.8rem;
}

.message.user .message-header {
  flex-direction: row-reverse;
}

.message-role {
  font-weight: 600;
  color: #ccc;
}

.message-time {
  color: #666;
}

.message-text {
  background-color: #2a2a2a;
  padding: 1rem;
  border-radius: 12px;
  color: #fff;
  line-height: 1.5;
  word-wrap: break-word;
}

.message.user .message-text {
  background-color: #10b981;
  color: #000;
}

.message-actions {
  display: flex;
  gap: 0.5rem;
  margin-top: 0.5rem;
  opacity: 0;
  transition: opacity 0.2s;
}

.message:hover .message-actions {
  opacity: 1;
}

.action-btn {
  background: none;
  border: none;
  color: #666;
  cursor: pointer;
  padding: 0.3rem;
  border-radius: 4px;
  font-size: 0.9rem;
  transition: all 0.2s;
}

.action-btn:hover {
  background-color: #333;
  color: #ccc;
}

/* Typing Indicator */
.typing-indicator {
  display: flex;
  align-items: center;
  gap: 1rem;
  padding: 1rem;
  background-color: #2a2a2a;
  border-radius: 12px;
}

.typing-dots {
  display: flex;
  gap: 0.3rem;
}

.typing-dots span {
  width: 6px;
  height: 6px;
  background-color: #666;
  border-radius: 50%;
  animation: typing 1.4s ease-in-out infinite both;
}

.typing-dots span:nth-child(1) { animation-delay: -0.32s; }
.typing-dots span:nth-child(2) { animation-delay: -0.16s; }

@keyframes typing {
  0%, 80%, 100% {
    transform: scale(0.8);
    opacity: 0.5;
  }
  40% {
    transform: scale(1);
    opacity: 1;
  }
}

.typing-text {
  color: #888;
  font-style: italic;
  font-size: 0.9rem;
}

/* Chat Input */
.chat-input-section {
  padding: 1rem;
  border-top: 1px solid #333;
  background-color: #1a1a1a;
}

.input-container {
  display: flex;
  gap: 0.8rem;
  align-items: flex-end;
  margin-bottom: 1rem;
}

.message-input {
  flex: 1;
  min-height: 44px;
  max-height: 120px;
  padding: 0.8rem 1rem;
  background-color: #2a2a2a;
  border: 1px solid #444;
  border-radius: 8px;
  color: #fff;
  font-size: 0.9rem;
  line-height: 1.4;
  resize: none;
  font-family: inherit;
}

.message-input:focus {
  outline: none;
  border-color: #10b981;
}

.message-input:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.send-button {
  width: 44px;
  height: 44px;
  background-color: #374151;
  border: 1px solid #6b7280;
  border-radius: 8px;
  color: #ccc;
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 1.2rem;
  transition: all 0.2s;
}

.send-button:hover:not(:disabled) {
  background-color: #4b5563;
}

.send-button.ready {
  background-color: #10b981;
  border-color: #10b981;
  color: #fff;
}

.send-button.ready:hover:not(:disabled) {
  background-color: #059669;
}

.send-button:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

/* Input Actions */
.input-actions {
  display: flex;
  justify-content: space-between;
  align-items: center;
  font-size: 0.8rem;
}

.left-actions,
.right-actions {
  display: flex;
  align-items: center;
  gap: 1rem;
}

.char-count {
  color: #666;
}

.clear-btn,
.export-btn,
.clear-chat-btn {
  background: none;
  border: none;
  color: #888;
  cursor: pointer;
  padding: 0.3rem 0.6rem;
  border-radius: 4px;
  font-size: 0.8rem;
  transition: all 0.2s;
}

.clear-btn:hover,
.export-btn:hover,
.clear-chat-btn:hover {
  background-color: #333;
  color: #ccc;
}

.clear-chat-btn:hover {
  color: #ff6b6b;
}

.clear-btn:disabled,
.export-btn:disabled,
.clear-chat-btn:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

/* Responsive Design */
@media (max-width: 768px) {
  .chat-header {
    flex-direction: column;
    gap: 1rem;
    text-align: center;
  }
  
  .model-indicator {
    justify-content: center;
  }
  
  .lora-toggle-section {
    align-items: center;
  }
  
  .input-actions {
    flex-direction: column;
    gap: 1rem;
  }
  
  .suggestion-chips {
    gap: 0.3rem;
  }
  
  .suggestion-chip {
    font-size: 0.7rem;
    padding: 0.4rem 0.8rem;
  }
}
</style>