<template>
  <header class="header-bar">
    <div class="header-section">
      <!-- Model Selection Dropdown -->
      <div class="form-group">
        <label for="model-select">Model:</label>
        <select 
          id="model-select" 
          v-model="selectedModel" 
          @change="onModelChange"
          class="model-dropdown"
        >
          <option value="">Select a model...</option>
          <option 
            v-for="model in availableModels" 
            :key="model.id" 
            :value="model.id"
          >
            {{ model.name }} ({{ model.size }})
          </option>
        </select>
      </div>

      <!-- URL Input for Custom Model -->
      <div class="form-group">
        <label for="model-url">Custom Model URL:</label>
        <input 
          id="model-url"
          type="url" 
          v-model="modelUrl" 
          @input="onUrlChange"
          placeholder="https://example.com/model.onnx"
          class="url-input"
          :class="{ 'invalid': urlError }"
        />
        <span v-if="urlError" class="error-text">{{ urlError }}</span>
      </div>
    </div>

    <div class="header-section">
      <!-- Corpus File Upload -->
      <div class="form-group">
        <label for="corpus-file">Upload Corpus:</label>
        <input 
          id="corpus-file"
          type="file" 
          @change="onFileUpload"
          accept=".txt,.md,.json"
          class="file-input"
        />
        <div v-if="uploadedFile" class="file-status">
          <span class="file-name">{{ uploadedFile.name }}</span>
          <span class="file-size">({{ formatFileSize(uploadedFile.size) }})</span>
          <button @click="clearFile" class="clear-btn">×</button>
        </div>
      </div>

      <!-- Corpus Text Input Button -->
      <div class="form-group">
        <button 
          @click="showTextModal = true" 
          class="text-input-btn"
          :class="{ 'has-content': corpusText.length > 0 }"
        >
          {{ corpusText.length > 0 ? `Edit Text (${formatTokenCount(corpusText)})` : 'Paste Text' }}
        </button>
      </div>

      <!-- Train Button -->
      <div class="form-group">
        <button 
          @click="startTraining" 
          class="train-btn"
          :disabled="!canStartTraining"
          :class="{ 'ready': canStartTraining }"
        >
          {{ isTraining ? 'Training...' : 'Train' }}
        </button>
      </div>
    </div>

    <!-- Text Input Modal -->
    <div v-if="showTextModal" class="modal-overlay" @click="closeModal">
      <div class="modal-content" @click.stop>
        <div class="modal-header">
          <h3>Corpus Text Input</h3>
          <button @click="closeModal" class="close-btn">×</button>
        </div>
        <div class="modal-body">
          <textarea 
            v-model="corpusText"
            @input="onTextInput"
            placeholder="Paste your training corpus here..."
            class="corpus-textarea"
            rows="15"
          ></textarea>
          <div class="text-stats">
            <span>Characters: {{ corpusText.length }}</span>
            <span>Estimated tokens: {{ estimatedTokens }}</span>
          </div>
        </div>
        <div class="modal-footer">
          <button @click="clearText" class="clear-text-btn">Clear</button>
          <button @click="closeModal" class="apply-btn">Apply</button>
        </div>
      </div>
    </div>
  </header>
</template>

<script>
export default {
  name: 'HeaderBar',
  emits: ['model-selected', 'corpus-loaded', 'start-training'],
  data() {
    return {
      selectedModel: '',
      modelUrl: '',
      urlError: '',
      uploadedFile: null,
      corpusText: '',
      showTextModal: false,
      isTraining: false,
      availableModels: [
        { id: 'gemma-2b', name: 'Gemma 2B', size: '2.3GB' },
        { id: 'phi-2', name: 'Phi-2', size: '2.7GB' },
        { id: 'tinyllama-1.1b', name: 'TinyLlama 1.1B', size: '1.1GB' },
        { id: 'qwen2-1.5b', name: 'Qwen2 1.5B', size: '1.5GB' }
      ]
    };
  },
  computed: {
    estimatedTokens() {
      // Rough estimation: ~4 characters per token
      return Math.ceil(this.corpusText.length / 4);
    },
    canStartTraining() {
      const hasModel = this.selectedModel || this.isValidUrl(this.modelUrl);
      const hasCorpus = this.corpusText.length > 0 || this.uploadedFile;
      return hasModel && hasCorpus && !this.isTraining;
    }
  },
  methods: {
    onModelChange() {
      this.modelUrl = ''; // Clear URL when selecting from dropdown
      this.urlError = '';
      this.$emit('model-selected', {
        type: 'preset',
        modelId: this.selectedModel,
        model: this.availableModels.find(m => m.id === this.selectedModel)
      });
    },

    onUrlChange() {
      this.selectedModel = ''; // Clear dropdown when entering URL
      this.urlError = '';
      
      if (this.modelUrl && !this.isValidUrl(this.modelUrl)) {
        this.urlError = 'Please enter a valid URL';
      } else if (this.modelUrl && this.isValidUrl(this.modelUrl)) {
        this.$emit('model-selected', {
          type: 'custom',
          url: this.modelUrl
        });
      }
    },

    isValidUrl(string) {
      try {
        const url = new URL(string);
        return url.protocol === 'http:' || url.protocol === 'https:';
      } catch (_) {
        return false;
      }
    },

    async onFileUpload(event) {
      const file = event.target.files[0];
      if (!file) return;

      // Validate file type
      const validTypes = ['.txt', '.md', '.json'];
      const fileExtension = '.' + file.name.split('.').pop().toLowerCase();
      
      if (!validTypes.includes(fileExtension)) {
        alert(`Invalid file type. Please upload one of: ${validTypes.join(', ')}`);
        event.target.value = '';
        return;
      }

      this.uploadedFile = file;
      
      try {
        const text = await this.readFileContent(file);
        this.corpusText = ''; // Clear textarea when loading file
        await this.tokenizeText(text);
        
        this.$emit('corpus-loaded', {
          type: 'file',
          file: file,
          content: text,
          tokenCount: this.estimateTokens(text)
        });
      } catch (error) {
        console.error('Error reading file:', error);
        alert('Error reading file. Please try again.');
        this.clearFile();
      }
    },

    readFileContent(file) {
      return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = (e) => resolve(e.target.result);
        reader.onerror = reject;
        reader.readAsText(file);
      });
    },

    clearFile() {
      this.uploadedFile = null;
      const fileInput = document.getElementById('corpus-file');
      if (fileInput) fileInput.value = '';
      
      this.$emit('corpus-loaded', {
        type: 'clear',
        content: '',
        tokenCount: 0
      });
    },

    onTextInput() {
      if (this.uploadedFile) {
        this.clearFile(); // Clear file when typing in textarea
      }
      this.tokenizeText(this.corpusText);
    },

    async tokenizeText(text) {
      // Simple tokenization for now - can be enhanced with actual tokenizer
      const tokenCount = this.estimateTokens(text);
      
      this.$emit('corpus-loaded', {
        type: 'text',
        content: text,
        tokenCount: tokenCount
      });
    },

    estimateTokens(text) {
      // Rough estimation: ~4 characters per token
      return Math.ceil(text.length / 4);
    },

    formatTokenCount(text) {
      const tokens = this.estimateTokens(text);
      return `~${tokens.toLocaleString()} tokens`;
    },

    formatFileSize(bytes) {
      if (bytes === 0) return '0 Bytes';
      const k = 1024;
      const sizes = ['Bytes', 'KB', 'MB', 'GB'];
      const i = Math.floor(Math.log(bytes) / Math.log(k));
      return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    },

    clearText() {
      this.corpusText = '';
      this.onTextInput();
    },

    closeModal() {
      this.showTextModal = false;
    },

    startTraining() {
      if (!this.canStartTraining) return;
      
      this.isTraining = true;
      this.$emit('start-training', {
        model: this.selectedModel || this.modelUrl,
        corpus: this.corpusText || this.uploadedFile,
        estimatedTokens: this.corpusText ? this.estimatedTokens : null
      });
    }
  }
};
</script>

<style scoped>
.header-bar {
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  padding: 1rem 2rem;
  background-color: #2a2a2a;
  border-bottom: 1px solid #444;
  gap: 2rem;
  flex-wrap: wrap;
}

.header-section {
  display: flex;
  align-items: flex-start;
  gap: 1.5rem;
  flex-wrap: wrap;
}

.form-group {
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
  min-width: 150px;
}

.form-group label {
  font-size: 0.9rem;
  color: #ccc;
  font-weight: 500;
}

.model-dropdown,
.url-input {
  padding: 0.5rem;
  border: 1px solid #555;
  border-radius: 4px;
  background-color: #333;
  color: #fff;
  font-size: 0.9rem;
}

.model-dropdown {
  min-width: 200px;
}

.url-input {
  min-width: 300px;
}

.url-input.invalid {
  border-color: #ff6b6b;
}

.error-text {
  font-size: 0.8rem;
  color: #ff6b6b;
}

.file-input {
  padding: 0.3rem;
  border: 1px solid #555;
  border-radius: 4px;
  background-color: #333;
  color: #fff;
  font-size: 0.9rem;
}

.file-status {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  font-size: 0.8rem;
  color: #4ade80;
}

.file-name {
  font-weight: 500;
}

.file-size {
  color: #888;
}

.clear-btn {
  background: none;
  border: none;
  color: #ff6b6b;
  cursor: pointer;
  font-size: 1.2rem;
  padding: 0 0.3rem;
  border-radius: 3px;
}

.clear-btn:hover {
  background-color: #ff6b6b22;
}

.text-input-btn {
  padding: 0.6rem 1rem;
  border: 1px solid #555;
  border-radius: 4px;
  background-color: #444;
  color: #fff;
  cursor: pointer;
  font-size: 0.9rem;
  transition: all 0.2s;
}

.text-input-btn:hover {
  background-color: #555;
}

.text-input-btn.has-content {
  background-color: #4ade80;
  color: #000;
  border-color: #4ade80;
}

.train-btn {
  padding: 0.6rem 1.5rem;
  border: 1px solid #666;
  border-radius: 4px;
  background-color: #666;
  color: #fff;
  cursor: pointer;
  font-size: 0.9rem;
  font-weight: 600;
  transition: all 0.2s;
}

.train-btn:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.train-btn.ready {
  background-color: #10b981;
  border-color: #10b981;
  color: #fff;
}

.train-btn.ready:hover:not(:disabled) {
  background-color: #059669;
}

/* Modal Styles */
.modal-overlay {
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background-color: rgba(0, 0, 0, 0.7);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 1000;
}

.modal-content {
  background-color: #2a2a2a;
  border-radius: 8px;
  width: 90%;
  max-width: 600px;
  max-height: 80vh;
  display: flex;
  flex-direction: column;
}

.modal-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 1rem 1.5rem;
  border-bottom: 1px solid #444;
}

.modal-header h3 {
  margin: 0;
  color: #fff;
}

.close-btn {
  background: none;
  border: none;
  color: #ccc;
  cursor: pointer;
  font-size: 1.5rem;
  padding: 0;
  width: 2rem;
  height: 2rem;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
}

.close-btn:hover {
  background-color: #444;
}

.modal-body {
  padding: 1.5rem;
  flex: 1;
  overflow: auto;
}

.corpus-textarea {
  width: 100%;
  min-height: 300px;
  padding: 1rem;
  border: 1px solid #555;
  border-radius: 4px;
  background-color: #333;
  color: #fff;
  font-family: 'Courier New', monospace;
  font-size: 0.9rem;
  line-height: 1.4;
  resize: vertical;
}

.text-stats {
  display: flex;
  justify-content: space-between;
  margin-top: 1rem;
  font-size: 0.8rem;
  color: #888;
}

.modal-footer {
  padding: 1rem 1.5rem;
  border-top: 1px solid #444;
  display: flex;
  justify-content: space-between;
}

.clear-text-btn,
.apply-btn {
  padding: 0.6rem 1rem;
  border: 1px solid #555;
  border-radius: 4px;
  cursor: pointer;
  font-size: 0.9rem;
}

.clear-text-btn {
  background-color: #dc2626;
  color: #fff;
  border-color: #dc2626;
}

.clear-text-btn:hover {
  background-color: #b91c1c;
}

.apply-btn {
  background-color: #10b981;
  color: #fff;
  border-color: #10b981;
}

.apply-btn:hover {
  background-color: #059669;
}

/* Responsive Design */
@media (max-width: 768px) {
  .header-bar {
    flex-direction: column;
    gap: 1rem;
  }
  
  .header-section {
    width: 100%;
    justify-content: space-between;
  }
  
  .form-group {
    min-width: auto;
    flex: 1;
  }
  
  .url-input {
    min-width: auto;
  }
  
  .modal-content {
    width: 95%;
    margin: 1rem;
  }
}
</style>