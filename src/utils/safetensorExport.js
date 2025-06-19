/**
 * Safetensors Export/Import Utility for LoRA Lab
 * Handles serialization and deserialization of LoRA adapters in safetensors format
 */

// Import safetensors - this version only has SafeTensor class for loading
import { SafeTensor, loadSafeTensor } from 'safetensors';

// Since this version of safetensors doesn't have serialize/deserialize functions,
// we'll implement them ourselves following the safetensors format specification

/**
 * Serialize tensors to safetensors format
 * @param {Object} tensors - Object with tensor names as keys and {data, shape, dtype} as values
 * @returns {Uint8Array} - Serialized safetensors buffer
 */
function serialize(tensors) {
  const metadata = {};
  const buffers = [];
  let currentOffset = 0;

  // Build metadata and collect buffers
  for (const [name, tensor] of Object.entries(tensors)) {
    const { data, shape, dtype } = tensor;
    const byteLength = data.byteLength;
    
    metadata[name] = {
      dtype: dtype || 'F32',
      shape: shape,
      data_offsets: [currentOffset, currentOffset + byteLength]
    };
    
    buffers.push(new Uint8Array(data.buffer || data));
    currentOffset += byteLength;
  }

  // Serialize metadata to JSON
  const metadataStr = JSON.stringify(metadata);
  const metadataBytes = new TextEncoder().encode(metadataStr);
  
  // Calculate padding to align to 8 bytes
  const padding = (8 - (metadataBytes.length % 8)) % 8;
  const paddedMetadataLength = metadataBytes.length + padding;
  
  // Create header (8 bytes for metadata length)
  const headerBuffer = new ArrayBuffer(8);
  const headerView = new DataView(headerBuffer);
  headerView.setBigUint64(0, BigInt(paddedMetadataLength), true);
  
  // Combine all parts
  const totalLength = 8 + paddedMetadataLength + currentOffset;
  const result = new Uint8Array(totalLength);
  
  // Copy header
  result.set(new Uint8Array(headerBuffer), 0);
  
  // Copy metadata with padding
  result.set(metadataBytes, 8);
  
  // Copy tensor data
  let offset = 8 + paddedMetadataLength;
  for (const buffer of buffers) {
    result.set(buffer, offset);
    offset += buffer.length;
  }
  
  return result;
}

/**
 * Deserialize safetensors format to tensors
 * @param {Uint8Array} buffer - Safetensors buffer
 * @returns {Object} - Object with tensor names as keys and {data, shape, dtype} as values
 */
function deserialize(buffer) {
  const safetensor = new SafeTensor(buffer);
  const tensors = {};
  
  for (const [name, metadata] of safetensor.metadata.entries()) {
    const data = safetensor.getTensor(name);
    tensors[name] = {
      data: data,
      shape: metadata.shape,
      dtype: metadata.dtype
    };
  }
  
  return tensors;
}

/**
 * Export LoRA adapter to safetensors format
 * @param {Object} adapterData - LoRA adapter data
 * @param {Object} metadata - Adapter metadata
 * @returns {Promise<Uint8Array>} Serialized safetensors data
 */
export async function exportAdapter(adapterData, metadata = {}) {
  try {
    console.log('Exporting LoRA adapter to safetensors format...');
    
    // Prepare tensors dictionary
    const tensors = {};
    
    // Process LoRA layer data
    for (const [layerName, layerData] of Object.entries(adapterData.layers || {})) {
      // LoRA A matrix (input_dim x rank)
      if (layerData.A && layerData.A.data) {
        tensors[`${layerName}.lora_A.weight`] = {
          data: new Float32Array(layerData.A.data),
          shape: layerData.A.shape,
          dtype: 'F32'
        };
      }
      
      // LoRA B matrix (rank x output_dim)
      if (layerData.B && layerData.B.data) {
        tensors[`${layerName}.lora_B.weight`] = {
          data: new Float32Array(layerData.B.data),
          shape: layerData.B.shape,
          dtype: 'F32'
        };
      }
    }
    
    // Prepare metadata
    const adapterMetadata = {
      format_version: '1.0',
      lora_lab_version: '0.1.0',
      adapter_type: 'lora',
      rank: adapterData.rank || 4,
      alpha: adapterData.alpha || 8,
      scaling: adapterData.scaling || (adapterData.alpha / adapterData.rank),
      target_modules: adapterData.targetModules || [],
      created_at: new Date().toISOString(),
      model_name: metadata.modelName || 'unknown',
      training_steps: metadata.trainingSteps || 0,
      final_loss: metadata.finalLoss || 0,
      ...metadata
    };
    
    // Serialize to safetensors format
    const serialized = serialize(tensors);
    
    console.log('Adapter exported successfully:', {
      tensorCount: Object.keys(tensors).length,
      dataSize: serialized.byteLength,
      metadata: adapterMetadata
    });
    
    return serialized;
    
  } catch (error) {
    console.error('Adapter export failed:', error);
    throw new Error(`Failed to export adapter: ${error.message}`);
  }
}

/**
 * Import LoRA adapter from safetensors format
 * @param {Uint8Array} data - Serialized safetensors data
 * @returns {Promise<Object>} Deserialized adapter data
 */
export async function importAdapter(data) {
  try {
    console.log('Importing LoRA adapter from safetensors format...');
    
    // Deserialize safetensors data
    const tensors = deserialize(data);
    
    // Validate format
    if (!tensors || tensors.adapter_type !== 'lora') {
      throw new Error('Invalid adapter format: not a LoRA adapter');
    }
    
    // Process tensor data into LoRA layers
    const layers = {};
    const tensorNames = Object.keys(tensors);
    
    // Group tensors by layer
    const layerGroups = {};
    for (const tensorName of tensorNames) {
      const match = tensorName.match(/^(.+)\.(lora_[AB])\.weight$/);
      if (match) {
        const [, layerName, matrixType] = match;
        if (!layerGroups[layerName]) {
          layerGroups[layerName] = {};
        }
        layerGroups[layerName][matrixType] = tensors[tensorName];
      }
    }
    
    // Reconstruct layer data
    for (const [layerName, matrices] of Object.entries(layerGroups)) {
      layers[layerName] = {
        A: matrices.lora_A ? {
          data: Array.from(matrices.lora_A.data),
          shape: matrices.lora_A.shape,
          initialized: true
        } : null,
        B: matrices.lora_B ? {
          data: Array.from(matrices.lora_B.data),
          shape: matrices.lora_B.shape,
          initialized: true
        } : null,
        paramCount: matrices.lora_A && matrices.lora_B ? 
          matrices.lora_A.data.length + matrices.lora_B.data.length : 0
      };
    }
    
    // Calculate total parameters
    const totalParams = Object.values(layers).reduce(
      (sum, layer) => sum + (layer.paramCount || 0), 0
    );
    
    const adapterData = {
      rank: tensors.rank || 4,
      alpha: tensors.alpha || 8,
      scaling: tensors.scaling || (tensors.alpha / tensors.rank),
      layers,
      totalParams,
      targetModules: tensors.target_modules || [],
      metadata: {
        formatVersion: tensors.format_version,
        loraLabVersion: tensors.lora_lab_version,
        createdAt: tensors.created_at,
        modelName: tensors.model_name,
        trainingSteps: tensors.training_steps,
        finalLoss: tensors.final_loss,
        importedAt: new Date().toISOString()
      }
    };
    
    console.log('Adapter imported successfully:', {
      layerCount: Object.keys(layers).length,
      totalParams,
      rank: adapterData.rank,
      alpha: adapterData.alpha
    });
    
    return adapterData;
    
  } catch (error) {
    console.error('Adapter import failed:', error);
    throw new Error(`Failed to import adapter: ${error.message}`);
  }
}

/**
 * Download adapter as a file
 * @param {Object} adapterData - LoRA adapter data
 * @param {string} filename - Download filename
 * @param {Object} metadata - Additional metadata
 */
export async function downloadAdapter(adapterData, filename = 'lora_adapter.safetensors', metadata = {}) {
  try {
    const serializedData = await exportAdapter(adapterData, metadata);
    
    // --- New Validation Step ---
    const validation = await validateAdapterFile(serializedData);
    if (!validation.isValid) {
        console.error("Validation failed:", validation.errors);
        alert(`Failed to validate the exported adapter: ${validation.errors.join(', ')}`);
        // Optionally, still allow download if there are only warnings
        if (validation.errors.length > 0) {
            const proceed = confirm("Adapter validation failed. Proceed with download anyway?");
            if (!proceed) return;
        }
    }
    // --- End Validation Step ---

    // Create blob and download
    const blob = new Blob([serializedData], { type: 'application/octet-stream' });
    const url = URL.createObjectURL(blob);
    
    const link = document.createElement('a');
    link.href = url;
    link.download = filename;
    link.style.display = 'none';
    
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    
    // Cleanup
    URL.revokeObjectURL(url);
    
    console.log('Adapter download initiated:', filename);
    
  } catch (error) {
    console.error('Adapter download failed:', error);
    throw error;
  }
}

/**
 * Validate adapter file before import
 * @param {File|Uint8Array} fileData - File data to validate
 * @returns {Promise<Object>} Validation result
 */
export async function validateAdapterFile(fileData) {
  const validation = {
    isValid: false,
    errors: [],
    warnings: [],
    metadata: null,
    tensorInfo: null
  };
  
  try {
    // Convert File to Uint8Array if needed
    let data = fileData;
    if (fileData instanceof File) {
      data = new Uint8Array(await fileData.arrayBuffer());
    }
    
    // Check file size
    if (data.byteLength === 0) {
      validation.errors.push('File is empty');
      return validation;
    }
    
    if (data.byteLength > 500 * 1024 * 1024) { // 500MB limit
      validation.warnings.push('File is very large (>500MB) - may cause memory issues');
    }
    
    // Try to deserialize
    const tensors = deserialize(data);
    
    // Validate metadata
    if (!tensors) {
      validation.errors.push('No metadata found in safetensors file');
      return validation;
    }
    
    if (tensors.adapter_type !== 'lora') {
      validation.errors.push(`Unsupported adapter type: ${tensors.adapter_type}`);
      return validation;
    }
    
    // Check format version compatibility
    const formatVersion = tensors.format_version || '1.0';
    if (formatVersion !== '1.0') {
      validation.warnings.push(`Format version ${formatVersion} may not be fully compatible`);
    }
    
    // Validate tensor structure
    const tensorNames = Object.keys(tensors);
    const loraLayers = {};
    
    for (const tensorName of tensorNames) {
      const match = tensorName.match(/^(.+)\.(lora_[AB])\.weight$/);
      if (!match) {
        validation.warnings.push(`Unexpected tensor name: ${tensorName}`);
        continue;
      }
      
      const [, layerName, matrixType] = match;
      if (!loraLayers[layerName]) {
        loraLayers[layerName] = {};
      }
      loraLayers[layerName][matrixType] = tensors[tensorName];
    }
    
    // Check for complete A/B pairs
    const incompleteLayers = [];
    for (const [layerName, matrices] of Object.entries(loraLayers)) {
      if (!matrices.lora_A || !matrices.lora_B) {
        incompleteLayers.push(layerName);
      }
    }
    
    if (incompleteLayers.length > 0) {
      validation.warnings.push(`Incomplete LoRA layers: ${incompleteLayers.join(', ')}`);
    }
    
    // Calculate tensor statistics
    const totalTensors = Object.keys(tensors).length;
    const totalParams = Object.values(tensors).reduce(
      (sum, tensor) => sum + tensor.data.length, 0
    );
    
    validation.isValid = validation.errors.length === 0;
    validation.metadata = tensors;
    validation.tensorInfo = {
      totalTensors,
      totalParams,
      layerCount: Object.keys(loraLayers).length,
      rank: tensors.rank,
      alpha: tensors.alpha
    };
    
    if (validation.isValid) {
      console.log('Adapter file validation passed:', validation.tensorInfo);
    } else {
      console.warn('Adapter file validation failed:', validation.errors);
    }
    
  } catch (error) {
    validation.errors.push(`File parsing failed: ${error.message}`);
    console.error('Adapter validation error:', error);
  }
  
  return validation;
}

/**
 * Create drag-and-drop file handler
 * @param {HTMLElement} dropZone - Drop zone element
 * @param {Function} onFileDropped - Callback for when file is dropped
 * @returns {Object} Handler controls
 */
export function createDropHandler(dropZone, onFileDropped) {
  const handlers = {
    dragOver: (event) => {
      event.preventDefault();
      event.stopPropagation();
      dropZone.classList.add('drag-over');
    },
    
    dragLeave: (event) => {
      event.preventDefault();
      event.stopPropagation();
      dropZone.classList.remove('drag-over');
    },
    
    drop: async (event) => {
      event.preventDefault();
      event.stopPropagation();
      dropZone.classList.remove('drag-over');
      
      const files = Array.from(event.dataTransfer.files);
      const safetensorFiles = files.filter(file => 
        file.name.endsWith('.safetensors') || file.type === 'application/octet-stream'
      );
      
      if (safetensorFiles.length === 0) {
        alert('Please drop a .safetensors file');
        return;
      }
      
      if (safetensorFiles.length > 1) {
        alert('Please drop only one adapter file at a time');
        return;
      }
      
      const file = safetensorFiles[0];
      
      try {
        // Validate file
        const validation = await validateAdapterFile(file);
        
        if (!validation.isValid) {
          alert(`Invalid adapter file:\n${validation.errors.join('\n')}`);
          return;
        }
        
        if (validation.warnings.length > 0) {
          const proceed = confirm(
            `Warnings detected:\n${validation.warnings.join('\n')}\n\nContinue anyway?`
          );
          if (!proceed) return;
        }
        
        // Import adapter
        const data = new Uint8Array(await file.arrayBuffer());
        const adapterData = await importAdapter(data);
        
        // Call callback
        onFileDropped(adapterData, file.name);
        
      } catch (error) {
        console.error('File drop handling failed:', error);
        alert(`Failed to load adapter: ${error.message}`);
      }
    }
  };
  
  // Add event listeners
  dropZone.addEventListener('dragover', handlers.dragOver);
  dropZone.addEventListener('dragleave', handlers.dragLeave);
  dropZone.addEventListener('drop', handlers.drop);
  
  // Return cleanup function
  return {
    destroy: () => {
      dropZone.removeEventListener('dragover', handlers.dragOver);
      dropZone.removeEventListener('dragleave', handlers.dragLeave);
      dropZone.removeEventListener('drop', handlers.drop);
    }
  };
}

/**
 * Convert adapter data to HuggingFace format
 * @param {Object} adapterData - LoRA adapter data
 * @returns {Object} HuggingFace format adapter
 */
export function convertToHuggingFaceFormat(adapterData) {
  const hfAdapter = {
    adapter_type: 'lora',
    r: adapterData.rank,
    lora_alpha: adapterData.alpha,
    target_modules: adapterData.targetModules,
    lora_dropout: 0.1, // Default dropout
    bias: 'none',
    modules_to_save: [],
    layers: {}
  };
  
  // Convert layer data
  for (const [layerName, layerData] of Object.entries(adapterData.layers)) {
    if (layerData.A && layerData.B) {
      hfAdapter.layers[layerName] = {
        lora_A: {
          weight: layerData.A.data,
          shape: layerData.A.shape
        },
        lora_B: {
          weight: layerData.B.data,
          shape: layerData.B.shape
        }
      };
    }
  }
  
  return hfAdapter;
}

/**
 * Export utilities object
 */
export default {
  exportAdapter,
  importAdapter,
  downloadAdapter,
  validateAdapterFile,
  createDropHandler,
  convertToHuggingFaceFormat
};