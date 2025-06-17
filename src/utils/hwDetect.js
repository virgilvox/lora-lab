/**
 * Hardware Detection Module
 * Detects WebGPU support, estimates TFLOPs, and checks device memory
 */

/**
 * Detect WebGPU Support and Request Adapter
 * @returns {Promise<GPUAdapter|null>} GPU adapter if available, null otherwise
 */
export async function detectWebGPU() {
  try {
    // Check if WebGPU is supported
    if (!navigator.gpu) {
      console.log('WebGPU not supported in this browser');
      return null;
    }

    // Request GPU adapter
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) {
      console.log('WebGPU adapter not available');
      return null;
    }

    console.log('WebGPU adapter detected:', adapter);
    return adapter;
  } catch (error) {
    console.error('Error detecting WebGPU:', error);
    return null;
  }
}

/**
 * Extract GPU Information and Estimate TFLOPs
 * @param {GPUAdapter} adapter - The GPU adapter
 * @returns {Object} GPU information including estimated TFLOPs
 */
export function getGPUInfo(adapter) {
  if (!adapter) {
    return {
      name: 'Unknown',
      vendor: 'Unknown',
      architecture: 'Unknown',
      estimatedTFLOPs: 0,
      features: [],
      limits: {},
      error: 'No adapter available'
    };
  }

  try {
    const info = adapter.info || {};
    const limits = adapter.limits || {};
    const features = Array.from(adapter.features || []);

    // Estimate TFLOPs based on adapter info
    const estimatedTFLOPs = estimateTFLOPs(info, limits);

    return {
      name: info.device || 'Unknown GPU',
      vendor: info.vendor || 'Unknown',
      architecture: info.architecture || 'Unknown',
      estimatedTFLOPs,
      features,
      limits: {
        maxComputeUnits: limits.maxComputeWorkgroupsPerDimension,
        maxWorkgroupSize: limits.maxComputeWorkgroupSizeX,
        maxBufferSize: limits.maxBufferSize,
        maxStorageBufferBindingSize: limits.maxStorageBufferBindingSize
      }
    };
  } catch (error) {
    console.error('Error extracting GPU info:', error);
    return {
      name: 'Error',
      vendor: 'Unknown',
      architecture: 'Unknown',
      estimatedTFLOPs: 0,
      features: [],
      limits: {},
      error: error.message
    };
  }
}

/**
 * Estimate TFLOPs based on GPU adapter information
 * @param {Object} info - GPU adapter info
 * @param {Object} limits - GPU adapter limits
 * @returns {number} Estimated TFLOPs
 */
function estimateTFLOPs(info, limits) {
  const deviceName = (info.device || '').toLowerCase();
  const vendor = (info.vendor || '').toLowerCase();

  // Known GPU TFLOPs mappings
  const knownGPUs = {
    // Apple Silicon
    'm1': 2.6,
    'm1 pro': 5.2,
    'm1 max': 10.4,
    'm2': 3.6,
    'm2 pro': 7.2,
    'm2 max': 13.6,
    'm3': 4.1,
    'm3 pro': 8.2,
    'm3 max': 14.2,
    
    // NVIDIA RTX
    'rtx 4090': 83.0,
    'rtx 4080': 48.7,
    'rtx 4070': 29.1,
    'rtx 4060': 15.1,
    'rtx 3090': 35.6,
    'rtx 3080': 29.8,
    'rtx 3070': 20.3,
    'rtx 3060': 12.7,
    
    // AMD RDNA
    'rx 7900 xtx': 61.4,
    'rx 7900 xt': 51.5,
    'rx 6900 xt': 23.0,
    'rx 6800 xt': 20.7,
    'rx 6700 xt': 13.2,
    
    // Intel Arc
    'arc a770': 17.2,
    'arc a750': 14.1,
    'arc a580': 12.0
  };

  // Try to match known GPUs
  for (const [gpu, tflops] of Object.entries(knownGPUs)) {
    if (deviceName.includes(gpu)) {
      return tflops;
    }
  }

  // Fallback estimation based on vendor and compute units
  if (vendor.includes('apple')) {
    // Estimate based on Apple Silicon
    return Math.max(2.6, (limits.maxComputeWorkgroupsPerDimension || 256) / 100);
  } else if (vendor.includes('nvidia')) {
    // Estimate based on NVIDIA architecture
    return Math.max(10.0, (limits.maxComputeWorkgroupsPerDimension || 512) / 50);
  } else if (vendor.includes('amd')) {
    // Estimate based on AMD architecture
    return Math.max(8.0, (limits.maxComputeWorkgroupsPerDimension || 512) / 60);
  } else if (vendor.includes('intel')) {
    // Estimate based on Intel integrated graphics
    return Math.max(2.0, (limits.maxComputeWorkgroupsPerDimension || 256) / 128);
  }

  // Generic fallback
  return Math.max(1.0, (limits.maxComputeWorkgroupsPerDimension || 256) / 256);
}

/**
 * Detect CPU and Device Memory Capabilities
 * @returns {Object} Memory and CPU information
 */
export function getDeviceMemory() {
  try {
    // Get device memory (available in secure contexts)
    const deviceMemory = navigator.deviceMemory || 4; // Default to 4GB
    
    // Get CPU logical cores
    const logicalCores = navigator.hardwareConcurrency || 4; // Default to 4 cores
    
    return {
      deviceMemoryGB: deviceMemory,
      logicalCores,
      estimatedAvailableMemoryGB: Math.max(1, deviceMemory * 0.7), // Assume 70% available
      cpuArchitecture: getCPUArchitecture()
    };
  } catch (error) {
    console.error('Error detecting device memory:', error);
    return {
      deviceMemoryGB: 4,
      logicalCores: 4,
      estimatedAvailableMemoryGB: 2.8,
      cpuArchitecture: 'unknown',
      error: error.message
    };
  }
}

/**
 * Get CPU architecture information
 * @returns {string} CPU architecture
 */
function getCPUArchitecture() {
  try {
    // Try to detect architecture from user agent
    const userAgent = navigator.userAgent.toLowerCase();
    
    if (userAgent.includes('arm') || userAgent.includes('aarch64')) {
      return 'ARM64';
    } else if (userAgent.includes('x86_64') || userAgent.includes('win64')) {
      return 'x86_64';
    } else if (userAgent.includes('x86')) {
      return 'x86';
    }
    
    return 'unknown';
  } catch (error) {
    return 'unknown';
  }
}

/**
 * Get TFLOPs Estimate (convenience function)
 * @returns {Promise<number>} Estimated TFLOPs
 */
export async function getTFLOPsEstimate() {
  try {
    const adapter = await detectWebGPU();
    if (adapter) {
      const gpuInfo = getGPUInfo(adapter);
      return gpuInfo.estimatedTFLOPs;
    }
    
    // Fallback to CPU-based estimation
    const memInfo = getDeviceMemory();
    return Math.max(0.1, memInfo.logicalCores * 0.1); // Very rough CPU estimate
  } catch (error) {
    console.error('Error estimating TFLOPs:', error);
    return 0.1;
  }
}

/**
 * Comprehensive hardware detection
 * @returns {Promise<Object>} Complete hardware information
 */
export async function detectHardware() {
  try {
    const adapter = await detectWebGPU();
    const gpuInfo = getGPUInfo(adapter);
    const memoryInfo = getDeviceMemory();
    
    return {
      webGPUSupported: !!adapter,
      gpu: gpuInfo,
      memory: memoryInfo,
      estimatedTFLOPs: gpuInfo.estimatedTFLOPs || (memoryInfo.logicalCores * 0.1),
      timestamp: new Date().toISOString()
    };
  } catch (error) {
    console.error('Error in comprehensive hardware detection:', error);
    return {
      webGPUSupported: false,
      gpu: { error: error.message },
      memory: getDeviceMemory(),
      estimatedTFLOPs: 0.1,
      timestamp: new Date().toISOString(),
      error: error.message
    };
  }
}

// Default export for backward compatibility
export default {
  detectWebGPU,
  getGPUInfo,
  getDeviceMemory,
  getTFLOPsEstimate,
  detectHardware
};