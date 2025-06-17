/**
 * Hardware Detection Utility for LoRA Lab
 * Detects WebGPU support, GPU capabilities, memory, and estimates TFLOPs
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
 * Comprehensive hardware detection
 * @returns {Promise<Object>} Complete hardware information
 */
export async function detectHardware() {
  const hardwareInfo = {
    webGPUSupported: false,
    gpu: null,
    estimatedTFLOPs: 0,
    memory: {
      deviceMemoryGB: 4,
      estimatedAvailableMemoryGB: 2.8,
      logicalCores: 4,
      cpuArchitecture: 'unknown'
    },
    capabilities: {
      canTrainAdapter: false,
      canTrainFull: false
    },
    warnings: []
  }

  try {
    // Check WebGPU support
    if (!navigator.gpu) {
      hardwareInfo.warnings.push('WebGPU not supported by browser')
      // For debugging - check if we're in a secure context
      if (!window.isSecureContext) {
        hardwareInfo.warnings.push('Not in secure context (HTTPS required for WebGPU)')
      }
      return hardwareInfo
    }

    // Request GPU adapter with better error handling
    let adapter = null
    try {
      // Try with high-performance preference first
      adapter = await navigator.gpu.requestAdapter({ 
        powerPreference: 'high-performance' 
      })
      
      // If that fails, try without preference
      if (!adapter) {
        adapter = await navigator.gpu.requestAdapter()
      }
    } catch (adapterError) {
      console.error('Adapter request failed:', adapterError)
      hardwareInfo.warnings.push(`Adapter request failed: ${adapterError.message}`)
    }

    if (!adapter) {
      hardwareInfo.warnings.push('No WebGPU adapter available - check browser settings')
      // Additional debugging info
      if (navigator.userAgent.includes('Chrome')) {
        hardwareInfo.warnings.push('For Chrome: ensure chrome://flags/#enable-unsafe-webgpu is enabled')
      }
      return hardwareInfo
    }

    // At this point, we have a valid adapter
    hardwareInfo.webGPUSupported = true

    // Get GPU information
    hardwareInfo.gpu = await getGPUInfo(adapter)
    
    // Estimate TFLOPs based on GPU
    hardwareInfo.estimatedTFLOPs = estimateTFLOPs(hardwareInfo.gpu)
    
    // Get memory information
    hardwareInfo.memory = getMemoryInfo()
    
    // Determine training capabilities
    hardwareInfo.capabilities = determineCapabilities(hardwareInfo)

    // Log successful detection
    console.log('WebGPU successfully detected:', {
      adapter: adapter.info || 'info not available',
      gpu: hardwareInfo.gpu.name,
      tflops: hardwareInfo.estimatedTFLOPs
    })

  } catch (error) {
    console.error('Hardware detection failed:', error)
    hardwareInfo.warnings.push('Hardware detection failed: ' + error.message)
  }

  return hardwareInfo
}

async function getGPUInfo(adapter) {
  const info = adapter.info || {}
  
  // Get device info
  const device = await adapter.requestDevice().catch(() => null)
  const limits = device?.limits || {}

  // Try to identify GPU from vendor and architecture info
  const vendor = info.vendor || detectVendorFromUA()
  const architecture = info.architecture || 'unknown'
  
  // Get a reasonable GPU name
  const gpuName = getGPUName(vendor, architecture, info)

  if (device) {
    device.destroy()
  }

  return {
    vendor: vendor,
    name: gpuName,
    architecture: architecture,
    limits: limits,
    features: device?.features ? Array.from(device.features) : []
  }
}

function detectVendorFromUA() {
  const ua = navigator.userAgent.toLowerCase()
  
  if (ua.includes('apple') || ua.includes('mac')) {
    return 'apple'
  } else if (ua.includes('nvidia')) {
    return 'nvidia'
  } else if (ua.includes('amd') || ua.includes('radeon')) {
    return 'amd'
  } else if (ua.includes('intel')) {
    return 'intel'
  }
  
  return 'unknown'
}

function getGPUName(vendor, architecture, info) {
  // Try to get a user-friendly GPU name
  
  // Apple Silicon detection
  if (vendor === 'apple' || navigator.userAgent.includes('Mac')) {
    const platform = navigator.platform.toLowerCase()
    if (platform.includes('arm') || navigator.userAgent.includes('Apple Silicon')) {
      if (navigator.userAgent.includes('M3')) return 'Apple M3'
      if (navigator.userAgent.includes('M2')) return 'Apple M2'
      if (navigator.userAgent.includes('M1')) return 'Apple M1'
      return 'Apple Silicon GPU'
    }
    return 'Apple GPU'
  }
  
  // Use info.description if available
  if (info.description) {
    return info.description
  }
  
  // Fallback based on vendor
  const vendorNames = {
    'nvidia': 'NVIDIA GPU',
    'amd': 'AMD Radeon GPU',
    'intel': 'Intel GPU',
    'unknown': 'Unknown GPU'
  }
  
  return vendorNames[vendor] || 'Unknown GPU'
}

function estimateTFLOPs(gpu) {
  const vendor = gpu.vendor
  const name = gpu.name.toLowerCase()
  
  // Apple Silicon estimations (based on known specs)
  if (vendor === 'apple' || name.includes('apple')) {
    if (name.includes('m3')) {
      if (name.includes('max')) return 14.2
      if (name.includes('pro')) return 10.9
      return 5.5 // Base M3
    }
    if (name.includes('m2')) {
      if (name.includes('ultra')) return 27.2
      if (name.includes('max')) return 13.6
      if (name.includes('pro')) return 8.7
      return 3.6 // Base M2
    }
    if (name.includes('m1')) {
      if (name.includes('ultra')) return 21.0
      if (name.includes('max')) return 10.4
      if (name.includes('pro')) return 5.2
      return 2.6 // Base M1
    }
    return 2.6 // Default Apple Silicon estimate
  }
  
  // NVIDIA estimations (rough FP32 estimates)
  if (vendor === 'nvidia' || name.includes('nvidia') || name.includes('geforce') || name.includes('rtx')) {
    if (name.includes('4090')) return 83.0
    if (name.includes('4080')) return 48.7
    if (name.includes('4070')) return 29.1
    if (name.includes('4060')) return 15.1
    if (name.includes('3090')) return 35.6
    if (name.includes('3080')) return 29.8
    if (name.includes('3070')) return 20.3
    if (name.includes('3060')) return 13.0
    if (name.includes('2080')) return 14.2
    if (name.includes('2070')) return 10.1
    if (name.includes('2060')) return 6.5
    if (name.includes('1660')) return 5.0
    return 8.0 // Default NVIDIA estimate
  }
  
  // AMD estimations
  if (vendor === 'amd' || name.includes('amd') || name.includes('radeon')) {
    if (name.includes('7900')) return 61.0
    if (name.includes('7800')) return 37.3
    if (name.includes('7700')) return 35.2
    if (name.includes('7600')) return 21.5
    if (name.includes('6900')) return 23.0
    if (name.includes('6800')) return 20.7
    if (name.includes('6700')) return 13.3
    if (name.includes('6600')) return 8.9
    return 6.0 // Default AMD estimate
  }
  
  // Intel estimations (typically integrated graphics)
  if (vendor === 'intel' || name.includes('intel')) {
    if (name.includes('arc')) {
      if (name.includes('a770')) return 8.1
      if (name.includes('a750')) return 6.8
      if (name.includes('a580')) return 5.1
      return 4.0 // Default Arc
    }
    return 1.5 // Default Intel integrated
  }
  
  // Conservative default for unknown GPUs
  return 2.0
}

function getMemoryInfo() {
  // Get system memory info
  const deviceMemoryGB = navigator.deviceMemory || 4 // GB
  
  // Estimate available memory (typically 70% of total)
  const estimatedAvailableMemoryGB = deviceMemoryGB * 0.7
  
  // Get CPU info
  const logicalCores = navigator.hardwareConcurrency || 4
  
  // Try to detect CPU architecture
  let cpuArchitecture = 'unknown'
  const ua = navigator.userAgent
  if (ua.includes('arm64') || ua.includes('aarch64')) {
    cpuArchitecture = 'arm64'
  } else if (ua.includes('x86_64') || ua.includes('x64')) {
    cpuArchitecture = 'x64'
  } else if (ua.includes('x86')) {
    cpuArchitecture = 'x86'
  }
  
  return {
    deviceMemoryGB,
    estimatedAvailableMemoryGB,
    logicalCores,
    cpuArchitecture
  }
}

function determineCapabilities(hardwareInfo) {
  const capabilities = {
    canTrainAdapter: false,
    canTrainFull: false,
    recommendedMode: null,
    maxBatchSize: 1,
    maxSequenceLength: 512
  }
  
  const tflops = hardwareInfo.estimatedTFLOPs
  const availableMemory = hardwareInfo.memory.estimatedAvailableMemoryGB
  const webGPUSupported = hardwareInfo.webGPUSupported
  
  // Adapter training requirements (more lenient)
  if (webGPUSupported && tflops >= 1.0 && availableMemory >= 2.0) {
    capabilities.canTrainAdapter = true
    capabilities.recommendedMode = 'adapter'
  }
  
  // Full training requirements (more strict)
  if (webGPUSupported && tflops >= 8.0 && availableMemory >= 4.0) {
    capabilities.canTrainFull = true
    
    // Only recommend full mode for very capable hardware
    if (tflops >= 15.0 && availableMemory >= 8.0) {
      capabilities.recommendedMode = 'full'
    }
  }
  
  // Adjust batch size based on memory
  if (availableMemory >= 8.0) {
    capabilities.maxBatchSize = 8
  } else if (availableMemory >= 6.0) {
    capabilities.maxBatchSize = 4
  } else if (availableMemory >= 4.0) {
    capabilities.maxBatchSize = 2
  } else {
    capabilities.maxBatchSize = 1
  }
  
  // Adjust sequence length based on TFLOPs and memory
  if (tflops >= 10.0 && availableMemory >= 6.0) {
    capabilities.maxSequenceLength = 2048
  } else if (tflops >= 5.0 && availableMemory >= 4.0) {
    capabilities.maxSequenceLength = 1024
  } else {
    capabilities.maxSequenceLength = 512
  }
  
  return capabilities
}

/**
 * Test WebGPU compute capabilities
 */
export async function testWebGPUCompute() {
  if (!navigator.gpu) {
    throw new Error('WebGPU not supported')
  }
  
  const adapter = await navigator.gpu.requestAdapter()
  if (!adapter) {
    throw new Error('No WebGPU adapter available')
  }
  
  const device = await adapter.requestDevice()
  
  try {
    // Simple compute shader test
    const computeShader = device.createShaderModule({
      code: `
        @group(0) @binding(0) var<storage, read_write> data: array<f32>;
        
        @compute @workgroup_size(64)
        fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
          let index = global_id.x;
          if (index >= arrayLength(&data)) {
            return;
          }
          
          data[index] = data[index] * 2.0;
        }
      `
    })
    
    const size = 1024
    const buffer = device.createBuffer({
      size: size * 4, // 4 bytes per f32
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
    })
    
    // Test data
    const testData = new Float32Array(size).fill(1.0)
    device.queue.writeBuffer(buffer, 0, testData)
    
    const bindGroupLayout = device.createBindGroupLayout({
      entries: [{
        binding: 0,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: 'storage' }
      }]
    })
    
    const bindGroup = device.createBindGroup({
      layout: bindGroupLayout,
      entries: [{
        binding: 0,
        resource: { buffer }
      }]
    })
    
    const computePipeline = device.createComputePipeline({
      layout: device.createPipelineLayout({
        bindGroupLayouts: [bindGroupLayout]
      }),
      compute: {
        module: computeShader,
        entryPoint: 'main'
      }
    })
    
    // Run compute
    const commandEncoder = device.createCommandEncoder()
    const passEncoder = commandEncoder.beginComputePass()
    passEncoder.setPipeline(computePipeline)
    passEncoder.setBindGroup(0, bindGroup)
    passEncoder.dispatchWorkgroups(Math.ceil(size / 64))
    passEncoder.end()
    
    device.queue.submit([commandEncoder.finish()])
    
    // Clean up
    buffer.destroy()
    device.destroy()
    
    return true
  } catch (error) {
    device.destroy()
    throw error
  }
}

/**
 * Get detailed WebGPU limits and features
 */
export async function getWebGPUDetails() {
  if (!navigator.gpu) {
    return null
  }
  
  const adapter = await navigator.gpu.requestAdapter()
  if (!adapter) {
    return null
  }
  
  const device = await adapter.requestDevice()
  
  const details = {
    limits: device.limits,
    features: Array.from(device.features),
    adapterInfo: adapter.info || {},
    supportedFormats: [],
    maxWorkgroupSize: device.limits.maxComputeWorkgroupSizeX || 256
  }
  
  // Test common texture formats
  const formats = ['rgba8unorm', 'rgba16float', 'rgba32float', 'r32float']
  for (const format of formats) {
    try {
      const texture = device.createTexture({
        size: [1, 1],
        format,
        usage: GPUTextureUsage.STORAGE_BINDING
      })
      details.supportedFormats.push(format)
      texture.destroy()
    } catch (e) {
      // Format not supported
    }
  }
  
  device.destroy()
  return details
}

// Default export for compatibility
export default {
  detectHardware,
  testWebGPUCompute,
  getWebGPUDetails
}