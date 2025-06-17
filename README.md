# LoRA Lab

An entirely in-browser fine-tuning studio for training LoRA (Low-Rank Adaptation) adapters on WebLLM-compatible ONNX models.

## Overview

LoRA Lab is a web-based application that enables users to:
- Train ultra-light LoRA/LowRA adapters on any WebLLM-compatible ONNX model (Adapter-only mode)
- Perform full fine-tuning on small models (Full-tune mode, hardware permitting)
- Auto-detect local GPU/CPU capabilities and estimate throughput
- Fine-tune models directly in the browser without external API calls

## Features

- **Hardware-aware guidance**: Automatically detects WebGPU support and recommends optimal training strategies
- **Fast training**: Target of 1M-token adapter training ≤ 10 minutes on M1 Air
- **Local processing**: All training and inference runs locally in your browser
- **Simple codebase**: Built with vanilla JavaScript and Vue Single File Components (no TypeScript)

## Project Structure

```
lora-lab/
├── src/
│   ├── ui/              # Vue components
│   ├── trainers/        # ONNX session and training logic
│   ├── workers/         # Background training workers
│   ├── data/            # Dataset loading utilities
│   ├── utils/           # Helper functions (hw detection, exports)
│   └── main.js          # Application entry point
├── public/              # Static assets
├── vite.config.js       # Vite configuration
└── README.md
```

## Getting Started

### Prerequisites

- Node.js 16+
- npm or yarn
- A browser with WebGPU support (Chrome/Edge 113+)

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/lora-lab.git
cd lora-lab

# Install dependencies
npm install

# Start the development server
npm run dev
```

The application will be available at `http://localhost:5173`

### Building for Production

```bash
npm run build
```

The production-ready files will be generated in the `dist/` directory.

## Usage

1. **Select a Model**: Choose from pre-hosted models or provide a custom ONNX model URL
2. **Input Corpus**: Upload a text file or paste your training data directly
3. **Choose Training Plan**: Select between Adapter-only or Full-tune mode based on hardware capabilities
4. **Monitor Training**: View real-time loss curves, throughput, and ETA
5. **Test & Export**: Chat with your fine-tuned model and download the adapter as a `.safetensors` file

## Technical Details

### Training Modes

| Mode | Scope | Hardware Requirements |
|------|-------|--------------------|
| **Adapter** | LowRA rank 2-4, 1-2 bit, frozen base | WebGPU, ≥2GB GPU RAM |
| **Full-tune** | All weights FP16 or INT4 | ≥8 TFLOPs device |

### Performance Optimizations

- Custom WebGPU INT4 matmul kernels
- Fused 8-bit Adam optimizer
- IO-Binding and GraphCapture for minimal overhead
- Dual-sequence packing for optimal GPU utilization

## Development

### Scripts

- `npm run dev` - Start development server with hot module replacement
- `npm run build` - Build for production
- `npm run preview` - Preview production build locally

### Contributing

Contributions are welcome! Please ensure:
- All code is written in vanilla JavaScript (no TypeScript)
- Vue components use `<script>` tags only
- Each module stays under 200 lines of code
- Follow the existing project structure

## License

MIT License - see LICENSE file for details 