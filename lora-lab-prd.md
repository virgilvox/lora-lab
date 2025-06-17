# Project: LoRA Lab

## Overview
LoRA Lab is an **entirely in‑browser** fine‑tuning studio.  Users can either:
1. **Adapter‑only mode** – train ultra‑light LoRA / LowRA adapters on any WebLLM‑compatible ONNX model.
2. **Full‑tune mode** – (if hardware permits) update *all* model weights for small models.

The app autodetects local GPU/CPU capability, estimates throughput, and guides the user toward the fastest viable path.

---

## Goals & Success Metrics

| Goal | KPI |
|------|-----|
| 1 M‑token adapter training ≤ 10 min **on M1 Air** | Adapter‑only, rank ≤ 4, ≤ 2‑bit, ≤ 9 min wall‑time |
| Hardware‑aware guidance | 95 % of users choose a feasible plan on first try |
| Minimal external API calls | All training + inference local; only CDN fetch for models |
| Simple, JS‑only codebase | No TypeScript; each module ≤ 200 LoC and loosely coupled |

---

## User Stories
1. *Developer* pastes a 50 k‑token corpus, selects Gemma‑2B, gets a trained adapter in 6 min.
2. *Researcher* on a desktop 4060 Ti chooses **full‑tune** for TinyLlama‑1.1 B; UI predicts ~28 min; training succeeds.
3. *Teacher* on a low‑end Chromebook gets a warning (“adapter‑only, 5 k tokens ≈ 12 min”) and proceeds.

---

## Functional Requirements

### Core
- **Corpus input**: upload `.txt` *or* paste into textarea (auto‑tokenized)
- **Model selector**: dropdown of verified WebLLM ONNX models *or* paste URL
- **Hardware diagnostic**: detects WebGPU, unified RAM, and theoretical TFLOPs
- **Plan estimator**: shows *Adapter* vs *Full* modes with ETA & memory footprint
- **Training dashboard**: tokens processed, loss curve, throughput, ETA
- **Chat panel**: switch between *Base* and *LoRA* on‑the‑fly
- **Adapter export**: `.safetensors` download and drag‑back‑in apply

### Training Capabilities
| Mode | Default Scope | Hardware Gate |
|------|---------------|---------------|
| **Adapter** | LowRA rank 2–4, 1–2 bit, frozen base | WebGPU available, ≥ 2 GB free GPU RAM |
| **Full‑tune** | All weights FP16 or INT4 | ≥ 8 TFLOPs device *and* user confirmation |

*Both* modes use:
- ONNX Runtime Web (WebGPU EP) with IO‑Binding + GraphCapture
- INT4/NF4 base weights (QLoRA), fused INT4 matmul + 8‑bit Adam/Adafactor
- Dual‑sequence packing + TF‑IDF curriculum (no early stop)

### Hardware Detection Algorithm
```js
const info = await navigator.gpu.requestAdapter();
const tfops = estimateTFLOPs(info);
const mem   = navigator.deviceMemory || 4; // GB
const canWebGPU = !!info;
```
The estimator computes: `ETA = tokens / (tfops / flopsPerToken)` where `flopsPerToken` = 2.5 GF (adapter) or 42 GF (full).

---

## UI Requirements

### Layout
```
Header: [Model▼] [Load URL] [Choose Corpus] [Paste Text] [Plan ▼] [Train]
───────────────────────────────────────────────────────────────
Chat (lhs) | Training Console (rhs)
───────────────────────────────────────────────────────────────
Footer: GPU status • Mem usage • ETA • Download Adapter
```

### Components (all *.vue with plain JS script blocks)
- **HeaderBar.vue** – model dropdown, URL input, corpus input (file or textarea modal), plan selector (Adapter / Full), Train button.
- **ChatPanel.vue** – conversation, prompt box, toggle “Use LoRA”.
- **TrainConsole.vue** – token counter, `<LossChart/>`, throughput, ETA, abort.
- **PlanModal.vue** – shows hardware check and predicted runtimes.
- **FooterStatus.vue** – GPU name, memory bar, current mode.

> **No TypeScript**: `<script>` tags use standard ES modules.  Helper libs kept in `/src/lib/` with vanilla JS.

---

## File Structure (JS only)
```
browser-lora-lab/
├─ public/
├─ src/
│  ├─ ui/
│  │  ├─ HeaderBar.vue
│  │  ├─ ChatPanel.vue
│  │  ├─ TrainConsole.vue
│  │  ├─ PlanModal.vue
│  │  ├─ LossChart.vue
│  │  └─ FooterStatus.vue
│  ├─ trainers/
│  │  ├─ onnxSession.js        # create training session
│  │  ├─ loraKernels.wgsl      # INT4 matmul + fused opt
│  │  └─ rankScheduler.js
│  ├─ workers/
│  │  └─ training.worker.js
│  ├─ data/
│  │  └─ datasetLoader.js
│  ├─ utils/
│  │  ├─ safetensorExport.js
│  │  └─ hwDetect.js           # GPU TFLOPs & RAM estimator
│  └─ main.js
├─ vite.config.js
└─ README.md
```

---

## Updated “10‑Minute” Strategy

### Adapter‑Only math (validated)
- LowRA rank 2, ~1 GFLOP/token ≈ (forward + low‑rank back‑prop)
- M1 Air ≈ 2.6 TFLOPs  ➜ **2 600 GF / 1 GF ≈ 2 600 tokens/s theoretical**
- Real‑world efficiency ≈ 40 % ➜ **≈ 1 000 tokens/s practical**
- **1 M tokens / 1 000 t/s ≈ 1 000 s ≈ 17 min**
- Aggressive kernel fusion + INT4 shaders (~2×) ➜ **~8.5 min** (stretch‑goal)

### Full‑Tune path
- Warn if `ETA > 30 min` or `mem use > gpuMem`.
- Allow anyway for power users.

---

## WebGPU Shader & Kernel Strategy (NEW)

| Layer | Technique | Purpose |
|-------|-----------|---------|
| **INT4 MatMul** | Custom WGSL kernels with `dot4_i8_packed` and `shader-f16` | 2× throughput vs. FP16 matmul |
| **Cooperative Matrices (Metal 3.3)** | Sub‑group tiling (8×8) | 1.2× extra speed on M‑series |
| **Fused Optimizer** | Integrate **8‑bit Adam** update in same compute pass | Removes extra GPU read/write |
| **LowRA Adapter Buffers** | 1‑bit / 2‑bit packed bitfields | Cuts memory traffic 4×–8× |
| **IO‑Binding** | ONNX WebGPU zero‑copy tensors | Eliminates CPU ↔ GPU copies |
| **GraphCapture** | Static kernel graph reused each step | Avoids per‑iteration pipeline build |
| **Dual‑Sequence Packing** | 2 docs per sequence window | Keeps GPU 95 %+ occupied |

> **Implementation path:**
> 1. Prototype INT4 WGSL matmul in `src/trainers/loraKernels.wgsl` (128×64 tile).
> 2. Wrap in ONNX custom op; bind LowRA A/B matrices as 1‑bit textures.
> 3. Emit optimizer step in same shader: `w += lr * grad8bit; grad = 0`.
> 4. Register custom op in `onnxSession.js` before `createTrainingSession()`.
> 5. Benchmark `tokens/sec` in `training.worker.js` and fall back to stock FP16 if < 300 t/s.

**Hard math sanity‑check:**
```
INT4 MatMul   : ~0.5 GFLOP/token
Optimizer Fuse: ~0.1 GFLOP/token
Total Adapter : ~0.6 GFLOP/token
Effective t/s : 2.6 TF / 0.6 GF ≈ 4 300 t/s (theoretical)
Real 35 % eff : ~1 500 t/s ➜ 1 M tokens ≈ 11 min
```
Achievable with best‑case caching + thermal headroom; still within stretch window for adapter‑only mode.

---

## Open Questions (revised)
- Which models should be pre‑hosted (Gemma‑2B, Phi‑2, TinyLlama‑1.1 B)?
- Provide preset adapter ranks (2/4/8) or slider?
- How to benchmark TFLOPs reliably across browsers?

---

## Future Work
- Local Ollama backend option for offline dev
- Audio corpus (Whisper.js) → token stream
- Adapter merge & comparison viewer
- Local Ollama backend option for offline dev
- Audio corpus (Whisper.js) → token stream
- Adapter merge & comparison viewer
