# LoRA Lab – Architecture & Refactor Notes

_Last updated: <DATE_PLACEHOLDER>_

---

## 1. High-Level Modules & Data-Flow

```
┌────────────┐    UI Events     ┌────────────────┐   messages   ┌────────────────────┐
│  Vue UI    │ ───────────────►│  main.js +     │─────────────►│ generation.worker.js│
│ components │                  │  ModelManager  │              │  (inference)        │
└────────────┘                  └────────────────┘◄─────────────└────────────────────┘
      ▲                                 │
      │                                 │
      │   events                        │ state Map / callbacks
      │                                 ▼
┌────────────┐    progress/ctl  ┌────────────────┐   messages   ┌────────────────────┐
│  Vue UI    │◄─────────────────│ TrainingEngine │─────────────►│  training.worker.js │
│ components │                  └────────────────┘◄─────────────└────────────────────┘
```

* **ModelManager**: singleton wrapper around a _single_ `generation.worker.js`. Manages Map of model states and streams generated tokens back to UI.
* **TrainingEngine**: orchestrator that spawns `training.worker.js`, forwards control / progress events to UI.
* **Workers**:
  * **generation.worker.js**: ONNX-Runtime inference only.
  * **training.worker.js**: Intended to run LoRA training but currently _mock-trains_ (simulated loss / gradients). Contains WebGPU boilerplate + placeholder WGSL shader.

Supporting libs:
* `trainers/onnxSession.js` – thin wrapper around ORT Web (inference only).
* `trainers/rankScheduler.js` – adaptive LoRA rank logic.
* `data/datasetLoader.js` – tokenise corpus and slice to sequences.

UI components of interest:
* `TrainConsole.vue` – renders loss chart & throughput.
* `PlanModal.vue` – displays hardware diagnostics.

## 2. Current Gaps vs PRD

| PRD Requirement | Status in Code |
|-----------------|----------------|
| True LoRA adapter training (update A/B matrices) | **Not implemented** – worker simulates loss, gradients. |
| WebGPU INT4 fused kernels | Skeleton WGSL shader exists but never executed with real buffers. |
| Hardware-aware plan estimator | `utils/hwDetect.js` present, PlanModal uses it (need verification). |
| Adapter export (.safetensors) | `utils/safetensorExport.js` placeholder – export path incomplete. |
| Switch chat between Base / LoRA | UI toggle exists but generation side cannot load LoRA deltas yet. |

## 3. Identified Bugs / Race-Conditions

1. **ModelManager.worker Singleton**
   * `initializeWorker()` guards by checking `if (worker)` but `clearAll()` terminates the worker **and immediately re-calls** `initializeWorker()` _without_ waiting for termination to finish – could lead to two workers.
2. **Training Config Leak**
   * `training.worker.js` sets `trainingConfig = config;` but `config` variable is undefined (should use the argument `trainingConfig`).
3. **WebGPU Pipeline Creation**
   * `createComputePipelines()` never called inside `training.worker` → WebGPU passes are no-op.
4. **ONNX Runtime Training**
   * ONNX Runtime Web does **not** support backward pass; current code pretends. Will never update weights ⇒ no real training.
5. **UI Progress**
   * Loss / throughput derived from simulated numbers – misleading to user.
6. **Shared Mutable State in ModelStates Map**
   * Concurrent `.generate()` calls on same `modelId` could race; Map access not synchronised – possible interleaved onMessage callbacks.

## 4. External Library Best-Practices (research summary)

* **ONNX Runtime Web** – only inference. For training one must either:
  * Compile model to WebGPU compute shaders manually, or
  * Use third-party libs (e.g., `tensorflow.js`, `onnxruntime-training` hazy). Currently browser training at scale relies on custom WGSL.
* **Transformers.js** provides _in-browser inference_ with pre-packed ONNX models, not training.
* **LoRA in browser** – emerging pattern: keep base weights frozen, update low-rank A/B with gradient descent via WebGPU (see projects _lllms_ & _webgpu-lora_).

## 5. Refactor Plan — *aligned with Transformers.js capabilities*

1. **Library boundaries**
   * Keep **Transformers.js** as the *single* high-level interface for model loading & inference.
   * Do **NOT** create second‐hand `ort.InferenceSession`s – Transformers.js already initialises ORT internally (per [Transformers.js docs](https://huggingface.co/docs/transformers.js/en/index)).
   * Custom code is needed **only** for the _training_ path (LoRA A/B update kernels and adapter persistence).

2. **Inference Path Cleanup**
   * Refactor `ModelManager`/`generation.worker.js` to build on the [pipeline API](https://huggingface.co/docs/transformers.js/en/custom_usage#pipeline-api):
     ```js
     import { pipeline } from '@huggingface/transformers';
     const chat = await pipeline('text-generation', modelId, { device: 'webgpu', dtype: 'q4' });
     ```
   * Streaming tokens: use `generate` with callback supported by Transformers.js v3 (`onToken`).
   * Remove homemade `onnxSession.js` for inference; keep *only* helpers shared with training.

3. **Training Path (new code)**
   * **WebGPU WGSL Kernels** for LoRA ΔW updates (no overlap with Transformers.js internals):
     1. `lora_forward.wgsl` – apply A × B and add scaled residual (used both in forward and backward).
     2. `lora_backward.wgsl` – compute ∂loss/∂A, ∂loss/∂B given hidden states & dY.
     3. `adam8bit_update.wgsl` – fused optimizer update on A/B params.
   * Training worker uses Transformers.js for **forward** logits / loss (frozen weights) and WGSL for gradients.
   * No ORT duplication: we pass tensors from Transformers.js forward pass into WGSL via `.data` views.

4. **Adapter Integration After Training**
   * Serialize INT4 A/B matrices to Safetensors with `utils/safetensorExport.js`.
   * At inference-time, load adapter tensors into GPU buffers once and add LoRA delta inside `generation.worker.js` before logits (keeping API identical for UI toggle).

5. **Bug Fix Sprint (unchanged)**
   * Fix singleton race, config leak, pipeline creation, etc.  (See Section 3.)

6. **Milestones (updated)**
   * M1 – Bug fixes + switch inference to Transformers.js pipeline.
   * M2 – Working WGSL kernel prototypes with unit tests.
   * M3 – End-to-end adapter training on small corpus; export + load path.
   * M4 – UI polish & docs.

## 6. KPI Feasibility: 1 M tokens ≤ 10 min on M1 Air

| Metric | Value | Source |
|--------|-------|--------|
| M1 Air theoretical GPU TFLOPs | **2.6 TF** | Apple spec / Medium benchmarks [1](https://medium.com/analytics-vidhya/machine-learning-on-m1-macbook-air-1674ac0ca777) |
| LoRA math / token (rank ≤ 4, INT4) | **0.6 GF** | PRD math |
| Max theoretical throughput | 2 600 GF / 0.6 GF ≈ **4 300 t/s** | calc |
| Real-world (35 % eff) | **≈ 1 500 t/s** | conservative |
| Wall-clock for 1 M tokens | 1 000 000 / 1 500 ≈ **11 min** | |

> We need an extra ~15 % speedup to hit 10 minutes → achievable via kernel fusion, IO-binding and throttling mitigation.

### Bottlenecks & Fixes
1. **WGSL bandwidth** – wire real buffers, 128×64 tiling, shared mem reuse.
2. **Data pipeline stalls** – IndexedDB + SharedArrayBuffer prefetch.
3. **Logging overhead** – batch JSON emits → 1 Hz binary summaries.

### Risks & Mitigations
* Safari shader-f16 bugs → fallback FP16 (≈ 13 min worst-case).
* Thermal throttle → brief `await queue.onSubmittedWorkDone()` every 200 steps.
* Transform-js WASM fallback → enforce `*-onnx-web` models w/ WebGPU.

--- 