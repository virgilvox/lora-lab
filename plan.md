# LoRA-Lab: End-to-End Fine-Tuning Road-Map

---
## 1. What We *Already* Have

- **Rich UI & UX** (Vue 3 components) – chat panel, corpus modal, training console, plan wizard, hardware inspector
- **Model Load & Inference** through `transformers.js` + **WebWorker** (`generation.worker.js`)
- **Adapter Upload/Download** utilities (`safetensorExport.js`)
- **Custom WGSL Kernels** for LoRA forward/backward & 8-bit Adam (file: `src/trainers/loraKernels.wgsl`)
- **Training Worker** scaffold (`training.worker.js`) with:
  - Rank scheduler
  - Dataset loader
  - Stubbed training loop calling dummy kernels
- **Hardware detection** and batch/rank recommendations

> **NOTE:** All of the above run in-browser and rely on WebGPU when available. Chat inference must *always* stay on `transformers.js` for ease of use.

---
## 2. Blockers / Gaps

| # | Area | Current Limitation | Suggested Fix |
|---|------|-------------------|---------------|
|B1|Loss & Gradients|Loss is simulated; no real backward pass|Add CrossEntropy WGSL kernel & back-prop only LoRA params|
|B2|Kernel Wiring|Kernels dispatched with dummy buffers|Wire real tensors, per-layer buffers, bind groups|
|B3|Per-Layer LoRA|Single global A/B buffer | Enumerate transformer projection layers, allocate adapter buffers for each |
|B4|Optimizer|Momentum/velocity not initialised or persisted | Initialise 8-bit buffers, update each step |
|B5|Export Accuracy|Safetensor contains random or single-layer weights | Export all layers with proper names |
|B6|Memory Pressure|`transformers.js` model kept resident during training, wasting VRAM|Option A: unload or move to CPU during training; Option B: keep weights but disable caches |
|B7|ONNX Training|If WGSL proves insufficient, need ONNX   "training" path (ORT-Web) | Prototype tiny ORT training session; still keep TFJS for inference |

---
## 3. Plan of Attack (Milestones)

1. **Kernel MVP (Single Layer)**
   - [x] Implement CE-loss WGSL kernel (commit ce_loss_wgsl)
   - [x] Replace simulated loss in `training.worker.js`
   - [x] Wire forward-A/B + backward-B/A + Adam for *one* projection layer (commit efd18a9)
     - Implemented helper functions in `training.worker.js` to encode and dispatch all necessary WGSL kernels for a full training pass.
     - Chained forward, backward, and optimizer passes in a single command encoder submission.
     - Replaced placeholder `outputGradient` and `inputActivations` with real values derived from the model's loss and embeddings.
   - [x] Verify adapter weights change after N steps (unit self-test) (commit a4e21b3)
     - Added a logging step after the training loop in the worker to read back LoRA matrix A weights from the GPU.
     - Implemented before/after comparison of weights to confirm they are not static.
     - Added `console.log` and `postMessage` for verification status.
2. **Multi-Layer Adapter Support**
   - [x] Parse loaded ONNX graph to list all `Linear`/`MatMul` targets (commit a4e21b3)
     - Implemented a `findLoraTargetLayers` function in `training.worker.js`.
     - This function traverses the loaded `transformers.js` model object graph to heuristically find common target layers (q_proj, k_proj, v_proj, etc.).
     - The list of discovered layer names is logged to the console upon starting training.
   - [x] Allocate A/B/momentum/velocity buffers per layer (commit f4c9a3d)
     - Refactored `training.worker.js` to manage a dictionary of buffers keyed by layer name.
     - The `ensureTrainingBuffers` function now dynamically allocates and cleans up buffers for all targeted layers.
   - [x] Adjust kernel dispatch loops (commit f4c9a3d)
      - The `performTrainingStep` function now iterates through each target layer.
      - In each iteration, it dispatches the forward, backward, and optimizer kernels for that specific layer's buffers.
3. **Exporter Reliability**
   - [x] Read all buffers back; write proper tensor names (e.g. `decoder.layers.0.attn.q_proj.lora_A.weight`) (commit 8b6a1c2)
     - The `handleTrainingCompletion` function in the training worker now iterates through all trained layers.
     - For each layer, it reads the A and B weight buffers back from the GPU.
     - It constructs an `adapterData` object where keys are the full layer names (e.g., `model.decoder.layers.0.self_attn.q_proj`), which `safetensorExport.js` then uses to create the final tensor names.
   - [x] Validate with `validateAdapterFile` (commit 2a3f5d1)
     - The `downloadAdapter` function in `safetensorExport.js` now validates the serialized adapter data before triggering a download.
     - If validation fails, it shows an alert to the user.
4. **Memory Strategy**
   - [x] Unload inference model during training to save VRAM (commit 3f5a1b3)
     - Added `unload` message to `generation.worker.js` to remove models from its state.
     - `modelManager.js` now sends this message to the worker upon `unloadModel`.
     - `LoRALabApp.vue` now unloads the inference model before starting training.
     - Upon training completion, it reloads the base model and then applies the newly trained adapter weights.
5. **ONNX-Fallback Prototype (optional)**
   - [ ] Build minimal ORT-Web training graph for LoRA parameters only
   - [ ] Benchmark vs WGSL kernels; pick faster path
6. **UI Integration & QA**
   - [ ] Train small corpus (<50k tok) end-to-end; confirm chat outputs differ
   - [ ] Add "Evaluate" button to compare perplexity pre/post.
   - [x] Refine adapter download flow (commit 5a8a7c1)
     - The trained adapter data is now stored in the main Vue app's state upon training completion.
     - The "Download Adapter" button in the footer now uses this stored data, making the download user-initiated.
   - [x] Add model compatibility check and error handling (commit 2e1b4a9)
     - The training worker now checks if any suitable LoRA target layers are found in the selected model. If not, it throws an error.
     - `LoRALabApp.vue` now listens for errors from the workers and other parts of the app and displays them as notifications in the footer.

---
## 4. Implementation Tracking Instructions

Use this **plan.md** as a living document. For **each task/sub-task**:
1. Prefix with one of:
   - `[ ]` not started
   - `[~]` in-progress
   - `[x]` completed
2. Append commit / PR reference in parentheses.
3. Keep a short bullet under task with key findings or blocking issues.

Example:
```
- [~] Implement CE-loss WGSL kernel (commit 9f2c1e4)
  • Verified forward output matches NumPy within 1e-5
```

---
## 5. Constraints & Guidelines

1. **Inference stays on transformers.js** – chat UI must remain snappy.
2. **Training Worker Isolation**
   - May unload or garbage-collect `transformers.js` objects during heavy training.
   - Always restore for generation after training finishes.
3. **WebGPU First, ORT-Web Fallback**
   - Prioritise custom WGSL for speed & browser portability.
   - If blocked (e.g. non-mac GPUs), enable ORT-Web training path.
4. **No Blocking the UI Thread** – all heavy ops in workers.
5. **Browsers Without WebGPU** – gracefully drop to CPU / **adapter-only** recommendation.
6. **Adapter Format** – output must be HuggingFace-compatible `.safetensors`.

---
## 6. Open Questions

- What is the minimal layer subset to LoRA-adapt for worthwhile gains? (q,k,v,o vs all)
- Required learning-rate schedule? (static vs cosine) – decide post-MVP.
- Need mixed-precision support (f16) in WGSL? Depends on GPU capabilities.

---
*Last updated:* <!-- keep this line, CI will append datetime --> 