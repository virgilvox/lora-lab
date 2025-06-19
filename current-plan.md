# LoRA Lab - Detailed Implementation Plan

**Objective:** Transform the LoRA Lab from a training *simulator* into a functional, in-browser LoRA fine-tuning tool using WebGPU.

**Confirmed Analysis:** The current codebase successfully runs a WebGPU-based simulation but does not perform real training. Gradients are synthetic, and the LoRA weights are not integrated into the model's inference graph. This plan addresses the architectural gaps to build a true learning pipeline.

---

## Phase 1: Core Training Mechanism - Proof of Concept

The goal of this phase is to achieve a demonstrable end-to-end training loop on a *single, hard-coded target layer* with real gradients. This validates the core approach before generalizing.

### Step 1.1: Refactor `training.worker.js` for a Differentiable Forward Pass

**Problem:** The worker currently uses a high-level `model()` call which is a black box and doesn't expose intermediate activations needed for the LoRA backward pass.

**Solution:**
1.  **Isolate the Target Layer:** Instead of `findLoraTargetLayers`, we will hard-code a single target layer for the initial PoC (e.g., `model.model.layers.0.self_attn.q_proj`).
2.  **Manual Forward Pass:** Restructure the forward pass to execute layer by layer. This is critical for intercepting the input to our target layer.
3.  **Capture Activations:** Before executing the target linear layer (`q_proj`), capture its input tensor. This tensor is the `x` in the LoRA equation `h' = h + B(Ax)`.
4.  **Get Real Gradients:** The final output from the model's forward pass (logits) must be used to calculate the loss against the labels. We need to find a way to get the gradient of the loss with respect to the output of our target layer (`dL/dh'`). This is the most challenging step.
    *   **Initial Approach:** As a simplification for the PoC, we can calculate a pseudo-gradient. After getting the `logits` from the forward pass, we can compute the cross-entropy loss. The gradient of the loss with respect to the logits can be calculated (it's `softmax(logits) - one_hot_labels`). We can then back-propagate this error signal one step to the output of our target layer. This is still a simplification but is far better than the current synthetic gradient.

### Step 1.2: Integrate LoRA Kernels into the Forward Pass

**Problem:** The LoRA kernels currently run in isolation on embeddings, not on the actual layer activations.

**Solution:**
1.  **Modify Model Execution:** After capturing the input `x` to the target layer, run the original layer's forward pass to get its output `h`.
2.  **Execute LoRA Forward Kernels:** Use the captured input `x` and the LoRA A/B weight buffers to execute the `lora_forward_A_main` and `lora_forward_B_main` kernels. This will produce the LoRA delta.
3.  **Combine Outputs:** Add the LoRA delta to the original layer's output `h` to get the modified output `h'`.
4.  **Resume Forward Pass:** Continue the rest of the model's forward pass using this modified `h'`.

### Step 1.3: Connect Backward Pass to Real Gradients

**Problem:** The backward pass uses synthetic gradients.

**Solution:**
1.  **Feed Real Gradients:** Use the calculated `dL/dh'` (from Step 1.1) as the `outputGradient` input to the `lora_backward_B_main` kernel.
2.  **Feed Real Activations:** Use the captured layer input `x` (from Step 1.1) as the `input_a` input to the `lora_backward_A_main` kernel.
3.  This will compute the gradients for LoRA matrices A (`dL/dA`) and B (`dL/dB`) based on the actual model loss.

### Step 1.4: Fix Hard-Coded Dimensions

**Problem:** The WGSL kernels and JS buffer allocations assume a hidden dimension of 768.

**Solution:**
1.  **Dynamic Configuration:** In `training.worker.js`, fetch the `hidden_size` from the model's configuration (`model.config.hidden_size`).
2.  **Pass Dimensions to Kernels:** Pass `hidden_size`, `rank`, etc., into the `LoRAParams` uniform buffer for the WGSL kernels to use.
3.  **Dynamic Buffer Allocation:** Ensure all GPU buffers (`weightsA`, `weightsB`, `gradientsA`, `gradientsB`, etc.) are allocated using the dynamic `hidden_size` and `rank`, not fixed values.

---

## Phase 2: Generalization and Integration

With a working PoC, we can now generalize the solution to support multiple layers and integrate it cleanly.

### Step 2.1: Abstract Layer-Hooking Mechanism

**Problem:** The manual forward pass from Phase 1 is brittle and model-specific.

**Solution:**
1.  **Monkey-Patching/Proxy:** Develop a robust method to "hook" into the forward pass of any `nn.Linear` layer identified by `findLoraTargetLayers`. This could involve replacing the layer's `forward` method with a custom function that performs the "capture -> original pass -> LoRA pass -> combine" logic.
2.  **Layer-Specific Buffers:** Generalize the `loraLayerBuffers` to correctly manage GPU buffers for each targeted layer, ensuring names match the model's structure.

### Step 2.2: Implement True Backpropagation (Advanced)

**Problem:** The pseudo-gradient from Phase 1 is an approximation.

**Solution:**
*   This is a research-heavy task. A full autograd engine in JS/WebGPU is a massive project.
*   **Feasible Next Step:** Instead of a full backward pass, we can calculate the gradient for the *last* layer of the model more accurately and use that to update the LoRA weights of that layer. This would still be a significant improvement.
*   **Long-Term Vision:** Explore libraries that might emerge for WebGPU autograd or contribute to `transformers.js` to expose more of the internal computational graph. For now, we accept the limitation of a simplified gradient.

### Step 2.3: Connect LoRA to Inference Worker

**Problem:** The `generation.worker.js` does not use the trained adapter.

**Solution:**
1.  **Send Trained Weights:** After training completes, the main thread (`LoRALabApp.vue`) will hold the trained `adapterData`.
2.  **Modify `modelManager`:** The `modelManager` will send this `adapterData` to the `generation.worker.js` via a new message type, e.g., `APPLY_ADAPTER`.
3.  **Modify `generation.worker.js`:** The generation worker must implement the same layer-hooking mechanism as the training worker. When `useLoRA` is true and an adapter is present, it will modify the forward pass of the target layers to include the LoRA computation. This ensures the fine-tuned adjustments are actually used during chat.

### Step 2.4: Code Cleanup and Validation

1.  **Remove Simulation Code:** Eliminate all uses of `simulateLossCalculation` and `simulateGradientNorm`.
2.  **Fix Bugs:** Address the out-of-scope `targetLayers` variable in `handleStopTraining`.
3.  **Safetensors Export:** Ensure the `safetensorExport.js` correctly handles data types and tensor shapes for the exported LoRA weights.
4.  **Add Validation Step:** Implement a simple validation check post-training. Run a sample prompt through the model *with* the new adapter and compare its output to a run *without* the adapter to confirm the model's behavior has changed.

---

## Files to be Modified

*   `src/workers/training.worker.js`: **(Major Overhaul)** Core logic for forward/backward pass, gradient calculation, and dynamic dimension handling.
*   `src/trainers/loraKernels.wgsl`: Update kernels to accept dynamic dimensions from uniform buffers.
*   `src/workers/generation.worker.js`: Implement layer-hooking to apply the trained LoRA adapter during inference.
*   `src/utils/modelManager.js`: Add logic to pass trained adapters to the generation worker.
*   `src/ui/LoRALabApp.vue`: Manage the state of the trained adapter and trigger its application.
*   `src/utils/safetensorExport.js`: Verify that data types and tensor shapes are handled correctly upon export.

This plan moves from a foundational proof-of-concept to a more robust and generalized solution, acknowledging the significant technical challenges while providing a clear, incremental path forward. 