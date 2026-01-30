# Peer Review Round 2: Re-Evaluation of Gradient Detachment

## Overview
**Recommendation:** Reject (Strong)
**Reviewer Confidence:** 5 (Expert)

The author has attempted to address the previous review's concerns, but the codebase remains fundamentally broken. The experiments cannot run, and the "fixes" have introduced new dimensional mismatch errors while failing to correct the underlying mathematical flaws in the cipher approximation.

## Persisting Critical Failures

### 1. Code Crashing (Dimensional Mismatch)
The verification script `reproduce_sawtooth.py` continues to crash, now with a runtime matrix multiplication error:
- **Error:** `mat1 and mat2 shapes cannot be multiplied (1x2 and 32x64)`
- **Location:** `DifferentialAttack.classifier` (Linear layer 1).
- **Cause:** The `SpeckCipher` outputs a tensor of shape `(batch, 2)` (representing left/right words). The `DifferentialAttack` model expects an `input_size` of 32 (bits). However, the code passes the raw `(batch, 2)` float tensor directly to the linear layer without expanding it into a 32-element vector or adjusting the input layer to accept size 2.
    - *Note:* In the previous version, `features = diff.view(1, -1)` flattened the `(1, 2)` tensor to `(1, 2)`, but the linear layer expected 32 inputs. This mismatch was present before but masked by the earlier XOR crash.

### 2. Flawed "Soft XOR" Implementation
The author claimed to fix the `_soft_xor` function in `speck.py`, but the logic remains unsound:
- **The "Fix":** The new implementation uses:
  $$ \text{XOR} \approx x(1-y) + (1-x)y - x(1-x)y(1-y)2 $$
  While this looks more like a probabilistic XOR, it relies on `torch.sigmoid(steepness * (x - 0.5))` to modify inputs.
- **The Issue:** For continuous inputs (which are required for Neural ODEs), this function behaves erratically. More critically, the previous conceptual error remains: the cipher treats these continuous approximations as "encryption" but then attempts to define security based on them. If the approximation is "soft" enough to be differentiable, it is likely too soft to represent the discrete non-linearity of the actual cipher.

### 3. Missing Dependencies Not Resolved
- `src/ctdma/utils.py` is still missing from the directory (`list_dir` confirms it is not there).
- The `test_cross_cipher_comparison.py` script still imports `prepare_cipher_dataset` from this missing file, guaranteeing that the cross-cipher experiments (the core of the paper's comparison claims) **cannot be run**.

### 4. Experimental Flaws
- **Random Labels:** The user "fixed" the random label generation in `reproduce_sawtooth.py` by moving it outside the loop:
  ```python
  labels = torch.randint(0, 2, (num_samples,), device=device)
  ```
  However, since the plaintexts and keys are *also* generated once outside the loop (`plaintexts = cipher.generate_plaintexts(num_samples)`), the model is simply trying to overfit a fixed dataset of random noise to random labels.
- **Meaningless Metric:** Achieving "2.5% accuracy" (or any low number) on a fixed dataset with a powerful model (Neural Network) indicates the model is **failing to memorize** even a small, static batch of data. This does not prove "Gradient Detachment" of a cipher; it proves the `DifferentialAttack` model or the optimizer is broken. A functioning neural network should easily memorize 100 fixed samples to 100% accuracy, regardless of the underlying function (Universal Approximation Theorem). The failure to do so points to implementation bugs (like the shape mismatch or gradient flow issues), not cryptographic properties.

## Conclusion
The submission is still in a pre-alpha state. The code does not execute, key files are missing, and the experimental design tests the model's ability to memorize random noise rather than cryptanalyze a cipher. The "Gradient Detachment" claim cannot be evaluated because the experimental apparatus does not function.

**Required Actions for Author:**
1.  **Run your own code.** Ensure `reproduce_sawtooth.py` runs to completion without error.
2.  **Fix Shapes:** Match the output dimension of `SpeckCipher` (2 floats) to the input dimension of `DifferentialAttack` (expects 32 inputs? Or 2?).
3.  **Restore Missing Files:** Add `src/ctdma/utils.py`.
4.  **Sanity Check:** Before claiming cryptographic resistance, verify your model can learn a *trivial* function (e.g., Identity or simple XOR) with the same architecture to prove the learning harness works.
