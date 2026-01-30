"""
Debug Scripts for GradientDetachment

This directory contains scripts for verifying the correctness
of the implementation and debugging issues.

Available Scripts:
------------------

1. critical_sanity_check.py
   - Tests on single sample
   - Expected: 100% accuracy
   - Purpose: Verify no implementation bugs
   - Run: python debug_scripts/critical_sanity_check.py

2. reproduce_sawtooth.py (in root directory)
   - Runs both single-batch and multi-sample tests
   - Demonstrates Gradient Detachment phenomenon
   - Run: python reproduce_sawtooth.py

Debugging Methodology:
----------------------

If experiments fail, follow this order:

1. Run critical_sanity_check.py
   - If PASS: Implementation is correct
   - If FAIL: Check for bugs in cipher or ODE solver

2. Run reproduce_sawtooth.py
   - Confirms Gradient Detachment on multi-sample tests
   - Should see ~2.5% accuracy on ARX (Speck)

3. Check gradient flow
   - Inspect gradient norms
   - Look for vanishing/exploding gradients

4. Visualize loss landscape
   - Plot loss vs parameter changes
   - Should see "sawtooth" pattern for ARX

Expected Results:
-----------------

Single-batch (1 sample): 100% accuracy
Multi-sample (100+):     ~2.5% accuracy (ARX, 1 round)
                         ~15% accuracy (Feistel, 1 round)
                         0% accuracy (all, 4+ rounds)

If you see different results:
- Check random seed
- Verify hyperparameters
- Check hardware (GPU vs CPU)
- Compare with paper's expected ranges

Contact:
--------
For issues, open a GitHub issue with:
- Script output
- Hardware specs
- PyTorch version
- Expected vs actual results
"""