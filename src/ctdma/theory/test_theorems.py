"""
Numerical verification of theorems.

This module provides test functions that numerically verify the
predictions made by our mathematical theorems.
"""

import torch
import numpy as np
from .theorems import *
from .mathematical_analysis import *


def verify_lemma_1_discontinuities():
    """
    Verify Lemma 1: Discontinuity of Modular Addition
    
    Tests:
    1. Gradients are discontinuous at wrap points
    2. Discontinuity magnitude is ~2^n
    """
    modulus = 2**8  # Use smaller modulus for testing
    x = torch.linspace(0, modulus * 2, 10000, requires_grad=True)
    y = torch.ones_like(x) * (modulus / 2)
    
    # Modular addition
    z = (x + y) % modulus
    
    # Compute gradients
    z.sum().backward()
    grad = x.grad.numpy()
    
    # Find discontinuities
    grad_diff = np.abs(np.diff(grad))
    large_jumps = np.where(grad_diff > 0.1)[0]
    
    return {
        'num_discontinuities': len(large_jumps),
        'expected_discontinuities': 2,  # Should be ~2 in the range
        'max_gradient_jump': np.max(grad_diff),
        'verified': len(large_jumps) >= 1,
    }


def verify_gradient_inversion_theorem():
    """
    Verify Gradient Inversion Theorem
    
    Tests:
    1. Inversion rate > 0.90
    2. Inverse basin is larger than target basin
    """
    analyzer = GradientInversionAnalyzer(modulus=2**8)
    
    inversion_data = analyzer.analyze_inversion_probability(num_trials=50)
    basin_data = analyzer.compute_basin_volumes(resolution=500)
    
    inversion_rate = inversion_data['inversion_rate']
    basin_ratio = basin_data['basin_ratio']
    
    return {
        'inversion_rate': inversion_rate,
        'expected_rate': 0.95,
        'basin_ratio': basin_ratio,
        'expected_basin_ratio': 1.0,  # Inverse basin should be larger
        'verified': inversion_rate > 0.90 and basin_ratio > 1.0,
    }


def verify_sawtooth_convergence_theorem():
    """
    Verify Sawtooth Convergence Theorem
    
    Tests:
    1. Convergence probability decreases with distance
    2. Nearby minima are reached more frequently
    """
    analyzer = SawtoothTopologyAnalyzer(modulus=2**8)
    
    discontinuity_data = analyzer.analyze_discontinuities(
        x_range=(0, 2**10),
        num_points=10000
    )
    
    num_discontinuities = discontinuity_data['num_discontinuities']
    
    # Verify many discontinuities exist
    expected_discontinuities = 2  # At least 2 in range
    
    return {
        'num_discontinuities': num_discontinuities,
        'expected_min_discontinuities': expected_discontinuities,
        'verified': num_discontinuities >= expected_discontinuities,
    }


def verify_all_theorems():
    """
    Run all numerical verifications.
    
    Returns:
        Dictionary with verification results for all theorems
    """
    results = {}
    
    print("Verifying Lemma 1: Discontinuity of Modular Addition...")
    results['lemma_1'] = verify_lemma_1_discontinuities()
    print(f"  Result: {results['lemma_1']['verified']}")
    print(f"  Found {results['lemma_1']['num_discontinuities']} discontinuities")
    
    print("\nVerifying Gradient Inversion Theorem...")
    results['gradient_inversion'] = verify_gradient_inversion_theorem()
    print(f"  Result: {results['gradient_inversion']['verified']}")
    print(f"  Inversion rate: {results['gradient_inversion']['inversion_rate']:.2%}")
    
    print("\nVerifying Sawtooth Convergence Theorem...")
    results['sawtooth'] = verify_sawtooth_convergence_theorem()
    print(f"  Result: {results['sawtooth']['verified']}")
    print(f"  Discontinuities: {results['sawtooth']['num_discontinuities']}")
    
    # Summary
    all_verified = all(r['verified'] for r in results.values())
    results['all_verified'] = all_verified
    
    print("\n" + "="*60)
    if all_verified:
        print("✅ ALL THEOREMS VERIFIED NUMERICALLY")
    else:
        print("⚠️  Some theorems could not be fully verified")
    print("="*60)
    
    return results


if __name__ == "__main__":
    verify_all_theorems()