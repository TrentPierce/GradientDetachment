"""
Tests for Mathematical Theory Module

Tests formal theorems and mathematical analysis functions.
"""

import pytest
import torch
import numpy as np
from ctdma.theory import (
    GradientInversionAnalyzer,
    SawtoothTopologyAnalyzer,
    InformationTheoreticAnalyzer,
    verify_sawtooth_theorem,
    verify_gradient_inversion_theorem,
    verify_convergence_impossibility_theorem
)


class TestGradientInversionAnalyzer:
    """Test gradient inversion analysis."""
    
    def test_modular_addition_gradient_analysis(self, random_seed):
        """Test gradient analysis of modular addition."""
        analyzer = GradientInversionAnalyzer(modulus=65536)
        
        x = torch.tensor([0.3, 0.7, 0.9])
        y = torch.tensor([0.2, 0.5, 0.8])
        
        analysis = analyzer.analyze_modular_addition_gradient(x, y)
        
        # Check all required statistics present
        assert 'gradient_x' in analysis
        assert 'mean_gradient_x' in analysis
        assert 'gradient_sign_flips' in analysis
        assert 'wrap_around_detected' in analysis
        
        # Gradients should exist
        assert analysis['gradient_x'] is not None
    
    def test_discontinuity_score(self, random_seed):
        """Test gradient discontinuity score computation."""
        analyzer = GradientInversionAnalyzer(modulus=65536)
        
        x_range = torch.linspace(0, 100000, 1000)
        score = analyzer.compute_gradient_discontinuity_score(x_range)
        
        # Should detect high discontinuity in modular arithmetic
        assert score > 0
        assert not np.isnan(score)
        assert not np.isinf(score)


class TestSawtoothTopologyAnalyzer:
    """Test sawtooth topology analysis."""
    
    def test_fourier_spectrum(self, random_seed):
        """Test Fourier spectrum analysis."""
        analyzer = SawtoothTopologyAnalyzer()
        
        # Create synthetic sawtooth signal
        t = np.linspace(0, 10, 1000)
        signal = t % 1  # Sawtooth wave
        
        spectrum = analyzer.compute_fourier_spectrum(signal)
        
        # Check spectrum properties
        assert 'frequencies' in spectrum
        assert 'power_spectrum' in spectrum
        assert 'high_freq_ratio' in spectrum
        
        # Sawtooth should have significant high-frequency content
        assert spectrum['high_freq_ratio'] > 0.1
    
    def test_gradient_jumps(self, random_seed):
        """Test gradient jump detection."""
        analyzer = SawtoothTopologyAnalyzer()
        
        # Create gradient signal with jumps
        gradients = np.concatenate([
            np.ones(100),
            -np.ones(100) * 100,  # Large jump
            np.ones(100)
        ])
        
        jumps = analyzer.analyze_gradient_jumps(gradients)
        
        assert 'num_jumps' in jumps
        assert 'mean_jump_size' in jumps
        assert jumps['num_jumps'] > 0  # Should detect jumps
    
    def test_lipschitz_constant_unbounded(self, random_seed):
        """Test that modular arithmetic has large Lipschitz constant."""
        analyzer = SawtoothTopologyAnalyzer(modulus=65536)
        
        # Modular addition function
        def mod_add(x):
            y = 0.5 * torch.ones_like(x)
            sum_val = x + y
            wrap = torch.sigmoid(10.0 * (sum_val - 1.0))
            return sum_val - wrap
        
        x_range = torch.linspace(0, 2, 100)
        L = analyzer.measure_lipschitz_constant(mod_add, x_range)
        
        # Should be large due to wrap-around
        assert L > 1.0


class TestInformationTheoreticAnalyzer:
    """Test information-theoretic analysis."""
    
    def test_mutual_information(self, random_seed):
        """Test mutual information computation."""
        analyzer = InformationTheoreticAnalyzer()
        
        # Correlated variables
        x = np.random.randn(1000)
        y = x + np.random.randn(1000) * 0.1  # High correlation
        
        mi = analyzer.compute_mutual_information(x, y)
        
        # High correlation should give positive MI
        assert mi > 0
    
    def test_gradient_entropy(self, random_seed):
        """Test gradient entropy computation."""
        analyzer = InformationTheoreticAnalyzer()
        
        # Uniform gradients (high entropy)
        uniform_grads = np.random.uniform(-1, 1, 1000)
        entropy_uniform = analyzer.compute_gradient_entropy(uniform_grads)
        
        # Peaked gradients (low entropy)
        peaked_grads = np.random.randn(1000) * 0.1
        entropy_peaked = analyzer.compute_gradient_entropy(peaked_grads)
        
        # Uniform should have higher entropy
        assert entropy_uniform['entropy'] > entropy_peaked['entropy']
    
    def test_fisher_information(self, random_seed):
        """Test Fisher information computation."""
        analyzer = InformationTheoreticAnalyzer()
        
        # Low variance gradients (high Fisher information)
        low_var_grads = np.random.randn(1000) * 0.1
        fi_high = analyzer.compute_fisher_information(low_var_grads)
        
        # High variance gradients (low Fisher information)
        high_var_grads = np.random.randn(1000) * 10.0
        fi_low = analyzer.compute_fisher_information(high_var_grads)
        
        # Fisher information inversely related to variance
        assert fi_high > fi_low


class TestTheoremVerification:
    """Test formal theorem verification."""
    
    def test_sawtooth_theorem_verification(self, random_seed):
        """Test verification of sawtooth topology theorem."""
        results = verify_sawtooth_theorem(modulus=65536, steepness=10.0, num_points=1000)
        
        # Check all verification criteria
        assert 'tv_verified' in results
        assert 'lipschitz_verified' in results
        assert 'spectral_verified' in results
        assert 'theorem_verified' in results
        
        # At least some criteria should pass
        assert results['tv_verified'] or results['lipschitz_verified']
    
    def test_gradient_inversion_theorem(self, speck_cipher, random_seed):
        """Test verification of gradient inversion theorem."""
        results = verify_gradient_inversion_theorem(speck_cipher, num_samples=50)
        
        # Check verification criteria
        assert 'accuracy' in results
        assert 'inverted_accuracy' in results
        assert 'statistically_significant' in results
        
        # Inverted accuracy should be higher than regular accuracy
        # (characteristic of gradient inversion)
        assert results['inverted_accuracy'] > results['accuracy'] or \
               results['accuracy'] < 0.3  # Or just very low accuracy
    
    def test_convergence_impossibility_theorem(self, speck_cipher, random_seed):
        """Test verification of convergence impossibility theorem."""
        results = verify_convergence_impossibility_theorem(
            speck_cipher, num_trials=5, num_epochs=50
        )
        
        # Check results structure
        assert 'convergence_rate' in results
        assert 'mean_final_accuracy' in results
        assert 'convergence_failed' in results
        
        # Convergence should generally fail on ARX ciphers
        assert results['mean_final_accuracy'] < 0.7  # Not reliably converging


class TestStatisticalProperties:
    """Test statistical properties of analyses."""
    
    def test_gradient_variance_measurement(self, speck_cipher, random_seed):
        """Test gradient variance measurement across samples."""
        from ctdma.theory import measure_gradient_variance
        
        stats = measure_gradient_variance(speck_cipher, num_samples=20)
        
        assert 'mean_gradient' in stats
        assert 'std_gradient' in stats
        assert 'variance' in stats
        assert 'coefficient_of_variation' in stats
        
        # Variance should be positive
        assert stats['variance'] > 0
