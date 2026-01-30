"""
Tests for Approximation Methods

Tests different approximation techniques and their properties.
"""

import pytest
import torch
import numpy as np
from ctdma.approximation import (
    SigmoidApproximation,
    StraightThroughEstimator,
    GumbelSoftmaxApproximation,
    TemperatureBasedSmoothing,
    ApproximationMetrics,
    compute_fidelity
)


class TestSigmoidApproximation:
    """Test sigmoid-based approximations."""
    
    def test_sigmoid_xor_basic(self, random_seed):
        """Test basic XOR approximation."""
        approx = SigmoidApproximation(initial_temperature=10.0)
        
        x = torch.tensor([0.0, 0.0, 1.0, 1.0])
        y = torch.tensor([0.0, 1.0, 0.0, 1.0])
        
        result = approx.approximate_xor(x, y)
        expected = torch.tensor([0.0, 1.0, 1.0, 0.0])
        
        # Should approximate XOR truth table
        assert torch.allclose(result, expected, atol=0.2)
    
    def test_sigmoid_modular_add(self, random_seed):
        """Test modular addition approximation."""
        approx = SigmoidApproximation(initial_temperature=10.0)
        
        x = torch.tensor([0.3, 0.7, 0.9])
        y = torch.tensor([0.2, 0.5, 0.8])
        
        result = approx.approximate_modular_add(x, y, modulus=1.0)
        
        # Results should be in [0, 1]
        assert (result >= 0).all()
        assert (result <= 1.0).all()
        
        # Should wrap around at modulus
        assert result[2] < 0.8  # 0.9 + 0.8 = 1.7, wraps to ~0.7
    
    def test_temperature_effect(self, random_seed):
        """Test effect of temperature parameter."""
        x = torch.tensor([0.5])
        y = torch.tensor([0.5])
        
        # Low temperature (smooth)
        approx_low = SigmoidApproximation(initial_temperature=0.1)
        result_low = approx_low.approximate_xor(x, y)
        
        # High temperature (sharp)
        approx_high = SigmoidApproximation(initial_temperature=100.0)
        result_high = approx_high.approximate_xor(x, y)
        
        # High temperature should be closer to discrete (0)
        assert abs(result_high.item()) < abs(result_low.item()) + 0.1


class TestStraightThroughEstimator:
    """Test straight-through estimator."""
    
    def test_ste_forward(self, random_seed):
        """Test STE forward pass."""
        ste = StraightThroughEstimator()
        
        x = torch.tensor([0.3, 0.7, 0.5], requires_grad=True)
        discrete_op = lambda x: (x > 0.5).float()
        
        result = ste.forward(x, discrete_op)
        
        # Should use discrete operation in forward
        expected = torch.tensor([0.0, 1.0, 0.0])
        assert torch.equal(result, expected)
    
    def test_ste_gradient(self, random_seed):
        """Test STE gradient computation."""
        ste = StraightThroughEstimator()
        
        x = torch.tensor([0.3, 0.7], requires_grad=True)
        discrete_op = lambda x: (x > 0.5).float()
        
        result = ste.forward(x, discrete_op)
        loss = result.sum()
        loss.backward()
        
        # STE should pass gradient straight through (identity)
        assert torch.allclose(x.grad, torch.ones_like(x))


class TestGumbelSoftmax:
    """Test Gumbel-Softmax approximation."""
    
    def test_gumbel_softmax_basic(self, random_seed):
        """Test basic Gumbel-Softmax."""
        approx = GumbelSoftmaxApproximation(initial_temperature=1.0)
        
        logits = torch.tensor([[1.0, 2.0, 0.5], [0.0, 0.0, 3.0]])
        result = approx.forward(logits)
        
        # Output should be probability distribution
        assert result.shape == (2, 3)
        assert torch.allclose(result.sum(dim=1), torch.ones(2), atol=1e-5)
        assert (result >= 0).all()
    
    def test_gumbel_temperature_effect(self, random_seed):
        """Test temperature effect on Gumbel-Softmax."""
        logits = torch.tensor([[2.0, 1.0, 0.0]])
        
        # Low temperature (more discrete)
        torch.manual_seed(42)
        approx_low = GumbelSoftmaxApproximation(initial_temperature=0.1)
        result_low = approx_low.forward(logits)
        
        # High temperature (more uniform)
        torch.manual_seed(42)
        approx_high = GumbelSoftmaxApproximation(initial_temperature=10.0)
        result_high = approx_high.forward(logits)
        
        # Low temp should be more peaked
        assert result_low.max() > result_high.max()


class TestTemperatureSmoothing:
    """Test temperature-based smoothing."""
    
    @pytest.mark.parametrize('schedule', ['linear', 'exponential', 'cosine'])
    def test_temperature_schedules(self, schedule):
        """Test different temperature schedules."""
        smoother = TemperatureBasedSmoothing(
            initial_temperature=10.0,
            final_temperature=0.1,
            schedule=schedule
        )
        
        initial_temp = smoother.get_temperature()
        
        # Update temperature halfway
        smoother.update_temperature(0.5)
        mid_temp = smoother.get_temperature()
        
        # Update to end
        smoother.update_temperature(1.0)
        final_temp = smoother.get_temperature()
        
        # Temperature should decrease
        assert initial_temp > mid_temp > final_temp
        assert abs(initial_temp - 10.0) < 0.1
        assert abs(final_temp - 0.1) < 0.1


class TestApproximationMetrics:
    """Test approximation quality metrics."""
    
    def test_fidelity_perfect(self, random_seed):
        """Test fidelity with perfect approximation."""
        metrics = ApproximationMetrics()
        
        x = torch.rand(100)
        
        fidelity_mse = metrics.compute_fidelity(x, x, metric='mse')
        fidelity_corr = metrics.compute_fidelity(x, x, metric='correlation')
        
        assert fidelity_mse < 1e-6  # Near zero MSE
        assert abs(fidelity_corr - 1.0) < 1e-5  # Perfect correlation
    
    def test_gradient_bias_measurement(self, random_seed):
        """Test gradient bias computation."""
        metrics = ApproximationMetrics()
        
        x = torch.tensor([0.5, 0.5])
        
        # Define discrete and smooth operations
        discrete_op = lambda x: (x > 0.5).float()
        smooth_op = lambda x: torch.sigmoid(10 * (x - 0.5))
        
        bias = metrics.compute_gradient_bias(x, discrete_op, smooth_op)
        
        # Should have bias statistics
        assert 'mean_bias' in bias
        assert 'std_bias' in bias
        assert 'relative_bias' in bias
    
    def test_information_preservation(self, random_seed):
        """Test information preservation metric."""
        metrics = ApproximationMetrics()
        
        x = np.random.randn(100)
        y = x + np.random.randn(100) * 0.1  # Slightly noisy version
        
        info = metrics.compute_information_preservation(x, y)
        
        # High correlation should give high mutual information
        assert info['mutual_information'] > 0
        assert 0 <= info['normalized_mi'] <= 1
    
    def test_lipschitz_constant(self, random_seed):
        """Test Lipschitz constant estimation."""
        metrics = ApproximationMetrics()
        
        # Linear function has Lipschitz constant = slope
        linear_op = lambda x: 2 * x
        x_range = torch.linspace(0, 1, 100)
        
        L = metrics.compute_lipschitz_constant(linear_op, x_range)
        
        # Should be close to 2
        assert 1.5 < L < 2.5


class TestApproximationComparison:
    """Test comparison between different approximation methods."""
    
    def test_compare_xor_approximations(self, random_seed):
        """Compare different XOR approximations."""
        x = torch.tensor([0.0, 0.0, 1.0, 1.0])
        y = torch.tensor([0.0, 1.0, 0.0, 1.0])
        expected = torch.tensor([0.0, 1.0, 1.0, 0.0])
        
        # Sigmoid approximation
        sigmoid_approx = SigmoidApproximation(initial_temperature=10.0)
        sigmoid_result = sigmoid_approx.approximate_xor(x, y)
        sigmoid_error = (sigmoid_result - expected).abs().mean()
        
        # All approximations should be reasonably close
        assert sigmoid_error < 0.3
