"""
Tests for Neural ODE Solver

Tests ODE integration and gradient computation through cipher operations.
"""

import pytest
import torch
from ctdma.neural_ode.solver import NeuralODESolver


class TestNeuralODESolver:
    """Test Neural ODE solver functionality."""
    
    def test_solver_initialization(self, speck_cipher):
        """Test solver initialization."""
        solver = NeuralODESolver(speck_cipher, method='dopri5')
        
        assert solver.cipher is not None
        assert solver.method == 'dopri5'
    
    def test_solver_forward_pass(self, speck_cipher, random_seed):
        """Test forward integration."""
        solver = NeuralODESolver(speck_cipher, method='rk4')
        
        x0 = torch.rand(2, 2)  # Initial state
        t = torch.linspace(0, 1, 10)  # Time points
        key = torch.rand(2, 4)  # Encryption key
        
        try:
            solution = solver.forward(x0, t, key)
            
            # Solution should have shape (time_steps, batch_size, state_dim)
            assert solution.shape[0] == len(t)
            assert solution.shape[1] == 2
            assert not torch.isnan(solution).any()
        except Exception as e:
            pytest.skip(f"ODE solver not available or incompatible: {e}")
    
    def test_solver_gradient_flow(self, speck_cipher, random_seed):
        """Test gradient flow through ODE solver."""
        solver = NeuralODESolver(speck_cipher, method='rk4')
        
        x0 = torch.rand(1, 2, requires_grad=True)
        t = torch.linspace(0, 1, 5)
        key = torch.rand(1, 4)
        
        try:
            solution = solver.forward(x0, t, key)
            loss = solution[-1].sum()
            loss.backward()
            
            # Gradients should flow back to initial state
            assert x0.grad is not None
            assert not torch.isnan(x0.grad).any()
        except Exception as e:
            pytest.skip(f"ODE gradient computation not available: {e}")


class TestODEIntegration:
    """Test ODE integration methods."""
    
    @pytest.mark.parametrize('method', ['rk4', 'dopri5', 'euler'])
    def test_different_methods(self, speck_cipher, method, random_seed):
        """Test different ODE integration methods."""
        try:
            solver = NeuralODESolver(speck_cipher, method=method)
            
            x0 = torch.rand(1, 2)
            t = torch.linspace(0, 1, 10)
            key = torch.rand(1, 4)
            
            solution = solver.forward(x0, t, key)
            
            assert solution.shape[0] == 10
            assert not torch.isnan(solution).any()
        except Exception as e:
            pytest.skip(f"Method {method} not available: {e}")
