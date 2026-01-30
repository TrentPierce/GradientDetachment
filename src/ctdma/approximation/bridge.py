"""
Approximation Bridge for Discrete Operations

Implements multiple approximation techniques for making discrete
cryptographic operations differentiable.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable, Optional, Dict
from abc import ABC, abstractmethod


class ApproximationBridge(ABC):
    """
    Abstract base class for discrete operation approximations.
    
    All approximation methods must implement:
    - forward: Compute approximation
    - backward: Define gradient computation
    - get_temperature: Return current temperature/steepness
    """
    
    def __init__(self, initial_temperature: float = 1.0):
        self.temperature = initial_temperature
        
    @abstractmethod
    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Compute approximation."""
        pass
    
    @abstractmethod
    def get_gradient(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Compute gradient (may differ from automatic differentiation)."""
        pass
    
    def set_temperature(self, temperature: float):
        """Update approximation temperature."""
        self.temperature = temperature
    
    def get_temperature(self) -> float:
        """Get current temperature."""
        return self.temperature


class SigmoidApproximation(ApproximationBridge):
    """
    Sigmoid-based approximation for discrete operations.
    
    This is the method used in the original Speck implementation.
    
    For XOR:
        XOR(x, y) ≈ sigmoid(β(x-0.5)) + sigmoid(β(y-0.5)) - 2*sigmoid(β(x-0.5))*sigmoid(β(y-0.5))
    
    For modular addition:
        (x + y) mod n ≈ x + y - n*sigmoid(β(x+y-n))
    
    Advantages:
    - Smooth everywhere
    - Well-behaved gradients for small β
    - Easy to implement
    
    Disadvantages:
    - Large gradients at boundaries for large β
    - Creates sawtooth topology
    - Leads to gradient inversion
    """
    
    def __init__(self, initial_temperature: float = 10.0):
        super().__init__(initial_temperature)
        
    def approximate_xor(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Smooth XOR approximation.
        
        Mathematical formulation:
        XOR(x, y) = (x AND NOT y) OR (NOT x AND y)
        
        For continuous x, y ∈ [0, 1]:
        XOR(x, y) ≈ x + y - 2xy (simple version)
        
        For sharper approximation:
        XOR(x, y) ≈ f(x)(1-f(y)) + (1-f(x))f(y)
        where f(z) = sigmoid(β(z - 0.5))
        """
        beta = self.temperature
        
        # Sharpen inputs
        x_sharp = torch.sigmoid(beta * (x - 0.5))
        y_sharp = torch.sigmoid(beta * (y - 0.5))
        
        # XOR logic
        term1 = x_sharp * (1 - y_sharp)
        term2 = (1 - x_sharp) * y_sharp
        
        # Soft OR
        result = term1 + term2 - term1 * term2
        
        return torch.clamp(result, 0, 1)
    
    def approximate_modular_add(self, x: torch.Tensor, y: torch.Tensor, 
                               modulus: float = 2.0) -> torch.Tensor:
        """
        Smooth modular addition.
        
        z = (x + y) mod modulus
        
        Smooth version:
        z ≈ x + y - modulus * sigmoid(β(x + y - modulus))
        """
        beta = self.temperature
        sum_val = x + y
        wrap_amount = torch.sigmoid(beta * (sum_val - modulus))
        result = sum_val - modulus * wrap_amount
        return result
    
    def approximate_rotation(self, x: torch.Tensor, r: int, 
                           word_size: int = 16) -> torch.Tensor:
        """
        Smooth bit rotation.
        
        For differentiable rotation, we use weighted interpolation.
        """
        # Convert to bit representation (simplified)
        x_normalized = x * (2 ** word_size)
        
        # Circular shift approximation
        shifted_left = x_normalized * (2 ** r)
        shifted_right = x_normalized / (2 ** (word_size - r))
        
        result = (shifted_left + shifted_right) % (2 ** word_size)
        return result / (2 ** word_size)
    
    def forward(self, x: torch.Tensor, operation: str = 'xor', **kwargs) -> torch.Tensor:
        """
        Apply approximation for specified operation.
        
        Args:
            x: Input tensor
            operation: 'xor', 'mod_add', or 'rotate'
            **kwargs: Operation-specific parameters
        """
        if operation == 'xor':
            y = kwargs.get('y')
            return self.approximate_xor(x, y)
        elif operation == 'mod_add':
            y = kwargs.get('y')
            modulus = kwargs.get('modulus', 2.0)
            return self.approximate_modular_add(x, y, modulus)
        elif operation == 'rotate':
            r = kwargs.get('r', 1)
            word_size = kwargs.get('word_size', 16)
            return self.approximate_rotation(x, r, word_size)
        else:
            raise ValueError(f"Unknown operation: {operation}")
    
    def get_gradient(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Compute gradient using automatic differentiation.
        """
        x_grad = x.requires_grad_(True)
        y = self.forward(x_grad, **kwargs)
        y.sum().backward()
        return x_grad.grad


class StraightThroughEstimator(ApproximationBridge):
    """
    Straight-Through Estimator (STE) for discrete operations.
    
    Key idea: Use discrete operation in forward pass, but pretend
    it's identity in backward pass.
    
    Forward:  y = discrete_op(x)
    Backward: dy/dx = 1 (identity gradient)
    
    Advantages:
    - Exact discrete operation in forward pass
    - Simple gradient computation
    - No hyperparameters to tune
    
    Disadvantages:
    - Biased gradients
    - May not converge
    - No theoretical guarantees
    
    Reference:
    Bengio et al. "Estimating or Propagating Gradients Through
    Stochastic Neurons for Conditional Computation" (2013)
    """
    
    class STEFunction(torch.autograd.Function):
        """Custom autograd function for straight-through estimation."""
        
        @staticmethod
        def forward(ctx, x, discrete_op):
            """Forward: Use discrete operation."""
            return discrete_op(x)
        
        @staticmethod
        def backward(ctx, grad_output):
            """Backward: Pass gradient straight through."""
            return grad_output, None
    
    def __init__(self):
        super().__init__()
        
    def forward(self, x: torch.Tensor, discrete_op: Callable, **kwargs) -> torch.Tensor:
        """
        Apply STE.
        
        Args:
            x: Input tensor
            discrete_op: Discrete operation to approximate
        """
        return self.STEFunction.apply(x, discrete_op)
    
    def get_gradient(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        STE gradient is identity.
        """
        return torch.ones_like(x)


class GumbelSoftmaxApproximation(ApproximationBridge):
    """
    Gumbel-Softmax approximation for discrete operations.
    
    Key idea: Add Gumbel noise and use softmax with temperature
    to create differentiable samples from categorical distributions.
    
    For discrete choice c ∈ {0, 1, ..., K-1}:
    
    Hard: c = argmax_i(log π_i + g_i)
    Soft: s_i = exp((log π_i + g_i)/τ) / ∑_j exp((log π_j + g_j)/τ)
    
    where g_i ~ Gumbel(0, 1) and τ is temperature.
    
    As τ → 0: s → one-hot (discrete)
    As τ → ∞: s → uniform (smooth)
    
    Advantages:
    - Unbiased gradients
    - Smooth interpolation between discrete and continuous
    - Theoretically grounded
    
    Disadvantages:
    - Requires reparameterization
    - Temperature annealing needed
    - More complex than sigmoid
    
    Reference:
    Jang et al. "Categorical Reparameterization with Gumbel-Softmax" (2017)
    """
    
    def __init__(self, initial_temperature: float = 1.0, hard: bool = False):
        super().__init__(initial_temperature)
        self.hard = hard
        
    def sample_gumbel(self, shape: tuple, device: str = 'cpu', eps: float = 1e-10) -> torch.Tensor:
        """
        Sample from Gumbel(0, 1) distribution.
        
        G = -log(-log(U)) where U ~ Uniform(0, 1)
        """
        u = torch.rand(shape, device=device)
        return -torch.log(-torch.log(u + eps) + eps)
    
    def gumbel_softmax_sample(self, logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        """
        Sample from Gumbel-Softmax distribution.
        
        Args:
            logits: Unnormalized log probabilities
            temperature: Softmax temperature
        """
        gumbel_noise = self.sample_gumbel(logits.shape, device=logits.device)
        y = logits + gumbel_noise
        return F.softmax(y / temperature, dim=-1)
    
    def forward(self, logits: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Apply Gumbel-Softmax trick.
        
        Args:
            logits: Unnormalized log probabilities (batch_size, num_classes)
        """
        y = self.gumbel_softmax_sample(logits, self.temperature)
        
        if self.hard:
            # Straight-through: forward uses argmax, backward uses soft
            y_hard = torch.zeros_like(y)
            y_hard.scatter_(-1, y.argmax(dim=-1, keepdim=True), 1.0)
            y = (y_hard - y).detach() + y
        
        return y
    
    def get_gradient(self, logits: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Compute Gumbel-Softmax gradient.
        """
        logits_grad = logits.requires_grad_(True)
        y = self.forward(logits_grad, **kwargs)
        y.sum().backward()
        return logits_grad.grad


class TemperatureBasedSmoothing(ApproximationBridge):
    """
    Temperature-based smoothing with annealing schedule.
    
    Key idea: Start with high temperature (smooth) and gradually
    decrease temperature (approach discrete).
    
    Temperature schedules:
    1. Linear: T(t) = T_0 - (T_0 - T_f) * t / T_max
    2. Exponential: T(t) = T_0 * (T_f / T_0)^(t / T_max)
    3. Cosine: T(t) = T_f + 0.5 * (T_0 - T_f) * (1 + cos(π * t / T_max))
    
    Advantages:
    - Smooth optimization early (easier convergence)
    - Approaches discrete operation (better accuracy)
    - Flexible scheduling
    
    Disadvantages:
    - Requires tuning schedule
    - May get stuck if cooled too fast
    - Gradient explosion if cooled too slow
    """
    
    def __init__(self, initial_temperature: float = 10.0, 
                 final_temperature: float = 0.1,
                 schedule: str = 'exponential'):
        super().__init__(initial_temperature)
        self.initial_temperature = initial_temperature
        self.final_temperature = final_temperature
        self.schedule = schedule
        self.step_count = 0
        
    def update_temperature(self, progress: float):
        """
        Update temperature based on training progress.
        
        Args:
            progress: Training progress in [0, 1]
        """
        T_0 = self.initial_temperature
        T_f = self.final_temperature
        
        if self.schedule == 'linear':
            self.temperature = T_0 - (T_0 - T_f) * progress
        elif self.schedule == 'exponential':
            self.temperature = T_0 * (T_f / T_0) ** progress
        elif self.schedule == 'cosine':
            self.temperature = T_f + 0.5 * (T_0 - T_f) * (1 + np.cos(np.pi * progress))
        else:
            raise ValueError(f"Unknown schedule: {self.schedule}")
        
        self.step_count += 1
    
    def forward(self, x: torch.Tensor, operation: str = 'sigmoid', **kwargs) -> torch.Tensor:
        """
        Apply temperature-smoothed operation.
        
        Args:
            x: Input tensor
            operation: Base operation type
        """
        # Use sigmoid approximation with current temperature
        approximator = SigmoidApproximation(self.temperature)
        return approximator.forward(x, operation, **kwargs)
    
    def get_gradient(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Compute gradient with current temperature.
        """
        x_grad = x.requires_grad_(True)
        y = self.forward(x_grad, **kwargs)
        y.sum().backward()
        return x_grad.grad


def create_approximator(method: str, **kwargs) -> ApproximationBridge:
    """
    Factory function to create approximation bridges.
    
    Args:
        method: Approximation method ('sigmoid', 'ste', 'gumbel', 'temperature')
        **kwargs: Method-specific parameters
        
    Returns:
        ApproximationBridge instance
    """
    if method == 'sigmoid':
        return SigmoidApproximation(**kwargs)
    elif method == 'ste':
        return StraightThroughEstimator(**kwargs)
    elif method == 'gumbel':
        return GumbelSoftmaxApproximation(**kwargs)
    elif method == 'temperature':
        return TemperatureBasedSmoothing(**kwargs)
    else:
        raise ValueError(f"Unknown approximation method: {method}")


import numpy as np  # For cosine schedule