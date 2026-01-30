"""
Approximation Bridge Module

Implements multiple approximation techniques for discrete cryptographic operations,
allowing gradient flow while maintaining varying degrees of fidelity to the
original discrete operations.

Techniques implemented:
1. Sigmoid-based approximations (existing, smooth gradients)
2. Straight-through estimators (discrete forward, continuous backward)
3. Gumbel-Softmax (stochastic continuous relaxation)
4. Temperature-based smoothing (annealed approximation)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, Callable
from enum import Enum


class ApproximationType(Enum):
    """Types of approximation methods."""
    SIGMOID = "sigmoid"
    STRAIGHT_THROUGH = "straight_through"
    GUMBEL_SOFTMAX = "gumbel_softmax"
    TEMPERATURE_ANNEALING = "temperature_annealing"


class ApproximationBridge(nn.Module):
    """
    Base class for approximation bridges between discrete and continuous operations.
    
    Provides a unified interface for different approximation techniques,
    allowing easy comparison and analysis.
    """
    
    def __init__(self, n_bits: int = 16, modulus: int = None):
        super().__init__()
        self.n_bits = n_bits
        self.modulus = modulus or (2 ** n_bits)
        
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Forward pass - must be implemented by subclasses."""
        raise NotImplementedError
        
    def discrete_op(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Exact discrete operation for comparison."""
        raise NotImplementedError
        
    def compute_approximation_error(
        self, 
        x: torch.Tensor, 
        y: torch.Tensor
    ) -> torch.Tensor:
        """Compute error between approximation and discrete operation."""
        approx = self.forward(x, y)
        discrete = self.discrete_op(x, y)
        return torch.abs(approx - discrete).mean()


class SigmoidApproximation(ApproximationBridge):
    """
    Sigmoid-based smooth approximation.
    
    Modular addition: z = x + y - m·σ(β(x + y - m))
    XOR: z = x + y - 2·σ(β·x)·σ(β·y)
    
    Properties:
    - Smooth gradients everywhere
    - High approximation error at boundaries
    - Steepness β controls smoothness/fidelity tradeoff
    """
    
    def __init__(
        self, 
        n_bits: int = 16, 
        steepness: float = 10.0,
        operation: str = 'modadd'
    ):
        super().__init__(n_bits)
        self.steepness = steepness
        self.operation = operation
        
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Smooth approximation using sigmoid."""
        if self.operation == 'modadd':
            return self._sigmoid_modadd(x, y)
        elif self.operation == 'xor':
            return self._sigmoid_xor(x, y)
        else:
            raise ValueError(f"Unknown operation: {self.operation}")
    
    def _sigmoid_modadd(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Sigmoid-based modular addition."""
        sum_val = x + y
        wrap = torch.sigmoid(self.steepness * (sum_val - self.modulus))
        return sum_val - self.modulus * wrap
    
    def _sigmoid_xor(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Sigmoid-based XOR approximation."""
        # XOR ≈ x + y - 2xy for normalized inputs
        x_sig = torch.sigmoid(self.steepness * (x - 0.5))
        y_sig = torch.sigmoid(self.steepness * (y - 0.5))
        return x_sig + y_sig - 2 * x_sig * y_sig
    
    def discrete_op(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Exact discrete operation."""
        if self.operation == 'modadd':
            return (x + y) % self.modulus
        elif self.operation == 'xor':
            return ((x + y) % 2).round()
        else:
            raise ValueError(f"Unknown operation: {self.operation}")


class StraightThroughEstimator(ApproximationBridge):
    """
    Straight-Through Estimator (STE).
    
    Forward pass: Uses discrete operation
    Backward pass: Assumes gradient = 1 (identity)
    
    Properties:
    - Exact forward pass (zero approximation error)
    - Biased gradients (assumes linear function)
    - Popular in binary neural networks
    
    Mathematical justification:
    For f(x) = sign(x), STE uses:
        Forward: y = sign(x)
        Backward: ∂L/∂x = ∂L/∂y (pretend f is identity)
    
    This gives biased but useful gradients.
    """
    
    def __init__(self, n_bits: int = 16, operation: str = 'modadd'):
        super().__init__(n_bits)
        self.operation = operation
        
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Forward: discrete operation with straight-through gradient."""
        discrete_out = self.discrete_op(x, y)
        
        # Straight-through: forward discrete, backward continuous
        # y_ste = y_discrete + (x - x).detach() = y_discrete
        # but ∂y_ste/∂x = 1 (identity gradient)
        return discrete_out + (x + y - x - y).detach()
    
    def discrete_op(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Exact discrete operation."""
        if self.operation == 'modadd':
            return (x + y) % self.modulus
        elif self.operation == 'xor':
            x_int = (x * (2**self.n_bits)).long()
            y_int = (y * (2**self.n_bits)).long()
            result = (x_int ^ y_int).float() / (2**self.n_bits)
            return result
        else:
            raise ValueError(f"Unknown operation: {self.operation}")


class GumbelSoftmaxApproximation(ApproximationBridge):
    """
    Gumbel-Softmax (Concrete) relaxation for discrete operations.
    
    Converts discrete sampling to continuous relaxation:
        z_discrete ~ Categorical(π)
        z_continuous = softmax((log(π) + g) / τ)
    
    where g ~ Gumbel(0, 1) and τ is temperature.
    
    Properties:
    - Stochastic (adds noise)
    - Converges to discrete as τ → 0
    - Unbiased gradient estimates
    - Used in VAEs with discrete latents
    """
    
    def __init__(
        self, 
        n_bits: int = 16, 
        temperature: float = 1.0,
        operation: str = 'modadd'
    ):
        super().__init__(n_bits)
        self.temperature = temperature
        self.operation = operation
        
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Gumbel-Softmax approximation."""
        if self.operation == 'modadd':
            return self._gumbel_modadd(x, y)
        elif self.operation == 'xor':
            return self._gumbel_xor(x, y)
        else:
            raise ValueError(f"Unknown operation: {self.operation}")
    
    def _gumbel_modadd(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Gumbel-Softmax for modular addition."""
        # Compute exact result
        exact = (x + y) % self.modulus
        
        # Add Gumbel noise
        if self.training:
            # Sample Gumbel noise
            gumbel_noise = -torch.log(-torch.log(
                torch.rand_like(exact) + 1e-10
            ) + 1e-10)
            
            # Apply temperature
            logits = exact + gumbel_noise
            soft_result = torch.sigmoid(logits / self.temperature)
            
            # Normalize back to original range
            return soft_result * self.modulus
        else:
            return exact
    
    def _gumbel_xor(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Gumbel-Softmax for XOR."""
        # XOR as categorical: output is 0 or 1
        x_bit = (x > 0.5).float()
        y_bit = (y > 0.5).float()
        xor_result = (x_bit + y_bit) % 2
        
        if self.training:
            # Logits for 2-class problem
            logits = torch.stack([1 - xor_result, xor_result], dim=-1)
            
            # Gumbel noise
            gumbel = -torch.log(-torch.log(
                torch.rand_like(logits) + 1e-10
            ) + 1e-10)
            
            # Gumbel-Softmax
            soft_output = F.softmax((torch.log(logits + 1e-10) + gumbel) / self.temperature, dim=-1)
            
            return soft_output[..., 1]  # Return probability of 1
        else:
            return xor_result
    
    def discrete_op(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Exact discrete operation."""
        if self.operation == 'modadd':
            return (x + y) % self.modulus
        elif self.operation == 'xor':
            return ((x > 0.5).float() + (y > 0.5).float()) % 2
        else:
            raise ValueError(f"Unknown operation: {self.operation}")


class TemperatureAnnealing(ApproximationBridge):
    """
    Temperature-based annealing approximation.
    
    Gradually transitions from smooth (high temperature) to discrete (low temperature):
        z(τ) = x + y - m·σ((x + y - m)/τ)
    
    As τ → 0, σ becomes step function, recovering discrete operation.
    As τ → ∞, operation becomes linear (pure addition).
    
    Properties:
    - Controllable smoothness via temperature
    - Can anneal during training
    - Smooth transition between continuous and discrete
    """
    
    def __init__(
        self,
        n_bits: int = 16,
        initial_temperature: float = 1.0,
        operation: str = 'modadd',
        anneal_rate: float = 0.01
    ):
        super().__init__(n_bits)
        self.register_buffer('temperature', torch.tensor(initial_temperature))
        self.operation = operation
        self.anneal_rate = anneal_rate
        
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Temperature-annealed approximation."""
        if self.operation == 'modadd':
            return self._temp_modadd(x, y)
        elif self.operation == 'xor':
            return self._temp_xor(x, y)
        else:
            raise ValueError(f"Unknown operation: {self.operation}")
    
    def _temp_modadd(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Temperature-annealed modular addition."""
        sum_val = x + y
        # Temperature controls sharpness of transition
        wrap = torch.sigmoid((sum_val - self.modulus) / (self.temperature + 1e-10))
        return sum_val - self.modulus * wrap
    
    def _temp_xor(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Temperature-annealed XOR."""
        # XOR decision boundary at x = y
        diff = (x - y) / (self.temperature + 1e-10)
        soft_xor = torch.sigmoid(diff)
        return soft_xor
    
    def anneal(self):
        """Decrease temperature (move toward discrete)."""
        self.temperature.data = torch.maximum(
            self.temperature * (1 - self.anneal_rate),
            torch.tensor(0.01)  # Minimum temperature
        )
    
    def discrete_op(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Exact discrete operation."""
        if self.operation == 'modadd':
            return (x + y) % self.modulus
        elif self.operation == 'xor':
            return ((x + y) % 2).round()
        else:
            raise ValueError(f"Unknown operation: {self.operation}")


def create_approximation_bridge(
    approximation_type: str,
    n_bits: int = 16,
    operation: str = 'modadd',
    **kwargs
) -> ApproximationBridge:
    """
    Factory function to create approximation bridges.
    
    Args:
        approximation_type: Type of approximation
        n_bits: Bit width
        operation: Operation type ('modadd', 'xor')
        **kwargs: Additional parameters for specific approximation
        
    Returns:
        Approximation bridge instance
    """
    if approximation_type == 'sigmoid':
        steepness = kwargs.get('steepness', 10.0)
        return SigmoidApproximation(n_bits, steepness, operation)
    elif approximation_type == 'straight_through':
        return StraightThroughEstimator(n_bits, operation)
    elif approximation_type == 'gumbel_softmax':
        temperature = kwargs.get('temperature', 1.0)
        return GumbelSoftmaxApproximation(n_bits, temperature, operation)
    elif approximation_type == 'temperature_annealing':
        initial_temp = kwargs.get('initial_temperature', 1.0)
        anneal_rate = kwargs.get('anneal_rate', 0.01)
        return TemperatureAnnealing(n_bits, initial_temp, operation, anneal_rate)
    else:
        raise ValueError(f"Unknown approximation type: {approximation_type}")
