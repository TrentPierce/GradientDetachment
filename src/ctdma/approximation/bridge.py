"""
Approximation Bridge Techniques for Discrete Operations

This module implements various differentiable approximations of discrete
cryptographic operations, allowing gradient-based analysis while maintaining
controllable fidelity to the original functions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable, Tuple, Optional
from abc import ABC, abstractmethod
import numpy as np


class ApproximationBridge(ABC):
    """
    Abstract base class for approximation techniques.
    
    An approximation bridge provides a differentiable version of a discrete
    operation while tracking the approximation error and convergence properties.
    """
    
    def __init__(self, discrete_op: Callable, device: str = 'cpu'):
        """
        Initialize approximation bridge.
        
        Args:
            discrete_op: The true discrete operation to approximate
            device: Torch device
        """
        self.discrete_op = discrete_op
        self.device = device
        
    @abstractmethod
    def forward(self, *args, **kwargs) -> torch.Tensor:
        """Compute differentiable approximation."""
        pass
    
    @abstractmethod
    def get_approximation_error(self, *args, **kwargs) -> float:
        """Compute approximation error vs true discrete operation."""
        pass
    
    @abstractmethod
    def get_gradient_estimate(self, *args, **kwargs) -> torch.Tensor:
        """Get gradient estimate for backpropagation."""
        pass


class SigmoidApproximation(ApproximationBridge):
    """
    Sigmoid-based smooth approximation (existing method).
    
    For binary operations, uses sigmoid to create smooth transitions:
        XOR: x + y - 2·σ(k(x+y-1))
        AND: σ(k(x+y-1.5))
        Modular: x + y - n·σ(k(x+y-n))
    
    Args:
        discrete_op: Discrete operation to approximate
        steepness: Sigmoid steepness parameter (higher = closer to discrete)
        operation_type: Type of operation ('xor', 'and', 'modular')
    """
    
    def __init__(self, discrete_op: Callable, steepness: float = 10.0,
                 operation_type: str = 'modular', modulus: int = 2**16,
                 device: str = 'cpu'):
        super().__init__(discrete_op, device)
        self.steepness = steepness
        self.operation_type = operation_type
        self.modulus = modulus
        
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute sigmoid-based approximation.
        
        Mathematical form:
            g(x, y) = f_approx(x, y; k)
        
        where k is the steepness parameter controlling smoothness.
        """
        if self.operation_type == 'xor':
            return self._sigmoid_xor(x, y)
        elif self.operation_type == 'and':
            return self._sigmoid_and(x, y)
        elif self.operation_type == 'modular':
            return self._sigmoid_modular(x, y)
        else:
            raise ValueError(f"Unknown operation type: {self.operation_type}")
    
    def _sigmoid(self, z: torch.Tensor) -> torch.Tensor:
        """Sigmoid function: σ(z) = 1/(1 + e^(-z))"""
        return torch.sigmoid(z)
    
    def _sigmoid_xor(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Smooth XOR using sigmoid.
        
        XOR truth table: 0⊕0=0, 0⊕1=1, 1⊕0=1, 1⊕1=0
        Approximation: x + y - 2·x·y
        With sharpening: use sigmoid to make inputs more binary-like
        """
        x_sharp = self._sigmoid(self.steepness * (x - 0.5))
        y_sharp = self._sigmoid(self.steepness * (y - 0.5))
        
        # XOR ≈ (1-x)y + x(1-y) = x + y - 2xy
        return x_sharp + y_sharp - 2 * x_sharp * y_sharp
    
    def _sigmoid_and(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Smooth AND using sigmoid.
        
        AND: output 1 only when both inputs are 1
        """
        return self._sigmoid(self.steepness * (x + y - 1.5))
    
    def _sigmoid_modular(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Smooth modular addition: (x + y) mod n
        
        Uses sigmoid to smoothly wrap around at modulus boundary.
        """
        sum_val = x + y
        wrap = self._sigmoid(self.steepness * (sum_val - self.modulus))
        return sum_val - self.modulus * wrap
    
    def get_approximation_error(self, x: torch.Tensor, y: torch.Tensor) -> float:
        """
        Compute mean absolute error vs discrete operation.
        
        Error = E[|g(x,y) - f(x,y)|]
        """
        approx = self.forward(x, y)
        discrete = self.discrete_op(x, y)
        return torch.abs(approx - discrete).mean().item()
    
    def get_gradient_estimate(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Get gradient via autograd."""
        x_var = x.clone().requires_grad_(True)
        out = self.forward(x_var, y)
        grad = torch.autograd.grad(out.sum(), x_var)[0]
        return grad


class StraightThroughEstimator(ApproximationBridge):
    """
    Straight-Through Estimator (STE) for gradient approximation.
    
    Forward pass: Use discrete operation (no approximation)
    Backward pass: Pretend the operation was identity (∇f = I)
    
    This technique from Bengio et al. (2013) allows gradients to flow
    through non-differentiable operations by approximating the gradient.
    
    Pros: Exact forward computation
    Cons: Biased gradient estimate
    
    Reference:
        Bengio, Y., Léonard, N., & Courville, A. (2013).
        Estimating or Propagating Gradients Through Stochastic Neurons for
        Conditional Computation. arXiv:1308.3432
    """
    
    def __init__(self, discrete_op: Callable, device: str = 'cpu'):
        super().__init__(discrete_op, device)
        
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Forward pass uses exact discrete operation.
        Backward pass uses identity gradient.
        """
        return STEFunction.apply(x, y, self.discrete_op)
    
    def get_approximation_error(self, x: torch.Tensor, y: torch.Tensor) -> float:
        """STE has zero forward error (uses exact discrete op)."""
        return 0.0
    
    def get_gradient_estimate(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Gradient is identity for STE."""
        return torch.ones_like(x)


class STEFunction(torch.autograd.Function):
    """Custom autograd function for straight-through estimator."""
    
    @staticmethod
    def forward(ctx, x, y, discrete_op):
        """Forward: exact discrete operation."""
        ctx.save_for_backward(x, y)
        return discrete_op(x, y)
    
    @staticmethod
    def backward(ctx, grad_output):
        """Backward: identity gradient (straight through)."""
        x, y = ctx.saved_tensors
        # Gradient flows straight through
        return grad_output, grad_output, None


class GumbelSoftmaxApproximation(ApproximationBridge):
    """
    Gumbel-Softmax (Concrete) approximation for discrete operations.
    
    Uses the Gumbel-Softmax trick to create a differentiable approximation
    of discrete sampling. Particularly useful for operations that can be
    expressed as categorical distributions.
    
    The Gumbel-Softmax distribution provides a continuous approximation
    to discrete distributions with a temperature parameter τ:
    - τ → 0: approaches discrete (one-hot)
    - τ → ∞: approaches uniform
    
    Reference:
        Jang, E., Gu, S., & Poole, B. (2017).
        Categorical Reparameterization with Gumbel-Softmax.
        ICLR 2017.
    """
    
    def __init__(self, discrete_op: Callable, temperature: float = 1.0,
                 num_categories: int = 2, device: str = 'cpu'):
        super().__init__(discrete_op, device)
        self.temperature = temperature
        self.num_categories = num_categories
        
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute Gumbel-Softmax approximation.
        
        For binary operations, convert inputs to logits and apply
        Gumbel-Softmax relaxation.
        """
        # Convert continuous inputs to logits
        x_logits = self._to_logits(x)
        y_logits = self._to_logits(y)
        
        # Sample Gumbel noise
        gumbel_x = self._sample_gumbel(x_logits.shape)
        gumbel_y = self._sample_gumbel(y_logits.shape)
        
        # Gumbel-Softmax: softmax((logits + gumbel) / τ)
        x_soft = F.softmax((x_logits + gumbel_x) / self.temperature, dim=-1)
        y_soft = F.softmax((y_logits + gumbel_y) / self.temperature, dim=-1)
        
        # Combine via discrete operation logic
        # For XOR: output = x[1]⊕y[1] where [0] is prob(0), [1] is prob(1)
        if x_soft.dim() > 1:
            result = torch.abs(x_soft[..., 1] - y_soft[..., 1])  # XOR approximation
        else:
            result = torch.abs(x_soft - y_soft)
        
        return result
    
    def _to_logits(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convert continuous values to logits.
        
        For x ∈ [0, 1], create logits [log(1-x), log(x)]
        """
        # Clamp to avoid log(0)
        x_clamp = torch.clamp(x, 1e-7, 1 - 1e-7)
        logit_0 = torch.log(1 - x_clamp)
        logit_1 = torch.log(x_clamp)
        return torch.stack([logit_0, logit_1], dim=-1)
    
    def _sample_gumbel(self, shape: Tuple) -> torch.Tensor:
        """
        Sample from Gumbel(0, 1) distribution.
        
        Gumbel: G = -log(-log(U)) where U ~ Uniform(0, 1)
        """
        uniform = torch.rand(shape, device=self.device)
        uniform = torch.clamp(uniform, 1e-7, 1 - 1e-7)
        gumbel = -torch.log(-torch.log(uniform))
        return gumbel
    
    def get_approximation_error(self, x: torch.Tensor, y: torch.Tensor) -> float:
        """Compute approximation error."""
        approx = self.forward(x, y)
        discrete = self.discrete_op(x, y)
        return torch.abs(approx - discrete).mean().item()
    
    def get_gradient_estimate(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Get gradient via Gumbel-Softmax relaxation."""
        x_var = x.clone().requires_grad_(True)
        out = self.forward(x_var, y)
        grad = torch.autograd.grad(out.sum(), x_var)[0]
        return grad


class TemperatureSmoothing(ApproximationBridge):
    """
    Temperature-based smoothing with annealing schedule.
    
    Gradually decreases temperature during training to transition from
    smooth approximation to discrete operation:
    
        g_τ(x, y) = smooth_op(x, y; τ)
    
    where τ decreases over time: τ(t) = τ_0 · exp(-λt)
    
    This provides:
    - Early training: smooth gradients (high τ)
    - Late training: accurate discrete behavior (low τ)
    
    Args:
        discrete_op: True discrete operation
        initial_temp: Starting temperature
        final_temp: Ending temperature
        anneal_rate: Rate of temperature decrease
    """
    
    def __init__(self, discrete_op: Callable, initial_temp: float = 10.0,
                 final_temp: float = 0.1, anneal_rate: float = 0.001,
                 device: str = 'cpu'):
        super().__init__(discrete_op, device)
        self.initial_temp = initial_temp
        self.final_temp = final_temp
        self.anneal_rate = anneal_rate
        self.current_temp = initial_temp
        self.step_count = 0
        
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute temperature-smoothed approximation.
        
        Uses sigmoid with temperature-controlled steepness.
        """
        # Effective steepness based on current temperature
        steepness = 1.0 / max(self.current_temp, self.final_temp)
        
        # Smooth modular addition with temperature
        sum_val = x + y
        sigmoid = torch.sigmoid(steepness * (sum_val - 1.0))
        result = sum_val - 2.0 * sigmoid
        
        return result
    
    def step(self):
        """
        Update temperature for next iteration.
        
        Temperature schedule: τ(t) = τ_final + (τ_init - τ_final)·exp(-λt)
        """
        self.step_count += 1
        decay = np.exp(-self.anneal_rate * self.step_count)
        self.current_temp = self.final_temp + (self.initial_temp - self.final_temp) * decay
        
    def get_approximation_error(self, x: torch.Tensor, y: torch.Tensor) -> float:
        """Compute current approximation error."""
        approx = self.forward(x, y)
        discrete = self.discrete_op(x, y)
        return torch.abs(approx - discrete).mean().item()
    
    def get_gradient_estimate(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Get gradient with current temperature."""
        x_var = x.clone().requires_grad_(True)
        out = self.forward(x_var, y)
        grad = torch.autograd.grad(out.sum(), x_var)[0]
        return grad
    
    def get_temperature(self) -> float:
        """Get current temperature."""
        return self.current_temp


def compare_approximations(x: torch.Tensor, y: torch.Tensor,
                          discrete_op: Callable,
                          device: str = 'cpu') -> dict:
    """
    Compare all approximation methods on given inputs.
    
    Args:
        x: Input tensor 1
        y: Input tensor 2
        discrete_op: True discrete operation
        device: Torch device
        
    Returns:
        Dictionary with comparison metrics for each method
    """
    methods = {
        'Sigmoid': SigmoidApproximation(discrete_op, steepness=10.0, device=device),
        'STE': StraightThroughEstimator(discrete_op, device=device),
        'Gumbel-Softmax': GumbelSoftmaxApproximation(discrete_op, temperature=1.0, device=device),
        'Temperature': TemperatureSmoothing(discrete_op, initial_temp=5.0, device=device)
    }
    
    results = {}
    discrete_output = discrete_op(x, y)
    
    for name, method in methods.items():
        approx_output = method.forward(x, y)
        gradient = method.get_gradient_estimate(x, y)
        error = method.get_approximation_error(x, y)
        
        results[name] = {
            'output': approx_output,
            'error': error,
            'gradient_mean': gradient.mean().item(),
            'gradient_std': gradient.std().item(),
            'output_correlation': torch.corrcoef(torch.stack([
                discrete_output.flatten(),
                approx_output.flatten()
            ]))[0, 1].item()
        }
    
    return results
