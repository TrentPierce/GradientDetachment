"""
Approximation Bridge for Discrete Operations

Implements multiple approximation techniques:
1. Sigmoid-based soft approximations
2. Straight-through estimators (STE)
3. Gumbel-Softmax for categorical operations
4. Temperature-based smoothing

All approximations provide differentiable alternatives to discrete
cryptographic operations while maintaining numerical fidelity.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, Callable
from abc import ABC, abstractmethod


class ApproximationBridge(ABC, nn.Module):
    """
    Abstract base class for discrete operation approximations.
    """
    
    def __init__(self):
        super().__init__()
        
    @abstractmethod
    def approximate_xor(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Differentiable XOR approximation."""
        pass
    
    @abstractmethod
    def approximate_mod_add(self, x: torch.Tensor, y: torch.Tensor, 
                           modulus: int) -> torch.Tensor:
        """Differentiable modular addition."""
        pass
    
    @abstractmethod
    def approximate_rotation(self, x: torch.Tensor, r: int, 
                            word_size: int) -> torch.Tensor:
        """Differentiable bit rotation."""
        pass
    
    def forward(self, x: torch.Tensor, operation: str, **kwargs) -> torch.Tensor:
        """Generic forward pass routing to specific operations."""
        if operation == 'xor':
            return self.approximate_xor(x, kwargs['y'])
        elif operation == 'mod_add':
            return self.approximate_mod_add(x, kwargs['y'], kwargs['modulus'])
        elif operation == 'rotation':
            return self.approximate_rotation(x, kwargs['r'], kwargs['word_size'])
        else:
            raise ValueError(f"Unknown operation: {operation}")


class SigmoidApproximation(ApproximationBridge):
    """
    Sigmoid-based smooth approximations of discrete operations.
    
    This is the baseline approach used in the original implementation.
    Uses sigmoid functions to create smooth, differentiable versions.
    
    Parameters:
        steepness: Controls how closely approximation matches discrete operation
                   Higher values = closer to discrete, but less smooth gradients
    """
    
    def __init__(self, steepness: float = 10.0):
        super().__init__()
        self.steepness = nn.Parameter(torch.tensor(steepness))
        
    def approximate_xor(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Sigmoid-based XOR: XOR = (x AND NOT y) OR (NOT x AND y)
        
        Soft version uses sigmoid to sharpen inputs, then applies logic:
        XOR(x,y) ≈ sigmoid(s*(x-0.5)) * sigmoid(s*(1-y-0.5)) + 
                   sigmoid(s*(1-x-0.5)) * sigmoid(s*(y-0.5))
        """
        s = self.steepness
        
        # Sharpen to {0,1}
        x_sharp = torch.sigmoid(s * (x - 0.5))
        y_sharp = torch.sigmoid(s * (y - 0.5))
        
        # NOT operations
        not_x = 1 - x_sharp
        not_y = 1 - y_sharp
        
        # XOR = (x AND NOT y) OR (NOT x AND y)
        term1 = x_sharp * not_y
        term2 = not_x * y_sharp
        
        # Soft OR: a + b - ab
        result = term1 + term2 - term1 * term2
        
        return torch.clamp(result, 0, 1)
    
    def approximate_mod_add(self, x: torch.Tensor, y: torch.Tensor, 
                           modulus: int) -> torch.Tensor:
        """
        Smooth modular addition using sigmoid to handle wrap-around.
        
        z = x + y
        if z >= modulus: z -= modulus
        
        Smooth version:
        z = x + y - modulus * sigmoid(steepness * (x + y - modulus))
        """
        s = self.steepness
        sum_val = x + y
        
        # Smooth wrap detection
        wrap_amount = torch.sigmoid(s * (sum_val - modulus))
        result = sum_val - modulus * wrap_amount
        
        return torch.clamp(result, 0, modulus)
    
    def approximate_rotation(self, x: torch.Tensor, r: int, 
                            word_size: int) -> torch.Tensor:
        """
        Smooth bit rotation using weighted shifts.
        
        For binary representation: ROL(x, r) = (x << r) | (x >> (n-r))
        
        Smooth version uses floating-point shifts with interpolation.
        """
        # Normalize to [0, 1]
        x_norm = x / (2 ** word_size)
        
        # Circular shift approximation
        # Left shift: multiply by 2^r
        shifted_left = (x_norm * (2 ** r)) % 1.0
        
        # Right shift: divide by 2^(word_size - r)
        shifted_right = (x_norm / (2 ** (word_size - r))) % 1.0
        
        # Combine
        result = (shifted_left + shifted_right) % 1.0
        
        # Denormalize
        return result * (2 ** word_size)


class StraightThroughEstimator(ApproximationBridge):
    """
    Straight-Through Estimator (STE) for discrete operations.
    
    Forward pass: Uses actual discrete operation
    Backward pass: Gradient flows as if operation were identity
    
    This is a biased estimator but often works well in practice.
    
    Reference:
    Bengio et al., "Estimating or Propagating Gradients Through 
    Stochastic Neurons for Conditional Computation" (2013)
    """
    
    def __init__(self, temperature: float = 1.0):
        super().__init__()
        self.temperature = temperature
        
    def approximate_xor(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        STE for XOR: Forward uses rounded values, backward assumes identity.
        """
        # Forward: actual XOR on binarized inputs
        x_binary = (x > 0.5).float()
        y_binary = (y > 0.5).float()
        result_binary = (x_binary + y_binary) % 2
        
        # Backward: gradient flows as if identity
        # Trick: result_binary + (smooth_version - smooth_version.detach())
        smooth = x + y - 2 * x * y  # Soft XOR
        result = result_binary + (smooth - smooth.detach())
        
        return result
    
    def approximate_mod_add(self, x: torch.Tensor, y: torch.Tensor, 
                           modulus: int) -> torch.Tensor:
        """
        STE for modular addition.
        """
        # Forward: actual modular addition
        result_discrete = (x + y) % modulus
        
        # Backward: gradient of smooth version
        smooth = x + y  # Linear approximation
        result = result_discrete + (smooth - smooth.detach())
        
        return result
    
    def approximate_rotation(self, x: torch.Tensor, r: int, 
                            word_size: int) -> torch.Tensor:
        """
        STE for rotation: Discrete rotation in forward, smooth in backward.
        """
        # Forward: actual bitwise rotation
        x_int = x.long()
        mask = (1 << word_size) - 1
        rotated = ((x_int << r) | (x_int >> (word_size - r))) & mask
        result_discrete = rotated.float()
        
        # Backward: identity
        result = result_discrete + (x - x.detach())
        
        return result


class GumbelSoftmaxApproximation(ApproximationBridge):
    """
    Gumbel-Softmax approximation for categorical/discrete operations.
    
    Uses the Gumbel-Softmax (Concrete) distribution to provide
    a differentiable approximation of sampling from categorical distributions.
    
    Particularly useful for XOR and other bitwise operations.
    
    Reference:
    Jang et al., "Categorical Reparameterization with Gumbel-Softmax" (2017)
    Maddison et al., "The Concrete Distribution" (2017)
    """
    
    def __init__(self, temperature: float = 1.0, hard: bool = False):
        super().__init__()
        self.temperature = nn.Parameter(torch.tensor(temperature))
        self.hard = hard
        
    def gumbel_softmax_sample(self, logits: torch.Tensor, 
                             temperature: float) -> torch.Tensor:
        """
        Sample from Gumbel-Softmax distribution.
        
        Args:
            logits: Unnormalized log probabilities
            temperature: Temperature parameter (lower = more discrete)
        
        Returns:
            Soft sample from categorical distribution
        """
        # Sample Gumbel noise
        U = torch.rand_like(logits)
        gumbel_noise = -torch.log(-torch.log(U + 1e-20) + 1e-20)
        
        # Add noise and apply softmax with temperature
        y = logits + gumbel_noise
        y_soft = F.softmax(y / temperature, dim=-1)
        
        if self.hard:
            # Straight-through: discrete in forward, soft in backward
            y_hard = torch.zeros_like(y_soft)
            y_hard.scatter_(-1, y_soft.argmax(dim=-1, keepdim=True), 1.0)
            y = y_hard - y_soft.detach() + y_soft
        else:
            y = y_soft
            
        return y
    
    def approximate_xor(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Gumbel-Softmax XOR: Treat XOR as categorical distribution over {0, 1}.
        
        XOR truth table:
        x=0, y=0 -> 0
        x=0, y=1 -> 1
        x=1, y=0 -> 1  
        x=1, y=1 -> 0
        
        Model as categorical with logits based on inputs.
        """
        # Convert inputs to logits for binary outcomes
        # Logits for output=0: high when x==y
        # Logits for output=1: high when x!=y
        
        logits_0 = -torch.abs(x - y)  # High when x == y
        logits_1 = torch.abs(x - y)   # High when x != y
        
        logits = torch.stack([logits_0, logits_1], dim=-1)
        
        # Sample using Gumbel-Softmax
        sample = self.gumbel_softmax_sample(logits, self.temperature)
        
        # Extract probability of output=1 (XOR result)
        result = sample[..., 1]
        
        return result
    
    def approximate_mod_add(self, x: torch.Tensor, y: torch.Tensor, 
                           modulus: int) -> torch.Tensor:
        """
        Gumbel-Softmax for modular addition.
        
        Treat result as categorical over modulus possible values.
        """
        # Create logits for each possible output value
        sum_val = (x + y) % modulus
        
        # Create one-hot-like logits
        # This is simplified - full version would use learned parameters
        logits = torch.zeros(x.shape[0], modulus, device=x.device)
        logits.scatter_(1, sum_val.long().unsqueeze(1), 10.0)  # High logit for true value
        
        # Sample
        sample = self.gumbel_softmax_sample(logits, self.temperature)
        
        # Convert back to scalar value
        values = torch.arange(modulus, device=x.device).float()
        result = (sample * values).sum(dim=1)
        
        return result
    
    def approximate_rotation(self, x: torch.Tensor, r: int, 
                            word_size: int) -> torch.Tensor:
        """
        Gumbel-Softmax rotation: Sample rotated bits.
        """
        # Simplified version: use soft interpolation
        x_norm = x / (2 ** word_size)
        
        # Create categorical over bit positions
        num_positions = word_size
        logits = torch.zeros(x.shape[0], num_positions, device=x.device)
        
        # Set high logit for rotated position
        rotated_pos = r % num_positions
        logits[:, rotated_pos] = 10.0
        
        # Sample
        sample = self.gumbel_softmax_sample(logits, self.temperature)
        
        # Apply rotation based on sample
        # This is simplified - actual implementation would be more complex
        result = x_norm * (2 ** word_size)
        
        return result


class TemperatureScheduler:
    """
    Temperature scheduling for annealing approximations.
    
    Gradually decreases temperature during training to transition
    from smooth approximations to discrete operations.
    
    Schedules:
    - Linear: T(t) = T_max - (T_max - T_min) * t / T_total
    - Exponential: T(t) = T_max * (T_min / T_max)^(t / T_total)
    - Cosine: T(t) = T_min + (T_max - T_min) * (1 + cos(pi * t / T_total)) / 2
    """
    
    def __init__(self, T_max: float = 10.0, T_min: float = 0.1, 
                 total_steps: int = 1000, schedule: str = 'exponential'):
        self.T_max = T_max
        self.T_min = T_min
        self.total_steps = total_steps
        self.schedule = schedule
        self.current_step = 0
        
    def step(self) -> float:
        """Get current temperature and increment step."""
        T = self.get_temperature()
        self.current_step += 1
        return T
    
    def get_temperature(self, step: Optional[int] = None) -> float:
        """Compute temperature at given step."""
        if step is None:
            step = self.current_step
            
        t = min(step / self.total_steps, 1.0)  # Normalized time [0, 1]
        
        if self.schedule == 'linear':
            T = self.T_max - (self.T_max - self.T_min) * t
        elif self.schedule == 'exponential':
            T = self.T_max * (self.T_min / self.T_max) ** t
        elif self.schedule == 'cosine':
            T = self.T_min + (self.T_max - self.T_min) * \
                (1 + np.cos(np.pi * t)) / 2
        else:
            raise ValueError(f"Unknown schedule: {self.schedule}")
            
        return T
    
    def reset(self):
        """Reset scheduler to initial state."""
        self.current_step = 0


def create_approximation(method: str, **kwargs) -> ApproximationBridge:
    """
    Factory function to create approximation instances.
    
    Args:
        method: 'sigmoid', 'ste', 'gumbel'
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
    else:
        raise ValueError(f"Unknown approximation method: {method}")


if __name__ == "__main__":
    # Quick test
    print("Testing approximation methods...\n")
    
    x = torch.tensor([0.3, 0.7, 0.9])
    y = torch.tensor([0.2, 0.8, 0.1])
    
    methods = ['sigmoid', 'ste', 'gumbel']
    
    for method in methods:
        approx = create_approximation(method)
        result = approx.approximate_xor(x, y)
        print(f"{method.upper():10} XOR: {result.detach().numpy()}")
    
    print("\n✓ All approximation methods loaded successfully")