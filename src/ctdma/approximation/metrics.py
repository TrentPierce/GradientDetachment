"""
Quantitative Metrics for Approximation Fidelity

This module provides comprehensive metrics to evaluate the quality of
differentiable approximations to discrete operations.
"""

import torch
import numpy as np
from typing import Callable, Tuple, Dict
from dataclasses import dataclass
from scipy.stats import spearmanr, pearsonr


@dataclass
class ApproximationMetrics:
    """
    Comprehensive metrics for approximation quality.
    
    Attributes:
        forward_error: Mean absolute error in forward pass
        gradient_correlation: Correlation between approximate and true gradients
        lipschitz_ratio: Ratio of Lipschitz constants
        information_loss: Mutual information loss (bits)
        convergence_rate: Rate of convergence to discrete operation
    """
    forward_error: float
    gradient_correlation: float
    lipschitz_ratio: float
    information_loss: float
    convergence_rate: float
    
    def __repr__(self) -> str:
        return f"""ApproximationMetrics(
    Forward Error: {self.forward_error:.6f}
    Gradient Correlation: {self.gradient_correlation:.4f}
    Lipschitz Ratio: {self.lipschitz_ratio:.4f}
    Information Loss: {self.information_loss:.4f} bits
    Convergence Rate: {self.convergence_rate:.6f}
)"""


def compute_fidelity(approx_func: Callable,
                    discrete_func: Callable,
                    x: torch.Tensor,
                    y: torch.Tensor,
                    epsilon: float = 1e-3) -> ApproximationMetrics:
    """
    Compute comprehensive fidelity metrics.
    
    Measures how well the approximation preserves the behavior of the
    discrete operation across multiple dimensions:
    
    1. Forward Error: |g(x,y) - f(x,y)|
    2. Gradient Correlation: cor(∇g, ∇f_fd)
    3. Lipschitz Ratio: L(g) / L(f)
    4. Information Loss: I(X; f(X)) - I(X; g(X))
    5. Convergence Rate: Rate at which g → f
    
    Args:
        approx_func: Approximation function g
        discrete_func: True discrete function f
        x: Input tensor 1
        y: Input tensor 2
        epsilon: Small perturbation for finite differences
        
    Returns:
        ApproximationMetrics object
    """
    # 1. Forward error
    with torch.no_grad():
        approx_output = approx_func(x, y)
        discrete_output = discrete_func(x, y)
        forward_error = torch.abs(approx_output - discrete_output).mean().item()
    
    # 2. Gradient correlation
    grad_corr = compute_gradient_similarity(approx_func, discrete_func, x, y, epsilon)
    
    # 3. Lipschitz ratio
    lipschitz_approx = estimate_lipschitz_constant(approx_func, x, y)
    lipschitz_discrete = estimate_lipschitz_constant(discrete_func, x, y)
    lipschitz_ratio = lipschitz_approx / max(lipschitz_discrete, 1e-6)
    
    # 4. Information loss
    info_loss = estimate_information_loss(approx_output, discrete_output)
    
    # 5. Convergence rate (estimate from error gradient)
    convergence = estimate_convergence_rate(approx_func, discrete_func, x, y)
    
    return ApproximationMetrics(
        forward_error=forward_error,
        gradient_correlation=grad_corr,
        lipschitz_ratio=lipschitz_ratio,
        information_loss=info_loss,
        convergence_rate=convergence
    )


def compute_gradient_similarity(approx_func: Callable,
                                discrete_func: Callable,
                                x: torch.Tensor,
                                y: torch.Tensor,
                                epsilon: float = 1e-3) -> float:
    """
    Compute correlation between approximate and discrete gradients.
    
    Since the discrete function is not differentiable, we use finite
    differences to estimate the discrete gradient:
    
        ∇_x f(x,y) ≈ (f(x+ε,y) - f(x,y)) / ε
    
    Then compute correlation:
        ρ = cor(∇_x g, ∇_x f_fd)
    
    Args:
        approx_func: Smooth approximation g
        discrete_func: Discrete function f
        x: Input tensor 1
        y: Input tensor 2
        epsilon: Step size for finite differences
        
    Returns:
        Gradient correlation in [-1, 1]
    """
    # Compute approximate gradient via autograd
    x_var = x.clone().requires_grad_(True)
    approx_out = approx_func(x_var, y)
    approx_grad = torch.autograd.grad(approx_out.sum(), x_var)[0]
    
    # Compute discrete gradient via finite differences
    with torch.no_grad():
        f_x = discrete_func(x, y)
        f_x_plus = discrete_func(x + epsilon, y)
        discrete_grad = (f_x_plus - f_x) / epsilon
    
    # Compute correlation
    approx_flat = approx_grad.flatten().detach().cpu().numpy()
    discrete_flat = discrete_grad.flatten().cpu().numpy()
    
    # Handle edge cases
    if len(approx_flat) < 2:
        return 0.0
    
    correlation, _ = pearsonr(approx_flat, discrete_flat)
    
    return float(correlation)


def compute_discrete_error(approx_func: Callable,
                          discrete_func: Callable,
                          x: torch.Tensor,
                          y: torch.Tensor,
                          threshold: float = 0.5) -> float:
    """
    Compute discrete classification error.
    
    For binary operations, measures how often the approximation produces
    the wrong discrete output after thresholding:
    
        Error = P(round(g(x,y)) ≠ f(x,y))
    
    Args:
        approx_func: Approximation function
        discrete_func: True discrete function
        x: Input tensor 1
        y: Input tensor 2
        threshold: Threshold for binarization
        
    Returns:
        Discrete error rate in [0, 1]
    """
    with torch.no_grad():
        approx_output = approx_func(x, y)
        discrete_output = discrete_func(x, y)
        
        # Binarize approximation
        approx_binary = (approx_output > threshold).float()
        
        # Compute error rate
        errors = (approx_binary != discrete_output).float()
        error_rate = errors.mean().item()
    
    return error_rate


def estimate_lipschitz_constant(func: Callable,
                                x: torch.Tensor,
                                y: torch.Tensor,
                                num_samples: int = 100) -> float:
    """
    Estimate Lipschitz constant of a function.
    
    The Lipschitz constant L is defined as:
        L = sup_{x≠x'} |f(x) - f(x')| / |x - x'|
    
    We estimate this empirically by sampling pairs and computing:
        L_empirical = max_i |f(x_i) - f(x'_i)| / |x_i - x'_i|
    
    Args:
        func: Function to analyze
        x: Base input tensor
        y: Second input tensor
        num_samples: Number of sample pairs
        
    Returns:
        Estimated Lipschitz constant
    """
    max_lipschitz = 0.0
    
    with torch.no_grad():
        for _ in range(num_samples):
            # Random perturbation
            delta = torch.randn_like(x) * 0.01
            x_perturbed = x + delta
            
            # Compute function values
            f_x = func(x, y)
            f_x_perturbed = func(x_perturbed, y)
            
            # Compute Lipschitz ratio
            numerator = torch.abs(f_x - f_x_perturbed).max().item()
            denominator = torch.norm(delta).item()
            
            if denominator > 1e-8:
                lipschitz = numerator / denominator
                max_lipschitz = max(max_lipschitz, lipschitz)
    
    return max_lipschitz


def estimate_information_loss(approx_output: torch.Tensor,
                              discrete_output: torch.Tensor,
                              num_bins: int = 50) -> float:
    """
    Estimate information loss using mutual information.
    
    Measures how much information about the discrete output is lost
    in the approximation:
    
        Loss = I(X; f(X)) - I(X; g(X))
    
    where I is mutual information.
    
    We approximate this by computing:
        Loss ≈ H(f(X)) - H(f(X)|g(X))
    
    Args:
        approx_output: Approximate function output
        discrete_output: True discrete output
        num_bins: Number of bins for discretization
        
    Returns:
        Information loss in bits
    """
    # Flatten tensors
    approx_flat = approx_output.flatten().detach().cpu().numpy()
    discrete_flat = discrete_output.flatten().cpu().numpy()
    
    # Discretize for entropy calculation
    discrete_unique = np.unique(discrete_flat)
    approx_bins = np.linspace(approx_flat.min(), approx_flat.max(), num_bins)
    
    # Compute joint distribution
    joint_hist = np.zeros((len(discrete_unique), num_bins))
    
    for i, d_val in enumerate(discrete_unique):
        mask = discrete_flat == d_val
        approx_subset = approx_flat[mask]
        hist, _ = np.histogram(approx_subset, bins=approx_bins)
        joint_hist[i, :] = hist
    
    # Normalize to probability
    joint_prob = joint_hist / (joint_hist.sum() + 1e-10)
    
    # Marginal probabilities
    p_discrete = joint_prob.sum(axis=1)
    p_approx = joint_prob.sum(axis=0)
    
    # Entropy H(f(X))
    h_discrete = -np.sum(p_discrete * np.log2(p_discrete + 1e-10))
    
    # Conditional entropy H(f(X)|g(X))
    h_conditional = 0.0
    for j in range(num_bins):
        if p_approx[j] > 0:
            conditional_dist = joint_prob[:, j] / p_approx[j]
            h_conditional -= p_approx[j] * np.sum(
                conditional_dist * np.log2(conditional_dist + 1e-10)
            )
    
    # Information loss = H(f) - I(f;g) = H(f) - (H(f) - H(f|g)) = H(f|g)
    info_loss = h_conditional
    
    return float(info_loss)


def estimate_convergence_rate(approx_func: Callable,
                              discrete_func: Callable,
                              x: torch.Tensor,
                              y: torch.Tensor,
                              num_steps: int = 10) -> float:
    """
    Estimate convergence rate as approximation improves.
    
    For parameterized approximations (e.g., with steepness k),
    measures how quickly error decreases as k increases:
    
        Rate = -d(log error)/dk
    
    Args:
        approx_func: Approximation function
        discrete_func: Discrete function
        x: Input tensor 1
        y: Input tensor 2
        num_steps: Number of parameter steps to test
        
    Returns:
        Convergence rate (higher = faster convergence)
    """
    # This is a simplified estimate - assumes error decreases exponentially
    with torch.no_grad():
        approx_out = approx_func(x, y)
        discrete_out = discrete_func(x, y)
        current_error = torch.abs(approx_out - discrete_out).mean().item()
    
    # Estimate rate from current error
    # Assumes exponential convergence: error ~ exp(-rate * k)
    if current_error < 1e-6:
        return 1.0  # Already converged
    
    # Simple estimate based on error magnitude
    convergence_rate = -np.log(max(current_error, 1e-10))
    
    return float(convergence_rate)


def analyze_gradient_flow(approx_func: Callable,
                          x: torch.Tensor,
                          y: torch.Tensor,
                          num_layers: int = 5) -> Dict[str, float]:
    """
    Analyze gradient flow through multiple layers.
    
    Simulates a deep network with the approximation as an operation
    and measures gradient vanishing/explosion.
    
    Args:
        approx_func: Approximation function
        x: Input tensor
        y: Input tensor
        num_layers: Number of simulated layers
        
    Returns:
        Dictionary with gradient flow statistics
    """
    # Create a simple chain of operations
    current = x.clone().requires_grad_(True)
    intermediates = [current]
    
    for _ in range(num_layers):
        current = approx_func(current, y)
        intermediates.append(current)
    
    # Compute gradients at each layer
    loss = current.sum()
    gradients = torch.autograd.grad(loss, intermediates[0])[0]
    
    # Analyze gradient statistics
    grad_norm = torch.norm(gradients).item()
    grad_mean = gradients.mean().item()
    grad_std = gradients.std().item()
    grad_max = gradients.abs().max().item()
    
    # Check for vanishing/exploding
    vanishing = grad_norm < 1e-4
    exploding = grad_norm > 1e4
    
    return {
        'gradient_norm': grad_norm,
        'gradient_mean': grad_mean,
        'gradient_std': grad_std,
        'gradient_max': grad_max,
        'vanishing': vanishing,
        'exploding': exploding
    }


def compute_approximation_stability(approx_func: Callable,
                                   x: torch.Tensor,
                                   y: torch.Tensor,
                                   noise_levels: list = [0.01, 0.05, 0.1]) -> Dict:
    """
    Test stability of approximation under input noise.
    
    Measures how sensitive the approximation is to input perturbations:
        Stability = E[|g(x+ε) - g(x)|] / |ε|
    
    Args:
        approx_func: Approximation function
        x: Input tensor
        y: Input tensor
        noise_levels: List of noise standard deviations to test
        
    Returns:
        Dictionary with stability metrics
    """
    stability_results = {}
    
    with torch.no_grad():
        baseline = approx_func(x, y)
        
        for noise_level in noise_levels:
            # Add noise to input
            noise = torch.randn_like(x) * noise_level
            x_noisy = x + noise
            
            # Compute output change
            noisy_output = approx_func(x_noisy, y)
            output_change = torch.abs(noisy_output - baseline).mean().item()
            
            # Stability ratio
            stability = output_change / noise_level
            
            stability_results[f'noise_{noise_level}'] = {
                'output_change': output_change,
                'stability_ratio': stability
            }
    
    return stability_results
