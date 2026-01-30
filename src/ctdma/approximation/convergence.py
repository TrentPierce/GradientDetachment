"""
Convergence Property Analysis for Approximation Methods

This module analyzes how approximation precision affects gradient flow
and convergence properties during optimization.
"""

import torch
import numpy as np
from typing import Callable, List, Dict, Tuple
from dataclasses import dataclass
import matplotlib.pyplot as plt


@dataclass
class ConvergenceMetrics:
    """Metrics characterizing convergence behavior."""
    convergence_time: int
    final_loss: float
    gradient_variance: float
    oscillation_frequency: float
    basin_escape_probability: float
    
    def __repr__(self) -> str:
        return f"""ConvergenceMetrics(
    Convergence Time: {self.convergence_time} steps
    Final Loss: {self.final_loss:.6f}
    Gradient Variance: {self.gradient_variance:.6f}
    Oscillation Frequency: {self.oscillation_frequency:.4f}
    Basin Escape Probability: {self.basin_escape_probability:.4f}
)"""


class ConvergenceAnalyzer:
    """
    Analyze convergence properties of approximation methods.
    
    This class provides tools to study how approximation quality affects:
    1. Convergence speed
    2. Final accuracy
    3. Gradient stability
    4. Escape from local minima
    """
    
    def __init__(self, approx_func: Callable, discrete_func: Callable,
                 device: str = 'cpu'):
        """
        Initialize convergence analyzer.
        
        Args:
            approx_func: Approximation function
            discrete_func: True discrete function
            device: Torch device
        """
        self.approx_func = approx_func
        self.discrete_func = discrete_func
        self.device = device
        
    def analyze_convergence_trajectory(self,
                                      x_init: torch.Tensor,
                                      y: torch.Tensor,
                                      target: torch.Tensor,
                                      learning_rate: float = 0.01,
                                      num_steps: int = 1000,
                                      convergence_threshold: float = 1e-3) -> ConvergenceMetrics:
        """
        Analyze full convergence trajectory.
        
        Simulates gradient descent on a simple optimization problem and
        measures convergence properties.
        
        Problem: min_x ||approx_func(x, y) - target||²
        
        Args:
            x_init: Initial parameter value
            y: Fixed second input
            target: Target output
            learning_rate: Step size for gradient descent
            num_steps: Maximum optimization steps
            convergence_threshold: Loss threshold for convergence
            
        Returns:
            ConvergenceMetrics object
        """
        x = x_init.clone().requires_grad_(True)
        optimizer = torch.optim.SGD([x], lr=learning_rate)
        
        loss_history = []
        gradient_history = []
        converged = False
        convergence_time = num_steps
        
        for step in range(num_steps):
            optimizer.zero_grad()
            
            # Forward pass
            output = self.approx_func(x, y)
            loss = torch.mean((output - target) ** 2)
            
            # Backward pass
            loss.backward()
            gradient = x.grad.clone()
            
            # Track history
            loss_history.append(loss.item())
            gradient_history.append(gradient.norm().item())
            
            # Check convergence
            if loss.item() < convergence_threshold and not converged:
                convergence_time = step
                converged = True
            
            # Update parameters
            optimizer.step()
        
        # Compute metrics
        final_loss = loss_history[-1]
        gradient_variance = np.var(gradient_history)
        
        # Oscillation frequency (using FFT)
        if len(loss_history) > 10:
            fft = np.fft.fft(loss_history)
            freqs = np.fft.fftfreq(len(loss_history))
            peak_freq_idx = np.argmax(np.abs(fft[1:len(fft)//2])) + 1
            oscillation_frequency = abs(freqs[peak_freq_idx])
        else:
            oscillation_frequency = 0.0
        
        # Basin escape probability (estimated from gradient reversals)
        gradient_signs = np.sign(gradient_history)
        sign_changes = np.sum(np.diff(gradient_signs) != 0)
        basin_escape_prob = sign_changes / len(gradient_history)
        
        return ConvergenceMetrics(
            convergence_time=convergence_time,
            final_loss=final_loss,
            gradient_variance=gradient_variance,
            oscillation_frequency=oscillation_frequency,
            basin_escape_probability=basin_escape_prob
        )
    
    def compare_approximation_quality(self,
                                     steepness_values: List[float],
                                     x: torch.Tensor,
                                     y: torch.Tensor) -> Dict:
        """
        Compare convergence across different approximation qualities.
        
        For sigmoid-based approximations, steepness controls quality:
        - Low steepness: smooth but inaccurate
        - High steepness: accurate but discontinuous gradients
        
        Args:
            steepness_values: List of steepness parameters to test
            x: Input tensor
            y: Input tensor
            
        Returns:
            Dictionary with results for each steepness
        """
        results = {}
        
        for k in steepness_values:
            # Create approximation with this steepness
            from .bridge import SigmoidApproximation
            approx = SigmoidApproximation(
                self.discrete_func,
                steepness=k,
                device=self.device
            )
            
            # Compute forward error
            output = approx.forward(x, y)
            discrete_output = self.discrete_func(x, y)
            forward_error = torch.abs(output - discrete_output).mean().item()
            
            # Compute gradient stability
            x_var = x.clone().requires_grad_(True)
            out = approx.forward(x_var, y)
            grad = torch.autograd.grad(out.sum(), x_var)[0]
            
            gradient_variance = grad.var().item()
            gradient_mean = grad.mean().item()
            
            # Check for gradient inversion
            true_grad_sign = 1.0  # Assume positive for modular addition
            inversion_rate = (grad < 0).float().mean().item()
            
            results[f'k={k}'] = {
                'forward_error': forward_error,
                'gradient_variance': gradient_variance,
                'gradient_mean': gradient_mean,
                'inversion_rate': inversion_rate
            }
        
        return results
    
    def measure_gradient_noise_amplification(self,
                                            x: torch.Tensor,
                                            y: torch.Tensor,
                                            noise_levels: List[float]) -> Dict:
        """
        Measure how approximation amplifies gradient noise.
        
        Gradient noise can be amplified by discontinuities in the
        approximation, leading to unstable optimization.
        
        Args:
            x: Input tensor
            y: Input tensor
            noise_levels: List of noise standard deviations
            
        Returns:
            Dictionary with noise amplification factors
        """
        results = {}
        
        # Baseline gradient
        x_var = x.clone().requires_grad_(True)
        output = self.approx_func(x_var, y)
        baseline_grad = torch.autograd.grad(output.sum(), x_var)[0]
        baseline_norm = baseline_grad.norm().item()
        
        for noise_std in noise_levels:
            amplifications = []
            
            for _ in range(10):  # Multiple trials
                # Add noise to input
                noise = torch.randn_like(x) * noise_std
                x_noisy = x + noise
                
                # Compute gradient with noise
                x_var = x_noisy.clone().requires_grad_(True)
                output_noisy = self.approx_func(x_var, y)
                noisy_grad = torch.autograd.grad(output_noisy.sum(), x_var)[0]
                
                # Measure amplification
                grad_change = torch.abs(noisy_grad - baseline_grad).norm().item()
                amplification = grad_change / (noise_std * np.sqrt(x.numel()))
                amplifications.append(amplification)
            
            results[f'noise_{noise_std}'] = {
                'mean_amplification': np.mean(amplifications),
                'std_amplification': np.std(amplifications)
            }
        
        return results


def analyze_temperature_schedule(initial_temp: float = 10.0,
                                 final_temp: float = 0.1,
                                 num_steps: int = 1000,
                                 schedule_type: str = 'exponential') -> np.ndarray:
    """
    Analyze temperature annealing schedules.
    
    Temperature controls approximation quality over time:
    - Start high: smooth gradients, easy optimization
    - End low: accurate discrete behavior
    
    Schedules:
    ----------
    1. Exponential: τ(t) = τ_final + (τ_init - τ_final)·exp(-λt)
    2. Linear: τ(t) = τ_init - (τ_init - τ_final)·t/T
    3. Cosine: τ(t) = τ_final + 0.5·(τ_init - τ_final)·(1 + cos(πt/T))
    4. Step: τ(t) = τ_init if t < T/2 else τ_final
    
    Args:
        initial_temp: Starting temperature
        final_temp: Ending temperature
        num_steps: Number of optimization steps
        schedule_type: Type of schedule ('exponential', 'linear', 'cosine', 'step')
        
    Returns:
        Array of temperature values over time
    """
    t = np.linspace(0, 1, num_steps)
    
    if schedule_type == 'exponential':
        # Exponential decay: τ(t) = τ_f + (τ_i - τ_f)·exp(-5t)
        temps = final_temp + (initial_temp - final_temp) * np.exp(-5 * t)
        
    elif schedule_type == 'linear':
        # Linear decay
        temps = initial_temp - (initial_temp - final_temp) * t
        
    elif schedule_type == 'cosine':
        # Cosine annealing
        temps = final_temp + 0.5 * (initial_temp - final_temp) * (1 + np.cos(np.pi * t))
        
    elif schedule_type == 'step':
        # Step decay (sudden change at midpoint)
        temps = np.where(t < 0.5, initial_temp, final_temp)
        
    else:
        raise ValueError(f"Unknown schedule type: {schedule_type}")
    
    return temps


def measure_approximation_quality(approx_func: Callable,
                                  discrete_func: Callable,
                                  x: torch.Tensor,
                                  y: torch.Tensor,
                                  temperature: float = 1.0) -> Dict[str, float]:
    """
    Comprehensive quality measurement for a single approximation.
    
    Measures:
    1. Forward fidelity: |g(x,y) - f(x,y)|
    2. Gradient fidelity: cor(∇g, ∇f_fd)
    3. Smoothness: Lipschitz constant
    4. Information preservation: Mutual information
    5. Robustness: Stability to perturbations
    
    Args:
        approx_func: Approximation function
        discrete_func: True discrete function
        x: Input tensor
        y: Input tensor
        temperature: Approximation temperature parameter
        
    Returns:
        Dictionary with quality metrics
    """
    # 1. Forward fidelity
    with torch.no_grad():
        approx_out = approx_func(x, y)
        discrete_out = discrete_func(x, y)
        forward_fidelity = 1.0 - torch.abs(approx_out - discrete_out).mean().item()
    
    # 2. Gradient fidelity
    epsilon = 1e-3
    x_var = x.clone().requires_grad_(True)
    approx_output = approx_func(x_var, y)
    approx_grad = torch.autograd.grad(approx_output.sum(), x_var)[0]
    
    with torch.no_grad():
        discrete_grad_fd = (discrete_func(x + epsilon, y) - discrete_func(x, y)) / epsilon
    
    grad_correlation = torch.corrcoef(torch.stack([
        approx_grad.flatten(),
        discrete_grad_fd.flatten()
    ]))[0, 1].item()
    
    # 3. Smoothness (inverse of max gradient magnitude)
    smoothness = 1.0 / (approx_grad.abs().max().item() + 1e-6)
    
    # 4. Information preservation (via discrete accuracy)
    approx_binary = (approx_out > 0.5).float()
    accuracy = (approx_binary == discrete_out).float().mean().item()
    
    # 5. Robustness
    noise = torch.randn_like(x) * 0.01
    approx_noisy = approx_func(x + noise, y)
    robustness = 1.0 - torch.abs(approx_noisy - approx_out).mean().item()
    
    # Overall quality score (weighted average)
    quality_score = (
        0.3 * forward_fidelity +
        0.2 * grad_correlation +
        0.2 * smoothness +
        0.2 * accuracy +
        0.1 * robustness
    )
    
    return {
        'forward_fidelity': forward_fidelity,
        'gradient_correlation': grad_correlation,
        'smoothness': smoothness,
        'discrete_accuracy': accuracy,
        'robustness': robustness,
        'overall_quality': quality_score
    }


def visualize_convergence_comparison(results: Dict[str, ConvergenceMetrics],
                                    save_path: Optional[str] = None):
    """
    Visualize convergence comparison across methods.
    
    Args:
        results: Dictionary mapping method names to ConvergenceMetrics
        save_path: Optional path to save figure
    """
    methods = list(results.keys())
    
    # Extract metrics
    conv_times = [results[m].convergence_time for m in methods]
    final_losses = [results[m].final_loss for m in methods]
    grad_variances = [results[m].gradient_variance for m in methods]
    
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Plot 1: Convergence time
    axes[0].bar(methods, conv_times)
    axes[0].set_ylabel('Convergence Time (steps)')
    axes[0].set_title('Convergence Speed')
    axes[0].tick_params(axis='x', rotation=45)
    
    # Plot 2: Final loss
    axes[1].bar(methods, final_losses)
    axes[1].set_ylabel('Final Loss')
    axes[1].set_title('Final Accuracy')
    axes[1].set_yscale('log')
    axes[1].tick_params(axis='x', rotation=45)
    
    # Plot 3: Gradient variance
    axes[2].bar(methods, grad_variances)
    axes[2].set_ylabel('Gradient Variance')
    axes[2].set_title('Gradient Stability')
    axes[2].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    return fig


def compute_theoretical_convergence_bound(lipschitz_constant: float,
                                         smoothness_constant: float,
                                         learning_rate: float,
                                         initial_error: float) -> Tuple[int, float]:
    """
    Compute theoretical convergence bound for gradient descent.
    
    For a function with Lipschitz constant L and smoothness μ:
    
        ||x_t - x*||² ≤ (1 - 2μα + L²α²)^t · ||x_0 - x*||²
    
    where α is the learning rate.
    
    Convergence guaranteed if: α < 2μ/L²
    
    Args:
        lipschitz_constant: L (gradient Lipschitz constant)
        smoothness_constant: μ (strong convexity parameter)
        learning_rate: α (step size)
        initial_error: ||x_0 - x*||²
        
    Returns:
        (estimated_iterations, final_error)
    """
    # Check convergence condition
    if learning_rate >= 2 * smoothness_constant / (lipschitz_constant ** 2):
        return float('inf'), float('inf')  # No convergence guarantee
    
    # Convergence rate
    convergence_rate = 1 - 2 * smoothness_constant * learning_rate + \
                      (lipschitz_constant ** 2) * (learning_rate ** 2)
    
    # Iterations to reach error threshold (e.g., 1e-6)
    error_threshold = 1e-6
    if convergence_rate < 1:
        iterations = int(np.log(error_threshold / initial_error) / np.log(convergence_rate))
        final_error = initial_error * (convergence_rate ** iterations)
    else:
        iterations = float('inf')
        final_error = float('inf')
    
    return iterations, final_error


from typing import Optional
