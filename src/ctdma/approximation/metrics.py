"""
Metrics for Quantifying Approximation Quality

Provides comprehensive metrics to evaluate how well continuous approximations
represent discrete cryptographic operations.
"""

import torch
import numpy as np
from typing import Callable, Dict, Tuple, List
import scipy.stats as stats


class ApproximationMetrics:
    """
    Comprehensive metrics for approximation quality.
    
    Metrics include:
    1. Fidelity: How close approximation is to discrete operation
    2. Gradient bias: How much gradient differs from true gradient
    3. Convergence rate: How fast optimization converges
    4. Stability: Variance of gradients
    5. Information preservation: Mutual information between discrete and smooth
    """
    
    def __init__(self):
        pass
    
    def compute_fidelity(self, discrete_output: torch.Tensor, 
                        approximation_output: torch.Tensor,
                        metric: str = 'mse') -> float:
        """
        Compute fidelity between discrete and approximate operations.
        
        Fidelity measures how well the approximation preserves
        the functionality of the discrete operation.
        
        Metrics:
        - MSE: Mean Squared Error
        - MAE: Mean Absolute Error  
        - Cosine: Cosine similarity
        - Correlation: Pearson correlation
        
        Args:
            discrete_output: Output of discrete operation
            approximation_output: Output of smooth approximation
            metric: Fidelity metric to use
            
        Returns:
            Fidelity score (higher = better, except for MSE/MAE)
        """
        discrete = discrete_output.detach().cpu().numpy().flatten()
        approx = approximation_output.detach().cpu().numpy().flatten()
        
        if metric == 'mse':
            return float(np.mean((discrete - approx) ** 2))
        elif metric == 'mae':
            return float(np.mean(np.abs(discrete - approx)))
        elif metric == 'cosine':
            dot_product = np.dot(discrete, approx)
            norm_product = np.linalg.norm(discrete) * np.linalg.norm(approx)
            return float(dot_product / (norm_product + 1e-10))
        elif metric == 'correlation':
            return float(np.corrcoef(discrete, approx)[0, 1])
        else:
            raise ValueError(f"Unknown fidelity metric: {metric}")
    
    def compute_gradient_bias(self, x: torch.Tensor,
                             discrete_op: Callable,
                             approximate_op: Callable,
                             perturbation: float = 1e-3) -> Dict[str, float]:
        """
        Compute bias in gradients between discrete and approximate operations.
        
        Gradient bias quantifies how much the smooth gradient differs
        from the finite-difference approximation of the discrete gradient.
        
        For discrete operation f and smooth approximation g:
        
        True gradient (finite diff): ∇f(x) ≈ (f(x + ε) - f(x)) / ε
        Smooth gradient: ∇g(x) from automatic differentiation
        
        Bias: E[|∇f(x) - ∇g(x)|]
        
        Args:
            x: Input point
            discrete_op: Discrete operation
            approximate_op: Smooth approximation
            perturbation: Finite difference step size
            
        Returns:
            Dictionary of gradient bias statistics
        """
        x_np = x.detach().cpu().numpy()
        
        # Compute finite difference gradient for discrete operation
        discrete_grad_estimates = []
        for i in range(x_np.size):
            x_plus = x.clone()
            x_plus.view(-1)[i] += perturbation
            
            with torch.no_grad():
                f_x = discrete_op(x)
                f_x_plus = discrete_op(x_plus)
            
            grad_i = (f_x_plus - f_x) / perturbation
            discrete_grad_estimates.append(grad_i.mean().item())
        
        discrete_grad = np.array(discrete_grad_estimates)
        
        # Compute smooth gradient via automatic differentiation
        x_smooth = x.requires_grad_(True)
        approx_output = approximate_op(x_smooth)
        approx_output.sum().backward()
        smooth_grad = x_smooth.grad.detach().cpu().numpy().flatten()
        
        # Compute bias metrics
        bias = smooth_grad - discrete_grad
        
        return {
            'mean_bias': float(np.mean(bias)),
            'std_bias': float(np.std(bias)),
            'max_bias': float(np.max(np.abs(bias))),
            'relative_bias': float(np.mean(np.abs(bias)) / (np.mean(np.abs(discrete_grad)) + 1e-10)),
            'correlation': float(np.corrcoef(discrete_grad, smooth_grad)[0, 1] if len(discrete_grad) > 1 else 0.0)
        }
    
    def compute_gradient_variance(self, x_samples: torch.Tensor,
                                 operation: Callable) -> Dict[str, float]:
        """
        Compute variance of gradients across multiple samples.
        
        High gradient variance indicates unstable optimization.
        
        Args:
            x_samples: Multiple input samples (batch_size, ...)
            operation: Operation to analyze
            
        Returns:
            Gradient variance statistics
        """
        gradients = []
        
        for i in range(len(x_samples)):
            x = x_samples[i:i+1].requires_grad_(True)
            y = operation(x)
            y.sum().backward()
            gradients.append(x.grad.detach().cpu().numpy())
        
        gradients = np.array(gradients)
        
        return {
            'mean_gradient': float(gradients.mean()),
            'std_gradient': float(gradients.std()),
            'variance': float(gradients.var()),
            'coefficient_variation': float(gradients.std() / (abs(gradients.mean()) + 1e-10)),
            'max_gradient': float(gradients.max()),
            'min_gradient': float(gradients.min())
        }
    
    def compute_information_preservation(self, discrete_output: np.ndarray,
                                        approximation_output: np.ndarray,
                                        num_bins: int = 20) -> Dict[str, float]:
        """
        Compute mutual information between discrete and approximate outputs.
        
        High mutual information indicates the approximation preserves
        the information content of the discrete operation.
        
        I(D; A) = H(D) + H(A) - H(D, A)
        
        Args:
            discrete_output: Discrete operation outputs
            approximation_output: Approximation outputs
            num_bins: Number of bins for discretization
            
        Returns:
            Information preservation metrics
        """
        # Discretize outputs for entropy calculation
        discrete_bins = np.digitize(discrete_output, bins=np.linspace(discrete_output.min(), 
                                                                       discrete_output.max(), 
                                                                       num_bins))
        approx_bins = np.digitize(approximation_output, bins=np.linspace(approximation_output.min(),
                                                                          approximation_output.max(),
                                                                          num_bins))
        
        # Compute entropies
        h_discrete = self._entropy(discrete_bins)
        h_approx = self._entropy(approx_bins)
        h_joint = self._joint_entropy(discrete_bins, approx_bins)
        
        # Mutual information
        mi = h_discrete + h_approx - h_joint
        
        # Normalized mutual information (NMI)
        nmi = 2 * mi / (h_discrete + h_approx) if (h_discrete + h_approx) > 0 else 0
        
        return {
            'mutual_information': float(mi),
            'normalized_mi': float(nmi),
            'entropy_discrete': float(h_discrete),
            'entropy_approx': float(h_approx),
            'information_loss': float(h_discrete - mi)
        }
    
    def _entropy(self, data: np.ndarray) -> float:
        """Compute Shannon entropy."""
        _, counts = np.unique(data, return_counts=True)
        probs = counts / len(data)
        return -np.sum(probs * np.log2(probs + 1e-10))
    
    def _joint_entropy(self, x: np.ndarray, y: np.ndarray) -> float:
        """Compute joint entropy H(X, Y)."""
        joint_data = np.column_stack([x, y])
        _, counts = np.unique(joint_data, axis=0, return_counts=True)
        probs = counts / len(joint_data)
        return -np.sum(probs * np.log2(probs + 1e-10))
    
    def compute_lipschitz_constant(self, operation: Callable,
                                   x_range: torch.Tensor) -> float:
        """
        Estimate Lipschitz constant of operation.
        
        L = sup_{x≠y} |f(x) - f(y)| / |x - y|
        
        Bounded Lipschitz constant is necessary for convergence.
        
        Args:
            operation: Operation to analyze
            x_range: Range of inputs to sample
            
        Returns:
            Estimated Lipschitz constant
        """
        with torch.no_grad():
            y_values = operation(x_range)
        
        # Sample pairs and compute ratios
        max_ratio = 0.0
        n = len(x_range)
        
        # Sample subset of pairs to avoid O(n^2) complexity
        num_samples = min(1000, n * (n - 1) // 2)
        
        for _ in range(num_samples):
            i, j = np.random.choice(n, size=2, replace=False)
            
            delta_y = torch.abs(y_values[j] - y_values[i]).item()
            delta_x = torch.abs(x_range[j] - x_range[i]).item()
            
            if delta_x > 1e-8:
                ratio = delta_y / delta_x
                max_ratio = max(max_ratio, ratio)
        
        return max_ratio


def compute_fidelity(discrete_fn: Callable, approximate_fn: Callable,
                    test_inputs: torch.Tensor, metric: str = 'mse') -> float:
    """
    Convenience function to compute approximation fidelity.
    
    Args:
        discrete_fn: Discrete operation
        approximate_fn: Smooth approximation
        test_inputs: Test input samples
        metric: Fidelity metric
        
    Returns:
        Fidelity score
    """
    with torch.no_grad():
        discrete_output = discrete_fn(test_inputs)
    
    approx_output = approximate_fn(test_inputs)
    
    metrics = ApproximationMetrics()
    return metrics.compute_fidelity(discrete_output, approx_output, metric)


def compute_gradient_bias(x: torch.Tensor, discrete_fn: Callable,
                         approximate_fn: Callable) -> Dict[str, float]:
    """
    Convenience function to compute gradient bias.
    
    Args:
        x: Input point
        discrete_fn: Discrete operation
        approximate_fn: Smooth approximation
        
    Returns:
        Gradient bias statistics
    """
    metrics = ApproximationMetrics()
    return metrics.compute_gradient_bias(x, discrete_fn, approximate_fn)


def compute_convergence_rate(losses: List[float]) -> Dict[str, float]:
    """
    Analyze convergence rate from loss history.
    
    Convergence rate measures how quickly loss decreases:
    - Linear convergence: loss(t) ~ loss(0) - ct
    - Exponential: loss(t) ~ loss(0) * exp(-ct)
    - No convergence: loss remains flat
    
    Args:
        losses: Loss values over training
        
    Returns:
        Convergence statistics
    """
    losses = np.array(losses)
    t = np.arange(len(losses))
    
    # Check for convergence
    final_window = losses[-10:] if len(losses) >= 10 else losses
    initial_loss = losses[0]
    final_loss = final_window.mean()
    
    improvement = (initial_loss - final_loss) / (initial_loss + 1e-10)
    
    # Fit exponential decay
    try:
        log_losses = np.log(losses - losses.min() + 1e-10)
        slope, _ = np.polyfit(t, log_losses, 1)
        decay_rate = -slope
    except:
        decay_rate = 0.0
    
    # Compute convergence speed (epochs to 90% of final)
    target = initial_loss - 0.9 * (initial_loss - final_loss)
    epochs_to_90 = np.argmax(losses < target) if np.any(losses < target) else len(losses)
    
    return {
        'improvement': float(improvement),
        'decay_rate': float(decay_rate),
        'epochs_to_90_percent': int(epochs_to_90),
        'final_loss': float(final_loss),
        'converged': improvement > 0.5,
        'convergence_speed': 'fast' if epochs_to_90 < len(losses) * 0.3 else 'slow'
    }
