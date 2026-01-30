"""
Approximation Fidelity Metrics

Provides quantitative metrics to measure how well smooth approximations
match discrete operations. Metrics include:
- Approximation error (L1, L2, L∞)
- Gradient correlation
- Smoothness measures
- Convergence properties
"""

import torch
import numpy as np
from typing import Dict, Tuple, Callable
import scipy.stats as stats


class ApproximationFidelityMetrics:
    """
    Comprehensive metrics for approximation quality.
    """
    
    def __init__(self):
        pass
        
    def compute_all_metrics(self, 
                           discrete_fn: Callable,
                           smooth_fn: Callable,
                           test_inputs: torch.Tensor,
                           **kwargs) -> Dict:
        """
        Compute all fidelity metrics.
        
        Args:
            discrete_fn: Discrete operation (ground truth)
            smooth_fn: Smooth approximation
            test_inputs: Test input samples
            **kwargs: Additional arguments for functions
            
        Returns:
            Dictionary of metrics
        """
        # Forward pass
        discrete_output = discrete_fn(test_inputs, **kwargs)
        smooth_output = smooth_fn(test_inputs, **kwargs)
        
        # Approximation error
        error_metrics = self.compute_approximation_error(
            discrete_output, smooth_output
        )
        
        # Gradient correlation
        grad_metrics = self.compute_gradient_correlation(
            test_inputs, smooth_fn, discrete_fn, **kwargs
        )
        
        # Smoothness
        smoothness_metrics = self.measure_smoothness(
            smooth_fn, test_inputs, **kwargs
        )
        
        # Combine all metrics
        metrics = {
            **error_metrics,
            **grad_metrics,
            **smoothness_metrics
        }
        
        return metrics
    
    def compute_approximation_error(self, 
                                   discrete: torch.Tensor,
                                   smooth: torch.Tensor) -> Dict:
        """
        Compute error metrics between discrete and smooth operations.
        
        Returns:
            L1, L2, L∞ errors and relative errors
        """
        diff = smooth - discrete
        
        # Absolute errors
        l1_error = torch.mean(torch.abs(diff)).item()
        l2_error = torch.sqrt(torch.mean(diff ** 2)).item()
        linf_error = torch.max(torch.abs(diff)).item()
        
        # Relative errors
        eps = 1e-8
        rel_l1 = torch.mean(torch.abs(diff) / (torch.abs(discrete) + eps)).item()
        rel_l2 = torch.sqrt(torch.mean((diff / (discrete + eps)) ** 2)).item()
        
        # Correlation
        correlation = torch.corrcoef(torch.stack([
            discrete.flatten(), 
            smooth.flatten()
        ]))[0, 1].item()
        
        return {
            'l1_error': l1_error,
            'l2_error': l2_error,
            'linf_error': linf_error,
            'relative_l1': rel_l1,
            'relative_l2': rel_l2,
            'correlation': correlation,
        }
    
    def compute_gradient_correlation(self,
                                    inputs: torch.Tensor,
                                    smooth_fn: Callable,
                                    discrete_fn: Callable = None,
                                    **kwargs) -> Dict:
        """
        Measure how well gradients match between approximation and discrete op.
        
        For discrete operations, we use finite differences.
        For smooth operations, we use autograd.
        
        Returns:
            Gradient correlation metrics
        """
        inputs_copy = inputs.clone().detach().requires_grad_(True)
        
        # Smooth gradients (via autograd)
        smooth_output = smooth_fn(inputs_copy, **kwargs)
        smooth_output.sum().backward()
        smooth_grad = inputs_copy.grad.clone()
        
        if discrete_fn is not None:
            # Discrete gradients (via finite differences)
            eps = 1e-4
            discrete_grad = torch.zeros_like(inputs)
            
            for i in range(inputs.shape[0]):
                inputs_plus = inputs.clone()
                inputs_plus[i] += eps
                
                out_plus = discrete_fn(inputs_plus, **kwargs)
                out_orig = discrete_fn(inputs, **kwargs)
                
                discrete_grad[i] = (out_plus[i] - out_orig[i]) / eps
            
            # Correlation between gradients
            grad_corr = torch.corrcoef(torch.stack([
                discrete_grad.flatten(),
                smooth_grad.flatten()
            ]))[0, 1].item()
            
            # Angular difference
            cos_sim = F.cosine_similarity(
                discrete_grad.flatten().unsqueeze(0),
                smooth_grad.flatten().unsqueeze(0)
            ).item()
            
            return {
                'gradient_correlation': grad_corr,
                'gradient_cosine_similarity': cos_sim,
                'gradient_l2_error': torch.norm(discrete_grad - smooth_grad).item(),
            }
        else:
            # No discrete comparison, just measure gradient properties
            return {
                'gradient_norm': torch.norm(smooth_grad).item(),
                'gradient_mean': torch.mean(smooth_grad).item(),
                'gradient_std': torch.std(smooth_grad).item(),
            }
    
    def measure_smoothness(self,
                          fn: Callable,
                          inputs: torch.Tensor,
                          **kwargs) -> Dict:
        """
        Measure smoothness of approximation via Lipschitz constant estimate.
        
        A smooth function has bounded derivatives (Lipschitz continuous).
        We estimate the Lipschitz constant via:
        L = max_{x,y} ||f(x) - f(y)|| / ||x - y||
        
        Returns:
            Smoothness metrics
        """
        num_samples = min(100, inputs.shape[0])
        indices = torch.randperm(inputs.shape[0])[:num_samples]
        samples = inputs[indices]
        
        # Compute pairwise distances
        lipschitz_estimates = []
        
        for i in range(num_samples - 1):
            x1 = samples[i:i+1]
            x2 = samples[i+1:i+2]
            
            y1 = fn(x1, **kwargs)
            y2 = fn(x2, **kwargs)
            
            input_dist = torch.norm(x1 - x2).item()
            output_dist = torch.norm(y1 - y2).item()
            
            if input_dist > 1e-6:
                lipschitz_estimates.append(output_dist / input_dist)
        
        if len(lipschitz_estimates) > 0:
            lipschitz_constant = max(lipschitz_estimates)
            avg_lipschitz = np.mean(lipschitz_estimates)
        else:
            lipschitz_constant = float('inf')
            avg_lipschitz = float('inf')
        
        # Compute second derivative (Hessian trace) for smoothness
        inputs_copy = inputs.clone().detach().requires_grad_(True)
        outputs = fn(inputs_copy, **kwargs)
        
        # First derivative
        grad_outputs = torch.ones_like(outputs)
        first_grad = torch.autograd.grad(
            outputs, inputs_copy,
            grad_outputs=grad_outputs,
            create_graph=True,
            retain_graph=True
        )[0]
        
        # Second derivative (Hessian diagonal)
        second_grad = []
        for i in range(min(10, first_grad.shape[0])):  # Sample for efficiency
            if first_grad[i].requires_grad:
                second = torch.autograd.grad(
                    first_grad[i].sum(), inputs_copy,
                    retain_graph=True,
                    allow_unused=True
                )[0]
                if second is not None:
                    second_grad.append(second[i].abs().mean().item())
        
        if len(second_grad) > 0:
            hessian_trace = np.mean(second_grad)
        else:
            hessian_trace = 0.0
        
        return {
            'lipschitz_constant': lipschitz_constant,
            'avg_lipschitz': avg_lipschitz,
            'hessian_trace_estimate': hessian_trace,
            'is_smooth': lipschitz_constant < 1000,  # Arbitrary threshold
        }


def compute_approximation_error(discrete: torch.Tensor, 
                               smooth: torch.Tensor) -> Dict:
    """
    Convenience function for error computation.
    """
    metrics = ApproximationFidelityMetrics()
    return metrics.compute_approximation_error(discrete, smooth)


def compute_gradient_correlation(inputs: torch.Tensor,
                                smooth_fn: Callable,
                                discrete_fn: Callable = None,
                                **kwargs) -> Dict:
    """
    Convenience function for gradient correlation.
    """
    metrics = ApproximationFidelityMetrics()
    return metrics.compute_gradient_correlation(
        inputs, smooth_fn, discrete_fn, **kwargs
    )


def measure_smoothness(fn: Callable,
                      inputs: torch.Tensor,
                      **kwargs) -> Dict:
    """
    Convenience function for smoothness measurement.
    """
    metrics = ApproximationFidelityMetrics()
    return metrics.measure_smoothness(fn, inputs, **kwargs)


class BenchmarkSuite:
    """
    Comprehensive benchmarking suite for comparing approximation methods.
    """
    
    def __init__(self):
        self.results = {}
        
    def benchmark_xor(self, approximations: Dict[str, Callable]) -> Dict:
        """
        Benchmark XOR approximations.
        
        Args:
            approximations: Dict mapping method names to approximation functions
            
        Returns:
            Benchmark results
        """
        # Generate test data
        num_samples = 1000
        x = torch.rand(num_samples)
        y = torch.rand(num_samples)
        
        # Ground truth (binarized XOR)
        x_bin = (x > 0.5).float()
        y_bin = (y > 0.5).float()
        xor_discrete = (x_bin + y_bin) % 2
        
        results = {}
        metrics_obj = ApproximationFidelityMetrics()
        
        for name, approx_fn in approximations.items():
            xor_smooth = approx_fn(x, y)
            
            # Compute metrics
            error = metrics_obj.compute_approximation_error(
                xor_discrete, xor_smooth
            )
            
            results[name] = error
        
        return results
    
    def benchmark_mod_add(self, approximations: Dict[str, Callable],
                         modulus: int = 256) -> Dict:
        """
        Benchmark modular addition approximations.
        """
        num_samples = 1000
        x = torch.randint(0, modulus, (num_samples,)).float()
        y = torch.randint(0, modulus, (num_samples,)).float()
        
        # Ground truth
        mod_add_discrete = (x + y) % modulus
        
        results = {}
        metrics_obj = ApproximationFidelityMetrics()
        
        for name, approx_fn in approximations.items():
            mod_add_smooth = approx_fn(x, y, modulus)
            
            error = metrics_obj.compute_approximation_error(
                mod_add_discrete, mod_add_smooth
            )
            
            results[name] = error
        
        return results
    
    def generate_report(self) -> str:
        """
        Generate formatted report of benchmark results.
        """
        report = "\n" + "="*70 + "\n"
        report += "APPROXIMATION BENCHMARKING REPORT\n"
        report += "="*70 + "\n\n"
        
        for operation, results in self.results.items():
            report += f"Operation: {operation.upper()}\n"
            report += "-" * 70 + "\n"
            
            for method, metrics in results.items():
                report += f"\n  Method: {method}\n"
                for metric_name, value in metrics.items():
                    if isinstance(value, float):
                        report += f"    {metric_name}: {value:.6f}\n"
                    else:
                        report += f"    {metric_name}: {value}\n"
            
            report += "\n"
        
        return report


if __name__ == "__main__":
    # Test metrics
    print("Testing approximation metrics...\n")
    
    # Generate test data
    discrete = torch.tensor([0., 1., 0., 1., 0.])
    smooth = torch.tensor([0.1, 0.9, 0.2, 0.8, 0.15])
    
    metrics = compute_approximation_error(discrete, smooth)
    
    print("Approximation Error Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")
    
    print("\n✓ Metrics module loaded successfully")