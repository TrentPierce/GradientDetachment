"""
Convergence Analysis for Different Approximation Methods

Analyzes how approximation precision affects gradient flow and convergence properties.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, Dict, List, Tuple, Optional
from .bridge import ApproximationBridge
from .metrics import ApproximationMetrics


class ConvergenceAnalyzer:
    """
    Analyzes convergence properties of different approximation methods.
    
    Key analyses:
    1. Convergence speed vs. temperature
    2. Gradient stability across temperatures
    3. Final accuracy vs. approximation fidelity
    4. Critical temperature thresholds
    """
    
    def __init__(self, approximation: ApproximationBridge):
        self.approximation = approximation
        self.metrics = ApproximationMetrics()
        
    def analyze_temperature_sweep(self, 
                                  operation: Callable,
                                  test_inputs: torch.Tensor,
                                  temperatures: List[float],
                                  discrete_op: Optional[Callable] = None) -> Dict[str, List[float]]:
        """
        Analyze approximation quality across temperature range.
        
        Tests how temperature parameter affects:
        - Approximation fidelity
        - Gradient stability
        - Lipschitz constant
        
        Args:
            operation: Smooth operation with temperature parameter
            test_inputs: Test input samples
            temperatures: Temperature values to test
            discrete_op: Discrete operation for comparison (optional)
            
        Returns:
            Dictionary of metrics vs. temperature
        """
        fidelities = []
        lipschitz_constants = []
        gradient_variances = []
        gradient_norms = []
        
        for temp in temperatures:
            self.approximation.set_temperature(temp)
            
            # Compute approximation output
            approx_output = operation(test_inputs)
            
            # Fidelity (if discrete operation provided)
            if discrete_op is not None:
                with torch.no_grad():
                    discrete_output = discrete_op(test_inputs)
                fidelity = self.metrics.compute_fidelity(discrete_output, approx_output)
                fidelities.append(fidelity)
            
            # Lipschitz constant
            lipschitz = self.metrics.compute_lipschitz_constant(operation, test_inputs)
            lipschitz_constants.append(lipschitz)
            
            # Gradient statistics
            grad_stats = self.metrics.compute_gradient_variance(test_inputs, operation)
            gradient_variances.append(grad_stats['variance'])
            gradient_norms.append(abs(grad_stats['mean_gradient']))
        
        results = {
            'temperatures': temperatures,
            'lipschitz_constants': lipschitz_constants,
            'gradient_variances': gradient_variances,
            'gradient_norms': gradient_norms
        }
        
        if fidelities:
            results['fidelities'] = fidelities
        
        return results
    
    def analyze_convergence_trajectory(self,
                                      loss_fn: Callable,
                                      initial_params: torch.Tensor,
                                      num_steps: int = 100,
                                      learning_rate: float = 0.01) -> Dict[str, List[float]]:
        """
        Analyze convergence trajectory with current approximation.
        
        Tracks:
        - Loss over time
        - Gradient norm over time
        - Parameter distance from initial
        - Convergence metrics
        
        Args:
            loss_fn: Loss function to optimize
            initial_params: Starting parameters
            num_steps: Number of optimization steps
            learning_rate: Optimization learning rate
            
        Returns:
            Trajectory statistics
        """
        params = initial_params.clone().requires_grad_(True)
        optimizer = torch.optim.SGD([params], lr=learning_rate)
        
        losses = []
        gradient_norms = []
        param_distances = []
        
        for step in range(num_steps):
            optimizer.zero_grad()
            loss = loss_fn(params)
            loss.backward()
            
            # Record metrics
            losses.append(loss.item())
            gradient_norms.append(params.grad.norm().item())
            param_distances.append((params - initial_params).norm().item())
            
            optimizer.step()
        
        return {
            'losses': losses,
            'gradient_norms': gradient_norms,
            'param_distances': param_distances,
            'converged': losses[-1] < losses[0] * 0.1,
            'convergence_rate': self._compute_convergence_rate(losses)
        }
    
    def _compute_convergence_rate(self, losses: List[float]) -> float:
        """
        Estimate convergence rate from loss trajectory.
        
        Fits exponential decay: loss(t) = loss(0) * exp(-r * t)
        
        Args:
            losses: Loss values over time
            
        Returns:
            Convergence rate r
        """
        losses = np.array(losses)
        if np.any(losses <= 0):
            return 0.0
        
        log_losses = np.log(losses)
        t = np.arange(len(losses))
        
        try:
            slope, _ = np.polyfit(t, log_losses, 1)
            return -slope
        except:
            return 0.0
    
    def find_critical_temperature(self,
                                 operation: Callable,
                                 test_inputs: torch.Tensor,
                                 metric: str = 'lipschitz',
                                 threshold: float = 100.0,
                                 temp_range: Tuple[float, float] = (0.1, 20.0),
                                 num_samples: int = 20) -> float:
        """
        Find critical temperature where approximation quality degrades.
        
        Critical temperature is where:
        - Lipschitz constant exceeds threshold
        - Gradient variance explodes
        - Fidelity drops below acceptable level
        
        Args:
            operation: Operation to analyze
            test_inputs: Test inputs
            metric: Metric to use ('lipschitz', 'variance', 'fidelity')
            threshold: Threshold value
            temp_range: Range of temperatures to search
            num_samples: Number of temperature samples
            
        Returns:
            Critical temperature
        """
        temperatures = np.linspace(temp_range[0], temp_range[1], num_samples)
        
        for temp in temperatures:
            self.approximation.set_temperature(temp)
            
            if metric == 'lipschitz':
                value = self.metrics.compute_lipschitz_constant(operation, test_inputs)
            elif metric == 'variance':
                grad_stats = self.metrics.compute_gradient_variance(test_inputs, operation)
                value = grad_stats['variance']
            else:
                raise ValueError(f"Unknown metric: {metric}")
            
            if value > threshold:
                return temp
        
        return temp_range[1]  # No critical point found
    
    def compare_approximations(self,
                              approximations: Dict[str, ApproximationBridge],
                              test_inputs: torch.Tensor,
                              loss_fn: Callable,
                              num_steps: int = 50) -> Dict[str, Dict]:
        """
        Compare multiple approximation methods.
        
        Args:
            approximations: Dictionary of {name: approximation}
            test_inputs: Test inputs
            loss_fn: Loss function for convergence test
            num_steps: Training steps
            
        Returns:
            Comparison results
        """
        results = {}
        
        for name, approx in approximations.items():
            # Save current approximation
            original_approx = self.approximation
            self.approximation = approx
            
            # Analyze convergence
            initial_params = torch.randn_like(test_inputs[0]).requires_grad_(True)
            trajectory = self.analyze_convergence_trajectory(
                loss_fn, initial_params, num_steps
            )
            
            # Compute metrics
            grad_stats = self.metrics.compute_gradient_variance(test_inputs, 
                lambda x: loss_fn(x))
            
            results[name] = {
                'final_loss': trajectory['losses'][-1],
                'convergence_rate': trajectory['convergence_rate'],
                'converged': trajectory['converged'],
                'gradient_variance': grad_stats['variance'],
                'gradient_stability': 1.0 / (grad_stats['coefficient_variation'] + 1e-10)
            }
            
            # Restore original approximation
            self.approximation = original_approx
        
        return results


def analyze_convergence_properties(approximation: ApproximationBridge,
                                  operation: Callable,
                                  test_inputs: torch.Tensor,
                                  temperatures: Optional[List[float]] = None) -> Dict:
    """
    Comprehensive convergence analysis.
    
    Args:
        approximation: Approximation method
        operation: Operation to analyze
        test_inputs: Test inputs
        temperatures: Temperature values to test (optional)
        
    Returns:
        Complete analysis results
    """
    analyzer = ConvergenceAnalyzer(approximation)
    
    if temperatures is None:
        temperatures = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]
    
    # Temperature sweep
    temp_results = analyzer.analyze_temperature_sweep(
        operation, test_inputs, temperatures
    )
    
    # Find critical temperature
    critical_temp = analyzer.find_critical_temperature(
        operation, test_inputs, metric='lipschitz', threshold=1000.0
    )
    
    return {
        'temperature_analysis': temp_results,
        'critical_temperature': critical_temp,
        'summary': {
            'stable_range': (temperatures[0], critical_temp),
            'recommended_temperature': critical_temp * 0.5,
            'max_lipschitz': max(temp_results['lipschitz_constants']),
            'max_variance': max(temp_results['gradient_variances'])
        }
    }


def plot_convergence_curves(results: Dict[str, Dict], 
                           save_path: Optional[str] = None):
    """
    Plot convergence curves for multiple approximation methods.
    
    Args:
        results: Results from compare_approximations
        save_path: Path to save plot (optional)
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    methods = list(results.keys())
    
    # Extract metrics
    final_losses = [results[m]['final_loss'] for m in methods]
    convergence_rates = [results[m]['convergence_rate'] for m in methods]
    gradient_variances = [results[m]['gradient_variance'] for m in methods]
    gradient_stabilities = [results[m]['gradient_stability'] for m in methods]
    
    # Plot 1: Final Loss
    axes[0, 0].bar(methods, final_losses)
    axes[0, 0].set_ylabel('Final Loss')
    axes[0, 0].set_title('Final Loss Comparison')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # Plot 2: Convergence Rate
    axes[0, 1].bar(methods, convergence_rates)
    axes[0, 1].set_ylabel('Convergence Rate')
    axes[0, 1].set_title('Convergence Rate Comparison')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # Plot 3: Gradient Variance
    axes[1, 0].bar(methods, gradient_variances)
    axes[1, 0].set_ylabel('Gradient Variance')
    axes[1, 0].set_title('Gradient Variance Comparison')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # Plot 4: Gradient Stability
    axes[1, 1].bar(methods, gradient_stabilities)
    axes[1, 1].set_ylabel('Gradient Stability')
    axes[1, 1].set_title('Gradient Stability Comparison')
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
