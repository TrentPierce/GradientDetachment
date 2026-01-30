"""
Convergence Analysis Module

Analyzes convergence properties of different approximation methods,
including bias, variance, and convergence rates.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Callable, Optional
from scipy import stats
from dataclasses import dataclass


@dataclass
class ConvergenceResults:
    """Results from convergence analysis."""
    converged: bool
    convergence_rate: float
    bias: float
    variance: float
    iterations_to_converge: int
    final_error: float
    trajectory: List[float]


class ConvergenceAnalyzer:
    """
    Analyzes convergence properties of approximation methods.
    
    Studies:
    1. Convergence rate (exponential, polynomial, etc.)
    2. Bias (systematic error)
    3. Variance (random fluctuations)
    4. Asymptotic behavior
    """
    
    def __init__(self, tolerance: float = 1e-4, max_iterations: int = 10000):
        self.tolerance = tolerance
        self.max_iterations = max_iterations
        
    def analyze_convergence(
        self,
        approximation_fn: Callable,
        discrete_fn: Callable,
        input_generator: Callable,
        n_samples: int = 1000,
        parameter_schedule: Optional[List] = None
    ) -> ConvergenceResults:
        """
        Analyze convergence of an approximation method.
        
        Args:
            approximation_fn: Approximation function (parameterized)
            discrete_fn: True discrete function
            input_generator: Function to generate random inputs
            n_samples: Number of samples per iteration
            parameter_schedule: Schedule of approximation parameters (e.g., temperature)
            
        Returns:
            Convergence analysis results
        """
        if parameter_schedule is None:
            # Default: exponential annealing
            parameter_schedule = [10.0 * (0.95 ** i) for i in range(self.max_iterations)]
        
        errors = []
        biases = []
        variances = []
        
        converged = False
        convergence_iteration = self.max_iterations
        
        for iter_idx, param in enumerate(parameter_schedule):
            # Generate samples
            inputs = input_generator(n_samples)
            
            # Compute discrete and approximate outputs
            with torch.no_grad():
                discrete_outputs = []
                approx_outputs = []
                
                for inp in inputs:
                    if isinstance(inp, tuple):
                        x, y = inp
                    else:
                        x = inp
                        y = None
                    
                    discrete_out = discrete_fn(x, y) if y is not None else discrete_fn(x)
                    approx_out = approximation_fn(x, y, param) if y is not None else approximation_fn(x, param)
                    
                    discrete_outputs.append(discrete_out)
                    approx_outputs.append(approx_out)
                
                discrete_outputs = torch.stack(discrete_outputs)
                approx_outputs = torch.stack(approx_outputs)
            
            # Compute error metrics
            error = torch.abs(discrete_outputs - approx_outputs).mean().item()
            bias = (approx_outputs - discrete_outputs).mean().item()
            variance = torch.var(approx_outputs - discrete_outputs).item()
            
            errors.append(error)
            biases.append(bias)
            variances.append(variance)
            
            # Check convergence
            if error < self.tolerance and not converged:
                converged = True
                convergence_iteration = iter_idx
            
            # Early stopping if we have enough data
            if iter_idx > 100 and iter_idx % 100 == 0:
                # Check if error is plateauing
                recent_errors = errors[-100:]
                if np.std(recent_errors) < self.tolerance / 10:
                    break
        
        # Estimate convergence rate
        convergence_rate = self._estimate_convergence_rate(errors)
        
        return ConvergenceResults(
            converged=converged,
            convergence_rate=convergence_rate,
            bias=biases[-1] if biases else 0.0,
            variance=variances[-1] if variances else 0.0,
            iterations_to_converge=convergence_iteration,
            final_error=errors[-1] if errors else float('inf'),
            trajectory=errors
        )
    
    def _estimate_convergence_rate(self, errors: List[float]) -> float:
        """
        Estimate convergence rate from error trajectory.
        
        Fits exponential: error(t) = A * exp(-λt)
        Returns λ (convergence rate).
        
        Args:
            errors: List of errors over iterations
            
        Returns:
            Estimated convergence rate λ
        """
        if len(errors) < 10:
            return 0.0
        
        # Take log of errors (for exponential fit)
        errors_array = np.array(errors)
        errors_array = np.maximum(errors_array, 1e-10)  # Avoid log(0)
        log_errors = np.log(errors_array)
        
        # Fit linear model: log(error) = log(A) - λt
        t = np.arange(len(errors))
        
        # Use robust regression (in case of outliers)
        try:
            slope, intercept, r_value, p_value, std_err = stats.linregress(t, log_errors)
            convergence_rate = -slope  # Negative slope = convergence rate
            return max(0.0, convergence_rate)
        except:
            return 0.0
    
    def analyze_bias_variance_tradeoff(
        self,
        approximation_fn: Callable,
        discrete_fn: Callable,
        input_generator: Callable,
        parameter_values: List[float],
        n_trials: int = 100
    ) -> Dict[str, List[float]]:
        """
        Analyze bias-variance tradeoff for different parameter values.
        
        For each parameter value (e.g., temperature, steepness):
        - Compute bias (systematic error)
        - Compute variance (random fluctuations)
        - Total error = bias^2 + variance
        
        Args:
            approximation_fn: Approximation function
            discrete_fn: True discrete function
            input_generator: Input generator
            parameter_values: List of parameter values to test
            n_trials: Number of trials per parameter
            
        Returns:
            Dictionary with bias, variance, and total error for each parameter
        """
        biases = []
        variances = []
        total_errors = []
        
        for param in parameter_values:
            trial_outputs = []
            discrete_outputs_ref = None
            
            for trial in range(n_trials):
                # Generate same inputs but with different random seeds (for stochastic methods)
                inputs = input_generator(100)
                
                with torch.no_grad():
                    discrete_outs = []
                    approx_outs = []
                    
                    for inp in inputs:
                        if isinstance(inp, tuple):
                            x, y = inp
                        else:
                            x = inp
                            y = None
                        
                        discrete_out = discrete_fn(x, y) if y is not None else discrete_fn(x)
                        approx_out = approximation_fn(x, y, param) if y is not None else approximation_fn(x, param)
                        
                        discrete_outs.append(discrete_out)
                        approx_outs.append(approx_out)
                    
                    discrete_outputs = torch.stack(discrete_outs)
                    approx_outputs = torch.stack(approx_outs)
                
                if discrete_outputs_ref is None:
                    discrete_outputs_ref = discrete_outputs
                
                trial_outputs.append(approx_outputs)
            
            # Stack all trials
            trial_outputs = torch.stack(trial_outputs)  # (n_trials, n_samples)
            
            # Compute bias: E[f̂(x)] - f(x)
            expected_approx = trial_outputs.mean(dim=0)
            bias = (expected_approx - discrete_outputs_ref).mean().item()
            
            # Compute variance: Var[f̂(x)]
            variance = trial_outputs.var(dim=0).mean().item()
            
            # Total error: bias^2 + variance
            total_error = bias**2 + variance
            
            biases.append(bias)
            variances.append(variance)
            total_errors.append(total_error)
        
        return {
            'parameters': parameter_values,
            'bias': biases,
            'variance': variances,
            'total_error': total_errors,
            'bias_squared': [b**2 for b in biases]
        }
    
    def analyze_temperature_annealing(
        self,
        approximation_fn: Callable,
        discrete_fn: Callable,
        input_generator: Callable,
        initial_temp: float = 10.0,
        final_temp: float = 0.1,
        n_steps: int = 100,
        anneal_schedule: str = 'exponential'
    ) -> Dict[str, any]:
        """
        Analyze convergence under temperature annealing.
        
        Args:
            approximation_fn: Temperature-dependent approximation
            discrete_fn: True discrete function
            input_generator: Input generator
            initial_temp: Starting temperature
            final_temp: Final temperature
            n_steps: Number of annealing steps
            anneal_schedule: 'exponential', 'linear', or 'cosine'
            
        Returns:
            Annealing analysis results
        """
        # Generate temperature schedule
        if anneal_schedule == 'exponential':
            temperatures = [initial_temp * (final_temp / initial_temp) ** (i / n_steps) 
                          for i in range(n_steps)]
        elif anneal_schedule == 'linear':
            temperatures = [initial_temp - (initial_temp - final_temp) * (i / n_steps) 
                          for i in range(n_steps)]
        elif anneal_schedule == 'cosine':
            temperatures = [final_temp + 0.5 * (initial_temp - final_temp) * 
                          (1 + np.cos(np.pi * i / n_steps)) 
                          for i in range(n_steps)]
        else:
            raise ValueError(f"Unknown anneal schedule: {anneal_schedule}")
        
        # Analyze convergence
        results = self.analyze_convergence(
            approximation_fn,
            discrete_fn,
            input_generator,
            n_samples=100,
            parameter_schedule=temperatures
        )
        
        return {
            'convergence_results': results,
            'temperature_schedule': temperatures,
            'anneal_schedule': anneal_schedule,
            'converged': results.converged,
            'final_temperature': temperatures[-1],
            'convergence_rate': results.convergence_rate
        }


def analyze_approximation_convergence(
    approximation_fn: Callable,
    discrete_fn: Callable,
    input_generator: Callable,
    n_samples: int = 1000
) -> ConvergenceResults:
    """
    Convenience function to analyze approximation convergence.
    
    Args:
        approximation_fn: Approximation function
        discrete_fn: Discrete function
        input_generator: Function to generate inputs
        n_samples: Number of samples
        
    Returns:
        Convergence results
    """
    analyzer = ConvergenceAnalyzer()
    return analyzer.analyze_convergence(
        approximation_fn,
        discrete_fn,
        input_generator,
        n_samples
    )


def compute_approximation_bias(
    approximation_outputs: torch.Tensor,
    discrete_outputs: torch.Tensor
) -> float:
    """
    Compute bias of approximation.
    
    Bias = E[f̂(x) - f(x)]
    
    Args:
        approximation_outputs: Outputs from approximation
        discrete_outputs: True discrete outputs
        
    Returns:
        Bias value
    """
    return (approximation_outputs - discrete_outputs).mean().item()


def estimate_convergence_rate(errors: List[float]) -> float:
    """
    Estimate convergence rate from error trajectory.
    
    Args:
        errors: List of errors over time
        
    Returns:
        Estimated convergence rate
    """
    analyzer = ConvergenceAnalyzer()
    return analyzer._estimate_convergence_rate(errors)


def compare_convergence_properties(
    approximations: Dict[str, Callable],
    discrete_fn: Callable,
    input_generator: Callable,
    n_samples: int = 1000
) -> Dict[str, ConvergenceResults]:
    """
    Compare convergence properties of multiple approximation methods.
    
    Args:
        approximations: Dictionary of approximation functions
        discrete_fn: True discrete function
        input_generator: Input generator
        n_samples: Number of samples
        
    Returns:
        Dictionary mapping method names to convergence results
    """
    analyzer = ConvergenceAnalyzer()
    results = {}
    
    for name, approx_fn in approximations.items():
        print(f"Analyzing {name}...")
        results[name] = analyzer.analyze_convergence(
            approx_fn,
            discrete_fn,
            input_generator,
            n_samples
        )
    
    return results
