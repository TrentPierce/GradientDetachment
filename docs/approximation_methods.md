# Approximation Methods API

Complete API reference for the approximation framework (`ctdma.approximation`).

## Table of Contents

- [ApproximationBridge](#approximationbridge)
- [Approximation Methods](#approximation-methods)
- [Metrics](#metrics)
- [Convergence Analysis](#convergence-analysis)
- [Utility Functions](#utility-functions)

---

## ApproximationBridge

Unified interface for all approximation methods.

### Factory Function

```python
def create_approximation_bridge(
    method: str,
    **kwargs
) -> ApproximationBridge:
    """
    Create an approximation bridge instance.
    
    Args:
        method: Approximation method name
            - 'sigmoid': Sigmoid approximation
            - 'straight_through' or 'ste': Straight-Through Estimator
            - 'gumbel_softmax' or 'gumbel': Gumbel-Softmax relaxation
            - 'temperature' or 'annealing': Temperature annealing
        **kwargs: Method-specific parameters
    
    Returns:
        ApproximationBridge instance
    
    Example:
        >>> bridge = create_approximation_bridge('sigmoid', steepness=10.0)
        >>> z = bridge.forward(x, y, modulus=2**16)
    """
```

### Base Class

```python
class ApproximationBridge:
    """Base class for approximation methods."""
    
    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        modulus: int
    ) -> torch.Tensor:
        """
        Forward pass with smooth approximation.
        
        Args:
            x: First operand
            y: Second operand
            modulus: Modulus for operation
        
        Returns:
            Smooth approximation of (x + y) mod modulus
        """
    
    def compute_error(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        modulus: int
    ) -> Dict[str, float]:
        """
        Compute approximation error metrics.
        
        Returns:
            Dictionary with L1, L2, Linf errors
        """
```

---

## Approximation Methods

### SigmoidApproximation

Smooth approximation using sigmoid function.

```python
class SigmoidApproximation(ApproximationBridge):
    """
    Sigmoid-based smooth approximation.
    
    Formula: z = x + y - m·σ(β(x + y - m))
    
    where σ(t) = 1/(1 + exp(-t))
    """
    
    def __init__(self, steepness: float = 10.0):
        """
        Initialize sigmoid approximation.
        
        Args:
            steepness: Steepness parameter β (default: 10.0)
                - Higher values: Closer to discrete, larger gradients
                - Lower values: Smoother, smaller gradients
        
        Example:
            >>> approx = SigmoidApproximation(steepness=10.0)
            >>> z = approx.forward(x, y, modulus=2**16)
        """
```

**Properties:**
- **Smoothness**: Differentiable everywhere
- **Boundary Error**: O(m·β) at wrap-around
- **Gradient**: Continuous but large near discontinuities

**Use Case:** When you need smooth gradients and can tolerate boundary errors

### StraightThroughEstimator

Zero forward error, biased gradients.

```python
class StraightThroughEstimator(ApproximationBridge):
    """
    Straight-Through Estimator (STE).
    
    Forward: z = (x + y) mod m (exact)
    Backward: ∂z/∂x = 1 (identity)
    """
    
    def __init__(self):
        """
        Initialize STE.
        
        Example:
            >>> ste = StraightThroughEstimator()
            >>> z = ste.forward(x, y, modulus=2**16)
        """
```

**Properties:**
- **Forward Error**: Zero (uses exact operation)
- **Backward Error**: High (identity gradient)
- **Gradient**: Biased but bounded

**Use Case:** When forward accuracy is critical and gradient bias is acceptable

### GumbelSoftmaxApproximation

Stochastic relaxation with unbiased gradients.

```python
class GumbelSoftmaxApproximation(ApproximationBridge):
    """
    Gumbel-Softmax continuous relaxation.
    
    z = softmax((log(π) + g) / τ)
    
    where g ~ Gumbel(0, 1)
    """
    
    def __init__(self, temperature: float = 0.5, hard: bool = False):
        """
        Initialize Gumbel-Softmax approximation.
        
        Args:
            temperature: Temperature parameter τ (default: 0.5)
                - Lower: Closer to discrete (sharper)
                - Higher: Smoother (more uniform)
            hard: Use hard (straight-through) during forward
        
        Example:
            >>> gumbel = GumbelSoftmaxApproximation(temperature=0.5)
            >>> z = gumbel.forward(x, y, modulus=2**16)
        """
```

**Properties:**
- **Stochasticity**: Adds noise for exploration
- **Gradient**: Unbiased in expectation
- **Convergence**: Converges to discrete as τ → 0

**Use Case:** When unbiased gradient estimates are needed

### TemperatureAnnealing

Controllable smoothness with annealing.

```python
class TemperatureAnnealing(ApproximationBridge):
    """
    Temperature-based annealing approximation.
    
    z(τ) = x + y - m·σ((x+y-m)/τ)
    """
    
    def __init__(
        self,
        initial_temp: float = 1.0,
        final_temp: float = 0.01,
        schedule: str = 'exponential'
    ):
        """
        Initialize temperature annealing.
        
        Args:
            initial_temp: Starting temperature (smoother)
            final_temp: Ending temperature (sharper)
            schedule: Annealing schedule
                - 'exponential': τ(t) = τ₀ · exp(-λt)
                - 'linear': τ(t) = τ₀ - (τ₀-τ_f)·t/T
                - 'cosine': τ(t) = τ_f + (τ₀-τ_f)·(1+cos(πt/T))/2
        
        Example:
            >>> annealing = TemperatureAnnealing(
            ...     initial_temp=1.0, 
            ...     final_temp=0.01,
            ...     schedule='exponential'
            ... )
            >>> z = annealing.forward(x, y, modulus=2**16)
        """
    
    def set_temperature(self, temp: float):
        """Update current temperature."""
    
    def step(self):
        """Advance one step in annealing schedule."""
```

**Properties:**
- **Flexibility**: Smooth transition from continuous to discrete
- **Control**: Explicit control over approximation quality
- **Convergence**: Guaranteed with proper schedule

**Use Case:** When you want gradual transition from smooth to discrete

---

## Metrics

Comprehensive quality metrics for approximations.

### ApproximationMetrics

```python
class ApproximationMetrics:
    """Compute quality metrics for approximations."""
    
    @staticmethod
    def compute_error_metrics(
        discrete: torch.Tensor,
        smooth: torch.Tensor
    ) -> Dict[str, float]:
        """
        Compute error metrics.
        
        Returns:
            - 'l1_error': Mean absolute error
            - 'l2_error': Root mean squared error
            - 'linf_error': Maximum absolute error
            - 'relative_error': Relative error (%)
            - 'correlation': Pearson correlation
        
        Example:
            >>> metrics = ApproximationMetrics.compute_error_metrics(
            ...     discrete_output, smooth_output
            ... )
            >>> print(f"L2 error: {metrics['l2_error']:.4f}")
        """
    
    @staticmethod
    def compute_gradient_fidelity(
        discrete_grads: torch.Tensor,
        smooth_grads: torch.Tensor
    ) -> Dict[str, float]:
        """
        Compute gradient fidelity metrics.
        
        Returns:
            - 'cosine_similarity': Gradient direction agreement
            - 'magnitude_ratio': Gradient magnitude ratio
            - 'angular_error': Angular error in degrees
            - 'sign_agreement': Fraction with same sign
        
        Example:
            >>> fidelity = ApproximationMetrics.compute_gradient_fidelity(
            ...     grad_discrete, grad_smooth
            ... )
            >>> print(f"Cosine similarity: {fidelity['cosine_similarity']:.3f}")
        """
    
    @staticmethod
    def compute_information_metrics(
        discrete: torch.Tensor,
        smooth: torch.Tensor,
        num_bins: int = 50
    ) -> Dict[str, float]:
        """
        Compute information-theoretic metrics.
        
        Returns:
            - 'discrete_entropy': H(discrete) in bits
            - 'smooth_entropy': H(smooth) in bits
            - 'mutual_information': I(discrete; smooth)
            - 'kl_divergence': D_KL(discrete || smooth)
            - 'js_divergence': JS divergence (symmetric)
        
        Example:
            >>> info = ApproximationMetrics.compute_information_metrics(
            ...     discrete_output, smooth_output
            ... )
            >>> print(f"Information loss: {info['discrete_entropy'] - info['smooth_entropy']:.3f} bits")
        """
```

### Comparison Functions

```python
def compare_approximation_methods(
    discrete_operation: Callable,
    approximations: Dict[str, ApproximationBridge],
    test_data: Tuple[torch.Tensor, torch.Tensor],
    modulus: int = 2**16
) -> Dict[str, Dict[str, float]]:
    """
    Compare multiple approximation methods.
    
    Args:
        discrete_operation: Exact discrete operation
        approximations: Dictionary of {name: approximation}
        test_data: (x, y) test inputs
        modulus: Modulus for operations
    
    Returns:
        Nested dictionary: {method_name: {metric_name: value}}
    
    Example:
        >>> methods = {
        ...     'sigmoid': SigmoidApproximation(steepness=10.0),
        ...     'ste': StraightThroughEstimator(),
        ...     'gumbel': GumbelSoftmaxApproximation(temperature=0.5)
        ... }
        >>> results = compare_approximation_methods(
        ...     modular_add_exact, methods, (x, y)
        ... )
        >>> for method, metrics in results.items():
        ...     print(f"{method}: L2={metrics['l2_error']:.4f}")
    """
```

---

## Convergence Analysis

Tools for analyzing convergence properties.

### ConvergenceAnalyzer

```python
class ConvergenceAnalyzer:
    """Analyze convergence of approximation methods."""
    
    def estimate_convergence_rate(
        self,
        loss_trajectory: List[float],
        method: str = 'exponential_fit'
    ) -> float:
        """
        Estimate convergence rate from loss trajectory.
        
        Args:
            loss_trajectory: List of loss values over time
            method: Estimation method
                - 'exponential_fit': Fit L(t) = L∞ + A·exp(-λt)
                - 'linear_regression': Linear fit on log scale
        
        Returns:
            Convergence rate λ (higher = faster)
        
        Example:
            >>> analyzer = ConvergenceAnalyzer()
            >>> rate = analyzer.estimate_convergence_rate(training_losses)
            >>> print(f"Convergence rate: {rate:.4f}")
        """
    
    def analyze_bias_variance_tradeoff(
        self,
        approximation: ApproximationBridge,
        num_trials: int = 100
    ) -> Dict[str, float]:
        """
        Analyze bias-variance tradeoff.
        
        Args:
            approximation: Approximation method to analyze
            num_trials: Number of trials for variance estimation
        
        Returns:
            Dictionary with:
                - 'bias': Systematic error (mean)
                - 'variance': Random error (variance)
                - 'mse': Mean squared error (bias² + variance)
        
        Example:
            >>> tradeoff = analyzer.analyze_bias_variance_tradeoff(
            ...     gumbel_approx, num_trials=100
            ... )
            >>> print(f"Bias: {tradeoff['bias']:.4f}, Variance: {tradeoff['variance']:.4f}")
        """
    
    def create_annealing_schedule(
        self,
        schedule_type: str,
        initial_temp: float,
        final_temp: float,
        num_steps: int
    ) -> List[float]:
        """
        Create temperature annealing schedule.
        
        Args:
            schedule_type: 'exponential', 'linear', or 'cosine'
            initial_temp: Starting temperature
            final_temp: Ending temperature
            num_steps: Number of annealing steps
        
        Returns:
            List of temperatures for each step
        
        Example:
            >>> schedule = analyzer.create_annealing_schedule(
            ...     'exponential', initial_temp=1.0, 
            ...     final_temp=0.01, num_steps=1000
            ... )
            >>> for step, temp in enumerate(schedule):
            ...     approximation.set_temperature(temp)
            ...     # ... train ...
        """
```

---

## Utility Functions

### Discrete Operations

```python
def modular_add_exact(
    x: torch.Tensor,
    y: torch.Tensor,
    modulus: int
) -> torch.Tensor:
    """Exact modular addition: (x + y) mod m"""

def modular_subtract_exact(
    x: torch.Tensor,
    y: torch.Tensor,
    modulus: int
) -> torch.Tensor:
    """Exact modular subtraction: (x - y) mod m"""

def circular_shift_exact(
    x: torch.Tensor,
    shift: int,
    bit_width: int
) -> torch.Tensor:
    """Exact circular shift (rotation)"""

def xor_exact(
    x: torch.Tensor,
    y: torch.Tensor
) -> torch.Tensor:
    """Exact XOR operation"""
```

### Gradient Utilities

```python
def compute_gradient_similarity(
    grad1: torch.Tensor,
    grad2: torch.Tensor
) -> float:
    """
    Compute cosine similarity between gradients.
    
    Returns:
        Similarity in [-1, 1] (1 = same direction)
    """

def analyze_gradient_flow(
    model: torch.nn.Module,
    loss: torch.Tensor
) -> Dict[str, Any]:
    """
    Analyze gradient flow through model.
    
    Returns:
        Statistics about gradient magnitudes and distributions
    """
```

---

## Best Practices

### Choosing an Approximation Method

| Method | Best For | Avoid When |
|--------|----------|------------|
| **Sigmoid** | Smooth optimization, general use | Need exact forward pass |
| **STE** | Forward accuracy critical | Need unbiased gradients |
| **Gumbel-Softmax** | Unbiased estimates, exploration | Deterministic required |
| **Temperature** | Gradual transition, fine control | Computational budget limited |

### Parameter Tuning

**Sigmoid Steepness:**
- β = 1: Very smooth, large errors
- β = 10: Balanced (recommended)
- β = 50: Sharp, numerical issues

**Temperature:**
- τ = 1.0: Very smooth
- τ = 0.1: Moderate sharpness
- τ = 0.01: Nearly discrete

**Annealing:**
- Exponential: Fast initial cooling
- Linear: Uniform progress
- Cosine: Smooth transition

---

## Performance Considerations

### Memory Usage

- Sigmoid: O(n) - same as input
- STE: O(n) - custom autograd
- Gumbel: O(n) - additional sampling
- Temperature: O(n) - schedule storage

### Computational Cost

- Sigmoid: 1.2x discrete operation
- STE: 1.0x (exact forward)
- Gumbel: 1.5x (sampling overhead)
- Temperature: 1.2x (schedule lookup)

---

## Examples

### Complete Comparison

```python
import torch
from ctdma.approximation.bridge import create_approximation_bridge
from ctdma.approximation.metrics import compare_approximation_methods

# Generate test data
x = torch.randint(0, 2**16, (1000,))
y = torch.randint(0, 2**16, (1000,))

# Create approximations
methods = {
    'sigmoid_10': create_approximation_bridge('sigmoid', steepness=10.0),
    'sigmoid_20': create_approximation_bridge('sigmoid', steepness=20.0),
    'ste': create_approximation_bridge('straight_through'),
    'gumbel': create_approximation_bridge('gumbel_softmax', temperature=0.5),
    'annealing': create_approximation_bridge('temperature', initial_temp=1.0)
}

# Compare
results = compare_approximation_methods(
    discrete_operation=lambda x, y, m: (x + y) % m,
    approximations=methods,
    test_data=(x, y),
    modulus=2**16
)

# Print results
for method, metrics in results.items():
    print(f"\n{method}:")
    print(f"  L2 error: {metrics['l2_error']:.4f}")
    print(f"  Gradient cosine similarity: {metrics['cosine_similarity']:.3f}")
    print(f"  Information loss: {metrics['information_loss']:.3f} bits")
```

---

## See Also

- [Mathematical Theory API](mathematical_theory.md)
- [Cipher Implementations API](cipher_implementations.md)
- [Example: Approximation Comparison](../examples/approximation_comparison.ipynb)

---

*Last updated: January 30, 2026*
