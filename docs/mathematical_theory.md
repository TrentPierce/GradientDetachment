# Mathematical Theory API

Comprehensive API reference for the mathematical analysis module (`ctdma.theory`).

## Table of Contents

- [GradientInversionAnalyzer](#gradientinversionanalyzer)
- [SawtoothTopologyAnalyzer](#sawtoothto pologyanalyzer)
- [InformationTheoreticAnalyzer](#informationtheoreticanalyzer)
- [Theorem Classes](#theorem-classes)
- [Utility Functions](#utility-functions)

---

## GradientInversionAnalyzer

Analyzes gradient discontinuities and inversion probabilities in ARX operations.

### Class Definition

```python
class GradientInversionAnalyzer:
    """Analyzer for gradient inversion phenomena in ARX ciphers."""
    
    def __init__(self, device: str = 'cpu'):
        """
        Initialize gradient inversion analyzer.
        
        Args:
            device: Computing device ('cpu' or 'cuda')
        """
```

### Methods

#### compute_gradient_discontinuity

```python
def compute_gradient_discontinuity(
    self,
    x: torch.Tensor,
    y: torch.Tensor,
    modulus: int = 2**16,
    steepness: float = 10.0
) -> Dict[str, float]:
    """
    Compute gradient discontinuity at modular wrap-around points.
    
    Analyzes the gradient error between exact modular addition and smooth
    sigmoid approximation at points where (x + y) >= modulus.
    
    Args:
        x: Input tensor of shape (batch_size,) or (batch_size, dim)
        y: Input tensor of same shape as x
        modulus: Modulus for modular arithmetic (default: 2^16)
        steepness: Steepness parameter β for sigmoid approximation
    
    Returns:
        Dictionary containing:
            - 'max_discontinuity': Maximum gradient jump
            - 'avg_discontinuity': Average gradient jump
            - 'num_discontinuities': Number of wrap-around points
            - 'gradient_error': RMS gradient error
    
    Example:
        >>> analyzer = GradientInversionAnalyzer()
        >>> x = torch.randint(0, 2**16, (1000,))
        >>> y = torch.randint(0, 2**16, (1000,))
        >>> results = analyzer.compute_gradient_discontinuity(x, y)
        >>> print(f"Max discontinuity: {results['max_discontinuity']:.2f}")
    """
```

#### estimate_inversion_probability

```python
def estimate_inversion_probability(
    self,
    x: torch.Tensor,
    y: torch.Tensor,
    num_rounds: int = 1,
    modulus: int = 2**16
) -> float:
    """
    Estimate probability of gradient inversion for k rounds.
    
    Computes both theoretical prediction and empirical measurement:
    P_inv >= 1 - (1 - 1/m)^k
    
    Args:
        x: Input tensor
        y: Input tensor
        num_rounds: Number of cipher rounds
        modulus: Modulus for operations
    
    Returns:
        Estimated inversion probability (0 to 1)
    
    Example:
        >>> prob = analyzer.estimate_inversion_probability(
        ...     x, y, num_rounds=2, modulus=2**16
        ... )
        >>> print(f"Inversion probability: {prob:.2%}")
    """
```

#### analyze_cipher

```python
def analyze_cipher(
    self,
    cipher,
    num_samples: int = 1000,
    modulus: int = 2**16
) -> Dict[str, Any]:
    """
    Comprehensive analysis of a cipher implementation.
    
    Args:
        cipher: Cipher instance implementing BaseCipher interface
        num_samples: Number of samples for analysis
        modulus: Modulus for operations
    
    Returns:
        Dictionary with comprehensive analysis results:
            - 'inversion_probability': Estimated P_inv
            - 'max_discontinuity': Maximum gradient jump
            - 'gradient_error': Average gradient error
            - 'num_rounds': Number of cipher rounds
    
    Example:
        >>> from ctdma.ciphers import create_cipher
        >>> cipher = create_cipher('speck', num_rounds=4)
        >>> results = analyzer.analyze_cipher(cipher)
        >>> print(results)
    """
```

---

## SawtoothTopologyAnalyzer

Analyzes loss landscape topology and adversarial attractors.

### Class Definition

```python
class SawtoothTopologyAnalyzer:
    """Analyzer for sawtooth loss landscape topology."""
    
    def __init__(self, device: str = 'cpu'):
        """
        Initialize sawtooth topology analyzer.
        
        Args:
            device: Computing device
        """
```

### Methods

#### analyze_loss_landscape_geometry

```python
def analyze_loss_landscape_geometry(
    self,
    loss_function: Callable,
    parameter_range: Tuple[float, float],
    num_points: int = 1000
) -> Dict[str, Any]:
    """
    Analyze the geometric properties of a loss landscape.
    
    Identifies:
    - Sawtooth patterns
    - Local minima (both correct and inverted)
    - Gradient discontinuities
    - Basin of attraction sizes
    
    Args:
        loss_function: Function mapping parameters to loss values
        parameter_range: (min, max) range for parameter sweep
        num_points: Number of evaluation points
    
    Returns:
        Dictionary containing:
            - 'num_minima': Number of local minima
            - 'num_maxima': Number of local maxima
            - 'sawtooth_period': Average distance between discontinuities
            - 'minima_locations': Positions of local minima
            - 'inverted_minima': Minima corresponding to inverted solutions
    
    Example:
        >>> def loss_fn(theta):
        ...     return compute_loss(model, data, theta)
        >>> results = analyzer.analyze_loss_landscape_geometry(
        ...     loss_fn, parameter_range=(-1.0, 1.0)
        ... )
        >>> print(f"Found {results['num_minima']} local minima")
    """
```

#### prove_adversarial_attractor_existence

```python
def prove_adversarial_attractor_existence(
    self,
    cipher,
    target_function: Callable,
    num_initializations: int = 100
) -> Dict[str, Any]:
    """
    Prove existence of adversarial attractors (inverted minima).
    
    Uses random initialization and gradient descent to empirically
    demonstrate that optimization consistently converges to inverted
    solutions rather than correct ones.
    
    Args:
        cipher: Cipher implementation
        target_function: True function to approximate
        num_initializations: Number of random starts
    
    Returns:
        Dictionary containing:
            - 'num_correct_convergence': Convergences to correct solution
            - 'num_inverted_convergence': Convergences to inverted solution
            - 'inversion_rate': Fraction converging to inverted
            - 'basin_size_ratio': Relative basin sizes
    
    Example:
        >>> results = analyzer.prove_adversarial_attractor_existence(
        ...     cipher, target_fn, num_initializations=50
        ... )
        >>> print(f"Inversion rate: {results['inversion_rate']:.2%}")
    """
```

---

## InformationTheoreticAnalyzer

Quantifies information loss in smooth approximations.

### Class Definition

```python
class InformationTheoreticAnalyzer:
    """Analyzer for information-theoretic properties."""
    
    def __init__(self, num_bins: int = 50):
        """
        Initialize information-theoretic analyzer.
        
        Args:
            num_bins: Number of bins for histogram-based entropy estimation
        """
```

### Methods

#### analyze_information_loss_in_approximation

```python
def analyze_information_loss_in_approximation(
    self,
    discrete_outputs: torch.Tensor,
    smooth_outputs: torch.Tensor,
    num_bins: Optional[int] = None
) -> Dict[str, float]:
    """
    Analyze information loss between discrete and smooth operations.
    
    Computes:
    - Entropy of discrete outputs
    - Entropy of smooth outputs  
    - Information loss (difference)
    - Mutual information
    - KL divergence
    
    Args:
        discrete_outputs: Outputs from exact discrete operation
        smooth_outputs: Outputs from smooth approximation
        num_bins: Number of bins for entropy estimation (optional)
    
    Returns:
        Dictionary containing:
            - 'discrete_entropy': H(discrete) in bits
            - 'smooth_entropy': H(smooth) in bits
            - 'information_loss': H(discrete) - H(smooth)
            - 'mutual_information': I(discrete; smooth)
            - 'kl_divergence': D_KL(discrete || smooth)
    
    Example:
        >>> z_discrete = modular_add_exact(x, y, modulus)
        >>> z_smooth = modular_add_sigmoid(x, y, modulus)
        >>> results = analyzer.analyze_information_loss_in_approximation(
        ...     z_discrete, z_smooth
        ... )
        >>> print(f"Information loss: {results['information_loss']:.3f} bits")
    """
```

#### compute_mutual_information

```python
def compute_mutual_information(
    self,
    x: torch.Tensor,
    y: torch.Tensor
) -> float:
    """
    Compute mutual information I(X; Y).
    
    Uses histogram-based estimation:
    I(X; Y) = H(X) + H(Y) - H(X, Y)
    
    Args:
        x: First random variable
        y: Second random variable
    
    Returns:
        Mutual information in bits
    
    Example:
        >>> mi = analyzer.compute_mutual_information(plaintext, ciphertext)
        >>> print(f"Mutual information: {mi:.3f} bits")
    """
```

---

## Theorem Classes

Formal theorem implementations with verification methods.

### ModularAdditionTheorem

```python
class ModularAdditionTheorem:
    """
    Theorem 1: Gradient Discontinuity in Modular Addition.
    
    States that smooth approximations of modular addition have gradient
    error bounded by O(m·β) at wrap-around points.
    """
    
    def verify_discontinuity(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        modulus: int = 2**16,
        beta_values: List[float] = [1.0, 5.0, 10.0, 20.0]
    ) -> Dict[str, Dict[str, float]]:
        """
        Verify the gradient discontinuity theorem.
        
        Args:
            x: Input tensor
            y: Input tensor
            modulus: Modulus for operations
            beta_values: List of steepness values to test
        
        Returns:
            Results for each beta value
        
        Example:
            >>> theorem = ModularAdditionTheorem()
            >>> results = theorem.verify_discontinuity(x, y)
            >>> for beta, metrics in results.items():
            ...     print(f"{beta}: error = {metrics['gradient_error']}")
        """
```

### GradientInversionTheorem

```python
class GradientInversionTheorem:
    """
    Theorem 2: Systematic Gradient Inversion.
    
    States that gradient inversion occurs with probability:
    P_inv >= 1 - (1 - 1/m)^k
    """
    
    def verify_inversion_probability(
        self,
        k: int,
        modulus: int = 2**16,
        num_samples: int = 1000
    ) -> Dict[str, float]:
        """
        Verify the inversion probability theorem.
        
        Args:
            k: Number of rounds
            modulus: Modulus for operations
            num_samples: Number of samples for empirical measurement
        
        Returns:
            Theoretical and empirical probabilities
        """
```

### SawtoothConvergenceTheorem

```python
class SawtoothConvergenceTheorem:
    """
    Theorem 3: Non-Convergence in Sawtooth Landscapes.
    
    States that gradient descent fails when learning rate exceeds:
    α > T / ||∇L||
    """
    
    def simulate_gradient_descent(
        self,
        learning_rate: float,
        num_steps: int = 500,
        sawtooth_period: float = 10.0,
        gradient_magnitude: float = 1.0
    ) -> Dict[str, Any]:
        """
        Simulate gradient descent in sawtooth landscape.
        
        Returns:
            Trajectory and convergence information
        """
```

### InformationLossTheorem

```python
class InformationLossTheorem:
    """
    Theorem 4: Information Loss Lower Bound.
    
    States that approximation information loss satisfies:
    Δ >= n·log(2)/4 bits
    """
    
    def compute_information_loss(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        modulus: int = 2**16
    ) -> Dict[str, float]:
        """
        Compute information loss and verify bound.
        
        Returns:
            Measured loss and theoretical bound
        """
```

---

## Utility Functions

### modular_add_exact

```python
def modular_add_exact(
    x: torch.Tensor,
    y: torch.Tensor,
    modulus: int
) -> torch.Tensor:
    """
    Exact modular addition (discrete).
    
    Args:
        x: First operand
        y: Second operand
        modulus: Modulus m
    
    Returns:
        (x + y) mod m
    """
```

### modular_add_sigmoid

```python
def modular_add_sigmoid(
    x: torch.Tensor,
    y: torch.Tensor,
    modulus: int,
    steepness: float = 10.0
) -> torch.Tensor:
    """
    Smooth modular addition using sigmoid approximation.
    
    Approximates: z = x + y - m·σ(β(x + y - m))
    
    Args:
        x: First operand
        y: Second operand
        modulus: Modulus m
        steepness: Steepness parameter β
    
    Returns:
        Smooth approximation of (x + y) mod m
    """
```

---

## Type Definitions

```python
from typing import Dict, List, Tuple, Optional, Callable, Any
import torch

# Common type aliases
Tensor = torch.Tensor
AnalysisResults = Dict[str, Any]
Theorem VerificationResults = Dict[str, Dict[str, float]]
```

---

## Examples

### Complete Analysis Workflow

```python
import torch
from ctdma.theory.mathematical_analysis import (
    GradientInversionAnalyzer,
    SawtoothTopologyAnalyzer,
    InformationTheoreticAnalyzer
)
from ctdma.ciphers import create_cipher

# Initialize analyzers
gradient_analyzer = GradientInversionAnalyzer()
topology_analyzer = SawtoothTopologyAnalyzer()
info_analyzer = InformationTheoreticAnalyzer()

# Create cipher
cipher = create_cipher('speck', num_rounds=2)

# Generate test data
x = torch.randint(0, 2**16, (1000,))
y = torch.randint(0, 2**16, (1000,))

# Analyze gradient behavior
grad_results = gradient_analyzer.compute_gradient_discontinuity(x, y)
print(f"Max discontinuity: {grad_results['max_discontinuity']:.2f}")

# Estimate inversion probability
inv_prob = gradient_analyzer.estimate_inversion_probability(
    x, y, num_rounds=2
)
print(f"Inversion probability: {inv_prob:.2%}")

# Analyze information loss
z_discrete = (x + y) % (2**16)
z_smooth = modular_add_sigmoid(x, y, 2**16)
info_results = info_analyzer.analyze_information_loss_in_approximation(
    z_discrete, z_smooth
)
print(f"Information loss: {info_results['information_loss']:.3f} bits")
```

---

## Performance Notes

- **Gradient analysis**: O(n) complexity, ~1ms per 1000 samples
- **Topology analysis**: O(n·m) where m is number of evaluation points
- **Information theory**: O(n·b) where b is number of bins

## See Also

- [Approximation Methods API](approximation_methods.md)
- [Cipher Implementations API](cipher_implementations.md)
- [Example Notebooks](../examples/)
- [Mathematical Foundations](../MATHEMATICAL_FOUNDATIONS.md)

---

*Last updated: January 30, 2026*
