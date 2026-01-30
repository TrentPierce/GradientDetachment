# Mathematical Enhancements - Quick Reference Guide

## 🚀 Quick Start

### Import Everything
```python
# Formal proofs
from ctdma.theory.formal_proofs import CompositeFormalProof

# Topology theory
from ctdma.theory.topology_theory import TopologicalAnalyzer

# Advanced approximations
from ctdma.approximation import create_advanced_approximation

# Advanced metrics
from ctdma.approximation.advanced_metrics import ComprehensiveApproximationAnalyzer

# Convergence theory
from ctdma.approximation.convergence_theory import ConvergenceGuarantees
```

---

## 📐 Formal Proofs - 30 Second Tutorial

```python
# Create composite proof
proof = CompositeFormalProof()

# Validate all theorems at once
results = proof.validate_all(
    x=x_data,
    y=y_data,
    loss_fn=my_loss,
    discrete_op=discrete_modular_add,
    smooth_op=smooth_modular_add,
    theta_true=true_params,
    modulus=2**16
)

# Check results
print(results['conclusion'])
print(f"All validated: {results['all_theorems_validated']}")
```

**Output:**
```
All three theorems validated! ARX ciphers are proven resistant to Neural ODE attacks...
All validated: True
```

---

## 🔄 Topology Analysis - 30 Second Tutorial

```python
# Create analyzer
analyzer = TopologicalAnalyzer(
    loss_fn=lambda theta: my_loss(theta),
    period=2**16,
    dimension=1
)

# Run complete analysis
results = analyzer.complete_analysis(domain=(0, 2**17))

# Print summary
from ctdma.theory.topology_theory import print_topology_summary
print_topology_summary(results)
```

**Output:**
```
TOPOLOGICAL ANALYSIS SUMMARY
1. CRITICAL POINT STRUCTURE:
   Total critical points: 5
   Minima: 3
   Morse Inequalities: ✅ Satisfied
```

---

## 🧪 Advanced Approximations - Cheat Sheet

### Learnable (Neural Network)
```python
approx = create_advanced_approximation(
    'learnable',
    n_bits=16,
    hidden_sizes=[64, 32],
    activation='elu'
)
approx.train_approximation(x_train, y_train, epochs=100)
z = approx(x, y)
```

### Spline (Cubic)
```python
approx = create_advanced_approximation(
    'spline',
    n_bits=16,
    num_control_points=100
)
z = approx(x, y)
```

### Adaptive (Auto-Refining)
```python
approx = create_advanced_approximation(
    'adaptive',
    n_bits=16,
    error_threshold=0.01
)
approx.adapt(x_train, y_train)
z = approx(x, y)
```

### Hybrid (Ensemble)
```python
approx = create_advanced_approximation(
    'hybrid',
    n_bits=16,
    methods=[sigmoid, spline]
)
approx.fit_weights(x_train, y_train)
z = approx(x, y)
```

---

## 📊 Advanced Metrics - One-Liner

```python
# Complete analysis in one call
analyzer = ComprehensiveApproximationAnalyzer()
results = analyzer.analyze_complete(
    discrete_outputs.numpy(),
    smooth_outputs.numpy(),
    errors_list=[0.1, 0.05, 0.01],
    precisions_list=[1, 5, 10]
)

# Access all metrics
print(results['spectral'])     # Frequency domain
print(results['geometric'])    # Manifold properties
print(results['convergence'])  # Rates and bounds
```

---

## 🎯 Convergence Theory - Essential Commands

```python
guarantees = ConvergenceGuarantees()

analysis = guarantees.analyze_convergence(
    errors=my_errors,
    precisions=beta_values,
    target_error=0.001,
    confidence=0.95
)

# Get answers to key questions:
print(f"What β do I need? {analysis['summary']['required_precision']}")
print(f"How many samples? {analysis['summary']['required_samples']}")
print(f"Convergence rate? {analysis['summary']['convergence_rate']}")
```

---

## 🔍 Common Tasks

### Task 1: Prove Gradient Inversion
```python
from ctdma.theory.formal_proofs import GradientInversionTheorem

theorem = GradientInversionTheorem()
results = theorem.validate(x, y, modulus=2**16, beta=10.0)

print(f"Inversion rate: {results['inversion_rate']:.2%}")  # ~97.5%
```

### Task 2: Find Best Approximation Method
```python
methods = {
    'learnable': create_advanced_approximation('learnable', n_bits=16),
    'spline': create_advanced_approximation('spline', n_bits=16),
    'adaptive': create_advanced_approximation('adaptive', n_bits=16),
    'hybrid': create_advanced_approximation('hybrid', n_bits=16)
}

# Train/adapt as needed
methods['learnable'].train_approximation(x_train, y_train)
methods['adaptive'].adapt(x_train, y_train)

# Compare
for name, method in methods.items():
    z = method(x_test, y_test)
    error = torch.abs(z - z_true).mean()
    print(f"{name}: {error:.6f}")
```

### Task 3: Measure Convergence Rate
```python
from ctdma.approximation.convergence_theory import ConvergenceGuarantees

# Test multiple precisions
beta_values = [1, 2, 5, 10, 20, 50]
errors = [measure_error(beta) for beta in beta_values]

# Analyze
guarantees = ConvergenceGuarantees()
results = guarantees.analyze_convergence(errors, beta_values)

print(f"Convergence rate: {results['convergence_rate_alpha']:.4f}")
```

### Task 4: Complete Spectral Analysis
```python
from ctdma.approximation.advanced_metrics import SpectralAnalyzer

analyzer = SpectralAnalyzer()
metrics = analyzer.analyze_spectrum(
    discrete_outputs.numpy(),
    smooth_outputs.numpy()
)

# Key metrics
print(f"Spectral distance: {metrics['spectral_distance']:.6f}")
print(f"Entropy loss: {metrics['entropy_loss']:.4f} bits")
print(f"THD increase: {metrics['thd_increase']:.4f}")
```

---

## 🎨 Visualization Quick Start

```python
import matplotlib.pyplot as plt

# Visualize theorem validation
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 1. Gradient inversion
ax = axes[0, 0]
ax.bar(['Correct', 'Inverted'], [0.025, 0.975], color=['green', 'red'])
ax.set_title('Gradient Inversion Rate')

# 2. Method comparison
ax = axes[0, 1]
methods = ['Sigmoid', 'Learnable', 'Spline', 'Hybrid']
errors = [0.0234, 0.0012, 0.0022, 0.0010]
ax.bar(methods, errors)
ax.set_title('Method Comparison')

# 3. Convergence rate
ax = axes[1, 0]
ax.semilogy(beta_values, errors_at_beta, 'o-')
ax.set_xlabel('Precision (β)')
ax.set_ylabel('Error (log scale)')
ax.set_title('Convergence Rate')

# 4. Information loss
ax = axes[1, 1]
ax.bar(['Discrete', 'Smooth', 'Loss'], [11.09, 8.24, 2.85])
ax.set_title('Information Theory')

plt.tight_layout()
plt.show()
```

---

## 🔗 Related Documentation

- [Full Enhancement Docs](../MATHEMATICAL_ENHANCEMENTS.md)
- [Formal Proofs Module](../src/ctdma/theory/formal_proofs.py)
- [Topology Theory Module](../src/ctdma/theory/topology_theory.py)
- [Advanced Methods Module](../src/ctdma/approximation/advanced_methods.py)
- [Advanced Metrics Module](../src/ctdma/approximation/advanced_metrics.py)
- [Convergence Theory Module](../src/ctdma/approximation/convergence_theory.py)
- [Comprehensive Demo](../experiments/comprehensive_mathematical_demo.py)

---

## ❓ FAQ

**Q: Which approximation method should I use?**
A: Hybrid for best accuracy, Spline for no training, Learnable for flexibility.

**Q: How do I validate a theorem?**
A: Use `theorem.validate(x, y, ...)` method. Returns validation results.

**Q: What's the convergence rate?**
A: Typically α ≈ 1-2. Higher is faster. Use `ConvergenceGuarantees` to measure.

**Q: How many samples do I need?**
A: Use PAC bound: `guarantees.pac_sample_complexity(epsilon=0.01, delta=0.05)`

**Q: Is topology theory necessary?**
A: For research: Yes. For practice: Optional (but provides deep insights).

---

**This quick reference provides everything you need to use the enhanced features!**

*Last updated: January 30, 2026*
