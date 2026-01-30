# API Documentation

Comprehensive API reference for the Gradient Detachment framework.

## 📚 Available Documentation

### Core Modules

1. **[Mathematical Theory](mathematical_theory.md)**
   - GradientInversionAnalyzer
   - SawtoothTopologyAnalyzer
   - InformationTheoreticAnalyzer
   - Theorem classes and verification methods

2. **[Approximation Methods](approximation_methods.md)**
   - ApproximationBridge interface
   - Sigmoid, STE, Gumbel-Softmax, Temperature methods
   - Metrics and convergence analysis

3. **[Cipher Implementations](cipher_implementations.md)**
   - Base cipher interface
   - ARX, Feistel, and SPN implementations
   - Smooth approximation utilities

4. **[Testing Framework](testing_framework.md)**
   - Testing guidelines and standards
   - Fixture usage
   - Performance benchmarking

## 🚀 Quick Reference

### Mathematical Analysis

```python
from ctdma.theory.mathematical_analysis import GradientInversionAnalyzer

analyzer = GradientInversionAnalyzer()
results = analyzer.compute_gradient_discontinuity(x, y, modulus=2**16)
```

### Approximation Methods

```python
from ctdma.approximation.bridge import create_approximation_bridge

method = create_approximation_bridge('sigmoid', steepness=10.0)
z_smooth = method.forward(x, y, modulus=2**16)
```

### Cipher Usage

```python
from ctdma.ciphers import create_cipher

cipher = create_cipher('speck', num_rounds=4)
ciphertext = cipher.encrypt(plaintext, key)
```

## 📖 Documentation Standards

All API documentation follows these principles:

- **Complete**: Every public function and class documented
- **Clear**: Easy-to-understand explanations with examples
- **Consistent**: Uniform formatting and structure
- **Correct**: Technically accurate and verified

## 🔗 External Resources

- [Main README](../README.md)
- [Mathematical Foundations](../MATHEMATICAL_FOUNDATIONS.md)
- [Example Notebooks](../examples/)
- [Contributing Guide](../CONTRIBUTING.md)

## ❔ Getting Help

If you need assistance:
1. Check the relevant API documentation
2. Review example notebooks
3. Search GitHub issues
4. Open a new issue with the `documentation` label

---

*Last updated: January 30, 2026*
