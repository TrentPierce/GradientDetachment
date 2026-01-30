# Example Notebooks

This directory contains comprehensive Jupyter notebooks demonstrating the key features and capabilities of the Gradient Detachment framework.

## 📓 Available Notebooks

### 1. Mathematical Analysis Demo
**File**: `mathematical_analysis_demo.ipynb`

**Description**: Interactive demonstrations of the four core theorems with empirical verification.

**Contents**:
- Theorem 1: Gradient discontinuity analysis
- Theorem 2: Systematic gradient inversion
- Theorem 3: Sawtooth convergence behavior
- Theorem 4: Information loss bounds
- Interactive visualizations
- Parameter exploration

**Time**: ~15 minutes

**Prerequisites**: Basic understanding of calculus and PyTorch

---

### 2. Approximation Comparison
**File**: `approximation_comparison.ipynb`

**Description**: Side-by-side comparison of four approximation methods for modular arithmetic.

**Contents**:
- Sigmoid approximation
- Straight-Through Estimator (STE)
- Gumbel-Softmax
- Temperature annealing
- Error analysis
- Gradient fidelity comparison
- Convergence rate analysis

**Time**: ~20 minutes

**Prerequisites**: Understanding of gradient descent and approximation theory

---

### 3. Cipher Evaluation
**File**: `cipher_evaluation.ipynb`

**Description**: Testing and comparing different cipher implementations (ARX, Feistel, SPN).

**Contents**:
- Cross-cipher comparison
- Round-by-round security analysis
- Inversion probability measurement
- Performance benchmarking
- Security recommendations

**Time**: ~25 minutes

**Prerequisites**: Basic cryptography knowledge

---

### 4. Comprehensive Benchmark
**File**: `comprehensive_benchmark.ipynb`

**Description**: Full performance analysis and scalability testing.

**Contents**:
- End-to-end benchmark suite
- Scalability analysis (varying data sizes)
- Memory profiling
- GPU vs CPU comparison
- Bottleneck identification
- Optimization recommendations

**Time**: ~30 minutes

**Prerequisites**: Understanding of performance optimization

---

## 🚀 Getting Started

### Installation

```bash
# Install Jupyter if not already installed
pip install jupyter notebook

# Or use JupyterLab
pip install jupyterlab
```

### Running Notebooks

```bash
# From the project root
jupyter notebook

# Or JupyterLab
jupyter lab

# Navigate to examples/ and open desired notebook
```

### Required Dependencies

All notebooks require:
- torch >= 1.9.0
- numpy >= 1.19.0
- matplotlib >= 3.3.0
- scipy >= 1.5.0

Install with:
```bash
pip install -e .
```

---

## 📖 Notebook Structure

Each notebook follows this structure:

1. **Setup**: Import libraries and configure environment
2. **Introduction**: Overview of concepts
3. **Implementation**: Core analysis code
4. **Results**: Visualizations and metrics
5. **Discussion**: Interpretation of findings
6. **Exercises**: Optional hands-on activities

---

## 📊 Expected Outputs

### Visualizations
- Loss landscape plots
- Gradient flow diagrams
- Error comparison charts
- Convergence trajectories
- Information theory metrics

### Metrics
- Accuracy measurements
- Inversion probabilities
- Error statistics
- Performance benchmarks
- Convergence rates

---

## ⚡ Tips for Best Experience

1. **Run cells sequentially**: Notebooks are designed to be executed top-to-bottom
2. **Check kernel**: Ensure you're using the correct Python environment
3. **GPU optional**: Most examples work on CPU, but GPU accelerates larger experiments
4. **Explore parameters**: Try modifying hyperparameters to see effects
5. **Save outputs**: Export figures and results for your own analysis

---

## 🐛 Troubleshooting

### Common Issues

**Issue**: Module not found error
```bash
# Solution: Install package in development mode
pip install -e .
```

**Issue**: Kernel crashes on large datasets
```python
# Solution: Reduce batch size or use smaller datasets
batch_size = 32  # Instead of 128
```

**Issue**: Slow execution
```python
# Solution: Use GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

---

## 📝 Additional Resources

- [Mathematical Foundations](../MATHEMATICAL_FOUNDATIONS.md)
- [API Documentation](../docs/)
- [Research Paper](../RESEARCH_PAPER.md)
- [Contributing Guide](../CONTRIBUTING.md)

---

## ❓ Questions?

If you have questions or find issues:
- Open an issue on GitHub
- Check the main documentation
- Email: Pierce.trent@gmail.com

---

**Happy exploring!**
