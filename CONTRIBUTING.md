# Contributing to Gradient Detachment

First off, thank you for considering contributing to Gradient Detachment! This project aims to advance our understanding of gradient-based cryptanalysis and the inherent limitations of neural approaches to breaking ARX ciphers.

## 📋 Table of Contents

- [Code of Conduct](#code-of-conduct)
- [How Can I Contribute?](#how-can-i-contribute)
  - [Reporting Bugs](#reporting-bugs)
  - [Suggesting Features](#suggesting-features)
  - [Improving Documentation](#improving-documentation)
  - [Contributing Code](#contributing-code)
- [Development Setup](#development-setup)
- [Coding Standards](#coding-standards)
- [Testing Guidelines](#testing-guidelines)
- [Pull Request Process](#pull-request-process)
- [Research Contributions](#research-contributions)
- [Community Guidelines](#community-guidelines)

---

## Code of Conduct

### Our Pledge

We are committed to providing a welcoming and inspiring community for all. We pledge to:

- Use welcoming and inclusive language
- Be respectful of differing viewpoints and experiences
- Gracefully accept constructive criticism
- Focus on what is best for the community
- Show empathy towards other community members

### Our Standards

**Positive behaviors include:**
- Using welcoming and inclusive language
- Being respectful of differing viewpoints
- Providing and accepting constructive feedback
- Focusing on scientific rigor and reproducibility
- Acknowledging contributions appropriately

**Unacceptable behaviors include:**
- Harassment or discriminatory comments
- Publishing others' private information without permission
- Plagiarism or misrepresentation of research
- Malicious use of cryptanalysis tools
- Any conduct violating academic integrity

### Enforcement

Violations can be reported to Pierce.trent@gmail.com. All complaints will be reviewed and investigated promptly and fairly.

---

## How Can I Contribute?

### Reporting Bugs

Before creating a bug report:

1. **Check existing issues** to avoid duplicates
2. **Verify the bug** with the latest version
3. **Collect information** about your environment

**Bug Report Template:**

```markdown
**Description**
A clear and concise description of the bug.

**To Reproduce**
Steps to reproduce the behavior:
1. Go to '...'
2. Run command '...'
3. See error

**Expected Behavior**
What you expected to happen.

**Actual Behavior**
What actually happened.

**Environment**
- OS: [e.g., Ubuntu 20.04]
- Python version: [e.g., 3.8.10]
- PyTorch version: [e.g., 1.9.0]
- CUDA version (if applicable): [e.g., 11.1]

**Additional Context**
- Stack trace or error messages
- Relevant code snippets
- Screenshots if applicable
```

### Suggesting Features

We welcome feature suggestions that align with the project's research goals.

**Feature Request Template:**

```markdown
**Feature Description**
A clear and concise description of the feature.

**Motivation**
Why is this feature valuable? What problem does it solve?

**Proposed Solution**
Describe how you envision this feature working.

**Alternatives Considered**
Describe alternative solutions you've considered.

**Research Relevance**
How does this advance cryptanalysis or mathematical understanding?

**Implementation Complexity**
Estimate: Simple / Moderate / Complex
```

### Improving Documentation

Documentation improvements are always welcome!

**Types of documentation contributions:**
- Fixing typos or clarifying explanations
- Adding examples or tutorials
- Improving API documentation
- Translating documentation
- Creating video tutorials or blog posts

**Documentation standards:**
- Use clear, concise language
- Include code examples
- Provide mathematical notation in LaTeX
- Add visualizations where helpful
- Ensure technical accuracy

### Contributing Code

We welcome code contributions in the following areas:

#### 1. **New Approximation Methods**

Implement new techniques for approximating discrete operations:

```python
# Example: Custom approximation method
class MyApproximation(ApproximationMethod):
    def __init__(self, **kwargs):
        super().__init__()
        # Initialize parameters
    
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Forward pass with smooth approximation."""
        # Implementation
    
    def compute_gradients(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute gradients for backpropagation."""
        # Implementation
```

#### 2. **Additional Cipher Implementations**

Add support for new cipher families:

```python
# Example: New cipher implementation
class MyCipher(BaseCipher):
    def __init__(self, num_rounds: int = 4, **kwargs):
        super().__init__(num_rounds, **kwargs)
    
    def encrypt_round(self, state: torch.Tensor, round_key: torch.Tensor) -> torch.Tensor:
        """Single round encryption."""
        # Implementation
    
    def decrypt_round(self, state: torch.Tensor, round_key: torch.Tensor) -> torch.Tensor:
        """Single round decryption."""
        # Implementation
```

#### 3. **Mathematical Analysis Tools**

Extend the theoretical analysis framework:

```python
# Example: New analysis tool
class MyAnalyzer:
    def analyze_property(self, cipher, data):
        """Analyze specific mathematical property."""
        # Implementation
        return results
```

#### 4. **Performance Optimizations**

- GPU acceleration
- Vectorization improvements
- Memory optimization
- Caching strategies

#### 5. **Visualization Tools**

- Loss landscape visualization
- Gradient flow diagrams
- Interactive plots
- Animation utilities

---

## Development Setup

### Prerequisites

- Python 3.7 or higher
- Git
- Virtual environment tool (venv, conda, etc.)

### Setup Steps

```bash
# 1. Fork the repository on GitHub

# 2. Clone your fork
git clone https://github.com/YOUR_USERNAME/gradientdetachment.git
cd gradientdetachment

# 3. Add upstream remote
git remote add upstream https://github.com/TrentPierce/gradientdetachment.git

# 4. Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 5. Install in development mode with all dependencies
pip install -e ".[dev]"

# 6. Install pre-commit hooks (optional but recommended)
pre-commit install

# 7. Verify installation
pytest tests/
python reproduce_sawtooth.py
```

### Development Dependencies

The `[dev]` extra installs:
- pytest (testing)
- pytest-cov (coverage)
- black (formatting)
- mypy (type checking)
- flake8 (linting)
- sphinx (documentation)
- jupyter (notebooks)

---

## Coding Standards

### Python Style Guide

We follow **PEP 8** with some modifications:

```python
# Good example
def compute_gradient_discontinuity(
    x: torch.Tensor,
    y: torch.Tensor,
    modulus: int = 2**16,
    steepness: float = 10.0
) -> Dict[str, float]:
    """
    Compute gradient discontinuity at modular wrap-around points.
    
    Args:
        x: Input tensor of shape (batch_size, dim)
        y: Input tensor of shape (batch_size, dim)
        modulus: Modulus for modular arithmetic (default: 2^16)
        steepness: Steepness parameter for sigmoid approximation
    
    Returns:
        Dictionary containing:
            - 'max_discontinuity': Maximum gradient jump
            - 'avg_discontinuity': Average gradient jump
            - 'num_discontinuities': Number of wrap-around points
    
    Example:
        >>> x = torch.randint(0, 2**16, (100, 8))
        >>> y = torch.randint(0, 2**16, (100, 8))
        >>> results = compute_gradient_discontinuity(x, y)
        >>> print(results['max_discontinuity'])
    """
    # Implementation with clear variable names
    wrap_around_mask = (x + y) >= modulus
    # ...
```

### Key Standards

1. **Type Hints**: All function signatures must include type hints
2. **Docstrings**: Use Google-style docstrings for all public functions
3. **Variable Names**: Use descriptive names (no single letters except loop counters)
4. **Line Length**: Maximum 100 characters (flexible for readability)
5. **Imports**: Organize imports (standard library, third-party, local)

### Code Formatting

```bash
# Format code with black
black src/ tests/ experiments/

# Check with flake8
flake8 src/ tests/ experiments/

# Type checking with mypy
mypy src/
```

### Mathematical Notation

For mathematical formulas in docstrings:

```python
def compute_information_loss(p: torch.Tensor, q: torch.Tensor) -> float:
    """
    Compute information loss using KL divergence.
    
    The KL divergence is defined as:
    
        D_KL(P || Q) = Σ p(x) log(p(x) / q(x))
    
    For approximation error bounds:
    
        Δ ≥ n·log(2)/4 bits
    
    Args:
        p: True probability distribution
        q: Approximate distribution
    
    Returns:
        KL divergence in bits
    """
    # Implementation
```

---

## Testing Guidelines

### Test Organization

```
tests/
├── test_theory.py              # Mathematical analysis tests
├── test_approximation.py       # Approximation method tests
├── test_ciphers.py            # Cipher implementation tests
├── test_integration.py        # End-to-end tests
├── test_performance.py        # Performance benchmarks
└── conftest.py                # Pytest fixtures
```

### Writing Tests

```python
import pytest
import torch
from ctdma.theory.mathematical_analysis import GradientInversionAnalyzer

class TestGradientInversionAnalyzer:
    """Tests for GradientInversionAnalyzer."""
    
    @pytest.fixture
    def analyzer(self):
        """Create analyzer instance."""
        return GradientInversionAnalyzer()
    
    @pytest.fixture
    def sample_data(self):
        """Create sample test data."""
        x = torch.randint(0, 2**16, (100, 8))
        y = torch.randint(0, 2**16, (100, 8))
        return x, y
    
    def test_discontinuity_detection(self, analyzer, sample_data):
        """Test that discontinuities are detected correctly."""
        x, y = sample_data
        results = analyzer.compute_gradient_discontinuity(x, y)
        
        assert 'max_discontinuity' in results
        assert results['max_discontinuity'] > 0
        assert results['num_discontinuities'] >= 0
    
    def test_inversion_probability(self, analyzer, sample_data):
        """Test inversion probability calculation."""
        x, y = sample_data
        prob = analyzer.estimate_inversion_probability(x, y, num_rounds=1)
        
        assert 0 <= prob <= 1
        assert prob > 0.9  # Should be high for ARX ciphers
    
    @pytest.mark.parametrize("modulus", [2**8, 2**16, 2**32])
    def test_different_moduli(self, analyzer, modulus):
        """Test with different modulus values."""
        x = torch.randint(0, modulus, (100, 8))
        y = torch.randint(0, modulus, (100, 8))
        results = analyzer.compute_gradient_discontinuity(x, y, modulus=modulus)
        
        assert results is not None
```

### Test Coverage Requirements

- **Minimum coverage**: 80% for new code
- **Target coverage**: 95% for core modules
- **Integration tests**: All major workflows
- **Edge cases**: Boundary conditions, error handling

### Running Tests

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=src/ctdma --cov-report=html

# Run specific test file
pytest tests/test_theory.py

# Run specific test
pytest tests/test_theory.py::TestGradientInversionAnalyzer::test_discontinuity_detection

# Run with verbose output
pytest tests/ -v

# Run only fast tests (skip slow benchmarks)
pytest tests/ -m "not slow"
```

---

## Pull Request Process

### Before Submitting

1. **Create a feature branch**
   ```bash
   git checkout -b feature/my-new-feature
   ```

2. **Make your changes**
   - Write clear, documented code
   - Add tests for new functionality
   - Update documentation

3. **Run quality checks**
   ```bash
   black src/ tests/
   flake8 src/ tests/
   mypy src/
   pytest tests/ --cov=src/ctdma
   ```

4. **Update CHANGELOG.md**
   - Add entry under [Unreleased]
   - Describe changes clearly

5. **Commit changes**
   ```bash
   git add .
   git commit -m "Add new approximation method: XYZ"
   ```

### PR Checklist

- [ ] Code follows project style guidelines
- [ ] All tests pass
- [ ] New tests added for new functionality
- [ ] Documentation updated
- [ ] CHANGELOG.md updated
- [ ] Type hints added
- [ ] Docstrings complete
- [ ] No merge conflicts with main branch

### PR Template

```markdown
## Description
Brief description of changes.

## Motivation
Why are these changes needed?

## Changes Made
- Change 1
- Change 2
- Change 3

## Testing
How were these changes tested?

## Documentation
What documentation was updated?

## Checklist
- [ ] Code follows style guidelines
- [ ] Tests pass
- [ ] Documentation updated
- [ ] CHANGELOG.md updated

## Related Issues
Closes #123
```

### Review Process

1. **Automated checks**: CI/CD runs tests and linting
2. **Code review**: Maintainers review code quality
3. **Discussion**: Address reviewer feedback
4. **Approval**: Two approvals required for merge
5. **Merge**: Squash and merge to main

---

## Research Contributions

### Novel Research

We welcome research contributions:

1. **New Theorems**: Formal mathematical proofs
2. **Experimental Results**: Novel findings
3. **Analysis Methods**: New analytical techniques
4. **Cipher Studies**: Analysis of new cipher families

### Research Standards

- **Reproducibility**: All results must be reproducible
- **Documentation**: Clear methodology description
- **Validation**: Empirical verification of theoretical claims
- **Peer Review**: Submit to project maintainers for review

### Publishing

If your contribution leads to a publication:

1. **Acknowledge the project** in your paper
2. **Cite appropriately** using the BibTeX in README
3. **Share results** back with the community
4. **Link publications** in project documentation

---

## Community Guidelines

### Communication Channels

- **GitHub Issues**: Bug reports, feature requests
- **GitHub Discussions**: General questions, ideas
- **Email**: Pierce.trent@gmail.com for private matters
- **Twitter**: @severesig for updates

### Getting Help

- Check **documentation** first
- Search **existing issues**
- Ask in **GitHub Discussions**
- Be **specific** and provide **examples**

### Recognition

Contributors will be:
- Listed in CONTRIBUTORS.md
- Acknowledged in release notes
- Credited in academic publications (as appropriate)
- Invited to join the research team (significant contributions)

---

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

## Questions?

Don't hesitate to ask! We're here to help:

- **Email**: Pierce.trent@gmail.com
- **GitHub**: Open a discussion or issue
- **Twitter**: @severesig

---

**Thank you for contributing to Gradient Detachment!**

Your contributions help advance our understanding of cryptographic security and machine learning limitations.

---

*Last updated: January 30, 2026*
