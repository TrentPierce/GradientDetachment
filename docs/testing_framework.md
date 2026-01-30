# Testing Framework

Comprehensive testing guidelines for the Gradient Detachment project.

## Table of Contents

- [Testing Philosophy](#testing-philosophy)
- [Test Organization](#test-organization)
- [Unit Testing](#unit-testing)
- [Integration Testing](#integration-testing)
- [Performance Testing](#performance-testing)
- [Fixtures and Utilities](#fixtures-and-utilities)
- [Continuous Integration](#continuous-integration)

---

## Testing Philosophy

### Principles

1. **Comprehensive Coverage**: Aim for 95%+ code coverage
2. **Reproducibility**: All tests use fixed random seeds
3. **Independence**: Tests don't depend on each other
4. **Speed**: Fast tests run often, slow tests run on CI
5. **Clarity**: Test names describe what they test

### Test Categories

- **Unit tests**: Test individual functions/classes
- **Integration tests**: Test component interactions
- **Performance tests**: Benchmark and profiling
- **Regression tests**: Ensure bugs stay fixed
- **Property tests**: Test mathematical properties

---

## Test Organization

### Directory Structure

```
tests/
├── __init__.py
├── conftest.py                 # Pytest fixtures and config
├── test_theory.py             # Mathematical analysis tests
├── test_approximation.py      # Approximation method tests
├── test_ciphers.py           # Cipher implementation tests
├── test_neural_ode.py        # Neural ODE solver tests
├── test_integration.py       # End-to-end integration tests
├── test_performance.py       # Performance benchmarks
└── fixtures/
    ├── sample_data.py        # Test data generators
    ├── cipher_fixtures.py    # Cipher test fixtures
    └── model_fixtures.py     # Model test fixtures
```

### Naming Conventions

- Test files: `test_*.py`
- Test classes: `Test*`
- Test methods: `test_*`
- Fixtures: descriptive names (no prefix)

---

## Unit Testing

### Mathematical Analysis Tests

**File**: `tests/test_theory.py`

```python
import pytest
import torch
from ctdma.theory.mathematical_analysis import GradientInversionAnalyzer

class TestGradientInversionAnalyzer:
    """Tests for gradient inversion analysis."""
    
    @pytest.fixture
    def analyzer(self):
        """Create analyzer instance."""
        return GradientInversionAnalyzer(device='cpu')
    
    @pytest.fixture
    def sample_data(self):
        """Generate sample test data."""
        torch.manual_seed(42)  # Reproducibility
        x = torch.randint(0, 2**16, (100, 8))
        y = torch.randint(0, 2**16, (100, 8))
        return x, y
    
    def test_discontinuity_detection(self, analyzer, sample_data):
        """Test gradient discontinuity detection."""
        x, y = sample_data
        results = analyzer.compute_gradient_discontinuity(x, y)
        
        # Check return structure
        assert 'max_discontinuity' in results
        assert 'avg_discontinuity' in results
        assert 'num_discontinuities' in results
        
        # Check value ranges
        assert results['max_discontinuity'] >= 0
        assert results['num_discontinuities'] >= 0
        assert results['num_discontinuities'] <= len(x)
    
    def test_inversion_probability(self, analyzer, sample_data):
        """Test inversion probability calculation."""
        x, y = sample_data
        prob = analyzer.estimate_inversion_probability(x, y, num_rounds=1)
        
        # Check probability bounds
        assert 0 <= prob <= 1
        
        # For ARX ciphers, probability should be high
        assert prob > 0.9
    
    @pytest.mark.parametrize("modulus", [2**8, 2**16, 2**32])
    def test_different_moduli(self, analyzer, modulus):
        """Test with different modulus values."""
        x = torch.randint(0, modulus, (100, 8))
        y = torch.randint(0, modulus, (100, 8))
        results = analyzer.compute_gradient_discontinuity(
            x, y, modulus=modulus
        )
        
        assert results is not None
        assert results['max_discontinuity'] > 0
    
    def test_edge_cases(self, analyzer):
        """Test edge cases and boundary conditions."""
        # Empty input
        x_empty = torch.tensor([])
        y_empty = torch.tensor([])
        with pytest.raises(ValueError):
            analyzer.compute_gradient_discontinuity(x_empty, y_empty)
        
        # Single element
        x_single = torch.tensor([100])
        y_single = torch.tensor([200])
        results = analyzer.compute_gradient_discontinuity(
            x_single, y_single
        )
        assert results is not None
```

### Approximation Method Tests

**File**: `tests/test_approximation.py`

```python
import pytest
import torch
from ctdma.approximation.bridge import create_approximation_bridge
from ctdma.approximation.metrics import ApproximationMetrics

class TestSigmoidApproximation:
    """Tests for sigmoid approximation."""
    
    @pytest.fixture
    def approximation(self):
        return create_approximation_bridge('sigmoid', steepness=10.0)
    
    def test_forward_pass(self, approximation):
        """Test forward pass produces valid outputs."""
        x = torch.tensor([100, 200, 300])
        y = torch.tensor([150, 250, 350])
        z = approximation.forward(x, y, modulus=2**16)
        
        # Check output shape
        assert z.shape == x.shape
        
        # Check output range
        assert torch.all(z >= 0)
        assert torch.all(z < 2**16)
    
    def test_gradients(self, approximation):
        """Test gradient computation."""
        x = torch.tensor([100.0, 200.0], requires_grad=True)
        y = torch.tensor([150.0, 250.0], requires_grad=True)
        z = approximation.forward(x, y, modulus=2**16)
        
        # Compute gradients
        loss = z.sum()
        loss.backward()
        
        # Check gradients exist
        assert x.grad is not None
        assert y.grad is not None
        
        # Check gradient shapes
        assert x.grad.shape == x.shape
        assert y.grad.shape == y.shape
```

### Cipher Tests

**File**: `tests/test_ciphers.py`

```python
import pytest
import torch
from ctdma.ciphers import create_cipher

class TestSpeckCipher:
    """Tests for Speck cipher implementation."""
    
    @pytest.fixture
    def cipher(self):
        return create_cipher('speck', num_rounds=4)
    
    def test_encryption_decryption(self, cipher):
        """Test that decrypt(encrypt(x)) = x."""
        plaintext = torch.randint(0, 2**32, (10, 2))
        key = torch.randint(0, 2**64, (10, 2))
        
        ciphertext = cipher.encrypt(plaintext, key)
        recovered = cipher.decrypt(ciphertext, key)
        
        assert torch.all(recovered == plaintext)
    
    def test_different_keys_produce_different_outputs(self, cipher):
        """Test that different keys give different ciphertexts."""
        plaintext = torch.randint(0, 2**32, (1, 2))
        key1 = torch.randint(0, 2**64, (1, 2))
        key2 = torch.randint(0, 2**64, (1, 2))
        
        ciphertext1 = cipher.encrypt(plaintext, key1)
        ciphertext2 = cipher.encrypt(plaintext, key2)
        
        # Different keys should produce different outputs
        assert not torch.all(ciphertext1 == ciphertext2)
    
    def test_avalanche_effect(self, cipher):
        """Test avalanche effect (1-bit change → 50% bits change)."""
        plaintext = torch.randint(0, 2**32, (1, 2))
        key = torch.randint(0, 2**64, (1, 2))
        
        # Flip one bit in plaintext
        plaintext_flipped = plaintext.clone()
        plaintext_flipped[0, 0] ^= 1
        
        ciphertext1 = cipher.encrypt(plaintext, key)
        ciphertext2 = cipher.encrypt(plaintext_flipped, key)
        
        # Count bit differences
        diff = ciphertext1 ^ ciphertext2
        num_diff_bits = bin(diff.item()).count('1')
        total_bits = 32 * 2  # Two 32-bit words
        
        avalanche_ratio = num_diff_bits / total_bits
        
        # Should be close to 50%
        assert 0.3 < avalanche_ratio < 0.7
```

---

## Integration Testing

### End-to-End Tests

**File**: `tests/test_integration.py`

```python
import pytest
import torch
from ctdma.ciphers import create_cipher
from ctdma.theory.mathematical_analysis import GradientInversionAnalyzer
from ctdma.approximation.bridge import create_approximation_bridge

def test_complete_analysis_pipeline():
    """Test complete analysis pipeline from cipher to results."""
    # Create cipher
    cipher = create_cipher('speck', num_rounds=2)
    
    # Create analyzer
    analyzer = GradientInversionAnalyzer()
    
    # Analyze cipher
    results = analyzer.analyze_cipher(cipher, num_samples=100)
    
    # Check results structure
    assert 'inversion_probability' in results
    assert 'max_discontinuity' in results
    assert 'gradient_error' in results
    
    # Check values are reasonable
    assert results['inversion_probability'] > 0.9

def test_approximation_comparison_pipeline():
    """Test approximation comparison pipeline."""
    from ctdma.approximation.metrics import compare_approximation_methods
    
    # Create test data
    x = torch.randint(0, 2**16, (100,))
    y = torch.randint(0, 2**16, (100,))
    
    # Create approximations
    methods = {
        'sigmoid': create_approximation_bridge('sigmoid', steepness=10.0),
        'ste': create_approximation_bridge('straight_through')
    }
    
    # Compare
    results = compare_approximation_methods(
        discrete_operation=lambda x, y, m: (x + y) % m,
        approximations=methods,
        test_data=(x, y)
    )
    
    # Check results
    assert 'sigmoid' in results
    assert 'ste' in results
    assert 'l2_error' in results['sigmoid']
```

---

## Performance Testing

### Benchmarks

**File**: `tests/test_performance.py`

```python
import pytest
import time
import torch
from ctdma.ciphers import create_cipher

@pytest.mark.slow
class TestPerformance:
    """Performance benchmark tests."""
    
    def test_cipher_encryption_speed(self):
        """Benchmark cipher encryption speed."""
        cipher = create_cipher('speck', num_rounds=4)
        
        batch_sizes = [10, 100, 1000, 10000]
        
        for batch_size in batch_sizes:
            plaintext = torch.randint(0, 2**32, (batch_size, 2))
            key = torch.randint(0, 2**64, (batch_size, 2))
            
            start = time.time()
            _ = cipher.encrypt(plaintext, key)
            elapsed = time.time() - start
            
            throughput = batch_size / elapsed
            print(f"Batch {batch_size}: {throughput:.0f} encryptions/sec")
            
            # Should process at least 1000 per second
            assert throughput > 1000
    
    @pytest.mark.gpu
    def test_gpu_acceleration(self):
        """Test GPU acceleration provides speedup."""
        if not torch.cuda.is_available():
            pytest.skip("GPU not available")
        
        batch_size = 10000
        plaintext = torch.randint(0, 2**32, (batch_size, 2))
        key = torch.randint(0, 2**64, (batch_size, 2))
        
        # CPU timing
        cipher_cpu = create_cipher('speck', num_rounds=4)
        start = time.time()
        _ = cipher_cpu.encrypt(plaintext, key)
        cpu_time = time.time() - start
        
        # GPU timing
        cipher_gpu = create_cipher('speck', num_rounds=4)
        plaintext_gpu = plaintext.cuda()
        key_gpu = key.cuda()
        start = time.time()
        _ = cipher_gpu.encrypt(plaintext_gpu, key_gpu)
        torch.cuda.synchronize()
        gpu_time = time.time() - start
        
        speedup = cpu_time / gpu_time
        print(f"GPU speedup: {speedup:.2f}x")
        
        # Should see at least 2x speedup
        assert speedup > 2.0
```

---

## Fixtures and Utilities

### Common Fixtures

**File**: `tests/conftest.py`

```python
import pytest
import torch
import numpy as np

@pytest.fixture(scope="session")
def set_random_seeds():
    """Set random seeds for reproducibility."""
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

@pytest.fixture
def device():
    """Get computing device."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@pytest.fixture
def sample_plaintext():
    """Generate sample plaintext."""
    return torch.randint(0, 2**32, (100, 2))

@pytest.fixture
def sample_key():
    """Generate sample key."""
    return torch.randint(0, 2**64, (100, 2))

@pytest.fixture
def temp_directory(tmp_path):
    """Create temporary directory for test outputs."""
    return tmp_path
```

---

## Continuous Integration

### GitHub Actions Workflow

**File**: `.github/workflows/tests.yml`

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.7, 3.8, 3.9, '3.10']
    
    steps:
    - uses: actions/checkout@v2
    
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install dependencies
      run: |
        pip install -e ".[dev]"
    
    - name: Run tests
      run: |
        pytest tests/ --cov=src/ctdma --cov-report=xml
    
    - name: Upload coverage
      uses: codecov/codecov-action@v2
      with:
        file: ./coverage.xml
```

---

## Running Tests

### Basic Usage

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

# Run only fast tests
pytest tests/ -m "not slow"

# Run in parallel
pytest tests/ -n auto
```

### Markers

```python
# Mark slow tests
@pytest.mark.slow
def test_large_scale_experiment():
    pass

# Mark GPU tests
@pytest.mark.gpu
def test_cuda_acceleration():
    pass

# Mark integration tests
@pytest.mark.integration
def test_end_to_end():
    pass
```

---

## Best Practices

1. **Write tests first**: TDD approach
2. **Use fixtures**: Avoid code duplication
3. **Test edge cases**: Empty inputs, boundaries
4. **Parameterize tests**: Test multiple values
5. **Mock external dependencies**: Isolate units
6. **Use descriptive names**: Clear test intent
7. **Keep tests fast**: Move slow tests to separate suite
8. **Assert meaningful things**: Not just "no errors"

---

## Coverage Goals

- **Core modules**: 95%+ coverage
- **Utilities**: 90%+ coverage
- **Examples**: Not required
- **Overall**: 85%+ coverage

---

## See Also

- [Contributing Guide](../CONTRIBUTING.md)
- [API Documentation](README.md)
- [Example Notebooks](../examples/)

---

*Last updated: January 30, 2026*
