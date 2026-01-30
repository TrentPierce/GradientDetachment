# Cipher Implementations API

Complete API reference for cipher implementations (`ctdma.ciphers`).

## Table of Contents

- [Base Cipher Interface](#base-cipher-interface)
- [ARX Ciphers](#arx-ciphers)
- [Feistel Ciphers](#feistel-ciphers)
- [SPN Ciphers](#spn-ciphers)
- [Factory Functions](#factory-functions)
- [Utility Functions](#utility-functions)

---

## Base Cipher Interface

All cipher implementations inherit from `BaseCipher`.

### BaseCipher

```python
class BaseCipher(ABC):
    """
    Abstract base class for all cipher implementations.
    
    Provides common interface for encryption, decryption, and smooth approximations.
    """
    
    def __init__(
        self,
        num_rounds: int,
        use_smooth: bool = True,
        approximation_method: str = 'sigmoid',
        **approx_kwargs
    ):
        """
        Initialize cipher.
        
        Args:
            num_rounds: Number of cipher rounds
            use_smooth: Use smooth approximations (for gradient analysis)
            approximation_method: Method for smooth operations
            **approx_kwargs: Parameters for approximation method
        """
    
    @abstractmethod
    def encrypt(self, plaintext: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
        """Encrypt plaintext with key."""
    
    @abstractmethod
    def decrypt(self, ciphertext: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
        """Decrypt ciphertext with key."""
    
    @abstractmethod
    def encrypt_round(
        self,
        state: torch.Tensor,
        round_key: torch.Tensor
    ) -> torch.Tensor:
        """Single round encryption."""
    
    def set_approximation(self, method: str, **kwargs):
        """Change approximation method."""
    
    def get_round_keys(self, key: torch.Tensor) -> List[torch.Tensor]:
        """Generate round keys from master key."""
```

---

## ARX Ciphers

**ARX** = Addition, Rotation, XOR

Ciphers based on modular addition, bitwise rotation, and XOR operations.

### SpeckCipher

Implementation of the Speck lightweight block cipher.

```python
class SpeckCipher(BaseCipher):
    """
    Speck ARX cipher implementation.
    
    Speck is a family of lightweight block ciphers designed for
    constrained environments. Uses only Addition, Rotation, and XOR.
    
    Reference: "The Speck Family of Lightweight Block Ciphers"
               Beaulieu et al., 2013
    """
    
    def __init__(
        self,
        block_size: int = 32,
        key_size: int = 64,
        num_rounds: int = 4,
        rotation_alpha: int = 7,
        rotation_beta: int = 2,
        use_smooth: bool = True,
        approximation_method: str = 'sigmoid',
        steepness: float = 10.0
    ):
        """
        Initialize Speck cipher.
        
        Args:
            block_size: Block size in bits (default: 32)
            key_size: Key size in bits (default: 64)
            num_rounds: Number of rounds (default: 4)
            rotation_alpha: Left rotation amount (default: 7)
            rotation_beta: Right rotation amount (default: 2)
            use_smooth: Use smooth approximations
            approximation_method: Approximation method ('sigmoid', 'ste', etc.)
            steepness: Steepness for sigmoid approximation
        
        Supported configurations:
            - Speck32/64: 32-bit blocks, 64-bit keys, 22 rounds
            - Speck64/128: 64-bit blocks, 128-bit keys, 27 rounds
            - Speck128/256: 128-bit blocks, 256-bit keys, 34 rounds
        
        Example:
            >>> cipher = SpeckCipher(block_size=32, num_rounds=4)
            >>> plaintext = torch.randint(0, 2**32, (batch_size, 2))
            >>> key = torch.randint(0, 2**64, (batch_size, 2))
            >>> ciphertext = cipher.encrypt(plaintext, key)
        """
    
    def encrypt_round(
        self,
        state: torch.Tensor,
        round_key: torch.Tensor
    ) -> torch.Tensor:
        """
        Single Speck round.
        
        Round function:
            1. x = ((x >>> α) + y) mod 2^n
            2. y = (y <<< β) ⊕ x
        
        Args:
            state: Current state [x, y]
            round_key: Round key
        
        Returns:
            Updated state
        """
    
    def analyze_gradient_inversion(self) -> Dict[str, float]:
        """
        Analyze gradient inversion for this cipher.
        
        Returns:
            Dictionary with inversion statistics
        """
```

**Key Properties:**
- **Security**: 2^(n/2) differential/linear attacks after sufficient rounds
- **Gradient Inversion**: ~97.5% at 1 round, ~99% at 2 rounds
- **Computational Cost**: Very lightweight (ARX operations only)
- **Use Case**: Embedded systems, IoT devices

---

## Feistel Ciphers

Ciphers based on Feistel network structure.

### FeistelCipher

Generic Feistel network implementation.

```python
class FeistelCipher(BaseCipher):
    """
    Generic Feistel network cipher.
    
    Structure:
        L_{i+1} = R_i
        R_{i+1} = L_i ⊕ F(R_i, K_i)
    
    where F is the round function.
    """
    
    def __init__(
        self,
        block_size: int = 64,
        num_rounds: int = 16,
        round_function: str = 'sbox',
        use_smooth: bool = True,
        **kwargs
    ):
        """
        Initialize Feistel cipher.
        
        Args:
            block_size: Block size in bits
            num_rounds: Number of Feistel rounds
            round_function: Type of F function
                - 'sbox': S-box based (DES-like)
                - 'arx': ARX operations
                - 'linear': Simple linear function
            use_smooth: Use smooth approximations
        
        Example:
            >>> cipher = FeistelCipher(block_size=64, num_rounds=16)
            >>> ciphertext = cipher.encrypt(plaintext, key)
        """
    
    def feistel_round(
        self,
        left: torch.Tensor,
        right: torch.Tensor,
        round_key: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Single Feistel round.
        
        Args:
            left: Left half of block
            right: Right half of block
            round_key: Key for this round
        
        Returns:
            (new_left, new_right)
        """
```

**Key Properties:**
- **Structure**: Symmetric (same for encryption/decryption)
- **Gradient Inversion**: ~15% at 1 round (weaker than ARX)
- **Security**: Depends on round function quality
- **Examples**: DES, Blowfish, Camellia

---

## SPN Ciphers

**SPN** = Substitution-Permutation Network

Ciphers based on substitution (S-boxes) and permutation layers.

### SPNCipher

Generic SPN implementation (AES-like).

```python
class SPNCipher(BaseCipher):
    """
    Substitution-Permutation Network cipher.
    
    Structure:
        1. SubBytes (S-box substitution)
        2. ShiftRows (permutation)
        3. MixColumns (diffusion)
        4. AddRoundKey (key mixing)
    """
    
    def __init__(
        self,
        block_size: int = 128,
        num_rounds: int = 10,
        sbox_size: int = 8,
        use_smooth: bool = True,
        sbox_approximation: str = 'table_lookup',
        **kwargs
    ):
        """
        Initialize SPN cipher.
        
        Args:
            block_size: Block size in bits (default: 128)
            num_rounds: Number of rounds (default: 10)
            sbox_size: S-box input/output size in bits
            use_smooth: Use smooth approximations
            sbox_approximation: How to approximate S-boxes
                - 'table_lookup': Direct lookup (discrete)
                - 'interpolation': Smooth interpolation
                - 'neural': Neural network approximation
        
        Example:
            >>> cipher = SPNCipher(block_size=128, num_rounds=10)
            >>> ciphertext = cipher.encrypt(plaintext, key)
        """
    
    def substitute_bytes(
        self,
        state: torch.Tensor,
        sbox: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply S-box substitution.
        
        Args:
            state: Current state
            sbox: Substitution box (lookup table)
        
        Returns:
            Substituted state
        """
    
    def mix_columns(
        self,
        state: torch.Tensor,
        mix_matrix: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply MixColumns transformation.
        
        Args:
            state: Current state
            mix_matrix: Mixing matrix
        
        Returns:
            Mixed state
        """
```

**Key Properties:**
- **Structure**: Layers of substitution and permutation
- **Gradient Inversion**: ~12% at 1 round (intermediate)
- **Security**: High diffusion and confusion
- **Examples**: AES, Serpent, PRESENT

---

## Factory Functions

Convenient functions for creating cipher instances.

### create_cipher

```python
def create_cipher(
    cipher_type: str,
    num_rounds: int = 4,
    use_smooth: bool = True,
    **kwargs
) -> BaseCipher:
    """
    Factory function to create cipher instances.
    
    Args:
        cipher_type: Type of cipher
            - 'speck': Speck ARX cipher
            - 'feistel': Generic Feistel network
            - 'spn': Substitution-Permutation Network
            - 'aes': AES (SPN variant)
            - 'des': DES (Feistel variant)
        num_rounds: Number of rounds
        use_smooth: Use smooth approximations
        **kwargs: Cipher-specific parameters
    
    Returns:
        Configured cipher instance
    
    Example:
        >>> # Create Speck cipher
        >>> speck = create_cipher('speck', num_rounds=4)
        >>> 
        >>> # Create Feistel with custom function
        >>> feistel = create_cipher(
        ...     'feistel',
        ...     num_rounds=16,
        ...     round_function='arx'
        ... )
        >>> 
        >>> # Create SPN cipher
        >>> spn = create_cipher('spn', num_rounds=10)
    """
```

### compare_ciphers

```python
def compare_ciphers(
    cipher_types: List[str],
    num_samples: int = 1000,
    num_rounds: int = 4
) -> Dict[str, Dict[str, float]]:
    """
    Compare gradient inversion across cipher families.
    
    Args:
        cipher_types: List of cipher types to compare
        num_samples: Number of test samples
        num_rounds: Number of rounds for each cipher
    
    Returns:
        Dictionary: {cipher_type: {metric: value}}
    
    Metrics:
        - 'inversion_probability': Estimated P_inv
        - 'max_discontinuity': Max gradient jump
        - 'avg_gradient_error': Average gradient error
        - 'convergence_failure_rate': Fraction of failed convergences
    
    Example:
        >>> results = compare_ciphers(
        ...     ['speck', 'feistel', 'spn'],
        ...     num_samples=1000,
        ...     num_rounds=2
        ... )
        >>> for cipher, metrics in results.items():
        ...     print(f"{cipher}: P_inv = {metrics['inversion_probability']:.2%}")
    """
```

---

## Utility Functions

### Key Generation

```python
def generate_random_key(
    key_size: int,
    batch_size: int = 1
) -> torch.Tensor:
    """
    Generate random cryptographic keys.
    
    Args:
        key_size: Key size in bits
        batch_size: Number of keys
    
    Returns:
        Random keys tensor
    """

def expand_key(
    master_key: torch.Tensor,
    num_round_keys: int,
    key_schedule: str = 'simple'
) -> List[torch.Tensor]:
    """
    Expand master key into round keys.
    
    Args:
        master_key: Master key
        num_round_keys: Number of round keys needed
        key_schedule: Key schedule algorithm
    
    Returns:
        List of round keys
    """
```

### Testing Utilities

```python
def test_encryption_correctness(
    cipher: BaseCipher,
    num_tests: int = 100
) -> bool:
    """
    Test that encrypt(decrypt(x)) = x.
    
    Args:
        cipher: Cipher to test
        num_tests: Number of test cases
    
    Returns:
        True if all tests pass
    """

def measure_avalanche_effect(
    cipher: BaseCipher,
    num_samples: int = 1000
) -> float:
    """
    Measure avalanche effect (bit flip propagation).
    
    Args:
        cipher: Cipher to analyze
        num_samples: Number of test samples
    
    Returns:
        Avalanche coefficient (ideal: 0.5)
    """
```

---

## Cross-Cipher Comparison

### Security Analysis

```python
class CipherSecurityAnalyzer:
    """Analyze security properties of ciphers."""
    
    def analyze_differential_properties(
        self,
        cipher: BaseCipher,
        num_samples: int = 10000
    ) -> Dict[str, float]:
        """
        Analyze differential cryptanalysis resistance.
        
        Returns:
            Maximum differential probability
        """
    
    def analyze_linear_properties(
        self,
        cipher: BaseCipher,
        num_samples: int = 10000
    ) -> Dict[str, float]:
        """
        Analyze linear cryptanalysis resistance.
        
        Returns:
            Maximum linear bias
        """
    
    def analyze_ml_resistance(
        self,
        cipher: BaseCipher,
        attack_type: str = 'neural_ode'
    ) -> Dict[str, float]:
        """
        Analyze resistance to ML-based attacks.
        
        Args:
            cipher: Cipher to analyze
            attack_type: Type of ML attack
        
        Returns:
            Resistance metrics including gradient inversion rate
        """
```

---

## Performance Benchmarks

### Typical Performance (CPU, 1000 samples)

| Cipher | Encrypt | Decrypt | Gradient | Memory |
|--------|---------|---------|----------|---------|
| Speck (4 rounds) | 2ms | 2ms | 5ms | 4MB |
| Feistel (16 rounds) | 15ms | 15ms | 30ms | 8MB |
| SPN (10 rounds) | 12ms | 12ms | 25ms | 6MB |

### Scalability

- **Batch processing**: Linear scaling up to GPU memory limits
- **Round scaling**: Linear with number of rounds
- **Block size**: Quadratic for MixColumns operations

---

## Examples

### Complete Cipher Analysis

```python
import torch
from ctdma.ciphers import create_cipher, compare_ciphers
from ctdma.theory.mathematical_analysis import GradientInversionAnalyzer

# Create ciphers
speck = create_cipher('speck', num_rounds=4)
feistel = create_cipher('feistel', num_rounds=4)
spn = create_cipher('spn', num_rounds=4)

# Generate test data
batch_size = 1000
plaintext = torch.randint(0, 2**32, (batch_size, 2))
key = torch.randint(0, 2**64, (batch_size, 2))

# Test encryption
for cipher in [speck, feistel, spn]:
    ciphertext = cipher.encrypt(plaintext, key)
    recovered = cipher.decrypt(ciphertext, key)
    correct = torch.all(recovered == plaintext)
    print(f"{cipher.__class__.__name__}: Correctness = {correct}")

# Compare gradient inversion
results = compare_ciphers(['speck', 'feistel', 'spn'])
for cipher_type, metrics in results.items():
    print(f"\n{cipher_type}:")
    print(f"  Inversion probability: {metrics['inversion_probability']:.2%}")
    print(f"  Max discontinuity: {metrics['max_discontinuity']:.2f}")
```

---

## See Also

- [Mathematical Theory API](mathematical_theory.md)
- [Approximation Methods API](approximation_methods.md)
- [Example: Cipher Evaluation](../examples/cipher_evaluation.ipynb)

---

*Last updated: January 30, 2026*
