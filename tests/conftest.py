"""
Pytest Configuration and Fixtures

Provides common fixtures for all test modules.
"""

import pytest
import torch
import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from ctdma.ciphers.speck import SpeckCipher
from ctdma.ciphers.feistel import FeistelCipher
from ctdma.ciphers.spn import SPnCipher
from ctdma.ciphers.chacha import ChaCha20Cipher
from ctdma.ciphers.salsa20 import Salsa20Cipher
from ctdma.ciphers.blake import Blake2Cipher
from ctdma.ciphers.simon import SimonCipher
from ctdma.ciphers.present import PRESENTCipher


@pytest.fixture(scope='session')
def device():
    """Determine device to use for tests."""
    return 'cuda' if torch.cuda.is_available() else 'cpu'


@pytest.fixture
def random_seed():
    """Set random seed for reproducibility."""
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    return seed


@pytest.fixture
def speck_cipher(device):
    """Create Speck cipher instance."""
    return SpeckCipher(rounds=2, device=device)


@pytest.fixture
def feistel_cipher(device):
    """Create Feistel cipher instance."""
    return FeistelCipher(rounds=2, device=device)


@pytest.fixture
def spn_cipher(device):
    """Create SPN cipher instance."""
    return SPnCipher(rounds=2, device=device)


@pytest.fixture
def chacha_cipher(device):
    """Create ChaCha20 cipher instance."""
    return ChaCha20Cipher(rounds=8, device=device)  # Reduced rounds for testing


@pytest.fixture
def salsa20_cipher(device):
    """Create Salsa20 cipher instance."""
    return Salsa20Cipher(rounds=8, device=device)


@pytest.fixture
def blake_cipher(device):
    """Create Blake2 cipher instance."""
    return Blake2Cipher(rounds=4, device=device)


@pytest.fixture
def simon_cipher(device):
    """Create Simon cipher instance."""
    return SimonCipher(rounds=8, device=device)


@pytest.fixture
def present_cipher(device):
    """Create PRESENT cipher instance."""
    return PRESENTCipher(rounds=4, device=device)


@pytest.fixture
def all_ciphers(speck_cipher, feistel_cipher, spn_cipher, chacha_cipher,
                salsa20_cipher, blake_cipher, simon_cipher, present_cipher):
    """List of all cipher instances."""
    return {
        'speck': speck_cipher,
        'feistel': feistel_cipher,
        'spn': spn_cipher,
        'chacha': chacha_cipher,
        'salsa20': salsa20_cipher,
        'blake': blake_cipher,
        'simon': simon_cipher,
        'present': present_cipher
    }


@pytest.fixture
def arx_ciphers(speck_cipher, chacha_cipher, salsa20_cipher, blake_cipher, simon_cipher):
    """List of ARX cipher instances."""
    return {
        'speck': speck_cipher,
        'chacha': chacha_cipher,
        'salsa20': salsa20_cipher,
        'blake': blake_cipher,
        'simon': simon_cipher
    }


@pytest.fixture
def sample_data(device):
    """Generate sample data for testing."""
    return {
        'plaintexts_2d': torch.rand(10, 2, device=device),
        'plaintexts_16d': torch.rand(10, 16, device=device),
        'plaintexts_64d': torch.rand(10, 64, device=device),
        'keys_small': torch.rand(10, 4, device=device),
        'keys_medium': torch.rand(10, 8, device=device),
        'keys_large': torch.rand(10, 80, device=device)
    }


@pytest.fixture
def tolerance():
    """Numerical tolerance for comparisons."""
    return {
        'rtol': 1e-4,
        'atol': 1e-6
    }
