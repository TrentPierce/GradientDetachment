"""Unit tests for GradientDetachment."""

import unittest
import torch
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from ctdma.ciphers.speck import SpeckCipher
from ctdma.ciphers.feistel import FeistelCipher
from ctdma.ciphers.spn import SPnCipher


class TestCiphers(unittest.TestCase):
    """Test cipher implementations."""
    
    def test_speck_encryption(self):
        """Test Speck cipher encrypts without errors."""
        cipher = SpeckCipher(rounds=1)
        plaintext = cipher.generate_plaintexts(10)
        key = cipher.generate_keys(10)
        
        # Should not raise exception
        ciphertext = cipher.encrypt(plaintext, key)
        
        # Check output shape
        self.assertEqual(ciphertext.shape, plaintext.shape)
    
    def test_feistel_encryption(self):
        """Test Feistel cipher encrypts without errors."""
        cipher = FeistelCipher(rounds=1)
        plaintext = cipher.generate_plaintexts(10)
        key = cipher.generate_keys(10)
        
        ciphertext = cipher.encrypt(plaintext, key)
        self.assertEqual(ciphertext.shape, plaintext.shape)
    
    def test_spn_encryption(self):
        """Test SPN cipher encrypts without errors."""
        cipher = SPnCipher(rounds=1)
        plaintext = cipher.generate_plaintexts(10)
        key = cipher.generate_keys(10)
        
        ciphertext = cipher.encrypt(plaintext, key)
        self.assertEqual(ciphertext.shape, plaintext.shape)


class TestGradientFlow(unittest.TestCase):
    """Test that gradients flow correctly."""
    
    def test_speck_gradients(self):
        """Test gradients flow through Speck cipher."""
        cipher = SpeckCipher(rounds=1)
        plaintext = cipher.generate_plaintexts(1)
        key = cipher.generate_keys(1)
        
        # Enable gradient tracking
        plaintext.requires_grad = True
        key.requires_grad = True
        
        # Encrypt
        ciphertext = cipher.encrypt(plaintext, key)
        
        # Compute dummy loss
        loss = ciphertext.sum()
        
        # Should not raise exception
        loss.backward()
        
        # Check gradients exist
        self.assertIsNotNone(plaintext.grad)
        self.assertIsNotNone(key.grad)


if __name__ == '__main__':
    unittest.main()
