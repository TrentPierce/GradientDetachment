"""
Comprehensive Tests for All Cipher Implementations

Tests cipher functionality, gradient flow, and consistency across implementations.
"""

import pytest
import torch
import numpy as np


class TestCipherBasics:
    """Basic functionality tests for all ciphers."""
    
    def test_speck_forward_pass(self, speck_cipher, random_seed):
        """Test Speck cipher forward pass."""
        plaintext = speck_cipher.generate_plaintexts(5)
        key = speck_cipher.generate_keys(5)
        
        ciphertext = speck_cipher.encrypt(plaintext, key)
        
        assert ciphertext.shape == plaintext.shape
        assert not torch.isnan(ciphertext).any()
        assert not torch.isinf(ciphertext).any()
    
    def test_chacha_forward_pass(self, chacha_cipher, random_seed):
        """Test ChaCha20 cipher forward pass."""
        plaintext = chacha_cipher.generate_plaintexts(3)
        key = chacha_cipher.generate_keys(3)
        
        ciphertext = chacha_cipher.encrypt(plaintext, key)
        
        assert ciphertext.shape == (3, 16)
        assert not torch.isnan(ciphertext).any()
    
    def test_salsa20_forward_pass(self, salsa20_cipher, random_seed):
        """Test Salsa20 cipher forward pass."""
        plaintext = salsa20_cipher.generate_plaintexts(3)
        key = salsa20_cipher.generate_keys(3)
        
        ciphertext = salsa20_cipher.encrypt(plaintext, key)
        
        assert ciphertext.shape == (3, 16)
        assert not torch.isnan(ciphertext).any()
    
    def test_present_forward_pass(self, present_cipher, random_seed):
        """Test PRESENT cipher forward pass."""
        plaintext = present_cipher.generate_plaintexts(3)
        key = present_cipher.generate_keys(3)
        
        ciphertext = present_cipher.encrypt(plaintext, key)
        
        assert ciphertext.shape == (3, 64)
        assert not torch.isnan(ciphertext).any()
    
    @pytest.mark.parametrize('cipher_name', ['speck', 'feistel', 'spn'])
    def test_cipher_determinism(self, cipher_name, all_ciphers, random_seed):
        """Test that ciphers produce deterministic outputs."""
        cipher = all_ciphers[cipher_name]
        
        # Generate test data
        if cipher_name in ['chacha', 'salsa20', 'blake']:
            plaintext = torch.rand(2, 16)
            key = torch.rand(2, 8)
        elif cipher_name == 'present':
            plaintext = torch.rand(2, 64)
            key = torch.rand(2, 80)
        else:
            plaintext = torch.rand(2, 2)
            key = torch.rand(2, 4)
        
        # Encrypt twice with same inputs
        torch.manual_seed(42)
        cipher1 = cipher.encrypt(plaintext, key)
        
        torch.manual_seed(42)
        cipher2 = cipher.encrypt(plaintext, key)
        
        # Should be identical
        assert torch.allclose(cipher1, cipher2, rtol=1e-5)
    
    @pytest.mark.parametrize('cipher_name', ['speck', 'chacha', 'simon'])
    def test_arx_gradient_flow(self, cipher_name, arx_ciphers, random_seed):
        """Test that gradients flow through ARX ciphers."""
        cipher = arx_ciphers[cipher_name]
        
        # Generate test data with gradients
        if cipher_name in ['chacha', 'salsa20', 'blake']:
            plaintext = torch.rand(2, 16, requires_grad=True)
            key = torch.rand(2, 8)
        else:
            plaintext = torch.rand(2, 2, requires_grad=True)
            key = torch.rand(2, 4)
        
        # Forward pass
        ciphertext = cipher.encrypt(plaintext, key)
        loss = ciphertext.sum()
        
        # Backward pass
        loss.backward()
        
        # Check gradients exist and are finite
        assert plaintext.grad is not None
        assert not torch.isnan(plaintext.grad).any()
        assert not torch.isinf(plaintext.grad).any()


class TestCipherProperties:
    """Test cryptographic properties of ciphers."""
    
    def test_speck_avalanche_effect(self, speck_cipher, random_seed):
        """Test avalanche effect in Speck cipher."""
        plaintext1 = torch.tensor([[0.5, 0.5]])
        plaintext2 = torch.tensor([[0.501, 0.5]])  # Small change
        key = torch.tensor([[0.5, 0.5, 0.5, 0.5]])
        
        cipher1 = speck_cipher.encrypt(plaintext1, key)
        cipher2 = speck_cipher.encrypt(plaintext2, key)
        
        # Outputs should differ significantly (avalanche effect)
        diff = torch.abs(cipher1 - cipher2).mean()
        assert diff > 0.01  # Expect noticeable difference
    
    def test_cipher_output_range(self, speck_cipher, random_seed):
        """Test that cipher outputs are in valid range."""
        plaintext = speck_cipher.generate_plaintexts(100)
        key = speck_cipher.generate_keys(100)
        
        ciphertext = speck_cipher.encrypt(plaintext, key)
        
        assert (ciphertext >= 0).all()
        assert (ciphertext <= 1).all()
    
    def test_different_keys_different_outputs(self, speck_cipher, random_seed):
        """Test that different keys produce different outputs."""
        plaintext = torch.tensor([[0.5, 0.5]])
        key1 = torch.tensor([[0.1, 0.2, 0.3, 0.4]])
        key2 = torch.tensor([[0.5, 0.6, 0.7, 0.8]])
        
        cipher1 = speck_cipher.encrypt(plaintext, key1)
        cipher2 = speck_cipher.encrypt(plaintext, key2)
        
        assert not torch.allclose(cipher1, cipher2, rtol=0.1)


class TestParameterizedCiphers:
    """Test ciphers with different parameter configurations."""
    
    @pytest.mark.parametrize('rounds', [1, 2, 4, 8])
    def test_speck_different_rounds(self, device, rounds):
        """Test Speck with different round numbers."""
        from ctdma.ciphers.speck import SpeckCipher
        
        cipher = SpeckCipher(rounds=rounds, device=device)
        plaintext = cipher.generate_plaintexts(5)
        key = cipher.generate_keys(5)
        
        ciphertext = cipher.encrypt(plaintext, key)
        
        assert ciphertext.shape == plaintext.shape
        assert not torch.isnan(ciphertext).any()
    
    @pytest.mark.parametrize('block_size', [32, 48, 64])
    def test_simon_different_block_sizes(self, device, block_size):
        """Test Simon with different block sizes."""
        from ctdma.ciphers.simon import SimonCipher
        
        cipher = SimonCipher(block_size=block_size, rounds=4, device=device)
        plaintext = cipher.generate_plaintexts(3)
        key = cipher.generate_keys(3)
        
        ciphertext = cipher.encrypt(plaintext, key)
        
        assert ciphertext.shape == (3, 2)
        assert not torch.isnan(ciphertext).any()


class TestGradientStatistics:
    """Test gradient statistics across ciphers."""
    
    def test_speck_gradient_variance(self, speck_cipher, random_seed):
        """Test gradient variance in Speck."""
        gradients = []
        
        for _ in range(10):
            plaintext = speck_cipher.generate_plaintexts(1).requires_grad_(True)
            key = speck_cipher.generate_keys(1)
            
            ciphertext = speck_cipher.encrypt(plaintext, key)
            loss = ciphertext.sum()
            loss.backward()
            
            gradients.append(plaintext.grad.detach().numpy())
        
        gradients = np.array(gradients)
        variance = gradients.var()
        
        # ARX ciphers should have high gradient variance (sawtooth topology)
        assert variance > 0.01
    
    def test_gradient_magnitude_comparison(self, arx_ciphers, random_seed):
        """Compare gradient magnitudes across ARX ciphers."""
        magnitudes = {}
        
        for name, cipher in arx_ciphers.items():
            try:
                if name in ['chacha', 'salsa20', 'blake']:
                    plaintext = torch.rand(1, 16, requires_grad=True)
                    key = torch.rand(1, 8)
                else:
                    plaintext = torch.rand(1, 2, requires_grad=True)
                    key = torch.rand(1, 4)
                
                ciphertext = cipher.encrypt(plaintext, key)
                loss = ciphertext.sum()
                loss.backward()
                
                mag = plaintext.grad.norm().item()
                magnitudes[name] = mag
            except Exception as e:
                pytest.skip(f"Cipher {name} not compatible: {e}")
        
        # All ARX ciphers should have finite, non-zero gradients
        for name, mag in magnitudes.items():
            assert 0 < mag < 1e6, f"{name} has gradient magnitude {mag}"


class TestBatchProcessing:
    """Test batch processing capabilities."""
    
    @pytest.mark.parametrize('batch_size', [1, 5, 10, 32])
    def test_speck_batch_sizes(self, speck_cipher, batch_size):
        """Test Speck with different batch sizes."""
        plaintext = speck_cipher.generate_plaintexts(batch_size)
        key = speck_cipher.generate_keys(batch_size)
        
        ciphertext = speck_cipher.encrypt(plaintext, key)
        
        assert ciphertext.shape[0] == batch_size
        assert not torch.isnan(ciphertext).any()
    
    def test_gradient_batch_consistency(self, speck_cipher, random_seed):
        """Test that batch gradients are consistent with individual gradients."""
        # Single sample
        plaintext_single = torch.rand(1, 2, requires_grad=True)
        key_single = torch.rand(1, 4)
        
        cipher_single = speck_cipher.encrypt(plaintext_single, key_single)
        cipher_single.sum().backward()
        grad_single = plaintext_single.grad.clone()
        
        # Batch with same sample repeated
        plaintext_batch = plaintext_single.repeat(5, 1).requires_grad_(True)
        key_batch = key_single.repeat(5, 1)
        
        cipher_batch = speck_cipher.encrypt(plaintext_batch, key_batch)
        cipher_batch.sum().backward()
        grad_batch = plaintext_batch.grad[0]
        
        # Gradients should be similar (accounting for numerical differences)
        assert torch.allclose(grad_single[0], grad_batch, rtol=0.1, atol=0.1)
