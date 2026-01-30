"""Utility functions for CTDMA framework."""

import torch


def prepare_cipher_dataset(cipher, key, num_samples=1000):
    """
    Generate plaintext-ciphertext pairs for training.
    
    Args:
        cipher: Cipher instance
        key: Encryption key (single key, not batched)
        num_samples: Number of samples to generate
        
    Returns:
        plaintexts: (num_samples, 2) tensor
        ciphertexts: (num_samples, 2) tensor
    """
    # Generate random plaintexts
    plaintexts = cipher.generate_plaintexts(num_samples)
    
    # Expand key to match batch size
    if key.dim() == 1:
        keys = key.unsqueeze(0).expand(num_samples, -1)
    else:
        keys = key
    
    # Encrypt
    ciphertexts = cipher.encrypt(plaintexts, keys)
    
    return plaintexts, ciphertexts


def to_binary_vector(value, bit_width=16):
    """
    Convert integer value to binary tensor.
    
    Args:
        value: Integer or tensor of integers
        bit_width: Number of bits
        
    Returns:
        Binary tensor of shape (..., bit_width)
    """
    if isinstance(value, int):
        value = torch.tensor(value)
    
    # Create binary representation
    binary = []
    for i in range(bit_width):
        bit = (value >> i) & 1
        binary.append(bit.float())
    
    return torch.stack(binary, dim=-1)