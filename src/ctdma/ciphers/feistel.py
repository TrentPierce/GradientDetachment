"""
Feistel Cipher Implementation

A simple Feistel cipher using XOR and permutation (no modular addition),
used for comparison with ARX ciphers.
"""

import torch
import torch.nn as nn


class FeistelCipher(nn.Module):
    """
    Simple Feistel cipher with XOR and permutation.
    
    Demonstrates that ciphers without modular addition are more
    vulnerable to Neural ODE attacks than ARX designs.
    
    Args:
        rounds: Number of rounds (default: 4)
        block_size: Block size in bits (default: 32)
        device: torch device
    """
    
    def __init__(self, rounds=4, block_size=32, device='cpu'):
        super().__init__()
        self.rounds = rounds
        self.block_size = block_size
        self.half_size = block_size // 2
        self.device = device
        
        # Simple permutation pattern (bit shuffle)
        self.register_buffer('perm', self._create_permutation())
        
    def _create_permutation(self):
        """Create a bit permutation pattern."""
        # Simple bit shuffle: reverse order
        perm = torch.arange(self.half_size - 1, -1, -1, dtype=torch.long)
        return perm
    
    def _round_function(self, R, K):
        """
        Feistel round function: F(R, K) = P(R XOR K)
        
        Uses only XOR and permutation - no modular arithmetic.
        """
        # XOR with key
        x = R ^ K
        
        # Apply permutation (simple bit shuffle)
        # In differentiable version, we use soft permutation
        x = self._soft_permute(x)
        
        return x
    
    def _soft_permute(self, x):
        """Soft differentiable permutation approximation."""
        # For a simple Feistel, we can use a linear transformation
        # that approximates the permutation
        
        # Bit reversal as matrix multiplication (simplified)
        # In practice, this would be a learned or predefined matrix
        weights = torch.eye(self.half_size, device=x.device)
        weights = weights.flip(0)  # Reverse rows for bit reversal
        
        return torch.matmul(x.unsqueeze(-1), weights.t()).squeeze(-1)
    
    def forward(self, plaintext, key):
        """
        Encrypt using Feistel network.
        
        Args:
            plaintext: (batch_size, 2) left and right halves
            key: (batch_size, rounds) round keys
            
        Returns:
            ciphertext: (batch_size, 2) encrypted output
        """
        L = plaintext[:, 0]
        R = plaintext[:, 1]
        
        # Feistel rounds
        for i in range(self.rounds):
            K = key[:, i % key.shape[1]]
            
            # Standard Feistel: L' = R, R' = L XOR F(R, K)
            L_new = R
            R_new = L ^ self._round_function(R, K)
            
            L, R = L_new, R_new
        
        # Final swap (optional, depends on Feistel variant)
        return torch.stack([R, L], dim=1)
    
    def encrypt(self, plaintext, key):
        """Standard encryption interface."""
        return self.forward(plaintext, key)
    
    def generate_plaintexts(self, num_samples):
        """Generate random plaintexts."""
        return torch.randint(0, 2**self.half_size, (num_samples, 2), 
                           dtype=torch.float32, device=self.device) / (2**self.half_size)
    
    def generate_keys(self, num_samples):
        """Generate random keys."""
        return torch.randint(0, 2**self.half_size, (num_samples, self.rounds), 
                           dtype=torch.float32, device=self.device) / (2**self.half_size)