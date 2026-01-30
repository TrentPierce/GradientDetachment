"""
SPN (Substitution-Permutation Network) Cipher Implementation

A simple SPN cipher using S-boxes and bit permutations,
for comparison with ARX and Feistel ciphers.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SPnCipher(nn.Module):
    """
    Simple SPN cipher with S-boxes and permutation.
    
    Args:
        rounds: Number of rounds (default: 4)
        block_size: Block size in bits (default: 16)
        device: torch device
    """
    
    def __init__(self, rounds=4, block_size=16, device='cpu'):
        super().__init__()
        self.rounds = rounds
        self.block_size = block_size
        self.device = device
        
        # S-box: 4-bit to 4-bit substitution
        # Using a simple non-linear S-box
        self.sbox = torch.tensor([0xC, 0x5, 0x6, 0xB, 0x9, 0x0, 0xA, 0xD,
                                  0x3, 0xE, 0xF, 0x8, 0x4, 0x7, 0x1, 0x2],
                                 dtype=torch.long, device=device)
        
        # Permutation pattern (bit positions)
        self.register_buffer('perm', self._create_permutation())
        
    def _create_permutation(self):
        """Create bit permutation pattern for diffusion."""
        # Simple cyclic shift permutation
        perm = torch.arange(self.block_size, device=self.device)
        perm = torch.roll(perm, shifts=self.block_size // 4, dims=0)
        return perm
    
    def _soft_sbox(self, x):
        """
        Differentiable S-box approximation using embedding lookup.
        
        Instead of hard indexing, we use soft attention over S-box entries.
        """
        batch_size = x.shape[0]
        
        # x is in range [0, 1] representing 4-bit value / 16
        # Convert to 16-way one-hot (soft)
        x_scaled = x * 15  # Scale to 0-15
        
        # Create soft one-hot using softmax
        indices = torch.arange(16, dtype=torch.float32, device=x.device)
        soft_one_hot = F.softmax(-(x_scaled.unsqueeze(-1) - indices) ** 2 * 10, dim=-1)
        
        # Lookup S-box values (normalized to [0, 1])
        sbox_values = self.sbox.float() / 15.0
        result = (soft_one_hot * sbox_values).sum(dim=-1)
        
        return result
    
    def _soft_permute(self, x):
        """Soft bit permutation using learned weights."""
        # Approximate permutation as linear transformation
        # This is a simplification - real implementation would be more complex
        
        # For differentiable permutation, we use attention mechanism
        batch_size = x.shape[0]
        x_expanded = x.unsqueeze(-1)  # (batch, block_size, 1)
        
        # Create permutation matrix (learned or fixed)
        perm_matrix = torch.eye(self.block_size, device=x.device)[self.perm]
        
        result = torch.matmul(x_expanded.transpose(1, 2), perm_matrix.t()).squeeze(1)
        return result
    
    def _add_round_key(self, x, k):
        """XOR with round key."""
        return self._soft_xor(x, k)
    
    def _soft_xor(self, x, y, steepness=10.0):
        """Smooth XOR approximation."""
        smooth_and = (torch.tanh(steepness * (x + y - 1)) + 1) / 2
        return x + y - 2 * smooth_and
    
    def forward(self, plaintext, key):
        """
        Encrypt plaintext.
        
        Args:
            plaintext: (batch_size,) block values
            key: (batch_size, rounds) round keys
            
        Returns:
            ciphertext: (batch_size,) encrypted output
        """
        x = plaintext
        
        # SPN rounds
        for i in range(self.rounds):
            # Key addition (XOR)
            k = key[:, i]
            x = self._add_round_key(x, k)
            
            # Substitution (S-box)
            # Split into 4-bit nibbles and apply S-box
            x = self._soft_sbox(x)
            
            # Permutation (mixing) - skip on last round
            if i < self.rounds - 1:
                x = self._soft_permute(x)
        
        # Final key addition
        x = self._add_round_key(x, key[:, 0])
        
        return x
    
    def encrypt(self, plaintext, key):
        """Standard encryption interface."""
        return self.forward(plaintext, key)
    
    def generate_plaintexts(self, num_samples):
        """Generate random plaintexts."""
        return torch.rand(num_samples, device=self.device)
    
    def generate_keys(self, num_samples):
        """Generate random keys."""
        return torch.rand(num_samples, self.rounds, device=self.device)