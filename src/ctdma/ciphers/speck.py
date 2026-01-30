"""
Speck-32/64 ARX Cipher Implementation with Smooth Approximations

This module implements the Speck cipher using smooth differentiable approximations
of XOR, rotation, and modular addition operations for use with Neural ODEs.
"""

import torch
import torch.nn as nn
import numpy as np


class SpeckCipher(nn.Module):
    """
    Speck-32/64 cipher with smooth approximations for Neural ODE compatibility.
    
    Uses:
    - Soft XOR: tanh-based approximation
    - Smooth rotation: Interpolation-based
    - Soft modular addition: Sigmoid-modulated addition
    
    Args:
        rounds: Number of rounds (default: 4 for security)
        device: torch device
    """
    
    def __init__(self, rounds=4, device='cpu'):
        super().__init__()
        self.rounds = rounds
        self.block_size = 32
        self.key_size = 64
        self.word_size = 16
        self.alpha = 7  # Rotation amount 1
        self.beta = 2   # Rotation amount 2
        self.device = device
        self.mask = (1 << self.word_size) - 1
        
    def _soft_xor(self, x, y, steepness=10.0):
        """
        Smooth XOR approximation using tanh.
        
        XOR(x,y) = x + y - 2*AND(x,y)
        Using smooth AND: (tanh(steepness*(x+y-1)) + 1) / 2
        """
        # Normalize to [-1, 1] for tanh
        x_norm = 2 * x - 1
        y_norm = 2 * y - 1
        
        # Smooth multiplication (AND)
        smooth_and = (torch.tanh(steepness * (x + y - 1)) + 1) / 2
        
        # XOR from AND
        result = x + y - 2 * smooth_and
        return torch.clamp(result, 0, 1)
    
    def _smooth_rotate(self, x, r):
        """
        Differentiable rotation approximation.
        
        For 16-bit words, we use a weighted interpolation approach
        that's differentiable w.r.t. the input bits.
        """
        # For differentiable rotation, we use a soft shift
        # This is a simplified version - in practice you'd use
        # circular convolution with learned shift kernels
        
        # Convert to float representation for smooth rotation
        x_float = x.float()
        
        # Soft rotation: interpolate between shifted versions
        # For 16-bit: shift by r positions
        shifted_left = x_float * (2 ** r)
        shifted_right = x_float / (2 ** (16 - r))
        
        # Combine with modulo
        result = (shifted_left + shifted_right) % (2 ** 16)
        
        # Normalize back to [0, 1] range
        return result / (2 ** 16)
    
    def _soft_add(self, x, y, steepness=10.0):
        """
        Smooth modular addition with sigmoid modulation.
        
        Regular addition: z = x + y
        Modular addition: z = (x + y) mod 2^n
        
        Soft version uses sigmoid to smoothly wrap around.
        """
        # Standard addition in float space
        sum_val = x + y
        
        # Soft modulo using sigmoid
        # When sum approaches 2^n, smoothly wrap to 0
        modulus = 2.0  # For normalized [0,1] inputs
        
        # Create smooth transition at modulus boundary
        wrap_amount = torch.sigmoid(steepness * (sum_val - modulus))
        result = sum_val - modulus * wrap_amount
        
        return torch.clamp(result, 0, modulus)
    
    def _round_function(self, x, y, k):
        """
        Single round of Speck with smooth operations.
        
        Standard Speck round:
        x = ROR(x, 7); x = x + y; x = x ^ k
        y = ROL(y, 2); y = y ^ x
        
        Smooth version uses differentiable approximations.
        """
        # Apply soft rotation
        x = self._smooth_rotate(x, self.alpha)
        y = self._smooth_rotate(y, -self.beta)
        
        # Soft addition with key
        x = self._soft_add(x, y)
        x = self._soft_add(x, k)
        
        # Soft XOR
        y = self._soft_xor(y, x)
        
        return x, y
    
    def forward(self, plaintext, key):
        """
        Encrypt plaintext with key.
        
        Args:
            plaintext: (batch_size, 2) tensor with left and right halves
            key: (batch_size, 4) tensor with 4 key words
            
        Returns:
            ciphertext: (batch_size, 2) encrypted output
        """
        # Split plaintext into left and right
        x = plaintext[:, 0]
        y = plaintext[:, 1]
        
        # Generate round keys (simplified - in real Speck this is more complex)
        round_keys = self._key_schedule(key)
        
        # Apply rounds
        for i in range(self.rounds):
            k = round_keys[i % len(round_keys)]
            x, y = self._round_function(x, y, k)
        
        return torch.stack([x, y], dim=1)
    
    def _key_schedule(self, key):
        """Generate round keys from master key."""
        # Simplified key schedule - expand key words
        round_keys = []
        for i in range(self.rounds):
            # Cycle through key words
            idx = i % key.shape[1]
            round_keys.append(key[:, idx])
        return round_keys
    
    def encrypt(self, plaintext, key):
        """Standard PyTorch-style encryption interface."""
        return self.forward(plaintext, key)
    
    def generate_plaintexts(self, num_samples):
        """Generate random plaintext pairs."""
        return torch.rand(num_samples, 2, device=self.device)
    
    def generate_keys(self, num_samples):
        """Generate random keys."""
        return torch.rand(num_samples, 4, device=self.device)