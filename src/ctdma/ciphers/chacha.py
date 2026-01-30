"""
ChaCha20 Stream Cipher Implementation

ChaCha20 is an ARX-based stream cipher designed by Daniel J. Bernstein.
It uses addition, rotation, and XOR operations (no S-boxes).

This implementation provides smooth approximations for Neural ODE compatibility.

Reference:
- Bernstein, D.J. "ChaCha, a variant of Salsa20"
"""

import torch
import torch.nn as nn
import numpy as np


class ChaCha20Cipher(nn.Module):
    """
    ChaCha20 stream cipher with smooth approximations.
    
    ChaCha20 uses a 512-bit state (16 x 32-bit words) and performs
    20 rounds of the "quarter round" function.
    
    Quarter round: (a, b, c, d) where each is a 32-bit word
    a += b; d ^= a; d <<<= 16;
    c += d; b ^= c; b <<<= 12;
    a += b; d ^= a; d <<<= 8;
    c += d; b ^= c; b <<<= 7;
    
    Args:
        rounds: Number of rounds (default: 20, can use reduced for testing)
        device: torch device
    """
    
    def __init__(self, rounds: int = 20, device='cpu'):
        super().__init__()
        self.rounds = rounds
        self.device = device
        self.word_size = 32
        self.modulus = 2 ** self.word_size
        
    def _soft_add(self, x: torch.Tensor, y: torch.Tensor, 
                  steepness: float = 5.0) -> torch.Tensor:
        """
        Smooth modular addition (32-bit).
        """
        sum_val = x + y
        wrap = torch.sigmoid(steepness * (sum_val - self.modulus))
        return sum_val - self.modulus * wrap
    
    def _soft_xor(self, x: torch.Tensor, y: torch.Tensor, 
                  steepness: float = 5.0) -> torch.Tensor:
        """
        Smooth XOR approximation.
        """
        x_norm = x / self.modulus
        y_norm = y / self.modulus
        
        x_sharp = torch.sigmoid(steepness * (x_norm - 0.5))
        y_sharp = torch.sigmoid(steepness * (y_norm - 0.5))
        
        not_x = 1 - x_sharp
        not_y = 1 - y_sharp
        
        xor_result = x_sharp * not_y + not_x * y_sharp
        
        return xor_result * self.modulus
    
    def _smooth_rotate(self, x: torch.Tensor, r: int) -> torch.Tensor:
        """
        Smooth left rotation by r bits.
        
        ROL(x, r) = (x << r) | (x >> (32 - r))
        """
        x_norm = x / self.modulus
        
        # Left shift component
        left_shifted = (x_norm * (2 ** r)) % 1.0
        
        # Right shift component
        right_shifted = (x_norm / (2 ** (self.word_size - r))) % 1.0
        
        # Combine (smooth OR approximation)
        result = (left_shifted + right_shifted) % 1.0
        
        return result * self.modulus
    
    def quarter_round(self, a: torch.Tensor, b: torch.Tensor,
                     c: torch.Tensor, d: torch.Tensor) -> Tuple:
        """
        ChaCha20 quarter round function (smooth version).
        
        Original:
        a += b; d ^= a; d <<<= 16;
        c += d; b ^= c; b <<<= 12;
        a += b; d ^= a; d <<<= 8;
        c += d; b ^= c; b <<<= 7;
        """
        # Line 1
        a = self._soft_add(a, b)
        d = self._soft_xor(d, a)
        d = self._smooth_rotate(d, 16)
        
        # Line 2
        c = self._soft_add(c, d)
        b = self._soft_xor(b, c)
        b = self._smooth_rotate(b, 12)
        
        # Line 3
        a = self._soft_add(a, b)
        d = self._soft_xor(d, a)
        d = self._smooth_rotate(d, 8)
        
        # Line 4
        c = self._soft_add(c, d)
        b = self._soft_xor(b, c)
        b = self._smooth_rotate(b, 7)
        
        return a, b, c, d
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        ChaCha20 block function.
        
        Args:
            state: (batch_size, 16) tensor representing 16 x 32-bit words
            
        Returns:
            transformed_state: (batch_size, 16) output state
        """
        # Working state
        x = state.clone()
        
        # ChaCha20 performs rounds in column and diagonal patterns
        for round_idx in range(self.rounds // 2):
            # Column rounds
            x[:, 0], x[:, 4], x[:, 8], x[:, 12] = self.quarter_round(
                x[:, 0], x[:, 4], x[:, 8], x[:, 12]
            )
            x[:, 1], x[:, 5], x[:, 9], x[:, 13] = self.quarter_round(
                x[:, 1], x[:, 5], x[:, 9], x[:, 13]
            )
            x[:, 2], x[:, 6], x[:, 10], x[:, 14] = self.quarter_round(
                x[:, 2], x[:, 6], x[:, 10], x[:, 14]
            )
            x[:, 3], x[:, 7], x[:, 11], x[:, 15] = self.quarter_round(
                x[:, 3], x[:, 7], x[:, 11], x[:, 15]
            )
            
            # Diagonal rounds
            x[:, 0], x[:, 5], x[:, 10], x[:, 15] = self.quarter_round(
                x[:, 0], x[:, 5], x[:, 10], x[:, 15]
            )
            x[:, 1], x[:, 6], x[:, 11], x[:, 12] = self.quarter_round(
                x[:, 1], x[:, 6], x[:, 11], x[:, 12]
            )
            x[:, 2], x[:, 7], x[:, 8], x[:, 13] = self.quarter_round(
                x[:, 2], x[:, 7], x[:, 8], x[:, 13]
            )
            x[:, 3], x[:, 4], x[:, 9], x[:, 14] = self.quarter_round(
                x[:, 3], x[:, 4], x[:, 9], x[:, 14]
            )
        
        # Add original state (this is part of ChaCha20 spec)
        x = x + state
        
        return x % self.modulus
    
    def encrypt(self, plaintext: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
        """
        Simplified encryption using ChaCha20 operations.
        
        Args:
            plaintext: (batch_size, block_size)
            key: (batch_size, 8) - simplified key (8 x 32-bit words)
            
        Returns:
            ciphertext: (batch_size, block_size)
        """
        batch_size = plaintext.shape[0]
        
        # Build initial state (simplified version)
        # Real ChaCha20 uses constant, key, counter, nonce
        state = torch.zeros(batch_size, 16, device=self.device)
        
        # Constants ("expand 32-byte k")
        state[:, 0] = 0x61707865
        state[:, 1] = 0x3320646e
        state[:, 2] = 0x79622d32
        state[:, 3] = 0x6b206574
        
        # Key (8 words)
        state[:, 4:12] = key
        
        # Counter and nonce (simplified)
        state[:, 12] = torch.arange(batch_size, device=self.device)
        state[:, 13:16] = 0
        
        # Run ChaCha20 block function
        keystream = self.forward(state)
        
        # XOR with plaintext (simplified to first few words)
        output_size = min(plaintext.shape[1], keystream.shape[1])
        ciphertext = self._soft_xor(
            plaintext[:, :output_size],
            keystream[:, :output_size]
        )
        
        # Pad if necessary
        if plaintext.shape[1] > output_size:
            ciphertext = torch.cat([
                ciphertext,
                plaintext[:, output_size:]
            ], dim=1)
        
        return ciphertext
    
    def generate_plaintexts(self, num_samples: int) -> torch.Tensor:
        """Generate random plaintext blocks."""
        return torch.rand(num_samples, 4, device=self.device) * self.modulus
    
    def generate_keys(self, num_samples: int) -> torch.Tensor:
        """Generate random keys."""
        return torch.rand(num_samples, 8, device=self.device) * self.modulus