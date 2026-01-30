"""
Salsa20 Stream Cipher Implementation

Salsa20 is an ARX stream cipher, predecessor to ChaCha20.
Designed by Daniel J. Bernstein.

This implementation provides smooth approximations for Neural ODE compatibility.

Reference:
- Bernstein, D.J. "Salsa20 specification"
"""

import torch
import torch.nn as nn
from typing import Tuple


class Salsa20Cipher(nn.Module):
    """
    Salsa20 stream cipher with smooth approximations.
    
    Similar to ChaCha20 but with different mixing pattern.
    Uses quarterround: (y0, y1, y2, y3) where operations are:
    y1 ^= (y0 + y3) <<< 7
    y2 ^= (y1 + y0) <<< 9
    y3 ^= (y2 + y1) <<< 13
    y0 ^= (y3 + y2) <<< 18
    
    Args:
        rounds: Number of rounds (default: 20, typically use 20 or 12)
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
        """Smooth modular addition (32-bit)."""
        sum_val = x + y
        wrap = torch.sigmoid(steepness * (sum_val - self.modulus))
        return sum_val - self.modulus * wrap
    
    def _soft_xor(self, x: torch.Tensor, y: torch.Tensor,
                  steepness: float = 5.0) -> torch.Tensor:
        """Smooth XOR approximation."""
        x_norm = x / self.modulus
        y_norm = y / self.modulus
        
        x_sharp = torch.sigmoid(steepness * (x_norm - 0.5))
        y_sharp = torch.sigmoid(steepness * (y_norm - 0.5))
        
        xor_result = x_sharp * (1 - y_sharp) + (1 - x_sharp) * y_sharp
        
        return xor_result * self.modulus
    
    def _smooth_rotate(self, x: torch.Tensor, r: int) -> torch.Tensor:
        """Smooth left rotation by r bits."""
        x_norm = x / self.modulus
        left_shifted = (x_norm * (2 ** r)) % 1.0
        right_shifted = (x_norm / (2 ** (self.word_size - r))) % 1.0
        result = (left_shifted + right_shifted) % 1.0
        return result * self.modulus
    
    def quarterround(self, y0: torch.Tensor, y1: torch.Tensor,
                    y2: torch.Tensor, y3: torch.Tensor) -> Tuple:
        """
        Salsa20 quarterround function (smooth version).
        
        y1 ^= (y0 + y3) <<< 7
        y2 ^= (y1 + y0) <<< 9
        y3 ^= (y2 + y1) <<< 13
        y0 ^= (y3 + y2) <<< 18
        """
        temp = self._smooth_rotate(self._soft_add(y0, y3), 7)
        y1 = self._soft_xor(y1, temp)
        
        temp = self._smooth_rotate(self._soft_add(y1, y0), 9)
        y2 = self._soft_xor(y2, temp)
        
        temp = self._smooth_rotate(self._soft_add(y2, y1), 13)
        y3 = self._soft_xor(y3, temp)
        
        temp = self._smooth_rotate(self._soft_add(y3, y2), 18)
        y0 = self._soft_xor(y0, temp)
        
        return y0, y1, y2, y3
    
    def rowround(self, y: torch.Tensor) -> torch.Tensor:
        """
        Apply quarterround to rows of state matrix.
        
        Args:
            y: (batch_size, 16) state
            
        Returns:
            z: (batch_size, 16) transformed state
        """
        z = y.clone()
        
        z[:, 0], z[:, 1], z[:, 2], z[:, 3] = self.quarterround(
            y[:, 0], y[:, 1], y[:, 2], y[:, 3]
        )
        z[:, 5], z[:, 6], z[:, 7], z[:, 4] = self.quarterround(
            y[:, 5], y[:, 6], y[:, 7], y[:, 4]
        )
        z[:, 10], z[:, 11], z[:, 8], z[:, 9] = self.quarterround(
            y[:, 10], y[:, 11], y[:, 8], y[:, 9]
        )
        z[:, 15], z[:, 12], z[:, 13], z[:, 14] = self.quarterround(
            y[:, 15], y[:, 12], y[:, 13], y[:, 14]
        )
        
        return z
    
    def columnround(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply quarterround to columns of state matrix.
        
        Args:
            x: (batch_size, 16) state
            
        Returns:
            y: (batch_size, 16) transformed state
        """
        y = x.clone()
        
        y[:, 0], y[:, 4], y[:, 8], y[:, 12] = self.quarterround(
            x[:, 0], x[:, 4], x[:, 8], x[:, 12]
        )
        y[:, 5], y[:, 9], y[:, 13], y[:, 1] = self.quarterround(
            x[:, 5], x[:, 9], x[:, 13], x[:, 1]
        )
        y[:, 10], y[:, 14], y[:, 2], y[:, 6] = self.quarterround(
            x[:, 10], x[:, 14], x[:, 2], x[:, 6]
        )
        y[:, 15], y[:, 3], y[:, 7], y[:, 11] = self.quarterround(
            x[:, 15], x[:, 3], x[:, 7], x[:, 11]
        )
        
        return y
    
    def doubleround(self, x: torch.Tensor) -> torch.Tensor:
        """
        Salsa20 doubleround = columnround(rowround(x)).
        """
        return self.columnround(self.rowround(x))
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Salsa20 hash function.
        
        Args:
            state: (batch_size, 16) input state
            
        Returns:
            output: (batch_size, 16) hashed state
        """
        x = state.clone()
        
        # Apply doublerounds
        for _ in range(self.rounds // 2):
            x = self.doubleround(x)
        
        # Add original state
        x = x + state
        
        return x % self.modulus
    
    def encrypt(self, plaintext: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
        """
        Encrypt using Salsa20.
        
        Args:
            plaintext: (batch_size, block_size)
            key: (batch_size, 8) key words
            
        Returns:
            ciphertext: (batch_size, block_size)
        """
        batch_size = plaintext.shape[0]
        
        # Build initial state
        state = torch.zeros(batch_size, 16, device=self.device)
        
        # Constants
        state[:, 0] = 0x61707865
        state[:, 5] = 0x3320646e
        state[:, 10] = 0x79622d32
        state[:, 15] = 0x6b206574
        
        # Key
        state[:, 1:5] = key[:, :4]
        state[:, 11:15] = key[:, 4:8]
        
        # Nonce and counter (simplified)
        state[:, 6:8] = 0
        state[:, 8:10] = torch.arange(batch_size, device=self.device).unsqueeze(1).repeat(1, 2)
        
        # Generate keystream
        keystream = self.forward(state)
        
        # XOR with plaintext
        output_size = min(plaintext.shape[1], keystream.shape[1])
        ciphertext = self._soft_xor(
            plaintext[:, :output_size],
            keystream[:, :output_size]
        )
        
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