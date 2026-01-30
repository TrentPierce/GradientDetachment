"""
Simon Block Cipher Implementation

Simon is a family of lightweight block ciphers designed by the NSA.
It uses AND, XOR, and rotation operations.

This is a variant of ARX using AND instead of modular addition.
"""

import torch
import torch.nn as nn


class SimonCipher(nn.Module):
    """
    Simon block cipher with smooth approximations.
    
    Simon uses:
    - Circular shifts
    - Bitwise AND
    - XOR
    
    Args:
        block_size: Block size (32, 48, 64, 96, or 128)
        key_size: Key size
        rounds: Number of rounds (default: 32)
        device: torch device
    """
    
    def __init__(self, block_size=32, key_size=64, rounds=32, device='cpu'):
        super().__init__()
        self.block_size = block_size
        self.key_size = key_size
        self.word_size = block_size // 2
        self.rounds = rounds
        self.device = device
        
        # Simon constants
        self.z_seq = self._generate_z_sequence()
        
    def _generate_z_sequence(self):
        """Generate Simon's z sequence for key schedule."""
        # Using z0 sequence (simplified)
        z0 = [1,1,1,1,1,0,1,0,0,0,1,0,0,1,0,1,0,1,1,0,0,0,0,1,1,1,0,0,1,1,0,
              1,1,1,1,1,0,1,0,0,0,1,0,0,1,0,1,0,1,1,0,0,0,0,1,1,1,0,0,1,1,0]
        return torch.tensor(z0, dtype=torch.float32, device=self.device)
    
    def _soft_and(self, x, y, steepness=10.0):
        """Smooth AND approximation."""
        x_sharp = torch.sigmoid(steepness * (x - 0.5))
        y_sharp = torch.sigmoid(steepness * (y - 0.5))
        return x_sharp * y_sharp
    
    def _soft_xor(self, x, y, steepness=10.0):
        """Smooth XOR approximation."""
        x_sharp = torch.sigmoid(steepness * (x - 0.5))
        y_sharp = torch.sigmoid(steepness * (y - 0.5))
        
        term1 = x_sharp * (1 - y_sharp)
        term2 = (1 - x_sharp) * y_sharp
        
        result = term1 + term2 - term1 * term2
        return torch.clamp(result, 0, 1)
    
    def _rotate_left(self, x, n):
        """Smooth left circular shift."""
        x_scaled = x * (2 ** self.word_size)
        shifted_left = (x_scaled * (2 ** n)) % (2 ** self.word_size)
        shifted_right = x_scaled / (2 ** (self.word_size - n))
        result = (shifted_left + shifted_right) % (2 ** self.word_size)
        return result / (2 ** self.word_size)
    
    def _round_function(self, x, y, k):
        """
        Simon round function.
        
        f(x) = (S^1(x) & S^8(x)) ^ S^2(x)
        y' = x
        x' = y ^ f(x) ^ k
        
        Where S^j denotes left circular shift by j positions.
        """
        # Compute f(x) = (S^1(x) & S^8(x)) ^ S^2(x)
        s1 = self._rotate_left(x, 1)
        s8 = self._rotate_left(x, 8)
        s2 = self._rotate_left(x, 2)
        
        and_term = self._soft_and(s1, s8)
        f_x = self._soft_xor(and_term, s2)
        
        # Update
        x_new = self._soft_xor(y, f_x)
        x_new = self._soft_xor(x_new, k)
        y_new = x
        
        return x_new, y_new
    
    def _key_schedule(self, key):
        """Generate round keys from master key."""
        # Simplified key schedule
        # In full Simon, this is more complex
        m = self.key_size // self.word_size  # Number of key words
        round_keys = []
        
        # Initial keys from master key
        for i in range(min(m, self.rounds)):
            if i < key.shape[1]:
                round_keys.append(key[:, i])
            else:
                round_keys.append(key[:, 0])  # Reuse first key word
        
        # Generate remaining round keys
        while len(round_keys) < self.rounds:
            # Simplified: just cycle through existing keys
            round_keys.append(round_keys[len(round_keys) % m])
        
        return round_keys
    
    def forward(self, plaintext, key):
        """
        Encrypt with Simon.
        
        Args:
            plaintext: (batch_size, 2) left and right halves
            key: (batch_size, m) key words
            
        Returns:
            ciphertext: (batch_size, 2) encrypted output
        """
        x = plaintext[:, 0]
        y = plaintext[:, 1]
        
        round_keys = self._key_schedule(key)
        
        for i in range(self.rounds):
            k = round_keys[i]
            x, y = self._round_function(x, y, k)
        
        return torch.stack([x, y], dim=1)
    
    def encrypt(self, plaintext, key):
        """Standard encryption interface."""
        return self.forward(plaintext, key)
    
    def generate_plaintexts(self, num_samples):
        """Generate random plaintexts."""
        return torch.rand(num_samples, 2, device=self.device)
    
    def generate_keys(self, num_samples):
        """Generate random keys."""
        m = self.key_size // self.word_size
        return torch.rand(num_samples, m, device=self.device)
