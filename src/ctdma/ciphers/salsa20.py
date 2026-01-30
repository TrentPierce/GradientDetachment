"""
Salsa20 Stream Cipher Implementation

Salsa20 is Daniel J. Bernstein's predecessor to ChaCha20.
It's also an ARX cipher with similar structure but different mixing.

This implementation uses smooth approximations for Neural ODE compatibility.
"""

import torch
import torch.nn as nn


class Salsa20Cipher(nn.Module):
    """
    Salsa20 stream cipher with smooth approximations.
    
    Similar to ChaCha20 but with different quarter-round and mixing pattern.
    
    Args:
        rounds: Number of rounds (default: 20, can use 8 or 12 for testing)
        device: torch device
    """
    
    def __init__(self, rounds=20, device='cpu'):
        super().__init__()
        self.rounds = rounds
        self.device = device
        
        # Salsa20 constants ("expand 32-byte k")
        self.constants = torch.tensor([0x61707865, 0x3320646e, 0x79622d32, 0x6b206574],
                                     dtype=torch.float32, device=device) / (2**32)
        
    def _soft_xor(self, x, y, steepness=10.0):
        """Smooth XOR approximation."""
        x_sharp = torch.sigmoid(steepness * (x - 0.5))
        y_sharp = torch.sigmoid(steepness * (y - 0.5))
        
        term1 = x_sharp * (1 - y_sharp)
        term2 = (1 - x_sharp) * y_sharp
        
        result = term1 + term2 - term1 * term2
        return torch.clamp(result, 0, 1)
    
    def _soft_add(self, x, y, steepness=10.0):
        """Smooth modular addition (mod 2^32)."""
        sum_val = x + y
        modulus = 1.0  # Normalized to [0, 1]
        wrap_amount = torch.sigmoid(steepness * (sum_val - modulus))
        result = sum_val - modulus * wrap_amount
        return result
    
    def _rotate_left(self, x, n, word_size=32):
        """Smooth left rotation approximation."""
        x_scaled = x * (2 ** word_size)
        shifted_left = (x_scaled * (2 ** n)) % (2 ** word_size)
        shifted_right = x_scaled / (2 ** (word_size - n))
        result = (shifted_left + shifted_right) % (2 ** word_size)
        return result / (2 ** word_size)
    
    def _quarter_round(self, y0, y1, y2, y3):
        """
        Salsa20 quarter-round function.
        
        Standard operations:
        y1 ^= (y0 + y3) <<< 7
        y2 ^= (y1 + y0) <<< 9
        y3 ^= (y2 + y1) <<< 13
        y0 ^= (y3 + y2) <<< 18
        """
        # Step 1
        temp = self._soft_add(y0, y3)
        temp = self._rotate_left(temp, 7)
        y1 = self._soft_xor(y1, temp)
        
        # Step 2
        temp = self._soft_add(y1, y0)
        temp = self._rotate_left(temp, 9)
        y2 = self._soft_xor(y2, temp)
        
        # Step 3
        temp = self._soft_add(y2, y1)
        temp = self._rotate_left(temp, 13)
        y3 = self._soft_xor(y3, temp)
        
        # Step 4
        temp = self._soft_add(y3, y2)
        temp = self._rotate_left(temp, 18)
        y0 = self._soft_xor(y0, temp)
        
        return y0, y1, y2, y3
    
    def _salsa_block(self, state):
        """
        Salsa20 block function.
        
        Args:
            state: 16-element state vector
            
        Returns:
            Updated state after specified rounds
        """
        working_state = state.clone()
        
        for i in range(self.rounds // 2):  # Each iteration is 2 rounds
            # Column rounds (different indices than ChaCha20)
            working_state[0], working_state[4], working_state[8], working_state[12] = \
                self._quarter_round(working_state[0], working_state[4], 
                                  working_state[8], working_state[12])
            
            working_state[5], working_state[9], working_state[13], working_state[1] = \
                self._quarter_round(working_state[5], working_state[9], 
                                  working_state[13], working_state[1])
            
            working_state[10], working_state[14], working_state[2], working_state[6] = \
                self._quarter_round(working_state[10], working_state[14], 
                                  working_state[2], working_state[6])
            
            working_state[15], working_state[3], working_state[7], working_state[11] = \
                self._quarter_round(working_state[15], working_state[3], 
                                  working_state[7], working_state[11])
            
            # Row rounds
            working_state[0], working_state[1], working_state[2], working_state[3] = \
                self._quarter_round(working_state[0], working_state[1], 
                                  working_state[2], working_state[3])
            
            working_state[5], working_state[6], working_state[7], working_state[4] = \
                self._quarter_round(working_state[5], working_state[6], 
                                  working_state[7], working_state[4])
            
            working_state[10], working_state[11], working_state[8], working_state[9] = \
                self._quarter_round(working_state[10], working_state[11], 
                                  working_state[8], working_state[9])
            
            working_state[15], working_state[12], working_state[13], working_state[14] = \
                self._quarter_round(working_state[15], working_state[12], 
                                  working_state[13], working_state[14])
        
        # Add original state (feedforward)
        for i in range(16):
            working_state[i] = self._soft_add(working_state[i], state[i])
        
        return working_state
    
    def forward(self, plaintext, key, nonce=None):
        """
        Encrypt plaintext with Salsa20.
        
        Args:
            plaintext: (batch_size, 16) state representation
            key: (batch_size, 8) 256-bit key
            nonce: (batch_size, 2) optional nonce (64 bits)
            
        Returns:
            ciphertext: (batch_size, 16) encrypted state
        """
        batch_size = plaintext.shape[0]
        
        # Build initial state
        state = torch.zeros(batch_size, 16, device=self.device)
        
        # Layout different from ChaCha20
        # constants(4) at positions 0, 5, 10, 15
        # key(8) at positions 1-4, 11-14
        # nonce(2) at positions 6-7
        # counter(2) at positions 8-9
        
        state[:, 0] = self.constants[0].expand(batch_size)
        state[:, 5] = self.constants[1].expand(batch_size)
        state[:, 10] = self.constants[2].expand(batch_size)
        state[:, 15] = self.constants[3].expand(batch_size)
        
        # Key
        state[:, 1:5] = key[:, :4]
        state[:, 11:15] = key[:, 4:8]
        
        # Nonce
        if nonce is not None:
            state[:, 6:8] = nonce
        else:
            state[:, 6:8] = torch.zeros(batch_size, 2, device=self.device)
        
        # Counter (simplified: use 0)
        state[:, 8:10] = torch.zeros(batch_size, 2, device=self.device)
        
        # Process each sample in batch
        ciphertexts = []
        for i in range(batch_size):
            keystream = self._salsa_block(state[i])
            # XOR plaintext with keystream
            ciphertext = torch.stack([self._soft_xor(plaintext[i, j], keystream[j]) 
                                     for j in range(16)])
            ciphertexts.append(ciphertext)
        
        return torch.stack(ciphertexts)
    
    def encrypt(self, plaintext, key, nonce=None):
        """Standard encryption interface."""
        return self.forward(plaintext, key, nonce)
    
    def generate_plaintexts(self, num_samples):
        """Generate random plaintexts (16 words)."""
        return torch.rand(num_samples, 16, device=self.device)
    
    def generate_keys(self, num_samples):
        """Generate random keys (8 words = 256 bits)."""
        return torch.rand(num_samples, 8, device=self.device)
    
    def generate_nonces(self, num_samples):
        """Generate random nonces (2 words = 64 bits)."""
        return torch.rand(num_samples, 2, device=self.device)
