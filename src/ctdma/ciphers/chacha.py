"""
ChaCha20 Stream Cipher Implementation

ChaCha20 is a stream cipher designed by Daniel J. Bernstein.
It's an ARX cipher (Addition, Rotation, XOR) with a 256-bit key.

This implementation uses smooth approximations for Neural ODE compatibility.
"""

import torch
import torch.nn as nn


class ChaCha20Cipher(nn.Module):
    """
    ChaCha20 stream cipher with smooth approximations.
    
    ChaCha20 operates on a 4x4 matrix of 32-bit words.
    Quarter-round function: (a, b, c, d) with ARX operations.
    
    Args:
        rounds: Number of rounds (default: 20, can use 8 or 12 for testing)
        device: torch device
    """
    
    def __init__(self, rounds=20, device='cpu'):
        super().__init__()
        self.rounds = rounds
        self.device = device
        
        # ChaCha20 constants ("expand 32-byte k")
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
        """
        Smooth left rotation approximation.
        
        ROL(x, n) = (x << n) | (x >> (32 - n))
        """
        x_scaled = x * (2 ** word_size)
        shifted_left = (x_scaled * (2 ** n)) % (2 ** word_size)
        shifted_right = x_scaled / (2 ** (word_size - n))
        result = (shifted_left + shifted_right) % (2 ** word_size)
        return result / (2 ** word_size)
    
    def _quarter_round(self, a, b, c, d):
        """
        ChaCha20 quarter-round function.
        
        Standard operations:
        a += b; d ^= a; d <<<= 16;
        c += d; b ^= c; b <<<= 12;
        a += b; d ^= a; d <<<= 8;
        c += d; b ^= c; b <<<= 7;
        
        Smooth approximations used throughout.
        """
        # Round 1
        a = self._soft_add(a, b)
        d = self._soft_xor(d, a)
        d = self._rotate_left(d, 16)
        
        # Round 2
        c = self._soft_add(c, d)
        b = self._soft_xor(b, c)
        b = self._rotate_left(b, 12)
        
        # Round 3
        a = self._soft_add(a, b)
        d = self._soft_xor(d, a)
        d = self._rotate_left(d, 8)
        
        # Round 4
        c = self._soft_add(c, d)
        b = self._soft_xor(b, c)
        b = self._rotate_left(b, 7)
        
        return a, b, c, d
    
    def _chacha_block(self, state):
        """
        ChaCha20 block function.
        
        Args:
            state: 16-element state vector
            
        Returns:
            Updated state after specified rounds
        """
        working_state = state.clone()
        
        for i in range(self.rounds // 2):  # Each iteration is 2 rounds
            # Column rounds
            working_state[0], working_state[4], working_state[8], working_state[12] = \
                self._quarter_round(working_state[0], working_state[4], 
                                  working_state[8], working_state[12])
            
            working_state[1], working_state[5], working_state[9], working_state[13] = \
                self._quarter_round(working_state[1], working_state[5], 
                                  working_state[9], working_state[13])
            
            working_state[2], working_state[6], working_state[10], working_state[14] = \
                self._quarter_round(working_state[2], working_state[6], 
                                  working_state[10], working_state[14])
            
            working_state[3], working_state[7], working_state[11], working_state[15] = \
                self._quarter_round(working_state[3], working_state[7], 
                                  working_state[11], working_state[15])
            
            # Diagonal rounds
            working_state[0], working_state[5], working_state[10], working_state[15] = \
                self._quarter_round(working_state[0], working_state[5], 
                                  working_state[10], working_state[15])
            
            working_state[1], working_state[6], working_state[11], working_state[12] = \
                self._quarter_round(working_state[1], working_state[6], 
                                  working_state[11], working_state[12])
            
            working_state[2], working_state[7], working_state[8], working_state[13] = \
                self._quarter_round(working_state[2], working_state[7], 
                                  working_state[8], working_state[13])
            
            working_state[3], working_state[4], working_state[9], working_state[14] = \
                self._quarter_round(working_state[3], working_state[4], 
                                  working_state[9], working_state[14])
        
        # Add original state (feedforward)
        for i in range(16):
            working_state[i] = self._soft_add(working_state[i], state[i])
        
        return working_state
    
    def forward(self, plaintext, key, nonce=None):
        """
        Encrypt plaintext with ChaCha20.
        
        Args:
            plaintext: (batch_size, 16) state representation
            key: (batch_size, 8) 256-bit key (8x32-bit words)
            nonce: (batch_size, 3) optional nonce (3x32-bit words)
            
        Returns:
            ciphertext: (batch_size, 16) encrypted state
        """
        batch_size = plaintext.shape[0]
        
        # Build initial state
        # Layout: constants(4) | key(8) | counter(1) | nonce(3)
        state = torch.zeros(batch_size, 16, device=self.device)
        
        # Constants
        state[:, 0:4] = self.constants.unsqueeze(0).expand(batch_size, -1)
        
        # Key
        state[:, 4:12] = key
        
        # Counter (simplified: use 0)
        state[:, 12] = 0.0
        
        # Nonce
        if nonce is not None:
            state[:, 13:16] = nonce
        else:
            state[:, 13:16] = torch.zeros(batch_size, 3, device=self.device)
        
        # Process each sample in batch
        ciphertexts = []
        for i in range(batch_size):
            keystream = self._chacha_block(state[i])
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
        """Generate random nonces (3 words = 96 bits)."""
        return torch.rand(num_samples, 3, device=self.device)
