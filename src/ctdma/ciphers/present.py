"""
PRESENT Block Cipher Implementation

PRESENT is an ultra-lightweight SPN (Substitution-Permutation Network) block cipher.
Designed for extremely constrained environments (RFID, sensors).

This implementation uses smooth approximations for Neural ODE compatibility.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PRESENTCipher(nn.Module):
    """
    PRESENT block cipher with smooth approximations.
    
    PRESENT operates on 64-bit blocks with 80 or 128-bit keys.
    It uses a 4-bit S-box and bit permutation.
    
    Args:
        key_size: Key size (80 or 128 bits)
        rounds: Number of rounds (default: 31)
        device: torch device
    """
    
    def __init__(self, key_size=80, rounds=31, device='cpu'):
        super().__init__()
        self.key_size = key_size
        self.rounds = rounds
        self.block_size = 64
        self.device = device
        
        # PRESENT 4-bit S-box
        self.sbox = torch.tensor([0xC, 0x5, 0x6, 0xB, 0x9, 0x0, 0xA, 0xD,
                                  0x3, 0xE, 0xF, 0x8, 0x4, 0x7, 0x1, 0x2],
                                 dtype=torch.long, device=device)
        
        # Bit permutation (player layer)
        self.perm = self._generate_permutation()
        
    def _generate_permutation(self):
        """Generate PRESENT bit permutation."""
        perm = torch.zeros(64, dtype=torch.long, device=self.device)
        for i in range(64):
            perm[i] = (16 * (i % 4)) + (i // 4)
        return perm
    
    def _soft_sbox(self, x):
        """
        Differentiable S-box approximation.
        
        Uses soft attention over S-box entries.
        """
        # x is in range [0, 1] representing value / 16
        x_scaled = x * 15  # Scale to 0-15
        
        # Create soft one-hot encoding
        indices = torch.arange(16, dtype=torch.float32, device=x.device)
        soft_one_hot = F.softmax(-(x_scaled.unsqueeze(-1) - indices) ** 2 * 10, dim=-1)
        
        # Lookup S-box values
        sbox_values = self.sbox.float() / 15.0
        result = (soft_one_hot * sbox_values).sum(dim=-1)
        
        return result
    
    def _soft_permute(self, x):
        """
        Differentiable bit permutation.
        
        Approximates permutation as matrix multiplication.
        """
        # Create permutation matrix
        perm_matrix = torch.eye(64, device=x.device)[self.perm]
        
        # Apply permutation (approximate)
        if x.dim() == 1:
            x_expanded = x.unsqueeze(0)
            result = torch.matmul(x_expanded, perm_matrix.t())
            return result.squeeze(0)
        else:
            return torch.matmul(x, perm_matrix.t())
    
    def _soft_xor(self, x, y, steepness=10.0):
        """Smooth XOR approximation."""
        x_sharp = torch.sigmoid(steepness * (x - 0.5))
        y_sharp = torch.sigmoid(steepness * (y - 0.5))
        
        term1 = x_sharp * (1 - y_sharp)
        term2 = (1 - x_sharp) * y_sharp
        
        result = term1 + term2 - term1 * term2
        return torch.clamp(result, 0, 1)
    
    def _add_round_key(self, state, round_key):
        """XOR state with round key."""
        return self._soft_xor(state, round_key)
    
    def _key_schedule(self, key):
        """Generate round keys from master key."""
        # Simplified key schedule
        round_keys = []
        
        for i in range(self.rounds + 1):
            # Extract 64 bits for round key
            if key.dim() == 1:
                round_key = key[:self.block_size] if len(key) >= self.block_size else \
                           torch.cat([key, key[:self.block_size - len(key)]])
            else:
                # Batch processing
                round_key = key[:, :min(self.block_size, key.shape[1])]
                if round_key.shape[1] < self.block_size:
                    padding = torch.zeros(key.shape[0], self.block_size - round_key.shape[1], 
                                        device=self.device)
                    round_key = torch.cat([round_key, padding], dim=1)
            
            round_keys.append(round_key)
            
            # Update key register (simplified rotation)
            key = torch.roll(key, shifts=1, dims=-1)
        
        return round_keys
    
    def forward(self, plaintext, key):
        """
        Encrypt with PRESENT.
        
        Args:
            plaintext: (batch_size, 64) or (64,) plaintext bits
            key: (batch_size, key_size) or (key_size,) key bits
            
        Returns:
            ciphertext: (batch_size, 64) or (64,) encrypted bits
        """
        state = plaintext
        round_keys = self._key_schedule(key)
        
        for i in range(self.rounds):
            # Add round key
            state = self._add_round_key(state, round_keys[i])
            
            # S-box layer (apply to each 4-bit nibble)
            # Simplified: apply to whole state
            state = self._soft_sbox(state)
            
            # Permutation layer
            state = self._soft_permute(state)
        
        # Final round key addition
        state = self._add_round_key(state, round_keys[self.rounds])
        
        return state
    
    def encrypt(self, plaintext, key):
        """Standard encryption interface."""
        return self.forward(plaintext, key)
    
    def generate_plaintexts(self, num_samples):
        """Generate random plaintexts (64 bits)."""
        return torch.rand(num_samples, self.block_size, device=self.device)
    
    def generate_keys(self, num_samples):
        """Generate random keys."""
        return torch.rand(num_samples, self.key_size, device=self.device)
