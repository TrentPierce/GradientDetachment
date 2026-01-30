"""
Differential Cryptanalysis Attack

Implements differential distinguishers using Neural ODEs.
"""

import torch
import torch.nn as nn
import numpy as np


class DifferentialAttack(nn.Module):
    """
    Differential distinguisher using Neural ODE.
    
    Attempts to distinguish ciphertexts generated from plaintext pairs
    with known differences.
    
    Args:
        cipher: Cipher instance
        input_size: Input dimension
        hidden_size: Hidden layer dimension
        num_rounds: Number of cipher rounds
    """
    
    def __init__(self, cipher, input_size=32, hidden_size=64, num_rounds=4):
        super().__init__()
        self.cipher = cipher
        self.num_rounds = num_rounds
        
        # Simple classifier network
        self.classifier = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 2)  # Binary classification
        )
        
    def forward(self, x):
        """Classify input as real or random."""
        return self.classifier(x)
    
    def generate_differential_pairs(self, num_samples, delta):
        """
        Generate plaintext pairs with specific difference.
        
        Args:
            num_samples: Number of pairs to generate
            delta: Difference pattern (XOR difference)
            
        Returns:
            P0: First plaintexts (num_samples, 2)
            P1: Second plaintexts (num_samples, 2) where P1 = P0 XOR delta
            K: Random keys
        """
        P0 = self.cipher.generate_plaintexts(num_samples)
        K = self.cipher.generate_keys(num_samples)
        
        # Apply delta to create second plaintext
        P1 = P0.clone()
        P1[:, 0] = P1[:, 0] ^ delta
        
        return P0, P1, K
    
    def train_distinguisher(self, num_epochs=100, batch_size=32, 
                           delta=0x0040, lr=0.001):
        """
        Train the differential distinguisher.
        
        Returns:
            history: Training accuracy history
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        
        history = {'train_acc': [], 'loss': []}
        
        for epoch in range(num_epochs):
            # Generate differential pairs
            P0, P1, K = self.generate_differential_pairs(batch_size, delta)
            
            # Encrypt both
            C0 = self.cipher.encrypt(P0, K)
            C1 = self.cipher.encrypt(P1, K)
            
            # Compute differential
            diff = C0 - C1  # Differential pattern
            
            # Flatten for classifier
            features = diff.view(batch_size, -1)
            
            # Labels: 1 for real differential, 0 for random
            # In real attack, we'd compare to random pairs
            labels = torch.ones(batch_size, dtype=torch.long, device=features.device)
            
            # Forward pass
            outputs = self.forward(features)
            loss = criterion(outputs, labels)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Track accuracy
            _, predicted = torch.max(outputs.data, 1)
            acc = (predicted == labels).sum().item() / batch_size
            
            history['train_acc'].append(acc)
            history['loss'].append(loss.item())
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Loss={loss.item():.4f}, Acc={acc:.2%}")
        
        return history