"""
Tests for Attack Methods

Tests differential and other cryptanalytic attack implementations.
"""

import pytest
import torch
from ctdma.attacks.differential import DifferentialAttack


class TestDifferentialAttack:
    """Test differential cryptanalysis attack."""
    
    def test_attack_initialization(self, speck_cipher):
        """Test attack initialization."""
        attack = DifferentialAttack(speck_cipher, input_size=2, hidden_size=32)
        
        assert attack.cipher is not None
        assert attack.classifier is not None
    
    def test_differential_pair_generation(self, speck_cipher, random_seed):
        """Test generation of differential pairs."""
        attack = DifferentialAttack(speck_cipher)
        
        P0, P1, K = attack.generate_differential_pairs(10, delta=0.1)
        
        assert P0.shape == (10, 2)
        assert P1.shape == (10, 2)
        assert K.shape == (10, 4)
        
        # P1 should differ from P0
        assert not torch.equal(P0, P1)
    
    def test_attack_forward_pass(self, speck_cipher, random_seed):
        """Test attack forward pass."""
        attack = DifferentialAttack(speck_cipher, input_size=2, hidden_size=32)
        
        features = torch.rand(5, 2)
        output = attack(features)
        
        assert output.shape == (5, 2)  # Binary classification
    
    def test_attack_training_single_batch(self, speck_cipher, random_seed):
        """Test attack training on single batch."""
        attack = DifferentialAttack(speck_cipher, input_size=2, hidden_size=32)
        
        # Generate training data
        P0, P1, K = attack.generate_differential_pairs(10, delta=0.1)
        C0 = speck_cipher.encrypt(P0, K)
        C1 = speck_cipher.encrypt(P1, K)
        
        diff = C0 - C1
        features = diff.view(10, -1)
        labels = torch.ones(10, dtype=torch.long)
        
        # Single training step
        optimizer = torch.optim.Adam(attack.parameters(), lr=0.01)
        criterion = torch.nn.CrossEntropyLoss()
        
        optimizer.zero_grad()
        outputs = attack(features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        # Loss should be finite
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)


class TestAttackPerformance:
    """Test attack performance metrics."""
    
    def test_attack_accuracy_random_baseline(self, speck_cipher, random_seed):
        """Test that untrained attack performs at random baseline."""
        attack = DifferentialAttack(speck_cipher, input_size=2, hidden_size=32)
        
        # Generate test data
        features = torch.rand(100, 2)
        labels = torch.randint(0, 2, (100,))
        
        with torch.no_grad():
            outputs = attack(features)
            _, predicted = torch.max(outputs, 1)
            accuracy = (predicted == labels).float().mean().item()
        
        # Untrained model should be near 50% (random)
        assert 0.3 < accuracy < 0.7
