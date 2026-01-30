#!/usr/bin/env python3
"""
Diagnostic Script for Gradient Inversion Phenomenon

This script investigates why accuracy drops to ~2.5% in binary classification tasks.
Hypothesis: The model is learning the INVERSE of the target function.
"""

import sys
import os
import torch
import torch.nn as nn
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from ctdma.ciphers.speck import SpeckCipher
from ctdma.attacks.differential import DifferentialAttack

def check_inversion():
    print("\n" + "="*60)
    print("DIAGNOSTIC: Gradient Inversion Check")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Setup standard experiment
    cipher = SpeckCipher(rounds=1, device=device)
    model = DifferentialAttack(cipher, input_size=2, hidden_size=64, num_rounds=1).to(device)
    
    # Generate STATIC dataset
    num_samples = 100
    plaintexts = cipher.generate_plaintexts(num_samples)
    keys = cipher.generate_keys(num_samples)
    
    # Generate labels (half 0, half 1 for balance)
    labels = torch.cat([torch.zeros(50), torch.ones(50)]).long().to(device)
    # Shuffle to ensure randomness relative to inputs
    idx = torch.randperm(num_samples)
    labels = labels[idx]
    
    # Create diffs
    plaintexts_diff = plaintexts.clone()
    plaintexts_diff[:, 0] = (plaintexts_diff[:, 0] + 0.5) % 1.0
    ct1 = cipher.encrypt(plaintexts, keys)
    ct2 = cipher.encrypt(plaintexts_diff, keys)
    diff = ct1 - ct2
    features = diff.view(num_samples, -1)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    
    print("Training for 50 epochs...")
    inverted_count = 0
    
    for epoch in range(50):
        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        _, predicted = torch.max(outputs.data, 1)
        correct = (predicted == labels).sum().item()
        acc = correct / num_samples
        
        # Check if we are predicting the OPPOSITE
        # Invert predictions (0->1, 1->0)
        inverted_pred = 1 - predicted
        inverted_correct = (inverted_pred == labels).sum().item()
        inverted_acc = inverted_correct / num_samples
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Acc={acc:.2%} (Inverted Acc={inverted_acc:.2%})")
            
    print(f"Final Standard Accuracy: {acc:.2%}")
    print(f"Final Inverted Accuracy: {inverted_acc:.2%}")
    
    if acc < 0.10 and inverted_acc > 0.90:
        print("\n✅ DIAGNOSIS CONFIRMED: GRADIENT INVERSION")
        print("   The model is consistently predicting the OPPOSITE of the target.")
        print("   This indicates adversarial attractors in the loss landscape.")
    else:
        print("\n❌ Diagnosis Unclear: Results do not show systematic inversion.")

if __name__ == "__main__":
    check_inversion()
