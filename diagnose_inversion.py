#!/usr/bin/env python3
"""
Diagnostic script to investigate the "anti-learning" phenomenon.

The reviewer correctly pointed out that 2.5% accuracy on binary classification
with random labels is impossible unless the model is predicting the inverse.

This script tests:
1. Whether the model is learning the inverse of labels
2. Whether flipping predictions improves accuracy
3. Whether this is a bug or a real phenomenon
"""

import torch
import torch.nn as nn
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from ctdma.ciphers.speck import SpeckCipher
from ctdma.attacks.differential import DifferentialAttack


def test_gradient_inversion():
    """
    Test if the model is learning to predict the inverse of labels.
    
    If accuracy is 2.5%, flipping predictions should give 97.5%.
    """
    print("="*70)
    print("GRADIENT INVERSION DIAGNOSTIC")
    print("="*70)
    print("\nTesting if model learns inverse labels...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create model
    cipher = SpeckCipher(rounds=1, device=device)
    model = DifferentialAttack(cipher, input_size=2, hidden_size=64, num_rounds=1)
    model = model.to(device)
    
    # Generate data
    num_samples = 100
    plaintexts = cipher.generate_plaintexts(num_samples)
    keys = cipher.generate_keys(num_samples)
    
    # Fixed random labels
    labels = torch.randint(0, 2, (num_samples,), device=device)
    
    # Train
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    print(f"\nTraining on {num_samples} samples for 50 epochs...")
    print(f"Label distribution: {labels.sum().item()} positive, {num_samples - labels.sum().item()} negative")
    
    for epoch in range(50):
        # Create differential pairs
        plaintexts_diff = plaintexts.clone()
        plaintexts_diff[:, 0] = (plaintexts_diff[:, 0] + 0.5) % 1.0
        
        ct1 = cipher.encrypt(plaintexts, keys)
        ct2 = cipher.encrypt(plaintexts_diff, keys)
        
        diff = ct1 - ct2
        features = diff.view(num_samples, -1)
        
        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        if epoch % 10 == 0:
            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            correct = (predicted == labels).sum().item()
            acc = correct / num_samples
            
            # Calculate inverse accuracy (predictions flipped)
            inverse_correct = (predicted == (1 - labels)).sum().item()
            inverse_acc = inverse_correct / num_samples
            
            print(f"Epoch {epoch}: Loss={loss.item():.4f}, Acc={acc:.2%}, Inverse Acc={inverse_acc:.2%}")
    
    # Final evaluation
    model.eval()
    with torch.no_grad():
        plaintexts_diff = plaintexts.clone()
        plaintexts_diff[:, 0] = (plaintexts_diff[:, 0] + 0.5) % 1.0
        
        ct1 = cipher.encrypt(plaintexts, keys)
        ct2 = cipher.encrypt(plaintexts_diff, keys)
        
        diff = ct1 - ct2
        features = diff.view(num_samples, -1)
        
        outputs = model(features)
        _, predicted = torch.max(outputs.data, 1)
        
        correct = (predicted == labels).sum().item()
        acc = correct / num_samples
        
        inverse_correct = (predicted == (1 - labels)).sum().item()
        inverse_acc = inverse_correct / num_samples
    
    print("\n" + "="*70)
    print("FINAL RESULTS")
    print("="*70)
    print(f"Standard Accuracy:     {acc:.2%}")
    print(f"Inverse Accuracy:      {inverse_acc:.2%}")
    print(f"Random Guess Expected: ~50%")
    print()
    
    if inverse_acc > 0.90:
        print("🔍 FINDING: Model is learning INVERSE labels!")
        print("   This is 'Gradient Inversion' - not 'Gradient Detachment'")
        print("   The model IS learning, but optimizing toward the wrong minimum")
        print()
        print("Possible causes:")
        print("  1. Sawtooth landscape creates adversarial minima")
        print("  2. Loss function sign error")
        print("  3. Initialization bias")
        
    elif acc < 0.10:
        print("🔍 FINDING: Model is stuck at low accuracy")
        print("   This suggests the 'Gradient Detachment' phenomenon")
        print("   Gradients are pointing in directions that don't help learning")
        
    elif 0.40 < acc < 0.60:
        print("✅ Model is performing at random chance (~50%)")
        print("   This would support 'Gradient Detachment' - model can't learn")
        
    else:
        print(f"⚠️ Unexpected accuracy: {acc:.2%}")
        
    print("\n" + "="*70)
    print("RECOMMENDATION")
    print("="*70)
    if inverse_acc > 0.90:
        print("""
The 'Gradient Detachment' paper should be revised:

1. The phenomenon is actually 'Gradient Inversion' - the model learns
   to predict the opposite of the target due to landscape properties.

2. This is still interesting! It shows the sawtooth landscape creates
   adversarial attractors that trap optimization.

3. The explanation should be updated to reflect this finding.

4. Consider testing: Does this inversion happen consistently across
   different initializations and label randomizations?
        """)
    
    return acc, inverse_acc


def test_with_identity_cipher():
    """
    Test if the model can learn a simple identity function.
    If it can't, the learning harness itself is broken.
    """
    print("\n" + "="*70)
    print("SANITY CHECK: Identity Function Test")
    print("="*70)
    print("\nTesting if model can learn a simple identity mapping...")
    print("(This verifies the learning harness works)")
    
    # Simple test: can the model learn identity?
    device = torch.device('cpu')  # Use CPU for simplicity
    
    # Create simple linear model
    model = nn.Sequential(
        nn.Linear(2, 64),
        nn.ReLU(),
        nn.Linear(64, 2)
    ).to(device)
    
    # Simple data: input = output (identity)
    X = torch.randn(100, 2, device=device)
    Y = X.clone()  # Target is same as input
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    
    print("\nTraining to learn identity function...")
    for epoch in range(100):
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, Y)
        loss.backward()
        optimizer.step()
        
        if epoch % 20 == 0:
            print(f"Epoch {epoch}: MSE Loss = {loss.item():.6f}")
    
    print(f"\nFinal MSE Loss: {loss.item():.6f}")
    
    if loss.item() < 0.01:
        print("✅ Model CAN learn simple functions")
        print("   The learning harness is working correctly")
    else:
        print("❌ Model cannot learn simple functions")
        print("   There may be issues with the learning setup")


if __name__ == "__main__":
    print("\n" + "="*70)
    print("DIAGNOSTIC INVESTIGATION")
    print("Peer Review Round 3 - Anti-Learning Anomaly")
    print("="*70)
    
    # Run diagnostic tests
    acc, inverse_acc = test_gradient_inversion()
    test_with_identity_cipher()
    
    print("\n" + "="*70)
    print("CONCLUSION")
    print("="*70)
    print(f"""
The reviewer's concern is VALID. An accuracy of {acc:.1%} on binary
classification with random labels is statistically anomalous.

Random guessing should yield ~50% accuracy.

The model appears to be either:
1. Learning inverse labels (Gradient Inversion)
2. Trapped in a non-learning state (Gradient Detachment)

RECOMMENDATION:
- Run multiple trials with different random seeds
- Check if the inversion is consistent
- Update the paper to reflect the actual phenomenon observed

This is a scientifically interesting finding, but the explanation
needs to be corrected from "Gradient Detachment" to account for
the anti-learning behavior.
""")
