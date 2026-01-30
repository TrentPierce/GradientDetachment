"""
Cross-Cipher Comparison Test

Tests and compares three cipher families using Neural ODE:
1. ARX (Speck) - Addition, Rotation, XOR
2. Feistel - DES-style with F-function
3. SPN - AES-style with S-boxes and permutation

This test measures the learnability of each cipher family
using identical Neural ODE architectures and training regimes.

Hypothesis: Different cipher structures have different
resistance to continuous manifold analysis.
"""

import torch
import numpy as np
import json
from datetime import datetime
from typing import Dict, List, Tuple
from tqdm import tqdm

# CTDMA imports
from ctdma.ciphers import SimplifiedSpeck, SimplifiedFeistel, SimplifiedSPN
from ctdma.neural_ode import CryptoODESolver
from ctdma.utils import prepare_cipher_dataset


def test_cipher_learnability(
    cipher,
    cipher_name: str,
    key: torch.Tensor,
    num_samples: int = 1000,
    epochs: int = 100,
    hidden_dim: int = 128,
    num_layers: int = 3,
    batch_size: int = 32,
    lr: float = 1e-3,
    verbose: bool = True
) -> Dict:
    """Test how well Neural ODE can learn a cipher transformation.
    
    Args:
        cipher: Cipher instance
        cipher_name: Name of the cipher
        key: Encryption key
        num_samples: Number of training samples
        epochs: Training epochs
        hidden_dim: Hidden dimension
        num_layers: Number of layers
        batch_size: Batch size
        lr: Learning rate
        verbose: Print progress
        
    Returns:
        Dictionary with results
    """
    if verbose:
        print(f"\n{'='*70}")
        print(f" Testing {cipher_name}")
        print(f"{'='*70}")
        print(f"  Rounds: {cipher.rounds}")
        print(f"  Block Size: {cipher.block_size}")
        print(f"  Key Size: {cipher.key_size}")
        print(f"  Samples: {num_samples}, Epochs: {epochs}")
    
    # Generate plaintext-ciphertext pairs
    plaintexts, ciphertexts = prepare_cipher_dataset(cipher, key, num_samples=num_samples)
    
    # Convert to binary vectors
    X_binary = torch.stack([cipher.to_binary_vector(pt) for pt in plaintexts])
    Y_binary = torch.stack([cipher.to_binary_vector(ct) for ct in ciphertexts])
    
    # Create and train ODE solver
    solver = CryptoODESolver(
        input_dim=cipher.block_size,
        hidden_dim=hidden_dim,
        num_layers=num_layers
    )
    
    history = solver.train_model(
        X_binary,
        Y_binary,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        verbose=verbose
    )
    
    # Compute final metrics
    with torch.no_grad():
        predictions = solver.forward(X_binary)
        mse_loss = torch.nn.functional.mse_loss(predictions, Y_binary).item()
        
        # Binary accuracy
        pred_binary = (predictions > 0.5).float()
        target_binary = Y_binary
        bit_accuracy = (pred_binary == target_binary).float().mean().item()
    
    results = {
        "cipher_name": cipher_name,
        "cipher_type": type(cipher).__name__,
        "rounds": cipher.rounds,
        "block_size": cipher.block_size,
        "key_size": cipher.key_size,
        "num_samples": num_samples,
        "epochs": epochs,
        "final_loss": mse_loss,
        "best_loss": history['best_loss'],
        "bit_accuracy": bit_accuracy,
        "training_history": history['history']
    }
    
    if verbose:
        print(f"\n  Results:")
        print(f"    Final MSE Loss: {mse_loss:.6f}")
        print(f"    Best MSE Loss:  {history['best_loss']:.6f}")
        print(f"    Bit Accuracy:   {bit_accuracy:.2%}")
        print(f"{'='*70}")
    
    return results


def compare_all_ciphers(
    rounds: int = 4,
    num_samples: int = 1000,
    epochs: int = 100,
    hidden_dim: int = 128,
    key_hex_speck: str = "ABCD,EF01",
    key_hex_feistel: str = "DEADBEEF",
    key_hex_spn: str = "CAFEBABE",
    verbose: bool = True
) -> Dict:
    """Compare all three cipher families.
    
    Args:
        rounds: Number of rounds for all ciphers
        num_samples: Training samples
        epochs: Training epochs
        hidden_dim: Hidden dimension
        key_hex_speck: Key for Speck (32-bit = 2x16-bit words)
        key_hex_feistel: Key for Feistel (32-bit)
        key_hex_spn: Key for SPN (32-bit)
        verbose: Print progress
        
    Returns:
        Comparison results
    """
    if verbose:
        print("\n" + "="*70)
        print(" CROSS-CIPHER COMPARISON TEST")
        print(" Comparing ARX vs Feistel vs SPN using Neural ODE")
        print("="*70)
    
    # Parse keys
    key_speck = torch.tensor([int(x.strip(), 16) for x in key_hex_speck.split(",")], dtype=torch.long)
    key_feistel = torch.tensor(int(key_hex_feistel, 16), dtype=torch.long)
    key_spn = torch.tensor(int(key_hex_spn, 16), dtype=torch.long)
    
    # Initialize ciphers
    speck = SimplifiedSpeck(rounds=rounds, block_size=32, key_size=64)
    feistel = SimplifiedFeistel(rounds=rounds, block_size=16, key_size=32)
    spn = SimplifiedSPN(rounds=rounds, block_size=16, key_size=32)
    
    # Test each cipher
    results = {}
    
    # ARX (Speck)
    results['arx'] = test_cipher_learnability(
        speck, "ARX (Speck)", key_speck,
        num_samples=num_samples, epochs=epochs, hidden_dim=hidden_dim,
        verbose=verbose
    )
    
    # Feistel
    results['feistel'] = test_cipher_learnability(
        feistel, "Feistel", key_feistel,
        num_samples=num_samples, epochs=epochs, hidden_dim=hidden_dim,
        verbose=verbose
    )
    
    # SPN
    results['spn'] = test_cipher_learnability(
        spn, "SPN", key_spn,
        num_samples=num_samples, epochs=epochs, hidden_dim=hidden_dim,
        verbose=verbose
    )
    
    # Rank ciphers by learnability (lower loss = more learnable = less secure)
    sorted_by_loss = sorted(results.items(), key=lambda x: x[1]['final_loss'])
    
    if verbose:
        print("\n" + "="*70)
        print(" COMPARISON SUMMARY")
        print("="*70)
        print(f"\n{'Cipher Family':<20} {'Rounds':<8} {'Final Loss':<15} {'Bit Acc':<10}")
        print("-"*70)
        
        for cipher_type, res in sorted_by_loss:
            print(f"{res['cipher_name']:<20} {res['rounds']:<8} "
                  f"{res['final_loss']:<15.6f} {res['bit_accuracy']:<10.2%}")
        
        print("\n" + "="*70)
        print(" INTERPRETATION")
        print("="*70)
        print(f"  Most Learnable (Least Secure):    {sorted_by_loss[0][1]['cipher_name']}")
        print(f"  Least Learnable (Most Secure):    {sorted_by_loss[-1][1]['cipher_name']}")
        print("\n  Lower loss indicates the cipher has more 'continuous'")
        print("  structure that Neural ODE can exploit.")
        print("="*70)
    
    return {
        "timestamp": datetime.now().isoformat(),
        "configuration": {
            "rounds": rounds,
            "num_samples": num_samples,
            "epochs": epochs,
            "hidden_dim": hidden_dim
        },
        "results": results,
        "ranking": [{
            "rank": i + 1,
            "cipher": res['cipher_name'],
            "loss": res['final_loss'],
            "accuracy": res['bit_accuracy']
        } for i, (_, res) in enumerate(sorted_by_loss)]
    }


def run_multi_round_comparison(
    rounds_list: List[int] = [2, 4, 6],
    num_samples: int = 1000,
    epochs: int = 100,
    hidden_dim: int = 128
) -> Dict:
    """Compare ciphers across multiple round counts.
    
    Args:
        rounds_list: List of round counts to test
        num_samples: Training samples
        epochs: Training epochs
        hidden_dim: Hidden dimension
        
    Returns:
        Multi-round comparison results
    """
    print("\n" + "="*70)
    print(" MULTI-ROUND CROSS-CIPHER COMPARISON")
    print("="*70)
    print("\nTesting how security scales with rounds for each cipher family...")
    
    all_results = []
    
    for rounds in rounds_list:
        print(f"\n{'='*70}")
        print(f" Testing with {rounds} Rounds")
        print(f"{'='*70}")
        
        result = compare_all_ciphers(
            rounds=rounds,
            num_samples=num_samples,
            epochs=epochs,
            hidden_dim=hidden_dim,
            verbose=True
        )
        
        all_results.append(result)
    
    # Generate comparison table
    print("\n\n" + "="*70)
    print(" COMPREHENSIVE COMPARISON TABLE")
    print("="*70)
    print(f"\n{'Rounds':<8} {'Cipher':<15} {'Loss':<12} {'Accuracy':<10}")
    print("-"*70)
    
    for result in all_results:
        rounds = result['configuration']['rounds']
        for cipher_type, res in result['results'].items():
            print(f"{rounds:<8} {res['cipher_name']:<15} "
                  f"{res['final_loss']:<12.6f} {res['bit_accuracy']:<10.2%}")
    
    print("\n" + "="*70)
    print(" SECURITY ANALYSIS")
    print("="*70)
    print("  Expected behavior:")
    print("    - Low rounds (2): All ciphers should be highly learnable")
    print("    - Medium rounds (4-6): Differences between families emerge")
    print("    - High rounds (6+): All ciphers approach random-like complexity")
    print("\n  Cipher family characteristics:")
    print("    - ARX: Fast, parallelizable, good in software")
    print("    - Feistel: Same encrypt/decrypt structure, efficient hardware")
    print("    - SPN: Strong diffusion, commonly used in modern ciphers")
    print("="*70)
    
    return {
        "multi_round_results": all_results,
        "rounds_tested": rounds_list
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Cross-Cipher Comparison Test",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic comparison with 4 rounds
  python test_cross_cipher_comparison.py --rounds 4
  
  # Quick test with fewer samples/epochs
  python test_cross_cipher_comparison.py --rounds 4 --samples 500 --epochs 50
  
  # Multi-round comparison
  python test_cross_cipher_comparison.py --multi-round
        """
    )
    
    parser.add_argument('--rounds', type=int, default=4,
                        help='Number of cipher rounds (default: 4)')
    parser.add_argument('--samples', type=int, default=1000,
                        help='Number of training samples (default: 1000)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Training epochs (default: 100)')
    parser.add_argument('--hidden-dim', type=int, default=128,
                        help='Hidden dimension (default: 128)')
    parser.add_argument('--multi-round', action='store_true',
                        help='Test multiple round counts')
    parser.add_argument('--rounds-list', type=int, nargs='+', default=[2, 4, 6],
                        help='Round counts for multi-round test (default: 2 4 6)')
    
    args = parser.parse_args()
    
    if args.multi_round:
        results = run_multi_round_comparison(
            rounds_list=args.rounds_list,
            num_samples=args.samples,
            epochs=args.epochs,
            hidden_dim=args.hidden_dim
        )
    else:
        results = compare_all_ciphers(
            rounds=args.rounds,
            num_samples=args.samples,
            epochs=args.epochs,
            hidden_dim=args.hidden_dim
        )
    
    # Save results
    filename = f"cross_cipher_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n💾 Results saved to: {filename}")
