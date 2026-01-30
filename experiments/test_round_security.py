"""
Round Security Analysis

Tests the security of all cipher families (ARX, Feistel, SPN) across
varying round counts (1-6 rounds). Generates a comprehensive comparison
table showing how security increases with rounds for each cipher type.

Key Metrics:
    - Learnability: How well Neural ODE can model the cipher
    - Bit Accuracy: How many output bits can be predicted
    - Loss Trend: How loss changes with increasing rounds

Theory: As rounds increase, ciphers should approach the complexity
of a random permutation, making them resistant to Neural ODE analysis.
"""

import torch
import numpy as np
import json
from datetime import datetime
from typing import Dict, List
from tqdm import tqdm
import time

# CTDMA imports
from ctdma.ciphers import SimplifiedSpeck, SimplifiedFeistel, SimplifiedSPN
from ctdma.neural_ode import CryptoODESolver
from ctdma.utils import prepare_cipher_dataset


def analyze_cipher_rounds(
    cipher_class,
    cipher_name: str,
    rounds_list: List[int],
    key: torch.Tensor,
    num_samples: int = 1000,
    epochs: int = 100,
    hidden_dim: int = 128,
    verbose: bool = True
) -> List[Dict]:
    """Analyze a cipher across multiple round counts.
    
    Args:
        cipher_class: Cipher class to instantiate
        cipher_name: Name of the cipher
        rounds_list: List of round counts to test
        key: Encryption key
        num_samples: Training samples
        epochs: Training epochs
        hidden_dim: Hidden dimension
        verbose: Print progress
        
    Returns:
        List of results for each round count
    """
    if verbose:
        print(f"\n{'='*70}")
        print(f" Analyzing {cipher_name} Security")
        print(f"{'='*70}")
    
    results = []
    
    for rounds in rounds_list:
        if verbose:
            print(f"\n  Testing {rounds} rounds...")
        
        # Create cipher instance
        if cipher_name == "ARX (Speck)":
            cipher = cipher_class(rounds=rounds, block_size=32, key_size=64)
        else:
            cipher = cipher_class(rounds=rounds, block_size=16, key_size=32)
        
        # Generate dataset
        plaintexts, ciphertexts = prepare_cipher_dataset(cipher, key, num_samples=num_samples)
        
        # Convert to binary
        X_binary = torch.stack([cipher.to_binary_vector(pt) for pt in plaintexts])
        Y_binary = torch.stack([cipher.to_binary_vector(ct) for ct in ciphertexts])
        
        # Train Neural ODE
        solver = CryptoODESolver(
            input_dim=cipher.block_size,
            hidden_dim=hidden_dim,
            num_layers=3
        )
        
        start_time = time.time()
        history = solver.train_model(
            X_binary, Y_binary,
            epochs=epochs,
            batch_size=32,
            lr=1e-3,
            verbose=False
        )
        train_time = time.time() - start_time
        
        # Compute metrics
        with torch.no_grad():
            predictions = solver.forward(X_binary)
            mse_loss = torch.nn.functional.mse_loss(predictions, Y_binary).item()
            pred_binary = (predictions > 0.5).float()
            bit_accuracy = (pred_binary == Y_binary).float().mean().item()
        
        # Calculate security score (higher = more secure)
        # Based on: 1 - accuracy (random = 0.5, perfect learning = 1.0)
        security_score = (1 - bit_accuracy) * 200  # Scale to 0-100
        
        result = {
            "rounds": rounds,
            "final_loss": mse_loss,
            "best_loss": history['best_loss'],
            "bit_accuracy": bit_accuracy,
            "security_score": security_score,
            "train_time": train_time,
            "cipher_name": cipher_name
        }
        
        results.append(result)
        
        if verbose:
            print(f"    Loss: {mse_loss:.6f} | Acc: {bit_accuracy:.2%} | Security: {security_score:.1f}/100")
    
    return results


def generate_comparison_table(all_results: Dict[str, List[Dict]]) -> str:
    """Generate a formatted comparison table.
    
    Args:
        all_results: Dictionary mapping cipher names to their results
        
    Returns:
        Formatted table string
    """
    # Get all round counts
    rounds_list = sorted(set(r['rounds'] for results in all_results.values() for r in results))
    
    lines = []
    lines.append("\n" + "="*90)
    lines.append(" ROUND SECURITY ANALYSIS - COMPARISON TABLE")
    lines.append("="*90)
    lines.append("")
    lines.append(f"{'Rounds':<8} {'Cipher Family':<15} {'Loss':<12} {'Accuracy':<10} {'Security Score':<15}")
    lines.append("-"*90)
    
    for rounds in rounds_list:
        first = True
        for cipher_name, results in all_results.items():
            # Find result for this round count
            result = next((r for r in results if r['rounds'] == rounds), None)
            if result:
                if first:
                    lines.append(f"{rounds:<8} {result['cipher_name']:<15} "
                               f"{result['final_loss']:<12.6f} {result['bit_accuracy']:<10.2%} "
                               f"{result['security_score']:<15.1f}")
                    first = False
                else:
                    lines.append(f"{'':8} {result['cipher_name']:<15} "
                               f"{result['final_loss']:<12.6f} {result['bit_accuracy']:<10.2%} "
                               f"{result['security_score']:<15.1f}")
        if rounds != rounds_list[-1]:
            lines.append("")
    
    lines.append("="*90)
    return "\n".join(lines)


def analyze_trends(all_results: Dict[str, List[Dict]]) -> Dict:
    """Analyze security trends across round counts.
    
    Args:
        all_results: Results for all ciphers
        
    Returns:
        Trend analysis
    """
    trends = {}
    
    for cipher_name, results in all_results.items():
        if len(results) < 2:
            continue
        
        # Calculate rate of security increase
        security_scores = [r['security_score'] for r in results]
        round_counts = [r['rounds'] for r in results]
        
        # Linear fit to get security growth rate
        if len(security_scores) >= 2:
            x = np.array(round_counts)
            y = np.array(security_scores)
            
            # Simple slope calculation
            slope = (y[-1] - y[0]) / (x[-1] - x[0]) if x[-1] != x[0] else 0
            
            trends[cipher_name] = {
                "security_growth_rate": float(slope),
                "initial_security": float(security_scores[0]),
                "final_security": float(security_scores[-1]),
                "total_improvement": float(security_scores[-1] - security_scores[0])
            }
    
    return trends


def run_round_security_analysis(
    rounds_list: List[int] = [1, 2, 3, 4, 5, 6],
    num_samples: int = 1000,
    epochs: int = 100,
    hidden_dim: int = 128,
    save_results: bool = True
) -> Dict:
    """Run comprehensive round security analysis for all cipher families.
    
    Args:
        rounds_list: Round counts to test
        num_samples: Training samples
        epochs: Training epochs
        hidden_dim: Hidden dimension
        save_results: Save to JSON file
        
    Returns:
        Complete analysis results
    """
    print("\n" + "="*90)
    print(" ROUND SECURITY ANALYSIS")
    print(" Testing ARX vs Feistel vs SPN across multiple rounds")
    print("="*90)
    print(f"\nConfiguration:")
    print(f"  Round counts: {rounds_list}")
    print(f"  Samples: {num_samples}")
    print(f"  Epochs: {epochs}")
    print(f"  Hidden dim: {hidden_dim}")
    print("\nSecurity Score: 0 = fully learnable (insecure)")
    print("                100 = random-like (secure)")
    print("="*90)
    
    # Keys for each cipher
    key_speck = torch.tensor([0xABCD, 0xEF01], dtype=torch.long)
    key_feistel = torch.tensor(0xDEADBEEF, dtype=torch.long)
    key_spn = torch.tensor(0xCAFEBABE, dtype=torch.long)
    
    # Analyze each cipher family
    all_results = {}
    
    all_results['ARX'] = analyze_cipher_rounds(
        SimplifiedSpeck, "ARX (Speck)", rounds_list, key_speck,
        num_samples, epochs, hidden_dim
    )
    
    all_results['Feistel'] = analyze_cipher_rounds(
        SimplifiedFeistel, "Feistel", rounds_list, key_feistel,
        num_samples, epochs, hidden_dim
    )
    
    all_results['SPN'] = analyze_cipher_rounds(
        SimplifiedSPN, "SPN", rounds_list, key_spn,
        num_samples, epochs, hidden_dim
    )
    
    # Generate comparison table
    table = generate_comparison_table(all_results)
    print(table)
    
    # Analyze trends
    trends = analyze_trends(all_results)
    
    # Print trend analysis
    print("\n" + "="*90)
    print(" SECURITY TREND ANALYSIS")
    print("="*90)
    print(f"\n{'Cipher Family':<20} {'Growth Rate':<15} {'Initial':<12} {'Final':<12} {'Improvement':<12}")
    print("-"*90)
    
    for cipher_name, trend in trends.items():
        print(f"{cipher_name:<20} {trend['security_growth_rate']:<15.2f} "
              f"{trend['initial_security']:<12.1f} {trend['final_security']:<12.1f} "
              f"{trend['total_improvement']:<12.1f}")
    
    print("\n" + "="*90)
    print(" INTERPRETATION")
    print("="*90)
    print("  Growth Rate: Points of security gained per additional round")
    print("  Higher growth rate = Security improves faster with rounds")
    print("")
    print("  Security Thresholds:")
    print("    0-20:  Highly vulnerable to Neural ODE analysis")
    print("    20-40: Weak resistance, some structure remains")
    print("    40-60: Moderate security, approaching random")
    print("    60-80: Strong security, difficult to distinguish from random")
    print("    80-100: Very strong, indistinguishable from random")
    print("="*90)
    
    # Find best performing cipher at highest round count
    max_rounds = max(rounds_list)
    final_scores = {}
    for cipher_name, results in all_results.items():
        final_result = next((r for r in results if r['rounds'] == max_rounds), None)
        if final_result:
            final_scores[cipher_name] = final_result['security_score']
    
    if final_scores:
        best_cipher = max(final_scores.items(), key=lambda x: x[1])
        print(f"\n🏆 Most secure at {max_rounds} rounds: {best_cipher[0]} (Score: {best_cipher[1]:.1f}/100)")
    
    # Compile final results
    final_results = {
        "timestamp": datetime.now().isoformat(),
        "configuration": {
            "rounds_list": rounds_list,
            "num_samples": num_samples,
            "epochs": epochs,
            "hidden_dim": hidden_dim
        },
        "results_by_cipher": all_results,
        "trends": trends,
        "summary": {
            "most_secure": best_cipher[0] if final_scores else None,
            "max_rounds_tested": max_rounds,
            "best_security_score": best_cipher[1] if final_scores else None
        }
    }
    
    # Save results
    if save_results:
        filename = f"round_security_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w') as f:
            json.dump(final_results, f, indent=2)
        print(f"\n💾 Results saved to: {filename}")
    
    return final_results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Round Security Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full analysis (rounds 1-6)
  python test_round_security.py
  
  # Quick analysis with fewer rounds
  python test_round_security.py --rounds 2 4 6
  
  # Fast test with reduced samples/epochs
  python test_round_security.py --rounds 2 4 --samples 500 --epochs 50
        """
    )
    
    parser.add_argument('--rounds', type=int, nargs='+', default=[1, 2, 3, 4, 5, 6],
                        help='Round counts to test (default: 1 2 3 4 5 6)')
    parser.add_argument('--samples', type=int, default=1000,
                        help='Training samples (default: 1000)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Training epochs (default: 100)')
    parser.add_argument('--hidden-dim', type=int, default=128,
                        help='Hidden dimension (default: 128)')
    parser.add_argument('--no-save', action='store_true',
                        help='Do not save results to file')
    
    args = parser.parse_args()
    
    results = run_round_security_analysis(
        rounds_list=args.rounds,
        num_samples=args.samples,
        epochs=args.epochs,
        hidden_dim=args.hidden_dim,
        save_results=not args.no_save
    )
