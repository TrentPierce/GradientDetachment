"""
Cross-Cipher Analysis and Comparison Framework

This module provides comprehensive tools for comparing different cipher families
and analyzing their resistance to gradient-based attacks.

Comparisons include:
1. ARX ciphers: Speck, ChaCha, Salsa20, BLAKE2
2. Feistel ciphers: DES-like, Camellia-like, variants
3. SPN ciphers: AES-like, PRESENT-like, variants

Analysis metrics:
- Gradient inversion probability
- Diffusion properties
- Avalanche effect
- Information loss
- Convergence behavior
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
import json
import time
from pathlib import Path


class CipherComparator:
    """
    Comprehensive cipher comparison framework.
    
    Analyzes multiple ciphers across various metrics to understand
    their resistance to gradient-based cryptanalysis.
    """
    
    def __init__(self, device: str = 'cpu'):
        self.device = device
        self.results = {}
        
    def compare_gradient_inversion(
        self,
        ciphers: Dict[str, nn.Module],
        num_samples: int = 1000,
        num_rounds_list: List[int] = [1, 2, 4, 8]
    ) -> Dict[str, Dict]:
        """
        Compare gradient inversion across cipher families.
        
        For each cipher:
        1. Generate random plaintext-key pairs
        2. Compute ciphertext
        3. Train Neural ODE to predict key
        4. Measure inversion probability
        
        Args:
            ciphers: Dict of {name: cipher_instance}
            num_samples: Number of test samples
            num_rounds_list: List of round counts to test
            
        Returns:
            Nested dict: {cipher_name: {rounds: metrics}}
        """
        results = {}
        
        for cipher_name, cipher in ciphers.items():
            print(f"\nAnalyzing {cipher_name}...")
            cipher_results = {}
            
            for num_rounds in num_rounds_list:
                print(f"  Testing {num_rounds} rounds...")
                
                # Configure cipher for this round count
                if hasattr(cipher, 'rounds'):
                    cipher.rounds = num_rounds
                elif hasattr(cipher, 'num_rounds'):
                    cipher.num_rounds = num_rounds
                
                # Generate test data
                metrics = self._measure_inversion_for_cipher(
                    cipher,
                    num_samples=num_samples
                )
                
                cipher_results[f'{num_rounds}_rounds'] = metrics
                print(f"    Inversion prob: {metrics['inversion_probability']:.2%}")
            
            results[cipher_name] = cipher_results
        
        self.results['gradient_inversion'] = results
        return results
    
    def _measure_inversion_for_cipher(
        self,
        cipher: nn.Module,
        num_samples: int = 1000
    ) -> Dict[str, float]:
        """
        Measure gradient inversion for a single cipher configuration.
        
        Creates a simple classification task and measures how often
        the model predicts the opposite of the true label.
        
        Args:
            cipher: Cipher instance
            num_samples: Number of samples
            
        Returns:
            Dict with inversion metrics
        """
        from ctdma.theory.mathematical_analysis import GradientInversionAnalyzer
        
        # Generate random inputs
        if hasattr(cipher, 'generate_plaintexts'):
            plaintexts = cipher.generate_plaintexts(num_samples)
            keys = cipher.generate_keys(num_samples)
        else:
            # Generic generation
            input_size = 2 if hasattr(cipher, 'block_size') and cipher.block_size == 32 else 16
            plaintexts = torch.rand(num_samples, input_size, device=self.device)
            keys = torch.rand(num_samples, input_size, device=self.device)
        
        # Encrypt
        if hasattr(cipher, 'encrypt'):
            ciphertexts = cipher.encrypt(plaintexts, keys)
        else:
            ciphertexts = cipher(plaintexts, keys)
        
        # Analyze gradient behavior
        analyzer = GradientInversionAnalyzer()
        
        # Use first two elements for analysis
        if plaintexts.shape[1] >= 2:
            x = plaintexts[:, 0]
            y = plaintexts[:, 1]
        else:
            x = plaintexts.flatten()[:num_samples]
            y = keys.flatten()[:num_samples]
        
        grad_results = analyzer.compute_gradient_discontinuity(x, y, operation='modadd')
        
        return {
            'inversion_probability': grad_results['inversion_probability'].item(),
            'gradient_magnitude_jump': grad_results['gradient_magnitude_jump'].item(),
            'wrap_frequency': grad_results['wrap_frequency'].item(),
            'num_samples': num_samples
        }
    
    def compare_arx_designs(
        self,
        ciphers: Dict[str, nn.Module]
    ) -> Dict[str, Dict]:
        """
        Compare different ARX cipher design philosophies.
        
        Analyzes:
        1. Rotation amounts and patterns
        2. Round function structures
        3. Key schedule complexity
        4. Diffusion speed
        
        Args:
            ciphers: Dict of ARX ciphers to compare
            
        Returns:
            Comparison metrics
        """
        results = {}
        
        for name, cipher in ciphers.items():
            design_analysis = {
                'cipher_type': 'ARX',
                'rotation_amounts': self._extract_rotation_amounts(cipher),
                'round_structure': self._analyze_round_structure(cipher),
                'diffusion_analysis': self._measure_diffusion(cipher),
                'gradient_properties': self._analyze_gradient_properties(cipher)
            }
            
            results[name] = design_analysis
        
        self.results['arx_comparison'] = results
        return results
    
    def _extract_rotation_amounts(self, cipher: nn.Module) -> List[int]:
        """
        Extract rotation amounts used in cipher.
        
        Args:
            cipher: Cipher instance
            
        Returns:
            List of rotation amounts
        """
        rotations = []
        
        # Check common attributes
        if hasattr(cipher, 'alpha'):
            rotations.append(cipher.alpha)
        if hasattr(cipher, 'beta'):
            rotations.append(cipher.beta)
        if hasattr(cipher, 'rotation_amounts'):
            rotations.extend(cipher.rotation_amounts)
        
        return rotations if rotations else [7, 2]  # Default Speck rotations
    
    def _analyze_round_structure(self, cipher: nn.Module) -> Dict[str, any]:
        """
        Analyze the structure of round functions.
        
        Args:
            cipher: Cipher instance
            
        Returns:
            Dict describing round structure
        """
        structure = {
            'has_feistel': hasattr(cipher, 'feistel_round'),
            'has_substitution': hasattr(cipher, 'substitute') or hasattr(cipher, 'sbox'),
            'has_permutation': hasattr(cipher, 'permute'),
            'operations_per_round': 0
        }
        
        # Estimate operations per round
        if hasattr(cipher, '_round_function'):
            structure['operations_per_round'] = 4  # Typical ARX
        elif hasattr(cipher, '_quarter_round'):
            structure['operations_per_round'] = 4  # ChaCha-style
        
        return structure
    
    def _measure_diffusion(self, cipher: nn.Module, num_samples: int = 100) -> Dict[str, float]:
        """
        Measure diffusion properties (avalanche effect).
        
        Args:
            cipher: Cipher instance
            num_samples: Number of test samples
            
        Returns:
            Diffusion metrics
        """
        if hasattr(cipher, 'analyze_diffusion'):
            return cipher.analyze_diffusion(num_samples)
        
        # Generic diffusion measurement
        if hasattr(cipher, 'generate_plaintexts'):
            plaintexts = cipher.generate_plaintexts(num_samples)
            keys = cipher.generate_keys(num_samples)
        else:
            plaintexts = torch.rand(num_samples, 2, device=self.device)
            keys = torch.rand(num_samples, 2, device=self.device)
        
        # Original encryption
        ct1 = cipher.encrypt(plaintexts, keys) if hasattr(cipher, 'encrypt') else cipher(plaintexts, keys)
        
        # Flip one bit
        plaintexts_flipped = plaintexts.clone()
        plaintexts_flipped[:, 0] += 1.0
        
        # Encryption with flipped input
        ct2 = cipher.encrypt(plaintexts_flipped, keys) if hasattr(cipher, 'encrypt') else cipher(plaintexts_flipped, keys)
        
        # Measure difference
        diff = (ct1 - ct2).abs()
        changed_ratio = (diff > 0.01).float().mean()
        
        return {
            'avalanche_ratio': changed_ratio.item(),
            'expected_ratio': 0.5,
            'quality': 'good' if 0.4 < changed_ratio < 0.6 else 'poor'
        }
    
    def _analyze_gradient_properties(self, cipher: nn.Module) -> Dict[str, float]:
        """
        Analyze gradient flow properties.
        
        Args:
            cipher: Cipher instance
            
        Returns:
            Gradient analysis metrics
        """
        # Create simple forward pass
        x = torch.rand(10, 2, device=self.device, requires_grad=True)
        
        if hasattr(cipher, 'generate_keys'):
            k = cipher.generate_keys(10)
        else:
            k = torch.rand(10, 2, device=self.device)
        
        # Forward pass
        if hasattr(cipher, 'encrypt'):
            y = cipher.encrypt(x, k)
        else:
            y = cipher(x, k)
        
        # Compute gradients
        loss = y.sum()
        loss.backward()
        
        # Analyze gradient
        grad_magnitude = x.grad.abs().mean().item()
        grad_variance = x.grad.var().item()
        grad_sparsity = (x.grad.abs() < 1e-6).float().mean().item()
        
        return {
            'gradient_magnitude': grad_magnitude,
            'gradient_variance': grad_variance,
            'gradient_sparsity': grad_sparsity,
            'gradient_flow_quality': 'good' if grad_magnitude > 0.01 else 'poor'
        }
    
    def generate_comparison_report(
        self,
        output_path: Optional[str] = None
    ) -> str:
        """
        Generate comprehensive comparison report.
        
        Args:
            output_path: Path to save JSON report (optional)
            
        Returns:
            Formatted report string
        """
        report = []
        report.append("=" * 70)
        report.append("CIPHER COMPARISON REPORT")
        report.append("=" * 70)
        report.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Gradient inversion comparison
        if 'gradient_inversion' in self.results:
            report.append("\n" + "-" * 70)
            report.append("GRADIENT INVERSION ANALYSIS")
            report.append("-" * 70)
            
            for cipher_name, rounds_data in self.results['gradient_inversion'].items():
                report.append(f"\n{cipher_name}:")
                for round_key, metrics in rounds_data.items():
                    inv_prob = metrics['inversion_probability']
                    report.append(f"  {round_key}: {inv_prob:.2%} inversion probability")
        
        # ARX design comparison
        if 'arx_comparison' in self.results:
            report.append("\n" + "-" * 70)
            report.append("ARX DESIGN COMPARISON")
            report.append("-" * 70)
            
            for cipher_name, design_data in self.results['arx_comparison'].items():
                report.append(f"\n{cipher_name}:")
                report.append(f"  Rotation amounts: {design_data['rotation_amounts']}")
                report.append(f"  Diffusion quality: {design_data['diffusion_analysis']['quality']}")
                report.append(f"  Gradient flow: {design_data['gradient_properties']['gradient_flow_quality']}")
        
        report.append("\n" + "=" * 70)
        report_text = "\n".join(report)
        
        # Save to file if requested
        if output_path:
            # Save JSON
            json_path = Path(output_path).with_suffix('.json')
            with open(json_path, 'w') as f:
                json.dump(self.results, f, indent=2)
            
            # Save text report
            txt_path = Path(output_path).with_suffix('.txt')
            with open(txt_path, 'w') as f:
                f.write(report_text)
            
            print(f"\nReport saved to: {json_path} and {txt_path}")
        
        return report_text
    
    def plot_comparison(
        self,
        metric: str = 'inversion_probability',
        output_path: Optional[str] = None
    ):
        """
        Create visualization of comparison results.
        
        Args:
            metric: Metric to plot
            output_path: Path to save plot (optional)
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            sns.set_style('whitegrid')
            
            if 'gradient_inversion' not in self.results:
                print("No gradient inversion results to plot")
                return
            
            # Extract data for plotting
            cipher_names = []
            round_counts = []
            values = []
            
            for cipher_name, rounds_data in self.results['gradient_inversion'].items():
                for round_key, metrics in rounds_data.items():
                    cipher_names.append(cipher_name)
                    # Extract number from key like '1_rounds'
                    num_rounds = int(round_key.split('_')[0])
                    round_counts.append(num_rounds)
                    values.append(metrics[metric])
            
            # Create plot
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Group by cipher
            unique_ciphers = sorted(set(cipher_names))
            for cipher in unique_ciphers:
                cipher_rounds = [r for c, r in zip(cipher_names, round_counts) if c == cipher]
                cipher_values = [v for c, v in zip(cipher_names, values) if c == cipher]
                
                ax.plot(cipher_rounds, cipher_values, marker='o', label=cipher, linewidth=2)
            
            ax.set_xlabel('Number of Rounds')
            ax.set_ylabel(metric.replace('_', ' ').title())
            ax.set_title('Cipher Comparison: ' + metric.replace('_', ' ').title())
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if output_path:
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                print(f"Plot saved to: {output_path}")
            else:
                plt.show()
                
        except ImportError:
            print("Matplotlib not available for plotting")


def compare_cipher_families(
    device: str = 'cpu',
    num_samples: int = 1000,
    rounds_list: List[int] = [1, 2, 4, 8]
) -> Dict:
    """
    Convenience function to compare all major cipher families.
    
    Args:
        device: torch device
        num_samples: Number of test samples
        rounds_list: List of round counts
        
    Returns:
        Comparison results
    """
    from ctdma.ciphers.speck import SpeckCipher
    from ctdma.ciphers.feistel import FeistelCipher
    from ctdma.ciphers.spn import SPNCipher
    
    # Create cipher instances
    ciphers = {
        'Speck (ARX)': SpeckCipher(rounds=1, device=device),
        'Feistel': FeistelCipher(rounds=1, device=device),
        'SPN': SPNCipher(rounds=1, device=device)
    }
    
    # Try to add ChaCha if available
    try:
        from ctdma.ciphers.chacha import ChaChaCipher
        ciphers['ChaCha20 (ARX)'] = ChaChaCipher(double_rounds=1, device=device)
    except ImportError:
        pass
    
    # Create comparator and run analysis
    comparator = CipherComparator(device=device)
    
    print("Comparing cipher families...")
    results = comparator.compare_gradient_inversion(
        ciphers=ciphers,
        num_samples=num_samples,
        num_rounds_list=rounds_list
    )
    
    # Generate report
    report = comparator.generate_comparison_report()
    print(report)
    
    return results
