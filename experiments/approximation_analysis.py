#!/usr/bin/env python3
"""
Approximation Analysis Experiment

Comprehensive comparison of different approximation methods for discrete
ARX operations, including:
1. Error analysis
2. Gradient fidelity
3. Convergence properties
4. Information preservation
"""

import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict
import json
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from ctdma.approximation.bridge import (
    SigmoidApproximation,
    StraightThroughEstimator,
    GumbelSoftmaxApproximation,
    TemperatureAnnealing
)
from ctdma.approximation.metrics import (
    ApproximationMetrics,
    compare_approximation_methods
)
from ctdma.approximation.convergence import (
    ConvergenceAnalyzer,
    compare_convergence_properties
)
from ctdma.theory.mathematical_analysis import (
    GradientInversionAnalyzer,
    InformationTheoreticAnalyzer
)


def create_approximation_methods(n_bits=16, device='cpu'):
    """Create all approximation methods for comparison."""
    methods = {
        'sigmoid_steep_5': SigmoidApproximation(n_bits, steepness=5.0, operation='modadd').to(device),
        'sigmoid_steep_10': SigmoidApproximation(n_bits, steepness=10.0, operation='modadd').to(device),
        'sigmoid_steep_20': SigmoidApproximation(n_bits, steepness=20.0, operation='modadd').to(device),
        'straight_through': StraightThroughEstimator(n_bits, operation='modadd').to(device),
        'gumbel_temp_0.5': GumbelSoftmaxApproximation(n_bits, temperature=0.5, operation='modadd').to(device),
        'gumbel_temp_1.0': GumbelSoftmaxApproximation(n_bits, temperature=1.0, operation='modadd').to(device),
        'temp_anneal': TemperatureAnnealing(n_bits, initial_temperature=1.0, operation='modadd').to(device),
    }
    return methods


def discrete_modadd(x, y, modulus=2**16):
    """Exact discrete modular addition."""
    return (x + y) % modulus


def experiment_1_error_analysis(device='cpu', n_samples=1000):
    """
    Experiment 1: Compare approximation errors across methods.
    """
    print("\n" + "="*70)
    print("EXPERIMENT 1: Approximation Error Analysis")
    print("="*70)
    
    n_bits = 16
    modulus = 2 ** n_bits
    
    # Create approximation methods
    methods = create_approximation_methods(n_bits, device)
    
    # Generate test data
    x = torch.rand(n_samples, device=device) * modulus
    y = torch.rand(n_samples, device=device) * modulus
    
    # Compute discrete output
    discrete_output = discrete_modadd(x, y, modulus)
    
    # Compare all methods
    print(f"\nComparing {len(methods)} approximation methods on {n_samples} samples...")
    print(f"Input range: [0, {modulus})")
    
    results = {}
    metrics_calculator = ApproximationMetrics(n_bits)
    
    for name, method in methods.items():
        with torch.no_grad():
            approx_output = method(x, y)
        
        # Compute metrics
        error_metrics = metrics_calculator.compute_approximation_error(
            discrete_output,
            approx_output
        )
        
        info_metrics = metrics_calculator.compute_information_preservation(
            discrete_output,
            approx_output
        )
        
        boundary_metrics = metrics_calculator.compute_boundary_metrics(
            x, y, discrete_output, approx_output
        )
        
        results[name] = {
            **error_metrics,
            **info_metrics,
            **boundary_metrics
        }
        
        # Print summary
        print(f"\n{name}:")
        print(f"  L1 Error: {error_metrics['l1_error']:.4f}")
        print(f"  L2 Error: {error_metrics['l2_error']:.4f}")
        print(f"  Correlation: {error_metrics['correlation']:.4f}")
        print(f"  Info Preservation: {info_metrics['information_preservation_ratio']:.4f}")
        print(f"  Boundary Error Amp: {boundary_metrics['boundary_error_amplification']:.2f}x")
    
    return results


def experiment_2_gradient_fidelity(device='cpu', n_samples=100):
    """
    Experiment 2: Analyze gradient fidelity.
    """
    print("\n" + "="*70)
    print("EXPERIMENT 2: Gradient Fidelity Analysis")
    print("="*70)
    
    n_bits = 16
    modulus = 2 ** n_bits
    
    methods = create_approximation_methods(n_bits, device)
    
    # Generate test data (requires grad)
    x = torch.rand(n_samples, device=device, requires_grad=True) * modulus
    y = torch.rand(n_samples, device=device, requires_grad=True) * modulus
    
    # Target (for loss computation)
    with torch.no_grad():
        target = discrete_modadd(x, y, modulus)
    
    results = {}
    metrics_calculator = ApproximationMetrics(n_bits)
    
    for name, method in methods.items():
        # Forward pass
        output = method(x, y)
        
        # Compute loss
        loss = torch.nn.functional.mse_loss(output, target)
        
        # Backward pass
        if x.grad is not None:
            x.grad.zero_()
        loss.backward()
        
        grad_x = x.grad.clone()
        
        # Compute gradient metrics (compare to numerical gradient)
        with torch.no_grad():
            delta = 1e-4
            x_plus = x.detach() + delta
            output_plus = method(x_plus, y.detach())
            numerical_grad = (output_plus - output.detach()) / delta
        
        grad_metrics = metrics_calculator.compute_gradient_fidelity(
            numerical_grad,
            grad_x
        )
        
        results[name] = grad_metrics
        
        print(f"\n{name}:")
        print(f"  Cosine Similarity: {grad_metrics['gradient_cosine_similarity']:.4f}")
        print(f"  Sign Agreement: {grad_metrics['gradient_sign_agreement']:.4f}")
        print(f"  Angular Error: {grad_metrics['gradient_angular_error_deg']:.2f}°")
    
    return results


def experiment_3_convergence_analysis(device='cpu'):
    """
    Experiment 3: Analyze convergence properties with temperature annealing.
    """
    print("\n" + "="*70)
    print("EXPERIMENT 3: Convergence Analysis")
    print("="*70)
    
    n_bits = 16
    modulus = 2 ** n_bits
    
    analyzer = ConvergenceAnalyzer(tolerance=1e-3, max_iterations=200)
    
    # Test sigmoid with different steepness schedules
    def sigmoid_approx(x, y, beta):
        sum_val = x + y
        wrap = torch.sigmoid(beta * (sum_val - modulus))
        return sum_val - modulus * wrap
    
    def discrete_op(x, y):
        return (x + y) % modulus
    
    def input_generator(n):
        inputs = []
        for _ in range(n):
            x = torch.rand(1, device=device) * modulus
            y = torch.rand(1, device=device) * modulus
            inputs.append((x, y))
        return inputs
    
    # Analyze different annealing schedules
    schedules = {
        'exponential': 'exponential',
        'linear': 'linear',
        'cosine': 'cosine'
    }
    
    results = {}
    for schedule_name, schedule_type in schedules.items():
        print(f"\nAnalyzing {schedule_name} annealing...")
        
        anneal_results = analyzer.analyze_temperature_annealing(
            sigmoid_approx,
            discrete_op,
            input_generator,
            initial_temp=20.0,
            final_temp=0.5,
            n_steps=200,
            anneal_schedule=schedule_type
        )
        
        conv_results = anneal_results['convergence_results']
        
        print(f"  Converged: {conv_results.converged}")
        print(f"  Iterations: {conv_results.iterations_to_converge}")
        print(f"  Final Error: {conv_results.final_error:.6f}")
        print(f"  Convergence Rate: {conv_results.convergence_rate:.6f}")
        
        results[schedule_name] = anneal_results
    
    return results


def experiment_4_information_theory(device='cpu', n_samples=1000):
    """
    Experiment 4: Information-theoretic analysis.
    """
    print("\n" + "="*70)
    print("EXPERIMENT 4: Information-Theoretic Analysis")
    print("="*70)
    
    n_bits = 16
    modulus = 2 ** n_bits
    
    analyzer = InformationTheoreticAnalyzer(n_bits)
    
    # Generate samples
    x = torch.rand(n_samples, device=device) * modulus
    y = torch.rand(n_samples, device=device) * modulus
    
    # Discrete operation
    discrete_output = discrete_modadd(x, y, modulus)
    
    # Test different approximations
    methods = create_approximation_methods(n_bits, device)
    
    results = {}
    for name, method in methods.items():
        with torch.no_grad():
            approx_output = method(x, y)
        
        # Information metrics
        info_metrics = analyzer.analyze_information_loss_in_approximation(
            lambda inp: discrete_modadd(inp[:, 0], inp[:, 1], modulus),
            lambda inp: method(inp[:, 0], inp[:, 1]),
            torch.stack([x, y], dim=1)
        )
        
        results[name] = info_metrics
        
        print(f"\n{name}:")
        print(f"  Entropy Discrete: {info_metrics['entropy_discrete']:.4f}")
        print(f"  Entropy Smooth: {info_metrics['entropy_smooth']:.4f}")
        print(f"  Information Loss: {info_metrics['information_loss']:.4f}")
        print(f"  Relative Loss: {info_metrics['relative_information_loss']:.2%}")
    
    return results


def experiment_5_gradient_inversion_probability(device='cpu', n_samples=1000):
    """
    Experiment 5: Measure actual gradient inversion probability.
    """
    print("\n" + "="*70)
    print("EXPERIMENT 5: Gradient Inversion Probability")
    print("="*70)
    
    n_bits = 16
    modulus = 2 ** n_bits
    
    analyzer = GradientInversionAnalyzer(n_bits, modulus)
    
    # Generate samples
    x = torch.rand(n_samples, device=device) * modulus
    y = torch.rand(n_samples, device=device) * modulus
    
    # Analyze discontinuities
    discontinuity_results = analyzer.compute_gradient_discontinuity(x, y, 'modadd')
    
    print(f"\nGradient Discontinuity Analysis:")
    print(f"  Wrap Frequency: {discontinuity_results['wrap_frequency']:.4f}")
    print(f"  Gradient Jump Magnitude: {discontinuity_results['gradient_magnitude_jump']:.4f}")
    print(f"  Inversion Probability: {discontinuity_results['inversion_probability']:.4f}")
    
    # Theoretical prediction
    noise_variance = float(discontinuity_results['gradient_magnitude_jump']) ** 2
    signal_strength = 1.0  # Typical gradient magnitude
    
    theoretical_prob = analyzer.theoretical_inversion_probability(
        noise_variance,
        signal_strength
    )
    
    print(f"  Theoretical Inversion Prob: {theoretical_prob:.4f}")
    
    return discontinuity_results


def create_visualizations(all_results, output_dir='results'):
    """Create visualization plots."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot 1: Error comparison
    if 'error_analysis' in all_results:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Approximation Error Analysis', fontsize=14, fontweight='bold')
        
        methods = list(all_results['error_analysis'].keys())
        
        # L1 errors
        l1_errors = [all_results['error_analysis'][m]['l1_error'] for m in methods]
        axes[0, 0].bar(range(len(methods)), l1_errors)
        axes[0, 0].set_xticks(range(len(methods)))
        axes[0, 0].set_xticklabels(methods, rotation=45, ha='right')
        axes[0, 0].set_ylabel('L1 Error')
        axes[0, 0].set_title('L1 Approximation Error')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Information preservation
        info_pres = [all_results['error_analysis'][m]['information_preservation_ratio'] for m in methods]
        axes[0, 1].bar(range(len(methods)), info_pres)
        axes[0, 1].set_xticks(range(len(methods)))
        axes[0, 1].set_xticklabels(methods, rotation=45, ha='right')
        axes[0, 1].set_ylabel('Information Preservation Ratio')
        axes[0, 1].set_title('Information Preservation')
        axes[0, 1].axhline(y=1.0, color='r', linestyle='--', label='Perfect')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Boundary error amplification
        boundary_amp = [all_results['error_analysis'][m]['boundary_error_amplification'] for m in methods]
        axes[1, 0].bar(range(len(methods)), boundary_amp)
        axes[1, 0].set_xticks(range(len(methods)))
        axes[1, 0].set_xticklabels(methods, rotation=45, ha='right')
        axes[1, 0].set_ylabel('Amplification Factor')
        axes[1, 0].set_title('Boundary Error Amplification')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Correlation
        correlation = [all_results['error_analysis'][m]['correlation'] for m in methods]
        axes[1, 1].bar(range(len(methods)), correlation)
        axes[1, 1].set_xticks(range(len(methods)))
        axes[1, 1].set_xticklabels(methods, rotation=45, ha='right')
        axes[1, 1].set_ylabel('Correlation')
        axes[1, 1].set_title('Output Correlation')
        axes[1, 1].axhline(y=1.0, color='r', linestyle='--', label='Perfect')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'error_analysis.png'), dpi=300, bbox_inches='tight')
        print(f"\nSaved: {output_dir}/error_analysis.png")
        plt.close()
    
    # Plot 2: Gradient fidelity
    if 'gradient_fidelity' in all_results:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle('Gradient Fidelity Analysis', fontsize=14, fontweight='bold')
        
        methods = list(all_results['gradient_fidelity'].keys())
        
        # Cosine similarity
        cos_sim = [all_results['gradient_fidelity'][m]['gradient_cosine_similarity'] for m in methods]
        axes[0].bar(range(len(methods)), cos_sim)
        axes[0].set_xticks(range(len(methods)))
        axes[0].set_xticklabels(methods, rotation=45, ha='right')
        axes[0].set_ylabel('Cosine Similarity')
        axes[0].set_title('Gradient Direction Fidelity')
        axes[0].axhline(y=1.0, color='r', linestyle='--', label='Perfect')
        axes[0].axhline(y=0.0, color='k', linestyle=':', alpha=0.5)
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Sign agreement
        sign_agree = [all_results['gradient_fidelity'][m]['gradient_sign_agreement'] for m in methods]
        axes[1].bar(range(len(methods)), sign_agree)
        axes[1].set_xticks(range(len(methods)))
        axes[1].set_xticklabels(methods, rotation=45, ha='right')
        axes[1].set_ylabel('Sign Agreement')
        axes[1].set_title('Gradient Sign Agreement')
        axes[1].axhline(y=1.0, color='r', linestyle='--', label='Perfect')
        axes[1].axhline(y=0.5, color='orange', linestyle='--', label='Random')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'gradient_fidelity.png'), dpi=300, bbox_inches='tight')
        print(f"Saved: {output_dir}/gradient_fidelity.png")
        plt.close()


def main():
    """Run all experiments."""
    print("="*70)
    print("APPROXIMATION ANALYSIS - COMPREHENSIVE EVALUATION")
    print("="*70)
    print("\nComparing multiple approximation techniques for discrete ARX operations")
    print("Methods: Sigmoid, Straight-Through, Gumbel-Softmax, Temperature Annealing")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    all_results = {}
    
    # Run experiments
    all_results['error_analysis'] = experiment_1_error_analysis(device)
    all_results['gradient_fidelity'] = experiment_2_gradient_fidelity(device)
    all_results['convergence'] = experiment_3_convergence_analysis(device)
    all_results['information_theory'] = experiment_4_information_theory(device)
    all_results['gradient_inversion'] = experiment_5_gradient_inversion_probability(device)
    
    # Create visualizations
    print("\n" + "="*70)
    print("Creating visualizations...")
    print("="*70)
    create_visualizations(all_results)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"approximation_analysis_{timestamp}.json"
    
    # Convert tensors to lists for JSON serialization
    def convert_to_serializable(obj):
        if isinstance(obj, torch.Tensor):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_to_serializable(item) for item in obj]
        else:
            return obj
    
    serializable_results = convert_to_serializable(all_results)
    
    with open(output_file, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    
    return all_results


if __name__ == "__main__":
    results = main()
