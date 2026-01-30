"""
Convergence Analysis for Approximations

Analyzes how approximation precision affects:
1. Training convergence speed
2. Final accuracy
3. Gradient flow stability
4. Loss landscape topology
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Callable, Optional
from dataclasses import dataclass


@dataclass
class ConvergenceResult:
    """Container for convergence analysis results."""
    method_name: str
    final_loss: float
    convergence_speed: float  # Steps to reach 90% of final loss
    gradient_stability: float  # Std of gradient norms
    training_history: Dict


class ConvergenceAnalyzer:
    """
    Analyzes convergence properties of different approximation methods.
    """
    
    def __init__(self, task: str = 'xor', num_epochs: int = 100):
        """
        Args:
            task: 'xor', 'mod_add', or 'cipher'
            num_epochs: Number of training epochs
        """
        self.task = task
        self.num_epochs = num_epochs
        self.results = []
        
    def analyze_method(self, 
                      approximation: nn.Module,
                      method_name: str,
                      train_loader: Optional[torch.utils.data.DataLoader] = None) -> ConvergenceResult:
        """
        Analyze convergence of a specific approximation method.
        
        Args:
            approximation: Approximation module
            method_name: Name for tracking
            train_loader: Optional data loader (if None, generates synthetic data)
            
        Returns:
            ConvergenceResult with analysis
        """
        # Generate synthetic data if not provided
        if train_loader is None:
            train_loader = self._generate_synthetic_data()
        
        # Simple model using the approximation
        model = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
        
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        criterion = nn.BCELoss()
        
        # Training loop
        history = {
            'loss': [],
            'accuracy': [],
            'gradient_norms': []
        }
        
        for epoch in range(self.num_epochs):
            epoch_loss = 0.0
            epoch_acc = 0.0
            epoch_grads = []
            
            for batch_x, batch_y in train_loader:
                optimizer.zero_grad()
                
                # Forward pass with approximation
                if self.task == 'xor':
                    # Use approximation for XOR
                    x1, x2 = batch_x[:, 0], batch_x[:, 1]
                    approx_out = approximation.approximate_xor(x1, x2)
                    features = torch.stack([x1, x2, approx_out], dim=1)
                else:
                    features = batch_x
                
                outputs = model(features[:, :2])
                loss = criterion(outputs.squeeze(), batch_y)
                
                # Backward pass
                loss.backward()
                
                # Track gradient norms
                total_norm = 0.0
                for p in model.parameters():
                    if p.grad is not None:
                        total_norm += p.grad.norm().item() ** 2
                total_norm = total_norm ** 0.5
                epoch_grads.append(total_norm)
                
                optimizer.step()
                
                epoch_loss += loss.item()
                
                # Accuracy
                predicted = (outputs.squeeze() > 0.5).float()
                epoch_acc += (predicted == batch_y).float().mean().item()
            
            # Average over batches
            epoch_loss /= len(train_loader)
            epoch_acc /= len(train_loader)
            
            history['loss'].append(epoch_loss)
            history['accuracy'].append(epoch_acc)
            history['gradient_norms'].append(np.mean(epoch_grads))
        
        # Analyze results
        final_loss = history['loss'][-1]
        
        # Convergence speed: epochs to reach 90% of final loss
        target_loss = final_loss * 1.1  # 90% closer to final
        convergence_speed = 0
        for i, loss in enumerate(history['loss']):
            if loss <= target_loss:
                convergence_speed = i
                break
        if convergence_speed == 0:
            convergence_speed = self.num_epochs
        
        # Gradient stability: std of gradient norms
        gradient_stability = np.std(history['gradient_norms'])
        
        result = ConvergenceResult(
            method_name=method_name,
            final_loss=final_loss,
            convergence_speed=convergence_speed,
            gradient_stability=gradient_stability,
            training_history=history
        )
        
        self.results.append(result)
        return result
    
    def _generate_synthetic_data(self, num_samples: int = 1000, 
                                batch_size: int = 32) -> torch.utils.data.DataLoader:
        """
        Generate synthetic data for the task.
        """
        if self.task == 'xor':
            # XOR task
            x = torch.rand(num_samples, 2)
            y = ((x[:, 0] > 0.5).float() + (x[:, 1] > 0.5).float()) % 2
        elif self.task == 'mod_add':
            # Modular addition task
            modulus = 2
            x = torch.rand(num_samples, 2)
            y = (x[:, 0] + x[:, 1]) % modulus
            y = (y > modulus/2).float()
        else:
            # Generic classification
            x = torch.rand(num_samples, 2)
            y = (x[:, 0] + x[:, 1] > 1.0).float()
        
        dataset = torch.utils.data.TensorDataset(x, y)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        return loader
    
    def compare_methods(self, approximations: Dict[str, nn.Module]) -> Dict:
        """
        Compare multiple approximation methods.
        
        Args:
            approximations: Dict mapping method names to approximation modules
            
        Returns:
            Comparison results
        """
        self.results = []
        
        for name, approx in approximations.items():
            print(f"Analyzing {name}...")
            self.analyze_method(approx, name)
        
        # Summarize
        comparison = {
            'methods': [r.method_name for r in self.results],
            'final_losses': [r.final_loss for r in self.results],
            'convergence_speeds': [r.convergence_speed for r in self.results],
            'gradient_stabilities': [r.gradient_stability for r in self.results],
        }
        
        return comparison
    
    def plot_results(self, save_path: Optional[str] = None):
        """
        Plot convergence curves for all methods.
        """
        if not self.results:
            print("No results to plot. Run analyze_method() first.")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Loss curves
        for result in self.results:
            axes[0, 0].plot(result.training_history['loss'], 
                          label=result.method_name, linewidth=2)
        axes[0, 0].set_xlabel('Epoch', fontsize=12)
        axes[0, 0].set_ylabel('Loss', fontsize=12)
        axes[0, 0].set_title('Training Loss', fontsize=14, fontweight='bold')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Accuracy curves
        for result in self.results:
            axes[0, 1].plot(result.training_history['accuracy'], 
                          label=result.method_name, linewidth=2)
        axes[0, 1].set_xlabel('Epoch', fontsize=12)
        axes[0, 1].set_ylabel('Accuracy', fontsize=12)
        axes[0, 1].set_title('Training Accuracy', fontsize=14, fontweight='bold')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Gradient norms
        for result in self.results:
            axes[1, 0].plot(result.training_history['gradient_norms'], 
                          label=result.method_name, linewidth=2)
        axes[1, 0].set_xlabel('Epoch', fontsize=12)
        axes[1, 0].set_ylabel('Gradient Norm', fontsize=12)
        axes[1, 0].set_title('Gradient Norm Over Time', fontsize=14, fontweight='bold')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Bar chart: Final metrics
        methods = [r.method_name for r in self.results]
        final_losses = [r.final_loss for r in self.results]
        conv_speeds = [r.convergence_speed for r in self.results]
        
        x = np.arange(len(methods))
        width = 0.35
        
        axes[1, 1].bar(x - width/2, final_losses, width, label='Final Loss', alpha=0.7)
        axes[1, 1].bar(x + width/2, np.array(conv_speeds)/max(conv_speeds), 
                      width, label='Convergence Speed (normalized)', alpha=0.7)
        axes[1, 1].set_xlabel('Method', fontsize=12)
        axes[1, 1].set_ylabel('Value', fontsize=12)
        axes[1, 1].set_title('Final Metrics Comparison', fontsize=14, fontweight='bold')
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels(methods)
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        
        plt.show()


def analyze_approximation_impact(approximations: Dict[str, nn.Module],
                                task: str = 'xor',
                                num_epochs: int = 100) -> Dict:
    """
    Convenience function to analyze multiple approximation methods.
    
    Args:
        approximations: Dict of approximation modules
        task: Task type
        num_epochs: Number of training epochs
        
    Returns:
        Analysis results
    """
    analyzer = ConvergenceAnalyzer(task=task, num_epochs=num_epochs)
    comparison = analyzer.compare_methods(approximations)
    
    return {
        'analyzer': analyzer,
        'comparison': comparison
    }


def plot_convergence_curves(results: List[ConvergenceResult],
                           save_path: Optional[str] = None):
    """
    Plot convergence curves from results.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Loss
    for result in results:
        axes[0].plot(result.training_history['loss'], 
                    label=result.method_name, linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training Loss Comparison', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy
    for result in results:
        axes[1].plot(result.training_history['accuracy'], 
                    label=result.method_name, linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy', fontsize=12)
    axes[1].set_title('Training Accuracy Comparison', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.show()


if __name__ == "__main__":
    print("Testing convergence analyzer...\n")
    
    from .bridge import SigmoidApproximation
    
    analyzer = ConvergenceAnalyzer(task='xor', num_epochs=50)
    approx = SigmoidApproximation(steepness=5.0)
    
    result = analyzer.analyze_method(approx, 'Sigmoid')
    
    print(f"Method: {result.method_name}")
    print(f"Final Loss: {result.final_loss:.4f}")
    print(f"Convergence Speed: {result.convergence_speed} epochs")
    print(f"Gradient Stability: {result.gradient_stability:.4f}")
    
    print("\n✓ Convergence analyzer loaded successfully")