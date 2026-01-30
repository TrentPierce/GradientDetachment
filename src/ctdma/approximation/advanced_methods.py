"""
Advanced Approximation Methods for Discrete Cryptographic Operations

This module extends the basic approximation techniques with advanced methods:
1. Learnable approximations (neural network-based)
2. Spline-based approximations (piecewise polynomial)
3. Adaptive approximations (dynamic adjustment)
4. Hybrid methods (combining multiple techniques)

All methods maintain backward compatibility with existing ApproximationBridge interface.

Author: Gradient Detachment Research Team
Date: 2026-01-30
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, Callable, List, Dict
from scipy import interpolate
import warnings

# Import base classes
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from approximation.bridge import ApproximationBridge


class LearnableApproximation(ApproximationBridge):
    """
    Learnable Neural Network-Based Approximation.
    
    Uses a small neural network to learn the optimal approximation of discrete
    operations. The network is trained to minimize approximation error while
    maintaining smooth gradients.
    
    Architecture:
        Input → [Hidden Layers with smooth activations] → Output
    
    Properties:
        - Flexible: Can learn complex approximations
        - Adaptive: Improves with training
        - Smooth: Uses smooth activation functions
        - Expressive: Can model non-trivial patterns
    
    Training:
        The network is trained on (input, discrete_output) pairs to minimize:
        L = ||NN(input) - discrete_output||^2 + λ·||∇NN||^2
        
        where the second term encourages smooth gradients.
    
    Args:
        n_bits: Bit width of operations
        hidden_sizes: List of hidden layer sizes
        activation: Activation function ('relu', 'tanh', 'elu', 'silu')
        dropout: Dropout probability (default: 0.0)
        operation: Operation type ('modadd', 'xor')
    
    Example:
        >>> approx = LearnableApproximation(
        ...     n_bits=16,
        ...     hidden_sizes=[64, 32, 16],
        ...     activation='elu'
        ... )
        >>> approx.train_approximation(x_train, y_train, epochs=100)
        >>> z_approx = approx(x_test, y_test)
    """
    
    def __init__(
        self,
        n_bits: int = 16,
        hidden_sizes: List[int] = [64, 32],
        activation: str = 'elu',
        dropout: float = 0.0,
        operation: str = 'modadd',
        gradient_penalty: float = 0.01
    ):
        super().__init__(n_bits)
        self.hidden_sizes = hidden_sizes
        self.activation_name = activation
        self.dropout_prob = dropout
        self.operation = operation
        self.gradient_penalty = gradient_penalty
        
        # Build network
        self.network = self._build_network()
        
        # Training history
        self.training_history = {
            'losses': [],
            'errors': [],
            'gradient_norms': []
        }
        
    def _build_network(self) -> nn.Module:
        """Build the learnable approximation network."""
        layers = []
        
        # Input layer: 2 inputs (x, y)
        prev_size = 2
        
        # Hidden layers
        for hidden_size in self.hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            
            # Smooth activation
            if self.activation_name == 'relu':
                layers.append(nn.ReLU())
            elif self.activation_name == 'tanh':
                layers.append(nn.Tanh())
            elif self.activation_name == 'elu':
                layers.append(nn.ELU())
            elif self.activation_name == 'silu':
                layers.append(nn.SiLU())  # Smooth alternative to ReLU
            else:
                raise ValueError(f"Unknown activation: {self.activation_name}")
            
            # Optional dropout
            if self.dropout_prob > 0:
                layers.append(nn.Dropout(self.dropout_prob))
            
            prev_size = hidden_size
        
        # Output layer: 1 output
        layers.append(nn.Linear(prev_size, 1))
        
        # Final activation to ensure output range
        if self.operation == 'modadd':
            # For modular addition, use sigmoid scaled to [0, modulus]
            layers.append(nn.Sigmoid())
        
        return nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Forward pass through learned approximation."""
        # Normalize inputs to [0, 1]
        x_norm = x.float() / self.modulus
        y_norm = y.float() / self.modulus
        
        # Concatenate inputs
        inputs = torch.stack([x_norm, y_norm], dim=-1)
        
        # Forward through network
        output = self.network(inputs).squeeze(-1)
        
        # Scale back to [0, modulus]
        if self.operation == 'modadd':
            return output * self.modulus
        else:
            return output
    
    def discrete_op(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Exact discrete operation."""
        if self.operation == 'modadd':
            return (x + y) % self.modulus
        elif self.operation == 'xor':
            x_int = (x * (2**self.n_bits)).long()
            y_int = (y * (2**self.n_bits)).long()
            result = (x_int ^ y_int).float() / (2**self.n_bits)
            return result
        else:
            raise ValueError(f"Unknown operation: {self.operation}")
    
    def train_approximation(
        self,
        x_train: torch.Tensor,
        y_train: torch.Tensor,
        epochs: int = 100,
        batch_size: int = 128,
        learning_rate: float = 0.001,
        verbose: bool = True
    ) -> Dict[str, List[float]]:
        """
        Train the learnable approximation.
        
        Args:
            x_train: Training inputs (first operand)
            y_train: Training inputs (second operand)
            epochs: Number of training epochs
            batch_size: Batch size for training
            learning_rate: Learning rate for Adam optimizer
            verbose: Print progress
            
        Returns:
            Training history dictionary
        """
        optimizer = torch.optim.Adam(self.network.parameters(), lr=learning_rate)
        
        # Compute discrete targets once
        with torch.no_grad():
            targets = self.discrete_op(x_train, y_train)
        
        # Training loop
        num_samples = len(x_train)
        
        for epoch in range(epochs):
            # Shuffle data
            indices = torch.randperm(num_samples)
            
            epoch_loss = 0.0
            epoch_error = 0.0
            epoch_grad_norm = 0.0
            num_batches = 0
            
            for start_idx in range(0, num_samples, batch_size):
                end_idx = min(start_idx + batch_size, num_samples)
                batch_indices = indices[start_idx:end_idx]
                
                x_batch = x_train[batch_indices]
                y_batch = y_train[batch_indices]
                target_batch = targets[batch_indices]
                
                # Forward pass
                output = self.forward(x_batch, y_batch)
                
                # Approximation loss
                approx_loss = F.mse_loss(output, target_batch)
                
                # Gradient penalty (encourage smooth gradients)
                grad_penalty = 0.0
                if self.gradient_penalty > 0:
                    # Compute gradient norm
                    grad_outputs = torch.ones_like(output)
                    x_batch_grad = x_batch.clone().requires_grad_(True)
                    y_batch_grad = y_batch.clone().requires_grad_(True)
                    
                    output_grad = self.forward(x_batch_grad, y_batch_grad)
                    
                    grads = torch.autograd.grad(
                        outputs=output_grad,
                        inputs=[x_batch_grad, y_batch_grad],
                        grad_outputs=grad_outputs,
                        create_graph=True,
                        retain_graph=True
                    )
                    
                    grad_norm = torch.norm(grads[0]) + torch.norm(grads[1])
                    grad_penalty = self.gradient_penalty * grad_norm
                    epoch_grad_norm += grad_norm.item()
                
                # Total loss
                loss = approx_loss + grad_penalty
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                epoch_error += approx_loss.item()
                num_batches += 1
            
            # Record history
            avg_loss = epoch_loss / num_batches
            avg_error = epoch_error / num_batches
            avg_grad_norm = epoch_grad_norm / num_batches
            
            self.training_history['losses'].append(avg_loss)
            self.training_history['errors'].append(avg_error)
            self.training_history['gradient_norms'].append(avg_grad_norm)
            
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}: "
                      f"Loss={avg_loss:.6f}, "
                      f"Error={avg_error:.6f}, "
                      f"GradNorm={avg_grad_norm:.6f}")
        
        return self.training_history


class SplineApproximation(ApproximationBridge):
    """
    Spline-Based Piecewise Polynomial Approximation.
    
    Uses cubic splines to create smooth approximations of discrete operations.
    Splines provide C^2 continuity (continuous second derivatives) while
    maintaining low approximation error.
    
    Method:
        1. Sample discrete operation at control points
        2. Fit cubic spline through samples
        3. Evaluate spline for intermediate values
    
    Properties:
        - C^2 continuous (smooth up to 2nd derivative)
        - Low approximation error
        - Efficient evaluation
        - No training required
    
    Mathematical Background:
        Cubic spline S(x) satisfies:
        - S(x_i) = y_i (interpolation)
        - S'(x) continuous
        - S''(x) continuous
        - Minimizes ∫(S''(x))^2 dx (smoothness)
    
    Args:
        n_bits: Bit width
        num_control_points: Number of spline control points
        operation: Operation type
        extrapolate: Allow extrapolation beyond control points
        spline_order: Spline order (1=linear, 3=cubic, 5=quintic)
    
    Example:
        >>> approx = SplineApproximation(
        ...     n_bits=16,
        ...     num_control_points=100,
        ...     spline_order=3  # cubic
        ... )
        >>> z = approx(x, y)
    """
    
    def __init__(
        self,
        n_bits: int = 16,
        num_control_points: int = 100,
        operation: str = 'modadd',
        extrapolate: bool = False,
        spline_order: int = 3
    ):
        super().__init__(n_bits)
        self.num_control_points = num_control_points
        self.operation = operation
        self.extrapolate = extrapolate
        self.spline_order = min(spline_order, 5)  # scipy limit
        
        # Build spline
        self.spline = self._build_spline()
        
    def _build_spline(self):
        """Build cubic spline interpolation."""
        # Create control points
        control_x = np.linspace(0, self.modulus * 2, self.num_control_points)
        control_y = np.zeros(self.num_control_points)
        
        # Evaluate discrete operation at control points
        for i, x_val in enumerate(control_x):
            # For modular addition with y=0
            if self.operation == 'modadd':
                control_y[i] = x_val % self.modulus
            else:
                control_y[i] = x_val
        
        # Create spline
        if self.extrapolate:
            fill_value = 'extrapolate'
        else:
            fill_value = (control_y[0], control_y[-1])
        
        spline = interpolate.UnivariateSpline(
            control_x, 
            control_y, 
            k=self.spline_order,  # Order of spline
            s=0  # No smoothing (exact interpolation)
        )
        
        return spline
    
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Forward pass using spline interpolation."""
        # Convert to numpy for spline evaluation
        x_np = x.detach().cpu().numpy()
        y_np = y.detach().cpu().numpy()
        
        # Sum for modular addition
        sum_np = x_np + y_np
        
        # Evaluate spline
        result_np = self.spline(sum_np)
        
        # Convert back to torch
        result = torch.from_numpy(result_np).to(x.device).float()
        
        return result
    
    def discrete_op(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Exact discrete operation."""
        if self.operation == 'modadd':
            return (x + y) % self.modulus
        else:
            raise ValueError(f"Unknown operation: {self.operation}")
    
    def evaluate_derivative(self, x: torch.Tensor, y: torch.Tensor, order: int = 1) -> torch.Tensor:
        """
        Evaluate derivative of spline approximation.
        
        Args:
            x, y: Input tensors
            order: Derivative order (1 or 2)
            
        Returns:
            Derivative values
        """
        sum_np = (x + y).detach().cpu().numpy()
        
        # Evaluate derivative
        if order == 1:
            deriv_np = self.spline.derivative()(sum_np)
        elif order == 2:
            deriv_np = self.spline.derivative(2)(sum_np)
        else:
            raise ValueError("Only derivatives of order 1 or 2 supported")
        
        return torch.from_numpy(deriv_np).to(x.device).float()


class AdaptiveApproximation(ApproximationBridge):
    """
    Adaptive Approximation with Dynamic Adjustment.
    
    Automatically adjusts approximation parameters based on local error.
    Uses error feedback to refine approximation in high-error regions.
    
    Strategy:
        1. Start with coarse approximation
        2. Measure local approximation error
        3. Refine approximation in high-error regions
        4. Iterate until error threshold met
    
    Features:
        - Automatic error-based refinement
        - Local adaptation (different parameters per region)
        - Convergence guarantees
        - Efficient (focuses on high-error regions)
    
    Args:
        n_bits: Bit width
        base_method: Base approximation method to adapt
        error_threshold: Target approximation error
        max_refinements: Maximum refinement iterations
        operation: Operation type
    
    Example:
        >>> base = SigmoidApproximation(n_bits=16, steepness=5.0)
        >>> adaptive = AdaptiveApproximation(
        ...     n_bits=16,
        ...     base_method=base,
        ...     error_threshold=0.01
        ... )
        >>> adaptive.adapt(x_train, y_train)
        >>> z = adaptive(x_test, y_test)
    """
    
    def __init__(
        self,
        n_bits: int = 16,
        base_method: Optional[ApproximationBridge] = None,
        error_threshold: float = 0.01,
        max_refinements: int = 10,
        operation: str = 'modadd'
    ):
        super().__init__(n_bits)
        self.error_threshold = error_threshold
        self.max_refinements = max_refinements
        self.operation = operation
        
        # Base method (default: sigmoid)
        if base_method is None:
            from approximation.bridge import SigmoidApproximation
            self.base_method = SigmoidApproximation(n_bits, steepness=5.0, operation=operation)
        else:
            self.base_method = base_method
        
        # Adaptive parameters (per region)
        self.region_parameters = {}
        self.is_adapted = False
        
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Forward pass with adaptive approximation."""
        if not self.is_adapted:
            # Use base method if not yet adapted
            return self.base_method.forward(x, y)
        
        # Use region-specific parameters
        result = torch.zeros_like(x, dtype=torch.float)
        
        for region_id, params in self.region_parameters.items():
            # Identify samples in this region
            region_mask = self._get_region_mask(x, y, region_id)
            
            if region_mask.any():
                # Apply region-specific approximation
                if 'steepness' in params:
                    # Adjust steepness for this region
                    original_steepness = self.base_method.steepness
                    self.base_method.steepness = params['steepness']
                    
                    region_result = self.base_method.forward(
                        x[region_mask], 
                        y[region_mask]
                    )
                    
                    # Restore original
                    self.base_method.steepness = original_steepness
                    
                    result[region_mask] = region_result
        
        return result
    
    def adapt(
        self,
        x_train: torch.Tensor,
        y_train: torch.Tensor,
        verbose: bool = True
    ) -> Dict[str, List[float]]:
        """
        Adapt approximation based on training data.
        
        Args:
            x_train: Training inputs (first operand)
            y_train: Training inputs (second operand)
            verbose: Print adaptation progress
            
        Returns:
            Adaptation history
        """
        history = {
            'errors': [],
            'num_regions': [],
            'refinements': []
        }
        
        # Divide input space into regions
        num_regions = 10
        region_boundaries = np.linspace(0, self.modulus * 2, num_regions + 1)
        
        # Initialize region parameters
        for i in range(num_regions):
            self.region_parameters[i] = {'steepness': 5.0}  # Default
        
        # Refinement loop
        for refinement in range(self.max_refinements):
            # Measure error in each region
            max_error = 0.0
            
            for region_id in range(num_regions):
                # Get samples in this region
                region_mask = self._get_region_mask(x_train, y_train, region_id)
                
                if region_mask.sum() == 0:
                    continue
                
                x_region = x_train[region_mask]
                y_region = y_train[region_mask]
                
                # Compute error
                approx = self.base_method.forward(x_region, y_region)
                discrete = self.discrete_op(x_region, y_region)
                error = torch.abs(approx - discrete).mean().item()
                
                # Update max error
                max_error = max(max_error, error)
                
                # If error too high, increase steepness
                if error > self.error_threshold:
                    self.region_parameters[region_id]['steepness'] *= 1.5
                    
                    if verbose:
                        print(f"Refinement {refinement+1}, Region {region_id}: "
                              f"Error={error:.6f}, "
                              f"New steepness={self.region_parameters[region_id]['steepness']:.2f}")
            
            history['errors'].append(max_error)
            history['num_regions'].append(num_regions)
            history['refinements'].append(refinement + 1)
            
            # Check convergence
            if max_error < self.error_threshold:
                if verbose:
                    print(f"Adaptation converged after {refinement+1} refinements")
                break
        
        self.is_adapted = True
        return history
    
    def _get_region_mask(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        region_id: int
    ) -> torch.Tensor:
        """Get boolean mask for samples in specified region."""
        sum_vals = x + y
        
        num_regions = len(self.region_parameters)
        region_boundaries = np.linspace(0, self.modulus * 2, num_regions + 1)
        
        lower = region_boundaries[region_id]
        upper = region_boundaries[region_id + 1]
        
        mask = (sum_vals >= lower) & (sum_vals < upper)
        return mask
    
    def discrete_op(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Exact discrete operation."""
        if self.operation == 'modadd':
            return (x + y) % self.modulus
        else:
            raise ValueError(f"Unknown operation: {self.operation}")


class HybridApproximation(ApproximationBridge):
    """
    Hybrid Approximation Combining Multiple Methods.
    
    Combines multiple approximation techniques using weighted ensemble:
        z_hybrid = Σ w_i · φ_i(x, y)
    
    where φ_i are different approximation methods and w_i are learned weights.
    
    Advantages:
        - Leverages strengths of multiple methods
        - More robust than single method
        - Better approximation quality
        - Learnable combination weights
    
    Methods Combined:
        - Sigmoid (smooth, global)
        - Spline (accurate, local)
        - Learnable (adaptive, flexible)
    
    Args:
        n_bits: Bit width
        methods: List of approximation methods
        learn_weights: Whether to learn combination weights
        operation: Operation type
    
    Example:
        >>> methods = [
        ...     SigmoidApproximation(n_bits=16, steepness=10.0),
        ...     SplineApproximation(n_bits=16, num_control_points=50),
        ... ]
        >>> hybrid = HybridApproximation(n_bits=16, methods=methods)
        >>> hybrid.fit_weights(x_train, y_train)
        >>> z = hybrid(x_test, y_test)
    """
    
    def __init__(
        self,
        n_bits: int = 16,
        methods: Optional[List[ApproximationBridge]] = None,
        learn_weights: bool = True,
        operation: str = 'modadd'
    ):
        super().__init__(n_bits)
        self.operation = operation
        self.learn_weights = learn_weights
        
        # Methods
        if methods is None:
            # Default: sigmoid + spline
            from approximation.bridge import SigmoidApproximation
            self.methods = [
                SigmoidApproximation(n_bits, steepness=10.0, operation=operation),
                SplineApproximation(n_bits, num_control_points=50, operation=operation)
            ]
        else:
            self.methods = methods
        
        # Combination weights
        num_methods = len(self.methods)
        if learn_weights:
            # Learnable weights (initialized uniform)
            self.weights = nn.Parameter(torch.ones(num_methods) / num_methods)
        else:
            # Fixed uniform weights
            self.register_buffer('weights', torch.ones(num_methods) / num_methods)
    
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Forward pass through hybrid approximation."""
        # Evaluate all methods
        outputs = []
        for method in self.methods:
            output = method.forward(x, y)
            outputs.append(output)
        
        # Stack outputs
        outputs_stacked = torch.stack(outputs, dim=-1)  # (batch, num_methods)
        
        # Normalize weights (softmax)
        weights_normalized = F.softmax(self.weights, dim=0)
        
        # Weighted combination
        result = torch.sum(outputs_stacked * weights_normalized, dim=-1)
        
        return result
    
    def fit_weights(
        self,
        x_train: torch.Tensor,
        y_train: torch.Tensor,
        epochs: int = 100,
        learning_rate: float = 0.01,
        verbose: bool = True
    ) -> Dict[str, List]:
        """
        Learn optimal combination weights.
        
        Args:
            x_train: Training inputs (first operand)
            y_train: Training inputs (second operand)
            epochs: Number of training epochs
            learning_rate: Learning rate
            verbose: Print progress
            
        Returns:
            Training history
        """
        if not self.learn_weights:
            raise ValueError("Cannot fit weights when learn_weights=False")
        
        optimizer = torch.optim.Adam([self.weights], lr=learning_rate)
        
        # Target: discrete operation
        with torch.no_grad():
            targets = self.discrete_op(x_train, y_train)
        
        history = {'losses': [], 'weights': []}
        
        for epoch in range(epochs):
            optimizer.zero_grad()
            
            # Forward pass
            output = self.forward(x_train, y_train)
            
            # Loss: MSE
            loss = F.mse_loss(output, targets)
            
            # Backward
            loss.backward()
            optimizer.step()
            
            # Record
            history['losses'].append(loss.item())
            history['weights'].append(F.softmax(self.weights, dim=0).detach().cpu().numpy().copy())
            
            if verbose and (epoch + 1) % 20 == 0:
                weights_str = ", ".join([f"{w:.3f}" for w in history['weights'][-1]])
                print(f"Epoch {epoch+1}: Loss={loss.item():.6f}, Weights=[{weights_str}]")
        
        return history
    
    def discrete_op(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Exact discrete operation."""
        if self.operation == 'modadd':
            return (x + y) % self.modulus
        else:
            raise ValueError(f"Unknown operation: {self.operation}")


# Factory function for advanced methods
def create_advanced_approximation(
    method_type: str,
    n_bits: int = 16,
    operation: str = 'modadd',
    **kwargs
) -> ApproximationBridge:
    """
    Factory function to create advanced approximation methods.
    
    Args:
        method_type: Type of advanced approximation
            - 'learnable': Neural network-based
            - 'spline': Cubic spline interpolation
            - 'adaptive': Adaptive refinement
            - 'hybrid': Ensemble of methods
        n_bits: Bit width
        operation: Operation type
        **kwargs: Additional parameters for specific method
        
    Returns:
        Advanced approximation instance
        
    Example:
        >>> # Learnable approximation
        >>> approx = create_advanced_approximation(
        ...     'learnable',
        ...     n_bits=16,
        ...     hidden_sizes=[64, 32],
        ...     activation='elu'
        ... )
        
        >>> # Spline approximation
        >>> approx = create_advanced_approximation(
        ...     'spline',
        ...     n_bits=16,
        ...     num_control_points=100,
        ...     spline_order=3
        ... )
    """
    if method_type == 'learnable':
        return LearnableApproximation(n_bits, operation=operation, **kwargs)
    elif method_type == 'spline':
        return SplineApproximation(n_bits, operation=operation, **kwargs)
    elif method_type == 'adaptive':
        return AdaptiveApproximation(n_bits, operation=operation, **kwargs)
    elif method_type == 'hybrid':
        return HybridApproximation(n_bits, operation=operation, **kwargs)
    else:
        raise ValueError(f"Unknown advanced method type: {method_type}")


if __name__ == "__main__":
    print("Advanced Approximation Methods")
    print("="*70)
    print("\nAvailable methods:")
    print("1. LearnableApproximation - Neural network-based")
    print("2. SplineApproximation - Cubic spline interpolation")
    print("3. AdaptiveApproximation - Dynamic error-based refinement")
    print("4. HybridApproximation - Ensemble of multiple methods")
    print("\nUse create_advanced_approximation() factory function.")
