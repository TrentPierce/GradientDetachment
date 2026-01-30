"""
Convergence Theory for Gradient Descent in Sawtooth Landscapes

Formal analysis of convergence behavior using:
- Lyapunov stability theory
- Dynamical systems analysis
- Fixed point theory
- Ergodic theory

Notation:
==========================================
Dynamical Systems:
- θ_t: Parameters at time t
- Φ_t: Flow map at time t
- ω(θ_0): ω-limit set (asymptotic behavior)
- d/dt: Time derivative

Lyapunov Theory:
- V(θ): Lyapunov function candidate
- dV/dt: Time derivative along trajectories
- LaSalle invariant set

Stability:
- Stable fixed point
- Asymptotically stable
- Unstable equilibrium
"""

import torch
import numpy as np
from typing import Dict, List, Callable, Optional
from dataclasses import dataclass


class ConvergenceTheory:
    """
    Formal convergence analysis using dynamical systems theory.
    """
    
    @staticmethod
    def theorem_lyapunov_failure():
        """
        Theorem 8: Lyapunov Function Does Not Exist
        
        Statement:
        ==========
        For sawtooth loss landscape ℒ: Θ → ℝ with periodic discontinuities,
        there does NOT exist a Lyapunov function V: Θ → ℝ such that:
          (1) V(θ) ≥ 0 for all θ
          (2) V(θ*) = 0 at minimum θ*
          (3) dV/dt ≤ 0 along all trajectories
        
        Therefore, standard Lyapunov convergence proofs fail.
        
        Proof:
        ======
        [1] Standard Approach: Try V(θ) = ℒ(θ) - ℒ(θ*)
            This should work for smooth convex problems.
        
        [2] Compute dV/dt along gradient flow θ̇ = -∇ℒ(θ):
            dV/dt = ∇V·θ̇
                  = ∇ℒ(θ)·(-∇ℒ(θ))
                  = -||∇ℒ(θ)||²
                  ≤ 0  ✓ (seems good!)
        
        [3] BUT: At discontinuity manifold M_k:
            ∇ℒ is undefined (jump discontinuity)
            dV/dt undefined or infinite
            Trajectory may jump to higher loss region!
        
        [4] Counterexample:
            Consider θ_t approaching M_k from left:
            - Loss decreasing: ℒ(θ_t) → ℒ_min
            - Hit manifold: gradient flips sign
            - Jump to other side with higher loss!
            - V increases: dV/dt > 0 ✗
        
        [5] Alternative V functions also fail:
            - V = ||θ - θ*||²: dV/dt not always negative (gradient flips)
            - V = ℒ + regularization: same discontinuity problem
            - Any V based on ℒ inherits discontinuities
        
        [6] Conclusion: No Lyapunov function exists for sawtooth landscapes.
            Standard convergence theory inapplicable. ∎
        """
        pass
    
    @staticmethod
    def analyze_fixed_points(
        loss_fn: Callable,
        param_range: Tuple[float, float],
        num_points: int = 1000
    ) -> Dict:
        """
        Analyze fixed points of gradient flow.
        
        Fixed point: θ* where ∇ℒ(θ*) = 0
        
        Stability:
        - Stable: Small perturbations decay
        - Unstable: Small perturbations grow
        - Saddle: Stable in some directions, unstable in others
        
        Args:
            loss_fn: Loss function
            param_range: (min, max) for parameter sweep
            num_points: Resolution
            
        Returns:
            Fixed point analysis
        """
        # Sample parameter space
        theta_values = torch.linspace(param_range[0], param_range[1], num_points)
        theta_values.requires_grad_(True)
        
        # Compute gradients
        gradients = []
        losses = []
        
        for theta in theta_values:
            theta_single = theta.unsqueeze(0).clone().detach().requires_grad_(True)
            loss = loss_fn(theta_single)
            loss.backward()
            
            gradients.append(theta_single.grad.item())
            losses.append(loss.item())
        
        gradients = np.array(gradients)
        losses = np.array(losses)
        
        # Find fixed points (where gradient ≈ 0)
        threshold = np.std(gradients) * 0.1
        fixed_point_mask = np.abs(gradients) < threshold
        fixed_points = theta_values.detach().numpy()[fixed_point_mask]
        
        # Classify stability (via second derivative)
        second_derivative = np.gradient(gradients)
        
        stable_points = []
        unstable_points = []
        saddle_points = []
        
        for idx in np.where(fixed_point_mask)[0]:
            hessian_approx = second_derivative[idx]
            if hessian_approx > 0:
                stable_points.append(theta_values[idx].item())
            elif hessian_approx < 0:
                unstable_points.append(theta_values[idx].item())
            else:
                saddle_points.append(theta_values[idx].item())
        
        return {
            'fixed_points': fixed_points.tolist() if len(fixed_points) > 0 else [],
            'num_fixed_points': len(fixed_points),
            'stable_points': stable_points,
            'unstable_points': unstable_points,
            'saddle_points': saddle_points,
            'stability_ratio': len(stable_points) / (len(fixed_points) + 1e-10)
        }


class BasinOfAttractionAnalysis:
    """
    Analyze basins of attraction for different fixed points.
    """
    
    @staticmethod
    def measure_basin_sizes(
        loss_fn: Callable,
        fixed_points: List[torch.Tensor],
        num_initializations: int = 1000,
        tolerance: float = 1e-4
    ) -> Dict:
        """
        Measure relative sizes of basins of attraction.
        
        Basin of attraction for θ*:
        B(θ*) = {θ_0 : lim_{t→∞} Φ_t(θ_0) = θ*}
        
        Where Φ_t is gradient flow map.
        
        Args:
            loss_fn: Loss function
            fixed_points: List of fixed points
            num_initializations: Number of random starts
            tolerance: Convergence tolerance
            
        Returns:
            Basin size estimates
        """
        # Sample random initializations
        convergence_counts = {i: 0 for i in range(len(fixed_points))}
        
        for _ in range(num_initializations):
            # Random initialization
            theta = torch.randn_like(fixed_points[0])
            
            # Run gradient descent
            optimizer = torch.optim.SGD([theta], lr=0.01)
            
            for _ in range(100):  # Max iterations
                optimizer.zero_grad()
                loss = loss_fn(theta)
                loss.backward()
                optimizer.step()
            
            # Check which fixed point we converged to
            for i, fp in enumerate(fixed_points):
                if torch.norm(theta - fp) < tolerance:
                    convergence_counts[i] += 1
                    break
        
        # Compute basin sizes
        total_converged = sum(convergence_counts.values())
        basin_sizes = {
            i: count / num_initializations 
            for i, count in convergence_counts.items()
        }
        
        return {
            'convergence_counts': convergence_counts,
            'basin_sizes': basin_sizes,
            'total_converged': total_converged,
            'convergence_rate': total_converged / num_initializations
        }


def visualize_sawtooth_topology(
    modulus: int = 2**8,
    steepness: float = 10.0,
    save_path: Optional[str] = None
):
    """
    Visualize sawtooth topology with discontinuities.
    
    Args:
        modulus: Modular arithmetic modulus
        steepness: Sigmoid steepness
        save_path: Path to save figure (optional)
    """
    # Generate data
    x_vals = np.linspace(0, 2*modulus, 2000)
    y_fixed = modulus * 0.3  # Fixed y value
    
    # Exact modular addition
    z_exact = (x_vals + y_fixed) % modulus
    
    # Smooth approximation
    x_tensor = torch.tensor(x_vals, dtype=torch.float32)
    y_tensor = torch.full_like(x_tensor, y_fixed)
    z_smooth = (x_tensor + y_tensor - modulus * 
                torch.sigmoid(steepness * (x_tensor + y_tensor - modulus)))
    z_smooth = z_smooth.detach().numpy()
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Function comparison
    ax = axes[0, 0]
    ax.plot(x_vals, z_exact, 'b-', label='Exact (discrete)', linewidth=2, alpha=0.8)
    ax.plot(x_vals, z_smooth, 'r--', label=f'Smooth (β={steepness})', linewidth=2)
    ax.set_xlabel('x (y fixed)')
    ax.set_ylabel('z = (x+y) mod m')
    ax.set_title('Modular Addition: Exact vs Smooth')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Mark discontinuities
    for k in range(1, int(2*modulus/modulus) + 1):
        disc_x = k * modulus - y_fixed
        if 0 <= disc_x <= 2*modulus:
            ax.axvline(x=disc_x, color='green', linestyle=':', alpha=0.5, linewidth=1)
    
    # Plot 2: Error
    ax = axes[0, 1]
    error = np.abs(z_exact - z_smooth)
    ax.plot(x_vals, error, 'orange', linewidth=2)
    ax.set_xlabel('x')
    ax.set_ylabel('|Exact - Smooth|')
    ax.set_title('Approximation Error')
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Gradient (numerical)
    ax = axes[1, 0]
    grad_exact = np.gradient(z_exact, x_vals)
    grad_smooth = np.gradient(z_smooth, x_vals)
    ax.plot(x_vals, grad_exact, 'b-', label='Exact gradient', linewidth=2, alpha=0.8)
    ax.plot(x_vals, grad_smooth, 'r--', label='Smooth gradient', linewidth=2)
    ax.set_xlabel('x')
    ax.set_ylabel('∂z/∂x')
    ax.set_title('Gradient Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    # Plot 4: Gradient error
    ax = axes[1, 1]
    grad_error = np.abs(grad_exact - grad_smooth)
    ax.plot(x_vals, grad_error, 'purple', linewidth=2)
    ax.set_xlabel('x')
    ax.set_ylabel('|∂z/∂x (exact) - ∂z/∂x (smooth)|')
    ax.set_title('Gradient Error')
    ax.grid(True, alpha=0.3)
    
    plt.suptitle(f'Sawtooth Topology Visualization (m={modulus}, β={steepness})',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
    else:
        plt.show()


if __name__ == "__main__":
    print("Convergence Theory Module")
    print("="*80)
    print("Analyzing gradient descent convergence in sawtooth landscapes...\n")
    
    # Visualize topology
    visualize_sawtooth_topology(modulus=2**8, steepness=10.0)
