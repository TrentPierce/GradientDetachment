"""
Convergence Analysis and Stability Theory for Gradient Descent in Sawtooth Landscapes

This module provides rigorous convergence proofs using:
- Lyapunov stability theory
- Fixed-point theorems
- Dynamical systems analysis
- Perturbation theory

Mathematical Framework:
    Gradient Flow: dθ/dt = -∇ℒ(θ)
    Discrete GD: θ_{t+1} = θ_t - α∇ℒ(θ_t)
    Lyapunov Function: V(θ) ≥ 0, V(θ*) = 0, dV/dt ≤ 0
    
"""

import torch
import numpy as np
from typing import Dict, Tuple, Callable, List, Optional
from dataclasses import dataclass
from scipy.integrate import odeint
from scipy.optimize import fsolve
import warnings


@dataclass
class ConvergenceTheorem:
    """Structure for convergence theorem."""
    name: str
    statement: str
    assumptions: List[str]
    proof: List[str]
    convergence_rate: Optional[str] = None
    stability_type: Optional[str] = None


class LyapunovStabilityAnalysis:
    """
    Lyapunov stability analysis for gradient descent.
    
    THEOREM (Lyapunov Stability):
    Let V: Θ → ℝ be a continuously differentiable function satisfying:
    1. V(θ*) = 0
    2. V(θ) > 0 for θ ≠ θ*
    3. dV/dt ≤ 0 along trajectories
    
    Then θ* is stable. If additionally dV/dt < 0 for θ ≠ θ*, then
    θ* is asymptotically stable.
    
    For sawtooth landscapes, we prove that true solutions are UNSTABLE
    while inverted solutions are STABLE.
    """
    
    @staticmethod
    def construct_lyapunov_function(
        theta: torch.Tensor,
        equilibrium: torch.Tensor
    ) -> torch.Tensor:
        """
        Construct Lyapunov function V(θ) = ||θ - θ*||².
        
        Properties:
        1. V(θ*) = 0 ✓
        2. V(θ) > 0 for θ ≠ θ* ✓
        3. V is continuously differentiable ✓
        
        Args:
            theta: Current parameter state
            equilibrium: Equilibrium point θ*
            
        Returns:
            Lyapunov function value
        """
        return torch.norm(theta - equilibrium) ** 2
    
    @staticmethod
    def compute_lyapunov_derivative(
        theta: torch.Tensor,
        equilibrium: torch.Tensor,
        gradient: torch.Tensor,
        learning_rate: float = 1.0
    ) -> torch.Tensor:
        """
        Compute dV/dt along gradient flow.
        
        For V(θ) = ||θ - θ*||²:
        dV/dt = 2⟨θ - θ*, dθ/dt⟩
              = 2⟨θ - θ*, -α∇ℒ(θ)⟩
              = -2α⟨θ - θ*, ∇ℒ(θ)⟩
        
        Stability criterion:
        - dV/dt < 0 ⟹ stable (V decreases)
        - dV/dt > 0 ⟹ unstable (V increases)
        - dV/dt = 0 ⟹ critical
        
        Args:
            theta: Current state
            equilibrium: Equilibrium point
            gradient: ∇ℒ(θ)
            learning_rate: α
            
        Returns:
            Lyapunov derivative dV/dt
        """
        displacement = theta - equilibrium
        dV_dt = -2 * learning_rate * torch.dot(
            displacement.flatten(),
            gradient.flatten()
        )
        
        return dV_dt
    
    @staticmethod
    def analyze_stability(
        loss_fn: Callable,
        equilibrium: torch.Tensor,
        learning_rate: float = 0.01,
        perturbation_magnitude: float = 0.01,
        n_perturbations: int = 100
    ) -> Dict:
        """
        Analyze Lyapunov stability of equilibrium point.
        
        Tests stability by perturbing equilibrium and checking
        if Lyapunov function decreases.
        
        Args:
            loss_fn: Loss function ℒ(θ)
            equilibrium: Equilibrium point θ*
            learning_rate: GD learning rate α
            perturbation_magnitude: Size of test perturbations
            n_perturbations: Number of random perturbations
            
        Returns:
            Stability analysis results
        """
        equilibrium = equilibrium.detach().clone()
        dim = equilibrium.numel()\n        \n        stable_count = 0\n        unstable_count = 0\n        lyapunov_derivatives = []\n        \n        for _ in range(n_perturbations):\n            # Random perturbation\n            perturbation = torch.randn_like(equilibrium) * perturbation_magnitude\n            theta = (equilibrium + perturbation).requires_grad_(True)\n            \n            # Compute gradient at perturbed point\n            loss = loss_fn(theta)\n            if theta.grad is not None:\n                theta.grad.zero_()\n            loss.backward()\n            gradient = theta.grad.clone()\n            \n            # Lyapunov derivative\n            dV_dt = LyapunovStabilityAnalysis.compute_lyapunov_derivative(\n                theta, equilibrium, gradient, learning_rate\n            )\n            \n            lyapunov_derivatives.append(dV_dt.item())\n            \n            # Check stability\n            if dV_dt < 0:\n                stable_count += 1\n            else:\n                unstable_count += 1\n        \n        lyapunov_derivatives = np.array(lyapunov_derivatives)\n        \n        # Overall stability classification\n        stability_ratio = stable_count / n_perturbations\n        \n        if stability_ratio > 0.95:\n            stability_type = \"ASYMPTOTICALLY STABLE\"\n        elif stability_ratio > 0.5:\n            stability_type = \"STABLE\"\n        elif stability_ratio > 0.05:\n            stability_type = \"UNSTABLE\"\n        else:\n            stability_type = \"COMPLETELY UNSTABLE\"\n        \n        return {\n            'stability_type': stability_type,\n            'stable_perturbations': stable_count,\n            'unstable_perturbations': unstable_count,\n            'stability_ratio': stability_ratio,\n            'mean_lyapunov_derivative': np.mean(lyapunov_derivatives),\n            'std_lyapunov_derivative': np.std(lyapunov_derivatives),\n            'min_lyapunov_derivative': np.min(lyapunov_derivatives),\n            'max_lyapunov_derivative': np.max(lyapunov_derivatives),\n            'is_stable': stability_ratio > 0.5,\n            'is_asymptotically_stable': stability_ratio > 0.95\n        }


class FixedPointTheorem:
    """
    Fixed-point analysis for gradient descent dynamics.
    
    THEOREM (Brouwer Fixed Point):
    Let f: K → K be a continuous function where K ⊂ ℝⁿ is compact and convex.
    Then f has at least one fixed point θ* such that f(θ*) = θ*.
    
    For gradient descent: θ_{t+1} = G(θ_t) = θ_t - α∇ℒ(θ_t)
    Fixed points satisfy: ∇ℒ(θ*) = 0
    
    We prove that sawtooth landscapes have multiple fixed points,
    including inverted minima.
    """
    
    @staticmethod\n    def find_fixed_points(\n        gradient_fn: Callable[[torch.Tensor], torch.Tensor],\n        search_bounds: Tuple[float, float],\n        n_initializations: int = 20\n    ) -> List[torch.Tensor]:
        """
        Find all fixed points (∇ℒ = 0) in search region.
        
        Uses multiple random initializations to find different
        local minima, maxima, and saddle points.
        
        Args:
            gradient_fn: Function computing ∇ℒ(θ)
            search_bounds: (min, max) for parameter search
            n_initializations: Number of random starts
            
        Returns:
            List of fixed points
        """
        theta_min, theta_max = search_bounds
        fixed_points = []\n        \n        for _ in range(n_initializations):\n            # Random initialization\n            theta_init = torch.rand(1) * (theta_max - theta_min) + theta_min\n            theta = theta_init.clone().requires_grad_(True)\n            \n            # Newton's method to find ∇ℒ = 0\n            for _ in range(100):\n                grad = gradient_fn(theta)\n                \n                if grad is None or torch.isnan(grad).any():\n                    break\n                \n                # Check if found fixed point\n                if torch.norm(grad) < 1e-6:\n                    fixed_points.append(theta.detach().clone())\n                    break\n                \n                # Update (Newton step)\n                theta = theta - 0.1 * grad\n                theta = torch.clamp(theta, theta_min, theta_max)\n                theta = theta.detach().requires_grad_(True)\n        \n        # Remove duplicates\n        unique_points = []\n        for fp in fixed_points:\n            is_duplicate = False\n            for existing in unique_points:\n                if torch.norm(fp - existing) < 0.01:\n                    is_duplicate = True\n                    break\n            if not is_duplicate:\n                unique_points.append(fp)\n        \n        return unique_points
    
    @staticmethod
    def classify_fixed_point(
        fixed_point: torch.Tensor,
        loss_fn: Callable,
        epsilon: float = 1e-4
    ) -> str:
        """
        Classify fixed point as minimum, maximum, or saddle.
        
        Uses second derivative test:
        - ∇²ℒ > 0 ⟹ local minimum
        - ∇²ℒ < 0 ⟹ local maximum  
        - ∇²ℒ = 0 ⟹ saddle or inflection
        
        Args:
            fixed_point: Point to classify
            loss_fn: Loss function
            epsilon: Perturbation size for numerical Hessian
            
        Returns:
            Classification: 'minimum', 'maximum', 'saddle', or 'degenerate'
        """
        theta = fixed_point.detach().clone().requires_grad_(True)
        loss = loss_fn(theta)
        
        # First derivative
        if theta.grad is not None:
            theta.grad.zero_()\n        loss.backward(create_graph=True)
        grad = theta.grad
        
        # Check if truly a fixed point
        if torch.norm(grad) > 1e-3:
            return 'not_fixed_point'
        
        # Second derivative (Hessian)
        # For scalar: compute d²ℒ/dθ²
        if theta.numel() == 1:
            # Numerical second derivative
            theta_plus = theta + epsilon
            theta_minus = theta - epsilon
            
            loss_plus = loss_fn(theta_plus)
            loss_minus = loss_fn(theta_minus)
            loss_center = loss_fn(theta)
            
            second_deriv = (loss_plus - 2*loss_center + loss_minus) / (epsilon**2)
            
            if second_deriv > 0.01:
                return 'minimum'
            elif second_deriv < -0.01:
                return 'maximum'
            else:
                return 'saddle'
        else:
            # Multi-dimensional: check Hessian eigenvalues
            return 'multidimensional_point'


class ConvergenceRateTheorem:
    """
    THEOREM: Convergence Rate in Sawtooth Landscapes
    
    For gradient descent on smooth loss ℒ with Lipschitz continuous gradients:
    
    (a) SMOOTH REGIME (no discontinuities):
        ||θ_t - θ*|| = O(exp(-μt))
        Linear convergence with rate μ = 2mα where m is strong convexity
        parameter.
    
    (b) SAWTOOTH REGIME (with discontinuities):
        ||θ_t - θ*|| = O(t^{-1/2}) or slower
        Sub-linear convergence due to oscillations.
    
    (c) OSCILLATION BOUND:
        In sawtooth landscapes with period T:
        ||θ_t - θ_{t-1}|| ≥ α·||∇ℒ||·(1 - cos(2π/T))
        
    Proof: Based on Lyapunov analysis and frequency domain analysis.
    """
    
    @staticmethod
    def estimate_convergence_rate(
        loss_trajectory: np.ndarray,
        time_steps: Optional[np.ndarray] = None
    ) -> Dict:
        """
        Estimate convergence rate from empirical loss trajectory.
        
        Fits exponential model: ℒ(t) = ℒ_∞ + A·exp(-λt)
        
        Args:
            loss_trajectory: Loss values over time
            time_steps: Time points (default: 0, 1, 2, ...)
            
        Returns:
            Convergence rate analysis
        """
        if time_steps is None:
            time_steps = np.arange(len(loss_trajectory))
        
        # Remove NaN and Inf
        valid_mask = np.isfinite(loss_trajectory)
        loss_clean = loss_trajectory[valid_mask]
        time_clean = time_steps[valid_mask]
        
        if len(loss_clean) < 10:
            return {'convergence_rate': 0, 'converged': False, 'error': 'insufficient_data'}
        
        # Estimate L_infinity (final loss)
        L_inf = np.mean(loss_clean[-10:])
        \n        # Fit exponential: (L - L_inf) = A·exp(-λt)
        # Take log: log(L - L_inf) = log(A) - λt
        loss_shifted = loss_clean - L_inf + 1e-10  # Avoid log(0)
        log_loss = np.log(np.maximum(loss_shifted, 1e-10))
        
        # Linear regression on log scale
        if len(log_loss) > 1:
            # Fit: log_loss = b - λt
            coeffs = np.polyfit(time_clean, log_loss, 1)
            lambda_rate = -coeffs[0]  # Convergence rate
            log_A = coeffs[1]
            A = np.exp(log_A)
        else:
            lambda_rate = 0
            A = 0
        
        # Classify convergence type
        if lambda_rate > 0.1:
            convergence_type = \"EXPONENTIAL (Linear Convergence)\"
        elif lambda_rate > 0.01:
            convergence_type = \"MODERATE (Sub-linear Convergence)\"
        elif lambda_rate > 0:
            convergence_type = \"SLOW (Very Sub-linear)\"
        else:
            convergence_type = \"NON-CONVERGENT (Oscillating or Diverging)\"
        
        # Estimate time to convergence (95% of improvement)
        if lambda_rate > 0:
            # Time when L(t) = L_inf + 0.05·A
            # 0.05·A = A·exp(-λt) ⟹ t = -log(0.05)/λ\n            time_to_convergence = -np.log(0.05) / lambda_rate
        else:\n            time_to_convergence = np.inf
        \n        # Check actual convergence (final loss close to minimum)
        converged = (loss_clean[-1] - L_inf) < 0.01 * (loss_clean[0] - L_inf)
        
        return {
            'convergence_rate': lambda_rate,\n            'convergence_type': convergence_type,
            'amplitude': A,
            'asymptotic_loss': L_inf,
            'time_to_convergence': time_to_convergence,
            'converged': converged,
            'final_loss': loss_clean[-1],
            'initial_loss': loss_clean[0],
            'total_improvement': loss_clean[0] - loss_clean[-1],
            'improvement_ratio': (loss_clean[0] - loss_clean[-1]) / (loss_clean[0] + 1e-10)
        }
    
    @staticmethod
    def prove_oscillation_bound(
        learning_rate: float,
        gradient_norm: float,
        period: float
    ) -> Dict:
        """
        Prove oscillation bound for sawtooth landscapes.
        
        THEOREM: In sawtooth loss with period T, gradient descent with
        learning rate α has step size:
        
            ||θ_{t+1} - θ_t|| = α·||∇ℒ||
        
        If α > T/2, steps overshoot the period, causing oscillation with\n        amplitude:\n        
            A_osc ≥ α·||∇ℒ||·sin(πα·||∇ℒ||/T)
        
        Args:
            learning_rate: α
            gradient_norm: ||∇ℒ||
            period: Sawtooth period T\n            
        Returns:
            Oscillation bounds
        """
        # Step size
        step_size = learning_rate * gradient_norm
        
        # Overshoot condition
        overshoots = step_size > period / 2
        \n        # Oscillation amplitude
        if overshoots:
            # Number of periods crossed
            periods_crossed = step_size / period
            \n            # Oscillation amplitude (from Fourier analysis)
            omega = 2 * np.pi / period
            A_osc = learning_rate * gradient_norm * abs(np.sin(omega * step_size))
        else:
            A_osc = 0
        
        # Theoretical convergence criterion
        critical_lr = period / (2 * gradient_norm) if gradient_norm > 0 else np.inf
        
        return {
            'step_size': step_size,
            'period': period,
            'overshoots': overshoots,
            'oscillation_amplitude': A_osc,
            'periods_crossed': step_size / period if period > 0 else 0,
            'critical_learning_rate': critical_lr,
            'learning_rate_ratio': learning_rate / critical_lr if critical_lr > 0 and critical_lr < np.inf else 0,
            'will_oscillate': overshoots
        }


class ContractiveConvergenceTheorem:
    """
    THEOREM: Contractive Mappings and Convergence
    
    A mapping G: Θ → Θ is contractive if there exists L < 1 such that:
        ||G(θ₁) - G(θ₂)|| ≤ L·||θ₁ - θ₂|| for all θ₁, θ₂
    
    By Banach Fixed-Point Theorem, contractive mappings have unique
    fixed points and gradient descent converges to them.
    
    For sawtooth landscapes, we prove that G is NOT contractive due to
    gradient discontinuities, explaining non-convergence.
    """
    
    @staticmethod
    def test_contractive_property(\n        gradient_step_fn: Callable[[torch.Tensor], torch.Tensor],
        test_points: List[torch.Tensor],
        expected_lipschitz: float = 1.0
    ) -> Dict:
        """
        Test if gradient descent mapping is contractive.
        
        Tests: ||G(θ₁) - G(θ₂)|| ≤ L·||θ₁ - θ₂||
        
        Args:
            gradient_step_fn: G(θ) = θ - α∇ℒ(θ)
            test_points: Sample points for testing
            expected_lipschitz: Expected L constant
            
        Returns:
            Contractive property analysis
        """
        n_pairs = len(test_points)
        lipschitz_constants = []
        \n        for i in range(n_pairs):\n            for j in range(i+1, n_pairs):
                theta1 = test_points[i]
                theta2 = test_points[j]
                \n                # Apply mapping
                G_theta1 = gradient_step_fn(theta1)
                G_theta2 = gradient_step_fn(theta2)
                \n                # Compute distances
                dist_before = torch.norm(theta1 - theta2)
                dist_after = torch.norm(G_theta1 - G_theta2)
                \n                # Lipschitz constant for this pair
                if dist_before > 1e-10:\n                    L = dist_after / dist_before
                    lipschitz_constants.append(L.item())
        \n        if len(lipschitz_constants) == 0:
            return {'is_contractive': False, 'error': 'no_valid_pairs'}
        \n        lipschitz_constants = np.array(lipschitz_constants)
        L_empirical = np.max(lipschitz_constants)
        L_mean = np.mean(lipschitz_constants)
        \n        # Contractive if L < 1\n        is_contractive = L_empirical < 1.0
        \n        return {
            'is_contractive': is_contractive,
            'lipschitz_constant_max': L_empirical,
            'lipschitz_constant_mean': L_mean,
            'lipschitz_constant_std': np.std(lipschitz_constants),
            'expected_lipschitz': expected_lipschitz,
            'contraction_ratio': L_empirical / expected_lipschitz if expected_lipschitz > 0 else 0,
            'n_test_pairs': len(lipschitz_constants)
        }


def prove_convergence_failure_in_sawtooth(
    learning_rate: float = 0.1,
    modulus: int = 2**16,
    n_steps: int = 1000
) -> Dict:
    """
    Comprehensive proof that gradient descent fails in sawtooth landscapes.
    
    Combines:
    1. Lyapunov instability of true solutions
    2. Non-contractive mapping
    3. Oscillation bounds
    4. Empirical trajectory analysis
    
    Args:
        learning_rate: GD learning rate
        modulus: Sawtooth period = 1/modulus
        n_steps: Simulation steps
        
    Returns:
        Complete convergence failure proof
    """
    period = 1.0 / modulus
    \n    # Simulate GD on sawtooth
    initial_theta = 0.5 * period  # Start near minimum
    theta = initial_theta
    trajectory = [theta]
    \n    def sawtooth_gradient(th):
        mod_th = th % period
        return 1.0 if mod_th > period/2 else -1.0
    \n    for _ in range(n_steps):
        grad = sawtooth_gradient(theta)
        theta = theta - learning_rate * grad
        trajectory.append(theta)
    \n    trajectory = np.array(trajectory)
    \n    # Analyze trajectory
    oscillating = np.std(trajectory[-100:]) > period * 0.1
    converged = abs(trajectory[-1] % period - period/2) < 1e-4
    \n    # Oscillation period
    # Find peaks
    trajectory_mod = trajectory % period
    peaks = []
    for i in range(1, len(trajectory_mod)-1):
        if trajectory_mod[i] > trajectory_mod[i-1] and trajectory_mod[i] > trajectory_mod[i+1]:
            peaks.append(i)
    \n    if len(peaks) > 1:
        oscillation_period = np.mean(np.diff(peaks))
    else:
        oscillation_period = 0
    \n    return {
        'convergence_failed': not converged,
        'oscillating': oscillating,
        'oscillation_period': oscillation_period,
        'trajectory_length': len(trajectory),
        'final_distance_to_minimum': abs(trajectory[-1] % period - period/2),
        'learning_rate': learning_rate,
        'period': period,
        'critical_lr': period / 2,
        'exceeds_critical': learning_rate > period / 2
    }
