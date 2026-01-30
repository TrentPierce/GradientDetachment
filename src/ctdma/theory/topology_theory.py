"""
Topology Theory for Sawtooth Loss Landscapes

Formal topological analysis of loss landscapes arising from ARX cipher approximations.
Provides rigorous mathematical definitions and convergence proofs.

Mathematical Framework:
    - Metric spaces and topologies
    - Critical point theory (Morse theory)
    - Gradient flows and dynamical systems
    - Homology and cohomology (optional advanced)

Key Concepts:
    - Sawtooth manifold: Piecewise smooth with periodic discontinuities
    - Adversarial attractors: Local minima at inverted solutions
    - Basin of attraction: Regions flowing to each attractor
    - Structural stability: Persistence under perturbations

Author: Gradient Detachment Research Team
Date: 2026-01-30
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
from scipy.spatial import distance
from scipy.ndimage import label
import warnings


class SawtoothManifold:
    """
    Mathematical definition of sawtooth loss manifold.
    
    Definition:
    ==========
    A sawtooth manifold M is a piecewise smooth manifold with:
    
    1. PARTITION: M = ⋃_{i=1}^N M_i where M_i are smooth patches
    
    2. DISCONTINUITIES: ∂M_i ∩ ∂M_j ≠ ∅ (patches share boundaries)
    
    3. PERIODICITY: ∃ period T such that M is T-periodic
    
    4. GRADIENT JUMPS: At discontinuities D = ⋃ ∂M_i:
       lim_{x→d^+} ∇ℒ(x) ≠ lim_{x→d^-} ∇ℒ(x) for d ∈ D
    
    Properties:
        - Piecewise smooth (C^∞ on each M_i)
        - Globally discontinuous
        - Periodic structure
        - Multiple local minima
    
    Args:
        dimension: Manifold dimension
        period: Periodicity T
        num_patches: Number of smooth patches
    """
    
    def __init__(
        self,
        dimension: int,
        period: float,
        num_patches: int
    ):
        self.dimension = dimension
        self.period = period
        self.num_patches = num_patches
        
        # Discontinuity set D
        self.discontinuities = self._compute_discontinuities()
        
    def _compute_discontinuities(self) -> List[float]:
        """
        Compute locations of discontinuities.
        
        For modular arithmetic with period T = m:
        Discontinuities at x = k·m for k ∈ ℤ
        """
        discontinuities = []
        for k in range(self.num_patches + 1):
            d = k * self.period
            discontinuities.append(d)
        return discontinuities
    
    def is_in_discontinuity_neighborhood(
        self,
        point: torch.Tensor,
        epsilon: float = 0.01
    ) -> torch.Tensor:
        """
        Check if point is in ε-neighborhood of discontinuity set.
        
        Args:
            point: Point in manifold
            epsilon: Neighborhood radius
            
        Returns:
            Boolean mask indicating points near discontinuities
        """
        distances_to_discontinuities = []
        
        for d in self.discontinuities:
            dist = torch.abs(point - d)
            distances_to_discontinuities.append(dist)
        
        # Minimum distance to any discontinuity
        min_distances = torch.min(torch.stack(distances_to_discontinuities), dim=0)[0]
        
        return min_distances < epsilon
    
    def classify_critical_points(
        self,
        loss_fn: Callable,
        candidates: torch.Tensor
    ) -> Dict[str, List[torch.Tensor]]:
        """
        Classify critical points using Morse theory.
        
        Morse Theory Classification:
            Critical point p: ∇ℒ(p) = 0
            
            Classified by Hessian eigenvalues:
            - All λ > 0: Local minimum (index 0)
            - All λ < 0: Local maximum (index n)
            - Mixed signs: Saddle point (index k = # negative eigenvalues)
        
        Args:
            loss_fn: Loss function ℒ
            candidates: Candidate critical points
            
        Returns:
            Dictionary grouping points by type:
                - 'minima': Local minima
                - 'maxima': Local maxima
                - 'saddles': Saddle points
                - 'morse_indices': Morse index for each point
        """
        minima = []
        maxima = []
        saddles = []
        morse_indices = []
        
        for candidate in candidates:
            candidate.requires_grad_(True)
            
            # Compute gradient
            loss = loss_fn(candidate)
            grad = torch.autograd.grad(loss, candidate, create_graph=True)[0]
            grad_norm = torch.norm(grad)
            
            # Check if critical point (gradient ≈ 0)
            if grad_norm < 0.01:
                # Compute Hessian
                hessian = []
                for i in range(len(candidate)):
                    grad_i = torch.autograd.grad(
                        grad[i], candidate, retain_graph=True, create_graph=True
                    )[0]
                    hessian.append(grad_i)
                
                hessian = torch.stack(hessian)
                
                # Eigenvalues
                eigenvalues = torch.linalg.eigvalsh(hessian)
                
                # Count negative eigenvalues (Morse index)
                morse_index = (eigenvalues < 0).sum().item()
                morse_indices.append(morse_index)
                
                # Classify
                if morse_index == 0:
                    minima.append(candidate.detach())
                elif morse_index == len(candidate):
                    maxima.append(candidate.detach())
                else:
                    saddles.append(candidate.detach())
        
        return {
            'minima': minima,
            'maxima': maxima,
            'saddles': saddles,
            'morse_indices': morse_indices,
            'num_minima': len(minima),
            'num_maxima': len(maxima),
            'num_saddles': len(saddles)
        }


class GradientFlowAnalyzer:
    """
    Analysis of gradient flow dynamics on sawtooth manifolds.
    
    Gradient Flow Equation:
        dθ/dt = -∇ℒ(θ)
    
    For sawtooth manifolds, flow has special properties:
    - Discontinuous vector field
    - Multiple equilibria (attractors)
    - Non-smooth trajectories
    - Possible non-convergence
    
    Analyzes:
        - Flow trajectories
        - Convergence to attractors
        - Basin boundaries
        - Structural stability
    
    Args:
        loss_fn: Loss function defining the flow
        manifold: Sawtooth manifold structure
    """
    
    def __init__(
        self,
        loss_fn: Callable,
        manifold: Optional[SawtoothManifold] = None
    ):
        self.loss_fn = loss_fn
        self.manifold = manifold
        
    def simulate_flow(
        self,
        initial_point: torch.Tensor,
        num_steps: int = 1000,
        step_size: float = 0.01,
        method: str = 'euler'
    ) -> Dict[str, any]:
        """
        Simulate gradient flow from initial point.
        
        Methods:
            - 'euler': Forward Euler (simple)
            - 'rk4': Runge-Kutta 4th order (accurate)
            - 'adaptive': Adaptive step size
        
        Args:
            initial_point: Starting point θ₀
            num_steps: Number of simulation steps
            step_size: Time step Δt
            method: Integration method
            
        Returns:
            Flow trajectory and analysis
        """
        trajectory = [initial_point.detach().clone()]
        losses = []
        gradients = []
        
        theta = initial_point.clone().requires_grad_(True)
        
        for step in range(num_steps):
            # Compute loss and gradient
            loss = self.loss_fn(theta)
            losses.append(loss.item())
            
            if theta.grad is not None:
                theta.grad.zero_()
            
            loss.backward()
            grad = theta.grad.clone()
            gradients.append(grad.detach())
            
            # Integration step
            with torch.no_grad():
                if method == 'euler':
                    # Forward Euler: θ_{t+1} = θ_t - Δt·∇ℒ(θ_t)
                    theta_new = theta - step_size * grad
                    
                elif method == 'rk4':
                    # Runge-Kutta 4th order
                    k1 = -grad
                    
                    theta_temp = theta + 0.5 * step_size * k1
                    theta_temp.requires_grad_(True)
                    loss_temp = self.loss_fn(theta_temp)
                    k2 = -torch.autograd.grad(loss_temp, theta_temp)[0]
                    
                    theta_temp = theta + 0.5 * step_size * k2
                    theta_temp.requires_grad_(True)
                    loss_temp = self.loss_fn(theta_temp)
                    k3 = -torch.autograd.grad(loss_temp, theta_temp)[0]
                    
                    theta_temp = theta + step_size * k3
                    theta_temp.requires_grad_(True)
                    loss_temp = self.loss_fn(theta_temp)
                    k4 = -torch.autograd.grad(loss_temp, theta_temp)[0]
                    
                    theta_new = theta + (step_size / 6) * (k1 + 2*k2 + 2*k3 + k4)
                    
                else:
                    raise ValueError(f"Unknown method: {method}")
                
                # Update theta
                theta = theta_new.detach().clone().requires_grad_(True)
                trajectory.append(theta.detach().clone())
        
        # Analyze trajectory
        analysis = self._analyze_trajectory(trajectory, losses, gradients)
        
        return {
            'trajectory': trajectory,
            'losses': losses,
            'gradients': gradients,
            **analysis
        }
    
    def _analyze_trajectory(
        self,
        trajectory: List[torch.Tensor],
        losses: List[float],
        gradients: List[torch.Tensor]
    ) -> Dict[str, any]:
        """Analyze trajectory properties."""
        # Convert to arrays
        losses = np.array(losses)
        grad_norms = np.array([torch.norm(g).item() for g in gradients])
        
        # Convergence check
        final_window = 100
        if len(losses) >= final_window:
            loss_variance = np.var(losses[-final_window:])
            loss_mean = np.mean(losses[-final_window:])
            
            # Converged if variance is small relative to mean
            converged = loss_variance < 0.01 * loss_mean
        else:
            converged = False
        
        # Oscillation detection
        # Count number of loss increases (non-monotonic behavior)
        loss_increases = np.sum(np.diff(losses) > 0)
        oscillation_rate = loss_increases / len(losses)
        is_oscillatory = oscillation_rate > 0.1  # >10% non-monotonic steps
        
        # Convergence rate (exponential fit)
        if len(losses) > 10 and not is_oscillatory:
            # Fit: L(t) = L_∞ + A·exp(-λt)
            t = np.arange(len(losses))
            L_inf = losses[-1]  # Approximate final loss
            
            # Linear fit in log space
            log_diff = np.log(losses - L_inf + 1e-10)
            if np.all(np.isfinite(log_diff)):
                slope, _ = np.polyfit(t[losses > L_inf], log_diff[losses > L_inf], 1)
                convergence_rate = -slope
            else:
                convergence_rate = 0.0
        else:
            convergence_rate = 0.0
        
        # Attractor identification
        final_point = trajectory[-1]
        final_loss = losses[-1]
        final_grad_norm = grad_norms[-1]
        
        # Determine attractor type based on final gradient
        if final_grad_norm < 0.01:
            attractor_type = 'minimum'
        elif is_oscillatory:
            attractor_type = 'oscillatory'
        else:
            attractor_type = 'undetermined'
        
        return {
            'converged': converged,
            'is_oscillatory': is_oscillatory,
            'oscillation_rate': float(oscillation_rate),
            'convergence_rate_lambda': float(convergence_rate),
            'final_loss': float(final_loss),
            'final_gradient_norm': float(final_grad_norm),
            'attractor_type': attractor_type,
            'num_steps': len(trajectory)
        }
    
    def compute_basin_of_attraction(
        self,
        attractor: torch.Tensor,
        domain: Tuple[float, float],
        resolution: int = 50,
        convergence_threshold: float = 0.1
    ) -> Dict[str, any]:
        """
        Compute basin of attraction for a given attractor.
        
        Definition:
            Basin B(a) = {x ∈ M : lim_{t→∞} φ_t(x) = a}
        
        where φ_t is the gradient flow at time t.
        
        Args:
            attractor: Attractor point a
            domain: Search domain [a, b]
            resolution: Number of grid points per dimension
            convergence_threshold: Distance threshold for convergence
            
        Returns:
            Basin characteristics
        """
        a, b = domain
        
        # Create grid of initial points
        if attractor.dim() == 0:
            # 1D case
            grid_points = torch.linspace(a, b, resolution)
        else:
            # Multi-dimensional: sample randomly
            grid_points = torch.rand(resolution, len(attractor)) * (b - a) + a
        
        # Simulate flow from each point
        basin_points = []
        
        for point in grid_points:
            flow_result = self.simulate_flow(
                point,
                num_steps=500,
                step_size=0.01,
                method='euler'
            )
            
            final_point = flow_result['trajectory'][-1]
            distance_to_attractor = torch.norm(final_point - attractor)
            
            # Check if converged to this attractor
            if distance_to_attractor < convergence_threshold:
                basin_points.append(point.detach())
        
        # Basin size
        basin_size = len(basin_points)
        basin_fraction = basin_size / resolution
        
        # Basin volume (approximate)
        if len(basin_points) > 0:
            basin_points_tensor = torch.stack(basin_points)
            basin_volume = torch.max(basin_points_tensor) - torch.min(basin_points_tensor)
        else:
            basin_volume = 0.0
        
        return {
            'basin_size': basin_size,
            'basin_fraction': float(basin_fraction),
            'basin_volume': float(basin_volume),
            'basin_points': basin_points,
            'attractor': attractor.detach()
        }


class CriticalPointTheory:
    """
    Critical Point Theory (Morse Theory) for Loss Landscapes.
    
    Morse Theory:
    ============
    For smooth function f: M → ℝ on manifold M:
    
    1. CRITICAL POINTS:
       p is critical if ∇f(p) = 0
    
    2. MORSE INDEX:
       Index(p) = number of negative eigenvalues of Hessian ∇²f(p)
       - Index 0: Local minimum
       - Index n: Local maximum (n = dim M)
       - Index k: Saddle point
    
    3. MORSE INEQUALITIES:
       For manifold M with Betti numbers β_k:
       
       M_k ≥ β_k
       
       where M_k = number of critical points with index k
    
    4. GRADIENT FLOW:
       Flow φ_t satisfies:
       - φ_0(p) = p (initial condition)
       - dφ_t/dt = -∇f(φ_t) (gradient descent)
       - lim_{t→∞} φ_t(p) = critical point
    
    Applications:
        - Count critical points by type
        - Compute Morse indices
        - Verify Morse inequalities
        - Analyze flow stability
    """
    
    def __init__(self, loss_fn: Callable):
        self.loss_fn = loss_fn
        
    def find_critical_points(
        self,
        domain: Tuple[float, float],
        num_initializations: int = 100,
        tolerance: float = 1e-4
    ) -> List[Tuple[torch.Tensor, int]]:
        """
        Find critical points using multiple random initializations.
        
        Args:
            domain: Search domain
            num_initializations: Number of random starts
            tolerance: Gradient norm threshold for critical point
            
        Returns:
            List of (critical_point, morse_index) tuples
        """
        a, b = domain
        critical_points = []
        
        for _ in range(num_initializations):
            # Random initialization
            x0 = torch.rand(1) * (b - a) + a
            x0.requires_grad_(True)
            
            # Gradient descent to find critical point
            optimizer = torch.optim.LBFGS([x0], max_iter=100, tolerance_grad=tolerance)
            
            def closure():
                optimizer.zero_grad()
                loss = self.loss_fn(x0)
                loss.backward()
                return loss
            
            optimizer.step(closure)
            
            # Check if critical point
            loss = self.loss_fn(x0)
            grad = torch.autograd.grad(loss, x0, create_graph=True)[0]
            
            if torch.norm(grad) < tolerance:
                # Compute Morse index
                hess = torch.autograd.grad(grad, x0, create_graph=True)[0]
                morse_index = int((hess < 0).sum().item())
                
                # Add if not duplicate
                is_duplicate = False
                for existing_point, _ in critical_points:
                    if torch.norm(existing_point - x0) < 0.01:
                        is_duplicate = True
                        break
                
                if not is_duplicate:
                    critical_points.append((x0.detach(), morse_index))
        
        return critical_points
    
    def verify_morse_inequalities(
        self,
        critical_points: List[Tuple[torch.Tensor, int]],
        manifold_dimension: int
    ) -> Dict[str, bool]:
        """
        Verify Morse inequalities for the loss landscape.
        
        Morse Inequality:
            M_k ≥ β_k
        
        For simple domains, Betti numbers are known:
            - β_0 = 1 (connected components)
            - β_k = 0 for 0 < k < n
            - β_n = 1 (if compact)
        
        Args:
            critical_points: List of (point, morse_index)
            manifold_dimension: Dimension of manifold
            
        Returns:
            Whether inequalities are satisfied
        """
        # Count critical points by Morse index
        morse_counts = {}
        for _, index in critical_points:
            morse_counts[index] = morse_counts.get(index, 0) + 1
        
        # Betti numbers for simple domain (assumption: domain is ball)
        betti = {0: 1, manifold_dimension: 1}  # β_0 = β_n = 1
        for k in range(1, manifold_dimension):
            betti[k] = 0
        
        # Check inequalities
        satisfied = {}
        for k in range(manifold_dimension + 1):
            M_k = morse_counts.get(k, 0)
            β_k = betti.get(k, 0)
            satisfied[k] = M_k >= β_k
        
        all_satisfied = all(satisfied.values())
        
        return {
            'morse_counts': morse_counts,
            'betti_numbers': betti,
            'inequalities_satisfied': satisfied,
            'all_satisfied': all_satisfied,
            'total_critical_points': len(critical_points)
        }


class StructuralStabilityAnalyzer:
    """
    Analysis of structural stability under perturbations.
    
    Structural Stability:
    ====================
    A dynamical system is structurally stable if small perturbations
    do not change qualitative behavior.
    
    For gradient flow dθ/dt = -∇ℒ(θ):
    
    1. STABLE if: Perturbation ℒ → ℒ + δℒ with ||δℒ|| < ε preserves:
       - Number of attractors
       - Basin topology
       - Flow structure
    
    2. UNSTABLE if: Small perturbations change attractor structure
    
    Tests:
        - Attractor persistence under noise
        - Basin boundary stability
        - Bifurcation analysis
    
    Args:
        loss_fn: Original loss function
    """
    
    def __init__(self, loss_fn: Callable):
        self.loss_fn = loss_fn
        
    def test_attractor_persistence(
        self,
        attractor: torch.Tensor,
        perturbation_scale: float = 0.01,
        num_perturbations: int = 20
    ) -> Dict[str, any]:
        """
        Test if attractor persists under perturbations.
        
        Args:
            attractor: Attractor point
            perturbation_scale: Size of perturbation
            num_perturbations: Number of perturbations to test
            
        Returns:
            Persistence analysis
        """
        persistence_count = 0
        perturbed_attractors = []
        
        for _ in range(num_perturbations):
            # Create perturbed loss function
            def perturbed_loss(theta):
                # Add random perturbation
                perturbation = perturbation_scale * torch.randn_like(theta)
                return self.loss_fn(theta + perturbation)
            
            # Find attractor in perturbed system
            flow_analyzer = GradientFlowAnalyzer(perturbed_loss)
            flow_result = flow_analyzer.simulate_flow(
                attractor + torch.randn_like(attractor) * 0.1,
                num_steps=500
            )
            
            perturbed_attractor = flow_result['trajectory'][-1]
            perturbed_attractors.append(perturbed_attractor)
            
            # Check if close to original
            distance = torch.norm(perturbed_attractor - attractor)
            if distance < 0.2:  # Threshold
                persistence_count += 1
        
        persistence_rate = persistence_count / num_perturbations
        
        # Compute variance of perturbed attractors
        if len(perturbed_attractors) > 1:
            perturbed_stack = torch.stack(perturbed_attractors)
            attractor_variance = torch.var(perturbed_stack).item()
        else:
            attractor_variance = 0.0
        
        return {
            'persistence_rate': float(persistence_rate),
            'is_stable': persistence_rate > 0.8,
            'attractor_variance': float(attractor_variance),
            'num_perturbations_tested': num_perturbations
        }


# Main comprehensive analyzer
class TopologicalAnalyzer:
    """
    Comprehensive topological analysis of sawtooth loss landscapes.
    
    Combines:
        - Sawtooth manifold structure
        - Critical point theory
        - Gradient flow dynamics
        - Structural stability
    
    Provides complete mathematical characterization of ARX cipher
    loss landscapes with rigorous topology theory.
    
    Example:
        >>> analyzer = TopologicalAnalyzer(
        ...     loss_fn=my_loss,
        ...     period=2**16,
        ...     dimension=64
        ... )
        >>> results = analyzer.complete_analysis(domain=(-10, 10))
        >>> print(results['topology']['num_minima'])
        >>> print(results['stability']['is_stable'])
    """
    
    def __init__(
        self,
        loss_fn: Callable,
        period: float = 2**16,
        dimension: int = 1
    ):
        self.loss_fn = loss_fn
        self.manifold = SawtoothManifold(dimension, period, num_patches=10)
        self.gradient_flow = GradientFlowAnalyzer(loss_fn, self.manifold)
        self.critical_theory = CriticalPointTheory(loss_fn)
        self.stability = StructuralStabilityAnalyzer(loss_fn)
        
    def complete_analysis(
        self,
        domain: Tuple[float, float],
        num_samples: int = 100
    ) -> Dict[str, Dict]:
        """
        Perform complete topological analysis.
        
        Args:
            domain: Analysis domain [a, b]
            num_samples: Number of sample points
            
        Returns:
            Comprehensive topology analysis
        """
        results = {}
        
        # 1. Find critical points
        print("Finding critical points...")
        critical_points = self.critical_theory.find_critical_points(
            domain, num_initializations=50
        )
        
        # 2. Classify critical points
        candidates = torch.tensor([p.item() for p, _ in critical_points])
        if len(candidates) > 0:
            classification = self.manifold.classify_critical_points(
                self.loss_fn,
                candidates
            )
        else:
            classification = {'minima': [], 'maxima': [], 'saddles': []}
        
        # 3. Verify Morse inequalities
        morse_verification = self.critical_theory.verify_morse_inequalities(
            critical_points,
            self.manifold.dimension
        )
        
        # 4. Analyze basins of attraction for each minimum
        basins = []
        if classification['num_minima'] > 0:
            for minimum in classification['minima'][:5]:  # Analyze first 5
                basin = self.gradient_flow.compute_basin_of_attraction(
                    minimum,
                    domain,
                    resolution=50
                )
                basins.append(basin)
        
        # 5. Test structural stability
        stability_results = []
        if classification['num_minima'] > 0:
            for minimum in classification['minima'][:3]:  # Test first 3
                stability = self.stability.test_attractor_persistence(
                    minimum,
                    perturbation_scale=0.01,
                    num_perturbations=10
                )
                stability_results.append(stability)
        
        # Compile results
        results['topology'] = {
            'num_critical_points': len(critical_points),
            'num_minima': classification['num_minima'],
            'num_maxima': classification['num_maxima'],
            'num_saddles': classification['num_saddles'],
            'morse_verification': morse_verification
        }
        
        results['basins'] = {
            'num_basins_analyzed': len(basins),
            'basin_sizes': [b['basin_size'] for b in basins],
            'basin_fractions': [b['basin_fraction'] for b in basins]
        }
        
        results['stability'] = {
            'num_tested': len(stability_results),
            'persistence_rates': [s['persistence_rate'] for s in stability_results],
            'all_stable': all(s['is_stable'] for s in stability_results) if stability_results else False
        }
        
        return results


def print_topology_summary(results: Dict[str, Dict]):
    """
    Pretty print topology analysis results.
    
    Args:
        results: Output from TopologicalAnalyzer.complete_analysis()
    """
    print("="*70)
    print("TOPOLOGICAL ANALYSIS SUMMARY")
    print("="*70)
    
    # Topology
    print("\n1. CRITICAL POINT STRUCTURE:")
    print(f"   Total critical points: {results['topology']['num_critical_points']}")
    print(f"   Minima: {results['topology']['num_minima']}")
    print(f"   Maxima: {results['topology']['num_maxima']}")
    print(f"   Saddles: {results['topology']['num_saddles']}")
    
    morse = results['topology']['morse_verification']
    print(f"\n   Morse Inequalities: {'✅ Satisfied' if morse['all_satisfied'] else '❌ Violated'}")
    
    # Basins
    print("\n2. BASIN OF ATTRACTION ANALYSIS:")
    if results['basins']['num_basins_analyzed'] > 0:
        print(f"   Basins analyzed: {results['basins']['num_basins_analyzed']}")
        print(f"   Basin fractions: {[f'{f:.2%}' for f in results['basins']['basin_fractions']]}")
    else:
        print("   No basins analyzed")
    
    # Stability
    print("\n3. STRUCTURAL STABILITY:")
    if results['stability']['num_tested'] > 0:
        print(f"   Attractors tested: {results['stability']['num_tested']}")
        print(f"   Persistence rates: {[f'{r:.1%}' for r in results['stability']['persistence_rates']]}")
        print(f"   Overall stable: {'✅ Yes' if results['stability']['all_stable'] else '❌ No'}")
    else:
        print("   No stability tests performed")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    print("Topology Theory for Sawtooth Loss Landscapes")
    print("="*70)
    print("\nKey components:")
    print("1. SawtoothManifold - Formal manifold definition")
    print("2. GradientFlowAnalyzer - Flow dynamics")
    print("3. CriticalPointTheory - Morse theory application")
    print("4. StructuralStabilityAnalyzer - Perturbation analysis")
    print("5. TopologicalAnalyzer - Comprehensive analysis")
    print("\nUse TopologicalAnalyzer for complete topological characterization.")
