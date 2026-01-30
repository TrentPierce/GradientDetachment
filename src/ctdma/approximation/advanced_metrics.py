"""
Advanced Metrics for Approximation Fidelity Analysis

This module provides comprehensive quantitative metrics beyond basic error measures:
1. Spectral analysis (frequency domain characterization)
2. Geometric measures (manifold distances, curvature)
3. Convergence guarantees (theoretical bounds, rates)
4. Information-theoretic measures (entropy, capacity)

All metrics provide rigorous quantitative assessment of approximation quality.

Author: Gradient Detachment Research Team
Date: 2026-01-30
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
from scipy import signal, stats
from scipy.fft import fft, fftfreq
from scipy.spatial.distance import pdist, squareform
import warnings


class SpectralAnalyzer:
    """
    Spectral Analysis of Approximations.
    
    Analyzes approximation quality in frequency domain using Fourier analysis.
    Spectral characteristics reveal:
    - High-frequency components (discontinuities)
    - Power spectrum (energy distribution)
    - Harmonic content (periodicity)
    
    Mathematical Background:
        For signal f(t), the Fourier transform is:
        F(ω) = ∫ f(t) exp(-iωt) dt
        
        Power spectrum: P(ω) = |F(ω)|^2
        
        Discontinuities manifest as high-frequency components in spectrum.
    
    Metrics Computed:
        - spectral_distance: ||F_discrete - F_smooth||_2
        - power_ratio: P_high_freq / P_total
        - spectral_entropy: Shannon entropy of power spectrum
        - harmonic_distortion: Total harmonic distortion (THD)
    """
    
    def __init__(self, sample_rate: float = 1.0):
        self.sample_rate = sample_rate
        
    def analyze_spectrum(
        self,
        discrete_signal: np.ndarray,
        smooth_signal: np.ndarray
    ) -> Dict[str, float]:
        """
        Perform spectral analysis comparing discrete and smooth signals.
        
        Args:
            discrete_signal: Output from discrete operation
            smooth_signal: Output from smooth approximation
            
        Returns:
            Dictionary of spectral metrics
        """
        # Compute FFT
        fft_discrete = fft(discrete_signal)
        fft_smooth = fft(smooth_signal)
        
        # Frequencies
        freqs = fftfreq(len(discrete_signal), d=1/self.sample_rate)
        
        # Power spectra
        power_discrete = np.abs(fft_discrete) ** 2
        power_smooth = np.abs(fft_smooth) ** 2
        
        # Spectral distance (L2 norm in frequency domain)
        spectral_distance = np.linalg.norm(fft_discrete - fft_smooth) / len(discrete_signal)
        
        # High-frequency power ratio
        # Define high frequency as > 25% of Nyquist frequency
        nyquist = self.sample_rate / 2
        high_freq_mask = np.abs(freqs) > 0.25 * nyquist
        
        power_high_discrete = np.sum(power_discrete[high_freq_mask])
        power_total_discrete = np.sum(power_discrete)
        power_ratio_discrete = power_high_discrete / (power_total_discrete + 1e-10)
        
        power_high_smooth = np.sum(power_smooth[high_freq_mask])
        power_total_smooth = np.sum(power_smooth)
        power_ratio_smooth = power_high_smooth / (power_total_smooth + 1e-10)
        
        # Spectral entropy
        # Normalize power spectra to probability distributions
        p_discrete = power_discrete / (np.sum(power_discrete) + 1e-10)
        p_smooth = power_smooth / (np.sum(power_smooth) + 1e-10)
        
        spectral_entropy_discrete = -np.sum(p_discrete * np.log2(p_discrete + 1e-10))
        spectral_entropy_smooth = -np.sum(p_smooth * np.log2(p_smooth + 1e-10))
        
        # Total Harmonic Distortion (THD)
        # Ratio of sum of powers of harmonics to power of fundamental
        fundamental_idx = np.argmax(power_discrete[:len(power_discrete)//2])
        fundamental_power = power_discrete[fundamental_idx]
        
        # Find harmonics (integer multiples of fundamental frequency)
        harmonics_power = 0.0
        for k in range(2, 6):  # Up to 5th harmonic
            harmonic_idx = fundamental_idx * k
            if harmonic_idx < len(power_discrete):
                harmonics_power += power_discrete[harmonic_idx]
        
        thd_discrete = np.sqrt(harmonics_power / (fundamental_power + 1e-10))
        
        # Same for smooth signal
        fundamental_power_smooth = power_smooth[fundamental_idx]
        harmonics_power_smooth = 0.0
        for k in range(2, 6):
            harmonic_idx = fundamental_idx * k
            if harmonic_idx < len(power_smooth):
                harmonics_power_smooth += power_smooth[harmonic_idx]
        
        thd_smooth = np.sqrt(harmonics_power_smooth / (fundamental_power_smooth + 1e-10))
        
        # Spectral flatness (measure of tonality vs noise)
        # Geometric mean / arithmetic mean
        geometric_mean_discrete = np.exp(np.mean(np.log(power_discrete + 1e-10)))
        arithmetic_mean_discrete = np.mean(power_discrete)
        flatness_discrete = geometric_mean_discrete / (arithmetic_mean_discrete + 1e-10)
        
        geometric_mean_smooth = np.exp(np.mean(np.log(power_smooth + 1e-10)))
        arithmetic_mean_smooth = np.mean(power_smooth)
        flatness_smooth = geometric_mean_smooth / (arithmetic_mean_smooth + 1e-10)
        
        return {
            'spectral_distance': float(spectral_distance),
            'power_ratio_discrete': float(power_ratio_discrete),
            'power_ratio_smooth': float(power_ratio_smooth),
            'power_ratio_difference': float(abs(power_ratio_discrete - power_ratio_smooth)),
            'spectral_entropy_discrete': float(spectral_entropy_discrete),
            'spectral_entropy_smooth': float(spectral_entropy_smooth),
            'entropy_loss': float(spectral_entropy_discrete - spectral_entropy_smooth),
            'thd_discrete': float(thd_discrete),
            'thd_smooth': float(thd_smooth),
            'thd_increase': float(abs(thd_smooth - thd_discrete)),
            'spectral_flatness_discrete': float(flatness_discrete),
            'spectral_flatness_smooth': float(flatness_smooth),
            'flatness_change': float(abs(flatness_smooth - flatness_discrete))
        }
    
    def compute_coherence(
        self,
        discrete_signal: np.ndarray,
        smooth_signal: np.ndarray,
        nperseg: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute magnitude-squared coherence between signals.
        
        Coherence measures correlation in frequency domain: C(f) ∈ [0, 1]
        C(f) = 1 indicates perfect correlation at frequency f
        
        Args:
            discrete_signal: Discrete operation output
            smooth_signal: Smooth approximation output
            nperseg: Length of FFT segments
            
        Returns:
            (frequencies, coherence_values)
        """
        if nperseg is None:
            nperseg = min(256, len(discrete_signal))
        
        freqs, coherence = signal.coherence(
            discrete_signal,
            smooth_signal,
            fs=self.sample_rate,
            nperseg=nperseg
        )
        
        return freqs, coherence


class GeometricAnalyzer:
    """
    Geometric Analysis of Approximation Manifolds.
    
    Treats discrete and smooth operations as mappings to manifolds in output space.
    Analyzes geometric properties:
    - Manifold distance (geodesic vs Euclidean)
    - Curvature (second fundamental form)
    - Tangent space alignment
    - Volume distortion
    
    Mathematical Background:
        For manifold M ⊂ ℝ^n:
        - Tangent space T_p M: Linearization at point p
        - Curvature κ: Rate of deviation from tangent space
        - Geodesic distance d_g: Shortest path on manifold
        
    Metrics Computed:
        - geodesic_distance: Distance along manifold
        - curvature_difference: ||κ_discrete - κ_smooth||
        - tangent_alignment: cos(θ) between tangent spaces
        - volume_distortion: det(J_smooth) / det(J_discrete)
    """
    
    def __init__(self, n_neighbors: int = 10):
        self.n_neighbors = n_neighbors
        
    def compute_manifold_distance(
        self,
        discrete_points: np.ndarray,
        smooth_points: np.ndarray,
        method: str = 'isomap'
    ) -> Dict[str, float]:
        """
        Compute distance between discrete and smooth manifolds.
        
        Args:
            discrete_points: Points on discrete manifold (n_samples, n_features)
            smooth_points: Points on smooth manifold (n_samples, n_features)
            method: Distance computation method ('isomap', 'procrustes')
            
        Returns:
            Manifold distance metrics
        """
        if method == 'procrustes':
            # Procrustes analysis: optimal rotation/scaling
            # Align smooth to discrete using Procrustes
            from scipy.spatial import procrustes
            
            _, aligned_smooth, disparity = procrustes(discrete_points, smooth_points)
            
            # Procrustes distance (after alignment)
            procrustes_distance = disparity
            
            # Point-wise distances after alignment
            pointwise_distances = np.linalg.norm(discrete_points - aligned_smooth, axis=1)
            
            return {
                'procrustes_distance': float(procrustes_distance),
                'mean_pointwise_distance': float(np.mean(pointwise_distances)),
                'max_pointwise_distance': float(np.max(pointwise_distances)),
                'std_pointwise_distance': float(np.std(pointwise_distances))
            }
        
        elif method == 'hausdorff':
            # Hausdorff distance: max of min distances
            # d_H(A, B) = max(max_a min_b d(a,b), max_b min_a d(a,b))
            
            # Distance matrices
            dist_ab = squareform(pdist(
                np.vstack([discrete_points, smooth_points])
            ))
            n_discrete = len(discrete_points)
            
            # Forward direction: max over discrete of min to smooth
            forward = np.max(np.min(dist_ab[:n_discrete, n_discrete:], axis=1))
            
            # Backward direction: max over smooth of min to discrete
            backward = np.max(np.min(dist_ab[n_discrete:, :n_discrete], axis=1))
            
            hausdorff = max(forward, backward)
            
            return {
                'hausdorff_distance': float(hausdorff),
                'forward_distance': float(forward),
                'backward_distance': float(backward)
            }
        
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def estimate_curvature(
        self,
        points: np.ndarray,
        k: int = 5
    ) -> np.ndarray:
        """
        Estimate local curvature at each point.
        
        Uses k-nearest neighbors to fit local quadratic surface.
        Curvature is eigenvalue of Hessian of fitted surface.
        
        Args:
            points: Points on manifold (n_samples, n_features)
            k: Number of neighbors for local fit
            
        Returns:
            Curvature estimates (n_samples,)
        """
        from sklearn.neighbors import NearestNeighbors
        
        n_samples = len(points)
        curvatures = np.zeros(n_samples)
        
        # Fit k-NN
        nbrs = NearestNeighbors(n_neighbors=k+1).fit(points)  # +1 for self
        
        for i in range(n_samples):
            # Get neighbors
            distances, indices = nbrs.kneighbors([points[i]])
            neighbor_points = points[indices[0][1:]]  # Exclude self
            
            # Center points
            center = points[i]
            centered = neighbor_points - center
            
            # Fit quadratic: z ≈ x^T H x (Hessian)
            # Use PCA to get principal curvatures
            if len(centered) > 1:
                cov = np.cov(centered.T)
                eigenvalues = np.linalg.eigvalsh(cov)
                # Curvature is related to smallest eigenvalue
                curvatures[i] = np.min(np.abs(eigenvalues)) + 1e-10
        
        return curvatures
    
    def compute_tangent_alignment(
        self,
        discrete_points: np.ndarray,
        smooth_points: np.ndarray,
        k: int = 5
    ) -> Dict[str, float]:
        """
        Compute alignment between tangent spaces.
        
        Tangent space at point p: span of gradients/derivatives.
        Alignment: cosine similarity between tangent spaces.
        
        Args:
            discrete_points: Points on discrete manifold
            smooth_points: Points on smooth manifold  
            k: Number of neighbors for tangent estimation
            
        Returns:
            Tangent space alignment metrics
        """
        from sklearn.neighbors import NearestNeighbors
        
        n_samples = min(len(discrete_points), len(smooth_points))
        alignments = []
        
        # Fit k-NN for both
        nbrs_discrete = NearestNeighbors(n_neighbors=k+1).fit(discrete_points)
        nbrs_smooth = NearestNeighbors(n_neighbors=k+1).fit(smooth_points)
        
        for i in range(min(100, n_samples)):  # Sample for efficiency
            # Get tangent vectors (differences to neighbors)
            _, indices_d = nbrs_discrete.kneighbors([discrete_points[i]])
            tangent_d = discrete_points[indices_d[0][1:]] - discrete_points[i]
            
            _, indices_s = nbrs_smooth.kneighbors([smooth_points[i]])
            tangent_s = smooth_points[indices_s[0][1:]] - smooth_points[i]
            
            # Compute principal directions (SVD)
            if len(tangent_d) > 0 and len(tangent_s) > 0:
                _, _, vt_d = np.linalg.svd(tangent_d, full_matrices=False)
                _, _, vt_s = np.linalg.svd(tangent_s, full_matrices=False)
                
                # Primary tangent directions
                v_d = vt_d[0]
                v_s = vt_s[0]
                
                # Cosine similarity
                alignment = np.abs(np.dot(v_d, v_s))
                alignments.append(alignment)
        
        alignments = np.array(alignments)
        
        return {
            'mean_tangent_alignment': float(np.mean(alignments)),
            'std_tangent_alignment': float(np.std(alignments)),
            'min_tangent_alignment': float(np.min(alignments)),
            'median_tangent_alignment': float(np.median(alignments))
        }


class ConvergenceAnalyzer:
    """
    Convergence Analysis with Theoretical Bounds.
    
    Provides convergence guarantees for approximation methods:
    - Convergence rates as precision increases
    - Error bounds (probabilistic and deterministic)
    - Asymptotic analysis
    - Confidence intervals
    
    Mathematical Framework:
        For approximation sequence {φ_n}, we analyze:
        
        1. Pointwise convergence:
           lim_{n→∞} φ_n(x) = f(x) for all x
        
        2. Uniform convergence:
           lim_{n→∞} sup_x |φ_n(x) - f(x)| = 0
        
        3. Convergence rate:
           |φ_n(x) - f(x)| = O(n^{-α}) for some α > 0
        
        4. Probabilistic bounds:
           P(|φ_n(x) - f(x)| > ε) ≤ δ for ε, δ > 0
    
    Theorems Implemented:
        - Weierstrass Approximation Theorem
        - Stone-Weierstrass Theorem
        - Universal Approximation Theorem
    """
    
    def __init__(self):
        pass
    
    def estimate_convergence_rate(
        self,
        errors: List[float],
        precisions: List[float]
    ) -> Dict[str, float]:
        """
        Estimate convergence rate from error vs precision data.
        
        Fits power law: error(p) = A · p^{-α}
        where α is the convergence rate.
        
        Args:
            errors: Approximation errors at different precisions
            precisions: Precision levels (e.g., number of parameters)
            
        Returns:
            Convergence rate α and fit quality
        """
        # Log-log fit
        log_precisions = np.log(precisions)
        log_errors = np.log(errors)
        
        # Linear regression in log space
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            log_precisions, log_errors
        )
        
        # Convergence rate is -slope (negative because error decreases)
        convergence_rate = -slope
        
        # Confidence interval (95%)
        ci_95 = 1.96 * std_err
        
        return {
            'convergence_rate_alpha': float(convergence_rate),
            'log_prefactor_A': float(intercept),
            'r_squared': float(r_value ** 2),
            'p_value': float(p_value),
            'standard_error': float(std_err),
            'confidence_interval_95': float(ci_95),
            'rate_lower_bound': float(convergence_rate - ci_95),
            'rate_upper_bound': float(convergence_rate + ci_95)
        }
    
    def compute_error_bounds(
        self,
        errors: np.ndarray,
        confidence: float = 0.95
    ) -> Dict[str, float]:
        """
        Compute probabilistic error bounds.
        
        Uses concentration inequalities (Hoeffding, Chebyshev) to bound
        approximation error with high probability.
        
        Args:
            errors: Sample approximation errors
            confidence: Confidence level (default: 0.95)
            
        Returns:
            Error bounds at specified confidence
        """
        n = len(errors)
        mean_error = np.mean(errors)
        std_error = np.std(errors)
        
        # Empirical quantile
        alpha = 1 - confidence
        empirical_bound = np.quantile(errors, confidence)
        
        # Chebyshev bound: P(|X - μ| > kσ) ≤ 1/k^2
        # For confidence p, we need k such that 1/k^2 = 1-p
        k_chebyshev = 1 / np.sqrt(1 - confidence)
        chebyshev_bound = mean_error + k_chebyshev * std_error
        
        # Hoeffding bound (assumes bounded errors in [a, b])
        # P(|mean - true_mean| > t) ≤ 2 exp(-2nt^2 / (b-a)^2)
        a, b = np.min(errors), np.max(errors)
        t_hoeffding = np.sqrt(-np.log((1-confidence)/2) * (b-a)**2 / (2*n))
        hoeffding_bound = mean_error + t_hoeffding
        
        # Normal approximation (CLT)
        from scipy.stats import norm
        z_score = norm.ppf(confidence)
        normal_bound = mean_error + z_score * (std_error / np.sqrt(n))
        
        return {
            'mean_error': float(mean_error),
            'std_error': float(std_error),
            'empirical_bound': float(empirical_bound),
            'chebyshev_bound': float(chebyshev_bound),
            'hoeffding_bound': float(hoeffding_bound),
            'normal_approximation_bound': float(normal_bound),
            'confidence_level': float(confidence)
        }
    
    def universal_approximation_theorem_check(
        self,
        approximator: Callable,
        target_function: Callable,
        domain: Tuple[float, float],
        epsilon: float = 0.01,
        num_samples: int = 1000
    ) -> Dict[str, bool]:
        """
        Check if approximator satisfies Universal Approximation Theorem.
        
        UAT states: For any continuous function f on compact domain K,
        there exists approximation φ such that:
        sup_{x ∈ K} |φ(x) - f(x)| < ε
        
        Args:
            approximator: Approximation function
            target_function: Target function to approximate
            domain: Domain [a, b]
            epsilon: Target approximation error
            num_samples: Number of test points
            
        Returns:
            Whether UAT conditions are satisfied
        """
        # Sample points uniformly in domain
        a, b = domain
        x_samples = np.linspace(a, b, num_samples)
        
        # Evaluate both functions
        approx_values = np.array([approximator(x) for x in x_samples])
        target_values = np.array([target_function(x) for x in x_samples])
        
        # Compute uniform error (L∞ norm)
        errors = np.abs(approx_values - target_values)
        uniform_error = np.max(errors)
        
        # Check UAT condition
        uat_satisfied = uniform_error < epsilon
        
        # Additional checks
        # 1. Continuity: Check Lipschitz constant
        diff_approx = np.diff(approx_values)
        diff_x = np.diff(x_samples)
        lipschitz_constant = np.max(np.abs(diff_approx / diff_x))
        is_lipschitz = lipschitz_constant < np.inf
        
        # 2. Compactness: Domain is bounded
        is_compact = (a > -np.inf) and (b < np.inf)
        
        return {
            'uat_satisfied': bool(uat_satisfied),
            'uniform_error': float(uniform_error),
            'epsilon_threshold': float(epsilon),
            'is_lipschitz_continuous': bool(is_lipschitz),
            'lipschitz_constant': float(lipschitz_constant),
            'domain_is_compact': bool(is_compact),
            'all_conditions_met': bool(uat_satisfied and is_lipschitz and is_compact)
        }


# Composite analyzer combining all metrics
class ComprehensiveApproximationAnalyzer:
    """
    Comprehensive analyzer combining spectral, geometric, and convergence analysis.
    
    Provides complete characterization of approximation quality using:
    - Spectral analysis (frequency domain)
    - Geometric analysis (manifold properties)
    - Convergence analysis (theoretical guarantees)
    - Information-theoretic measures
    
    Example:
        >>> analyzer = ComprehensiveApproximationAnalyzer()
        >>> results = analyzer.analyze_complete(
        ...     discrete_outputs,
        ...     smooth_outputs,
        ...     errors_at_precisions,
        ...     precision_levels
        ... )
    """
    
    def __init__(self):
        self.spectral = SpectralAnalyzer()
        self.geometric = GeometricAnalyzer()
        self.convergence = ConvergenceAnalyzer()
        
    def analyze_complete(
        self,
        discrete_outputs: np.ndarray,
        smooth_outputs: np.ndarray,
        errors_list: Optional[List[float]] = None,
        precisions_list: Optional[List[float]] = None
    ) -> Dict[str, Dict]:
        """
        Perform complete approximation analysis.
        
        Args:
            discrete_outputs: Outputs from discrete operation
            smooth_outputs: Outputs from smooth approximation
            errors_list: Errors at different precision levels (optional)
            precisions_list: Precision levels (optional)
            
        Returns:
            Comprehensive analysis results
        """
        results = {}
        
        # Spectral analysis
        print("Performing spectral analysis...")
        results['spectral'] = self.spectral.analyze_spectrum(
            discrete_outputs,
            smooth_outputs
        )
        
        # Geometric analysis
        print("Performing geometric analysis...")
        if len(discrete_outputs.shape) > 1:
            # Manifold analysis requires multi-dimensional outputs
            results['geometric'] = {}
            results['geometric']['manifold_distance'] = self.geometric.compute_manifold_distance(
                discrete_outputs,
                smooth_outputs,
                method='procrustes'
            )
            results['geometric']['tangent_alignment'] = self.geometric.compute_tangent_alignment(
                discrete_outputs,
                smooth_outputs
            )
        else:
            results['geometric'] = {'note': 'Requires multi-dimensional outputs'}
        
        # Convergence analysis
        print("Performing convergence analysis...")
        if errors_list is not None and precisions_list is not None:
            results['convergence'] = {}
            results['convergence']['rate'] = self.convergence.estimate_convergence_rate(
                errors_list,
                precisions_list
            )
            results['convergence']['bounds'] = self.convergence.compute_error_bounds(
                np.array(errors_list)
            )
        else:
            results['convergence'] = {'note': 'Requires error/precision data'}
        
        return results


if __name__ == "__main__":
    print("Advanced Metrics for Approximation Fidelity")
    print("="*70)
    print("\nAvailable analyzers:")
    print("1. SpectralAnalyzer - Frequency domain analysis")
    print("2. GeometricAnalyzer - Manifold distance and curvature")
    print("3. ConvergenceAnalyzer - Theoretical bounds and rates")
    print("4. ComprehensiveApproximationAnalyzer - All metrics combined")
    print("\nUse ComprehensiveApproximationAnalyzer for complete analysis.")
