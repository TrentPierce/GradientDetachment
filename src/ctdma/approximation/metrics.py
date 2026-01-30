"""
Approximation Metrics Module

Quantitative metrics for measuring approximation fidelity between
discrete operations and their continuous approximations.

Metrics include:
1. Approximation error (L1, L2, L∞)
2. Gradient fidelity (cosine similarity, magnitude ratio)
3. Information preservation (mutual information, KL divergence)
4. Boundary behavior (error at discontinuities)
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional
from scipy.stats import entropy
from scipy.spatial.distance import jensenshannon


class ApproximationMetrics:
    """
    Comprehensive metrics for approximation quality analysis.
    
    Evaluates how well a smooth approximation captures the behavior
    of the discrete operation, both in terms of output values and gradients.
    """
    
    def __init__(self, n_bits: int = 16):
        self.n_bits = n_bits
        self.modulus = 2 ** n_bits
        
    def compute_all_metrics(
        self,
        discrete_output: torch.Tensor,
        approx_output: torch.Tensor,
        discrete_grad: Optional[torch.Tensor] = None,
        approx_grad: Optional[torch.Tensor] = None,
        input_x: Optional[torch.Tensor] = None,
        input_y: Optional[torch.Tensor] = None
    ) -> Dict[str, float]:
        """
        Compute comprehensive approximation metrics.
        
        Args:
            discrete_output: Output from discrete operation
            approx_output: Output from approximation
            discrete_grad: Gradient from discrete operation (if available)
            approx_grad: Gradient from approximation
            input_x, input_y: Input values (for boundary analysis)
            
        Returns:
            Dictionary of metrics
        """
        metrics = {}
        
        # Output approximation error
        metrics.update(self.compute_approximation_error(discrete_output, approx_output))
        
        # Gradient fidelity
        if discrete_grad is not None and approx_grad is not None:
            metrics.update(self.compute_gradient_fidelity(discrete_grad, approx_grad))
        
        # Information preservation
        metrics.update(self.compute_information_preservation(discrete_output, approx_output))
        
        # Boundary behavior
        if input_x is not None and input_y is not None:
            metrics.update(self.compute_boundary_metrics(
                input_x, input_y, discrete_output, approx_output
            ))
        
        return metrics
    
    def compute_approximation_error(
        self,
        discrete_output: torch.Tensor,
        approx_output: torch.Tensor
    ) -> Dict[str, float]:
        """
        Compute various error norms between discrete and approximate outputs.
        
        Metrics:
        - L1 (mean absolute error)
        - L2 (root mean squared error)
        - L∞ (maximum absolute error)
        - Relative error
        
        Returns:
            Dictionary of error metrics
        """
        error = discrete_output - approx_output
        
        l1_error = torch.abs(error).mean().item()
        l2_error = torch.sqrt((error ** 2).mean()).item()
        linf_error = torch.abs(error).max().item()
        
        # Relative error (normalized by range)
        output_range = discrete_output.max() - discrete_output.min()
        relative_error = l1_error / (output_range.item() + 1e-10)
        
        # Correlation coefficient
        if discrete_output.numel() > 1:
            correlation = torch.corrcoef(torch.stack([
                discrete_output.flatten(),
                approx_output.flatten()
            ]))[0, 1].item()
        else:
            correlation = 1.0
        
        return {
            'l1_error': l1_error,
            'l2_error': l2_error,
            'linf_error': linf_error,
            'relative_error': relative_error,
            'correlation': correlation
        }
    
    def compute_gradient_fidelity(
        self,
        discrete_grad: torch.Tensor,
        approx_grad: torch.Tensor
    ) -> Dict[str, float]:
        """
        Measure how well approximation gradients match discrete gradients.
        
        Metrics:
        - Cosine similarity (direction)
        - Magnitude ratio (scale)
        - Angular error
        - Sign agreement (for critical decisions)
        
        Returns:
            Dictionary of gradient metrics
        """
        # Flatten gradients
        discrete_flat = discrete_grad.flatten()
        approx_flat = approx_grad.flatten()
        
        # Cosine similarity (direction)
        cos_sim = F.cosine_similarity(
            discrete_flat.unsqueeze(0),
            approx_flat.unsqueeze(0),
            dim=1
        ).item()
        
        # Magnitude ratio
        discrete_mag = torch.norm(discrete_flat).item()
        approx_mag = torch.norm(approx_flat).item()
        magnitude_ratio = approx_mag / (discrete_mag + 1e-10)
        
        # Angular error (in degrees)
        angle_rad = torch.acos(torch.clamp(torch.tensor(cos_sim), -1, 1))
        angle_deg = angle_rad.item() * 180 / np.pi
        
        # Sign agreement (what fraction have same sign?)
        sign_agreement = (torch.sign(discrete_flat) == torch.sign(approx_flat)).float().mean().item()
        
        # Gradient error norms
        grad_error = discrete_flat - approx_flat
        grad_l1_error = torch.abs(grad_error).mean().item()
        grad_l2_error = torch.sqrt((grad_error ** 2).mean()).item()
        
        return {
            'gradient_cosine_similarity': cos_sim,
            'gradient_magnitude_ratio': magnitude_ratio,
            'gradient_angular_error_deg': angle_deg,
            'gradient_sign_agreement': sign_agreement,
            'gradient_l1_error': grad_l1_error,
            'gradient_l2_error': grad_l2_error
        }
    
    def compute_information_preservation(
        self,
        discrete_output: torch.Tensor,
        approx_output: torch.Tensor,
        bins: int = 50
    ) -> Dict[str, float]:
        """
        Measure information preservation using information theory.
        
        Metrics:
        - Entropy of discrete output
        - Entropy of approximate output
        - Mutual information I(discrete; approx)
        - KL divergence D_KL(discrete || approx)
        - Jensen-Shannon divergence (symmetric)
        
        Returns:
            Dictionary of information-theoretic metrics
        """
        # Convert to numpy
        discrete_np = discrete_output.detach().cpu().numpy().flatten()
        approx_np = approx_output.detach().cpu().numpy().flatten()
        
        # Compute histograms (discretize for entropy)
        discrete_hist, _ = np.histogram(discrete_np, bins=bins, density=True)
        approx_hist, _ = np.histogram(approx_np, bins=bins, density=True)
        
        # Normalize to probabilities
        discrete_prob = discrete_hist / (discrete_hist.sum() + 1e-10)
        approx_prob = approx_hist / (approx_hist.sum() + 1e-10)
        
        # Entropies
        H_discrete = entropy(discrete_prob + 1e-10)
        H_approx = entropy(approx_prob + 1e-10)
        
        # KL divergence
        kl_div = entropy(discrete_prob + 1e-10, approx_prob + 1e-10)
        
        # Jensen-Shannon divergence (symmetric)
        js_div = jensenshannon(discrete_prob + 1e-10, approx_prob + 1e-10)
        
        # Mutual information (via joint histogram)
        hist_2d, _, _ = np.histogram2d(discrete_np, approx_np, bins=bins, density=True)
        hist_2d_prob = hist_2d / (hist_2d.sum() + 1e-10)
        H_joint = entropy(hist_2d_prob.flatten() + 1e-10)
        mutual_info = H_discrete + H_approx - H_joint
        
        # Information preservation ratio
        info_preservation = H_approx / (H_discrete + 1e-10)
        
        return {
            'entropy_discrete': H_discrete,
            'entropy_approx': H_approx,
            'kl_divergence': kl_div,
            'js_divergence': js_div,
            'mutual_information': mutual_info,
            'information_preservation_ratio': info_preservation
        }
    
    def compute_boundary_metrics(
        self,
        input_x: torch.Tensor,
        input_y: torch.Tensor,
        discrete_output: torch.Tensor,
        approx_output: torch.Tensor
    ) -> Dict[str, float]:
        """
        Analyze approximation behavior at discontinuity boundaries.
        
        For modular addition, boundaries occur at x + y = k·modulus.
        Error is typically highest near these points.
        
        Returns:
            Dictionary of boundary-specific metrics
        """
        # Find points near boundaries (wrap-around points)
        sum_val = input_x + input_y
        
        # Distance to nearest boundary
        boundary_distance = torch.minimum(
            sum_val % self.modulus,
            self.modulus - (sum_val % self.modulus)
        )
        
        # Define "near boundary" as within 5% of modulus
        near_boundary_threshold = self.modulus * 0.05
        near_boundary = boundary_distance < near_boundary_threshold
        
        if near_boundary.sum() > 0:
            # Error near boundaries
            error_near_boundary = torch.abs(
                discrete_output[near_boundary] - approx_output[near_boundary]
            ).mean().item()
            
            # Error far from boundaries
            far_from_boundary = ~near_boundary
            if far_from_boundary.sum() > 0:
                error_far_boundary = torch.abs(
                    discrete_output[far_from_boundary] - approx_output[far_from_boundary]
                ).mean().item()
            else:
                error_far_boundary = 0.0
            
            # Boundary error amplification
            boundary_amplification = error_near_boundary / (error_far_boundary + 1e-10)
        else:
            error_near_boundary = 0.0
            error_far_boundary = 0.0
            boundary_amplification = 1.0
        
        # Fraction of points near boundaries
        boundary_fraction = near_boundary.float().mean().item()
        
        return {
            'error_near_boundary': error_near_boundary,
            'error_far_boundary': error_far_boundary,
            'boundary_error_amplification': boundary_amplification,
            'boundary_fraction': boundary_fraction
        }


def compute_approximation_error(
    discrete_output: torch.Tensor,
    approx_output: torch.Tensor
) -> Dict[str, float]:
    """
    Convenience function to compute approximation error.
    
    Args:
        discrete_output: True discrete output
        approx_output: Approximate output
        
    Returns:
        Error metrics
    """
    metrics = ApproximationMetrics()
    return metrics.compute_approximation_error(discrete_output, approx_output)


def compute_gradient_fidelity(
    discrete_grad: torch.Tensor,
    approx_grad: torch.Tensor
) -> Dict[str, float]:
    """
    Convenience function to compute gradient fidelity.
    
    Args:
        discrete_grad: True discrete gradient
        approx_grad: Approximate gradient
        
    Returns:
        Gradient fidelity metrics
    """
    metrics = ApproximationMetrics()
    return metrics.compute_gradient_fidelity(discrete_grad, approx_grad)


def compute_information_preservation(
    discrete_output: torch.Tensor,
    approx_output: torch.Tensor
) -> Dict[str, float]:
    """
    Convenience function to compute information preservation.
    
    Args:
        discrete_output: True discrete output
        approx_output: Approximate output
        
    Returns:
        Information-theoretic metrics
    """
    metrics = ApproximationMetrics()
    return metrics.compute_information_preservation(discrete_output, approx_output)


def compare_approximation_methods(
    discrete_op: torch.nn.Module,
    approximations: Dict[str, torch.nn.Module],
    input_x: torch.Tensor,
    input_y: torch.Tensor
) -> Dict[str, Dict[str, float]]:
    """
    Compare multiple approximation methods side-by-side.
    
    Args:
        discrete_op: Discrete operation module
        approximations: Dictionary of approximation modules
        input_x, input_y: Input tensors
        
    Returns:
        Dictionary mapping method names to their metrics
    """
    # Compute discrete output
    discrete_output = discrete_op(input_x, input_y)
    
    results = {}
    metrics_calculator = ApproximationMetrics()
    
    for name, approx_method in approximations.items():
        # Compute approximation
        approx_output = approx_method(input_x, input_y)
        
        # Compute all metrics
        method_metrics = metrics_calculator.compute_all_metrics(
            discrete_output,
            approx_output,
            input_x=input_x,
            input_y=input_y
        )
        
        results[name] = method_metrics
    
    return results
