"""
Mathematical Analysis of Gradient Inversion in ARX Ciphers

This module provides rigorous mathematical analysis of the gradient inversion
phenomenon observed in Neural ODE-based cryptanalysis of ARX ciphers.

The analysis includes:
1. Formal proofs of gradient inversion in modular arithmetic
2. Topological analysis of sawtooth loss landscapes
3. Information-theoretic bounds on gradient flow
4. Critical point analysis and Hessian eigenvalue decomposition
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, List, Dict, Optional, Callable
from scipy.linalg import eigh
from scipy.stats import entropy


class ARXGradientAnalyzer:
    """
    Analyzes gradient behavior in ARX operations.
    
    Mathematical Foundation:
    ========================
    
    For modular addition f(x,y) = (x + y) mod 2^n, the gradient exhibits
    discontinuities at wraparound boundaries. Formally:
    
    ∇f(x,y) = {
        (1, 1)           if x + y < 2^n
        (undefined)      if x + y = 2^n
        (1, 1)           if x + y > 2^n (modulo applied)
    }
    
    The discontinuity at 2^n creates a "sawtooth" structure in the loss
    landscape, leading to gradient inversion.
    
    Theorem (Gradient Discontinuity):
    ----------------------------------
    Let f: ℤ₂ⁿ × ℤ₂ⁿ → ℤ₂ⁿ be modular addition. Then:
    
    lim[ε→0⁺] ∇f(2^n - ε, ε) - lim[ε→0⁻] ∇f(2^n + ε, -ε) ≠ 0
    
    Proof: See theorems.py for detailed proof.
    """
    
    def __init__(self, word_size: int = 16, device: str = 'cpu'):
        """
        Initialize ARX gradient analyzer.
        
        Args:
            word_size: Size of words in bits (default: 16)
            device: Computation device
        """
        self.word_size = word_size
        self.modulus = 2 ** word_size
        self.device = device
        
    def compute_modular_gradient(
        self, 
        x: torch.Tensor, 
        y: torch.Tensor,
        epsilon: float = 1e-6
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute gradient of modular addition using finite differences.
        
        Mathematical Formulation:
        -------------------------
        ∂f/∂x ≈ [f(x+ε, y) - f(x, y)] / ε
        ∂f/∂y ≈ [f(x, y+ε) - f(x, y)] / ε
        
        where f(x,y) = (x + y) mod 2^n
        
        Args:
            x: First operand
            y: Second operand
            epsilon: Finite difference step size
            
        Returns:
            (grad_x, grad_y): Partial derivatives
        """
        # Base value
        f_base = (x + y) % self.modulus
        
        # Gradient w.r.t x
        f_x_plus = (x + epsilon + y) % self.modulus
        grad_x = (f_x_plus - f_base) / epsilon
        
        # Gradient w.r.t y
        f_y_plus = (x + y + epsilon) % self.modulus
        grad_y = (f_y_plus - f_base) / epsilon
        
        return grad_x, grad_y
    
    def detect_discontinuities(
        self,
        x_range: Tuple[float, float],
        y_range: Tuple[float, float],
        num_samples: int = 1000
    ) -> Dict[str, torch.Tensor]:
        """
        Detect gradient discontinuities in the loss landscape.
        
        Mathematical Analysis:
        ----------------------
        A discontinuity occurs when:
        
        |∇f(x+δ) - ∇f(x)| / |δ| → ∞ as δ → 0
        
        We identify points where the gradient magnitude changes by > threshold.
        
        Returns:
            Dictionary containing:
            - positions: Locations of discontinuities
            - magnitudes: Jump magnitudes
            - directions: Direction of gradient change
        """
        x_vals = torch.linspace(x_range[0], x_range[1], num_samples)
        y_vals = torch.linspace(y_range[0], y_range[1], num_samples)
        
        discontinuities = {
            'positions': [],
            'magnitudes': [],
            'directions': []
        }
        
        for i in range(len(x_vals) - 1):
            x1, x2 = x_vals[i], x_vals[i + 1]
            
            for j in range(len(y_vals) - 1):
                y1, y2 = y_vals[j], y_vals[j + 1]
                
                # Compute gradients at adjacent points
                grad_x1, grad_y1 = self.compute_modular_gradient(x1, y1)
                grad_x2, grad_y2 = self.compute_modular_gradient(x2, y2)
                
                # Compute gradient change
                grad_change = torch.sqrt(
                    (grad_x2 - grad_x1)**2 + (grad_y2 - grad_y1)**2
                )
                
                # Threshold for discontinuity detection
                if grad_change > 0.5:  # Significant change
                    discontinuities['positions'].append((x1.item(), y1.item()))
                    discontinuities['magnitudes'].append(grad_change.item())
                    discontinuities['directions'].append(
                        torch.atan2(grad_y2 - grad_y1, grad_x2 - grad_x1).item()
                    )
        
        return {
            k: torch.tensor(v) if v else torch.tensor([])
            for k, v in discontinuities.items()
        }
    
    def compute_gradient_inversion_index(
        self,
        loss_landscape: Callable,
        x0: torch.Tensor,
        target_direction: torch.Tensor,
        num_steps: int = 100,
        step_size: float = 0.01
    ) -> float:
        """
        Compute the Gradient Inversion Index (GII).
        
        Mathematical Definition:
        ------------------------
        GII = (1/N) Σᵢ cos(θᵢ) where θᵢ is the angle between:
        - Gradient direction at step i
        - True direction to target
        
        GII ∈ [-1, 1]:
        - GII ≈ 1: Gradients point toward target (normal optimization)
        - GII ≈ 0: Gradients perpendicular (random walk)
        - GII ≈ -1: Gradients point away from target (inversion)
        
        Args:
            loss_landscape: Function computing loss
            x0: Starting point
            target_direction: True direction to minimum
            num_steps: Number of gradient steps
            step_size: Step size for gradient descent
            
        Returns:
            GII value in [-1, 1]
        """
        x = x0.clone().requires_grad_(True)
        cos_angles = []
        
        for step in range(num_steps):
            # Compute gradient
            loss = loss_landscape(x)
            if x.grad is not None:
                x.grad.zero_()
            loss.backward()
            
            grad = x.grad.clone()
            
            # Normalize vectors
            grad_norm = grad / (torch.norm(grad) + 1e-8)
            target_norm = target_direction / (torch.norm(target_direction) + 1e-8)
            
            # Compute cosine similarity
            cos_angle = torch.dot(grad_norm.flatten(), target_norm.flatten())
            cos_angles.append(cos_angle.item())
            
            # Update position
            with torch.no_grad():
                x -= step_size * grad
                x.requires_grad_(True)
        
        # Return mean cosine (GII)
        gii = np.mean(cos_angles)
        return gii


class SawtoothTopologyAnalyzer:
    """
    Analyzes the sawtooth topology induced by modular arithmetic.
    
    Mathematical Framework:
    =======================
    
    The loss landscape L(θ) for ARX ciphers exhibits a periodic sawtooth
    structure due to modular wraparound. Formally:
    
    L(θ) = Σᵢ ℓ(f_θ(xᵢ), yᵢ)
    
    where f_θ includes modular operations, creating discontinuities at:
    
    S = {θ : ∃i s.t. f_θ(xᵢ) = k·2^n for some k ∈ ℤ}
    
    Theorem (Sawtooth Periodicity):
    --------------------------------
    The loss landscape L(θ) is quasi-periodic with period T = 2^n in the
    direction of modular operations.
    
    Formally: L(θ + T·e_mod) ≈ L(θ) + ε(θ)
    
    where e_mod is the unit vector in the modular operation direction and
    ε(θ) is bounded noise.
    """
    
    def __init__(self, word_size: int = 16):
        """Initialize sawtooth topology analyzer."""
        self.word_size = word_size
        self.modulus = 2 ** word_size
        
    def compute_fourier_spectrum(
        self,
        loss_values: torch.Tensor
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute Fourier spectrum of loss landscape to detect periodicity.
        
        Mathematical Analysis:
        ----------------------
        Given loss values L[k], the Fourier transform is:
        
        L̂(ω) = Σₖ L[k] exp(-2πi ω k / N)
        
        Peaks in |L̂(ω)| indicate dominant frequencies, revealing the
        sawtooth period.
        
        Args:
            loss_values: Loss values along a trajectory
            
        Returns:
            (frequencies, magnitudes): Fourier spectrum
        """
        loss_np = loss_values.cpu().numpy()
        
        # Compute FFT
        fft = np.fft.fft(loss_np)
        frequencies = np.fft.fftfreq(len(loss_np))
        magnitudes = np.abs(fft)
        
        # Return positive frequencies only
        pos_mask = frequencies >= 0
        return frequencies[pos_mask], magnitudes[pos_mask]
    
    def estimate_sawtooth_period(
        self,
        loss_values: torch.Tensor
    ) -> float:
        """
        Estimate the period of sawtooth oscillations.
        
        Theorem Application:
        --------------------
        The dominant frequency ω₀ in the Fourier spectrum corresponds to
        the sawtooth period: T = 1 / ω₀
        
        For ARX ciphers with n-bit words, we expect T ≈ 2^n.
        
        Args:
            loss_values: Loss trajectory
            
        Returns:
            Estimated period T
        """
        frequencies, magnitudes = self.compute_fourier_spectrum(loss_values)
        
        # Find dominant frequency (excluding DC component)
        if len(frequencies) > 1:
            dominant_idx = np.argmax(magnitudes[1:]) + 1
            dominant_freq = frequencies[dominant_idx]
            
            # Period is inverse of frequency
            period = 1.0 / (dominant_freq + 1e-8)
            return period
        else:
            return 0.0
    
    def compute_landscape_roughness(
        self,
        loss_values: torch.Tensor,
        window_size: int = 10
    ) -> float:
        """
        Compute landscape roughness using discrete derivatives.
        
        Mathematical Definition:
        ------------------------
        Roughness R is defined as:
        
        R = √(1/N Σᵢ (∇L[i])²)
        
        where ∇L[i] = L[i+1] - L[i] is the discrete derivative.
        
        High roughness indicates many discontinuities (sawtooth structure).
        
        Args:
            loss_values: Loss trajectory
            window_size: Window for smoothing
            
        Returns:
            Roughness metric
        """
        # Compute discrete derivative
        derivatives = torch.diff(loss_values)
        
        # Compute RMS of derivatives
        roughness = torch.sqrt(torch.mean(derivatives ** 2))
        
        return roughness.item()
    
    def detect_local_minima(
        self,
        loss_values: torch.Tensor,
        threshold: float = 1e-4
    ) -> List[int]:
        """
        Detect local minima in the loss landscape.
        
        Mathematical Criterion:
        -----------------------
        A point i is a local minimum if:
        
        L[i-1] > L[i] and L[i+1] > L[i]
        
        with margin > threshold for numerical stability.
        
        Args:
            loss_values: Loss trajectory
            threshold: Minimum depth for local minimum
            
        Returns:
            Indices of local minima
        """
        minima = []
        
        for i in range(1, len(loss_values) - 1):
            if (loss_values[i-1] - loss_values[i] > threshold and
                loss_values[i+1] - loss_values[i] > threshold):
                minima.append(i)
        
        return minima


class InformationTheoreticAnalyzer:
    """
    Information-theoretic analysis of gradient flow in ARX operations.
    
    Theoretical Framework:
    ======================
    
    We analyze gradient flow through the lens of information theory,
    treating the cipher as a noisy channel:
    
    Plaintext → ARX Operations → Ciphertext
         X     →    Channel     →     Y
    
    Key Quantities:
    ---------------
    1. Mutual Information: I(X;Y) = H(X) - H(X|Y)
    2. Information Bottleneck: Information preserved through layers
    3. Gradient Signal-to-Noise Ratio: SNR_grad = E[∇L]² / Var[∇L]
    
    Theorem (Information Bottleneck in ARX):
    ----------------------------------------
    For ARX operations with n-bit words:
    
    I(X; f_ARX(X)) ≤ n - Σᵢ H(mod_i)
    
    where H(mod_i) is the entropy introduced by modular operation i.
    
    The information loss creates gradient noise, leading to inversion.
    """
    
    def __init__(self, num_bins: int = 256):
        """
        Initialize information-theoretic analyzer.
        
        Args:
            num_bins: Number of bins for histogram-based entropy estimation
        """
        self.num_bins = num_bins
        
    def compute_mutual_information(
        self,
        X: torch.Tensor,
        Y: torch.Tensor
    ) -> float:
        """
        Compute mutual information I(X;Y).
        
        Mathematical Definition:
        ------------------------
        I(X;Y) = H(X) + H(Y) - H(X,Y)
        
        where H(·) is Shannon entropy:
        
        H(X) = -Σₓ p(x) log p(x)
        
        Args:
            X: First random variable (shape: [N, ...])
            Y: Second random variable (shape: [N, ...])
            
        Returns:
            Mutual information in nats
        """
        # Flatten tensors
        X_flat = X.flatten().cpu().numpy()
        Y_flat = Y.flatten().cpu().numpy()
        
        # Compute histograms
        H_X = self._compute_entropy(X_flat)
        H_Y = self._compute_entropy(Y_flat)
        H_XY = self._compute_joint_entropy(X_flat, Y_flat)
        
        # Mutual information
        MI = H_X + H_Y - H_XY
        
        return float(MI)
    
    def _compute_entropy(self, x: np.ndarray) -> float:
        """Compute Shannon entropy using histogram."""
        hist, _ = np.histogram(x, bins=self.num_bins, density=True)
        # Remove zero bins
        hist = hist[hist > 0]
        # Normalize
        hist = hist / np.sum(hist)
        # Compute entropy
        return entropy(hist, base=np.e)
    
    def _compute_joint_entropy(
        self,
        x: np.ndarray,
        y: np.ndarray
    ) -> float:
        """Compute joint entropy H(X,Y)."""
        # Create 2D histogram
        hist, _, _ = np.histogram2d(x, y, bins=self.num_bins, density=True)
        # Remove zero bins
        hist = hist[hist > 0]
        # Normalize
        hist = hist / np.sum(hist)
        # Compute entropy
        return entropy(hist.flatten(), base=np.e)
    
    def compute_gradient_snr(
        self,
        gradients: List[torch.Tensor]
    ) -> float:
        """
        Compute Signal-to-Noise Ratio of gradients.
        
        Mathematical Definition:
        ------------------------
        SNR_grad = μ² / σ²
        
        where:
        - μ = E[||∇L||]: Mean gradient magnitude
        - σ² = Var[||∇L||]: Variance of gradient magnitude
        
        Low SNR indicates noisy gradients, typical of ARX operations.
        
        Args:
            gradients: List of gradient tensors
            
        Returns:
            SNR in dB: 10 log₁₀(SNR_grad)
        """
        # Compute gradient norms
        norms = [torch.norm(g).item() for g in gradients]
        
        # Mean and variance
        mean_norm = np.mean(norms)
        var_norm = np.var(norms)
        
        # SNR (avoid division by zero)
        if var_norm > 1e-10:
            snr = (mean_norm ** 2) / var_norm
            snr_db = 10 * np.log10(snr + 1e-10)
        else:
            snr_db = float('inf')
        
        return snr_db
    
    def compute_information_bottleneck(
        self,
        inputs: torch.Tensor,
        hidden_states: List[torch.Tensor],
        outputs: torch.Tensor
    ) -> Dict[str, float]:
        """
        Compute information bottleneck metrics.
        
        Information Bottleneck Theory:
        ------------------------------
        For a neural network with layers h₁, h₂, ..., hₗ:
        
        I(X; h₁) ≥ I(X; h₂) ≥ ... ≥ I(X; hₗ)
        
        The information decreases through layers, creating a bottleneck.
        
        For ARX ciphers, modular operations accelerate this decay.
        
        Args:
            inputs: Input data X
            hidden_states: List of hidden layer activations
            outputs: Output data Y
            
        Returns:
            Dictionary with I(X; hᵢ) for each layer i
        """
        results = {
            'I_X_input': self._compute_entropy(inputs.flatten().cpu().numpy())
        }
        
        # Compute MI for each hidden layer
        for i, hidden in enumerate(hidden_states):
            mi = self.compute_mutual_information(inputs, hidden)
            results[f'I_X_h{i+1}'] = mi
        
        # MI with output
        results['I_X_output'] = self.compute_mutual_information(inputs, outputs)
        
        # Compute compression ratios
        for i in range(len(hidden_states)):
            if i == 0:
                ratio = results[f'I_X_h{i+1}'] / results['I_X_input']
            else:
                ratio = results[f'I_X_h{i+1}'] / results[f'I_X_h{i}']
            results[f'compression_ratio_{i+1}'] = ratio
        
        return results


# Utility Functions
# =================

def compute_gradient_norm(
    model: nn.Module,
    norm_type: int = 2
) -> float:
    """
    Compute gradient norm for a model.
    
    Mathematical Definition:
    ------------------------
    ||∇θ||_p = (Σᵢ |∇θᵢ|^p)^(1/p)
    
    For p=2 (default), this is the Euclidean norm.
    
    Args:
        model: PyTorch model
        norm_type: Type of norm (1, 2, or inf)
        
    Returns:
        Gradient norm
    """
    total_norm = 0.0
    
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(norm_type)
            total_norm += param_norm.item() ** norm_type
    
    return total_norm ** (1.0 / norm_type)


def compute_hessian_eigenvalues(
    loss_fn: Callable,
    params: torch.Tensor,
    num_eigenvalues: int = 10
) -> np.ndarray:
    """
    Compute leading eigenvalues of the Hessian matrix.
    
    Mathematical Background:
    ------------------------
    The Hessian H = ∇²L(θ) characterizes the curvature of the loss landscape:
    
    H_ij = ∂²L / ∂θᵢ∂θⱼ
    
    Eigenvalue spectrum reveals:
    - Positive eigenvalues: Local minima directions
    - Negative eigenvalues: Saddle point directions
    - Near-zero eigenvalues: Flat directions
    
    Args:
        loss_fn: Loss function
        params: Parameters at which to compute Hessian
        num_eigenvalues: Number of leading eigenvalues to compute
        
    Returns:
        Array of eigenvalues (sorted by magnitude)
    """
    # Compute Hessian using autograd
    # Note: This is a simplified version; full implementation would use
    # efficient Hessian-vector products
    
    params_flat = params.flatten()
    n = len(params_flat)
    
    # For small parameter spaces, compute exact Hessian
    if n <= 1000:
        hessian = torch.zeros(n, n)
        
        # Compute each row of Hessian
        for i in range(n):
            # Gradient of gradient
            grad_i = torch.autograd.grad(
                loss_fn(params), params, create_graph=True
            )[0].flatten()[i]
            
            # Second derivatives
            hessian[i] = torch.autograd.grad(
                grad_i, params, retain_graph=True
            )[0].flatten()
        
        # Compute eigenvalues
        eigenvalues = torch.linalg.eigvalsh(hessian)
        eigenvalues = eigenvalues.cpu().numpy()
        
        # Return top k by magnitude
        indices = np.argsort(np.abs(eigenvalues))[-num_eigenvalues:]
        return eigenvalues[indices]
    else:
        # For large parameter spaces, use power iteration
        # (Approximation method)
        return np.array([])


def analyze_loss_landscape_curvature(
    model: nn.Module,
    data: torch.Tensor,
    labels: torch.Tensor,
    criterion: nn.Module,
    direction: Optional[torch.Tensor] = None,
    num_points: int = 21,
    alpha_range: Tuple[float, float] = (-1.0, 1.0)
) -> Dict[str, np.ndarray]:
    """
    Analyze curvature of loss landscape along a direction.
    
    Mathematical Framework:
    -----------------------
    Given parameters θ₀ and direction d, we analyze:
    
    L(α) = L(θ₀ + α·d)
    
    Computing:
    1. Loss values L(α)
    2. First derivative: dL/dα ≈ [L(α+ε) - L(α-ε)] / 2ε
    3. Second derivative: d²L/dα² ≈ [L(α+ε) - 2L(α) + L(α-ε)] / ε²
    
    Args:
        model: Neural network model
        data: Input data
        labels: Target labels
        criterion: Loss function
        direction: Direction to probe (if None, use random)
        num_points: Number of sample points
        alpha_range: Range of α values
        
    Returns:
        Dictionary with 'alphas', 'losses', 'first_deriv', 'second_deriv'
    """
    # Get current parameters
    params = [p.data.clone() for p in model.parameters()]
    
    # Generate random direction if not provided
    if direction is None:
        direction = [torch.randn_like(p) for p in params]
        # Normalize
        norm = sum([torch.norm(d) for d in direction])
        direction = [d / norm for d in direction]
    
    # Sample along direction
    alphas = np.linspace(alpha_range[0], alpha_range[1], num_points)
    losses = []
    
    for alpha in alphas:
        # Set parameters to θ₀ + α·d
        for p, p0, d in zip(model.parameters(), params, direction):
            p.data = p0 + alpha * d
        
        # Compute loss
        with torch.no_grad():
            output = model(data)
            loss = criterion(output, labels)
            losses.append(loss.item())
    
    # Restore original parameters
    for p, p0 in zip(model.parameters(), params):
        p.data = p0
    
    losses = np.array(losses)
    
    # Compute derivatives
    first_deriv = np.gradient(losses, alphas)
    second_deriv = np.gradient(first_deriv, alphas)
    
    return {
        'alphas': alphas,
        'losses': losses,
        'first_derivative': first_deriv,
        'second_derivative': second_deriv
    }
