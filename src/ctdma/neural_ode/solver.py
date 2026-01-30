"""
Neural ODE Solver

Implements ODE solver for cryptanalysis using torchdiffeq.
"""

import torch
import torch.nn as nn
from torchdiffeq import odeint_adjoint as odeint


class NeuralODESolver(nn.Module):
    """
    Neural ODE solver for differential cryptanalysis.
    
    Args:
        cipher: Cipher instance (Speck, Feistel, SPN)
        method: ODE solver method ('dopri5', 'rk4', etc.)
        atol: Absolute tolerance
        rtol: Relative tolerance
    """
    
    def __init__(self, cipher, method='dopri5', atol=1e-7, rtol=1e-9):
        super().__init__()
        self.cipher = cipher
        self.method = method
        self.atol = atol
        self.rtol = rtol
        
    def forward(self, x0, t, key):
        """
        Solve ODE from x0 over time span t with key.
        
        Args:
            x0: Initial state (batch_size, state_dim)
            t: Time points (time_steps,)
            key: Encryption key
            
        Returns:
            solution: State trajectory (time_steps, batch_size, state_dim)
        """
        def ode_func(t, x):
            # The ODE function defines how state evolves
            # For cryptanalysis, this represents cipher operations
            return self.cipher.encrypt(x, key)
        
        solution = odeint(ode_func, x0, t, method=self.method, 
                         atol=self.atol, rtol=self.rtol)
        return solution
    
    def integrate(self, plaintext, key, time_span=(0, 1), num_points=100):
        """
        Convenience method for encryption simulation.
        
        Args:
            plaintext: Input plaintext (batch_size, state_dim)
            key: Encryption key
            time_span: Start and end time
            num_points: Number of time points to evaluate
            
        Returns:
            ciphertext: Final state (batch_size, state_dim)
        """
        t = torch.linspace(time_span[0], time_span[1], num_points, 
                          device=plaintext.device)
        trajectory = self.forward(plaintext, t, key)
        return trajectory[-1]  # Return final state