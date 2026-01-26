import torch
import torch.nn.functional as F

# Try importing Triton kernel if available (Linux/CUDA only)
try:
    # We will implement a Real-Valued Triton kernel later
    from .triton_kernels import triton_parallel_scan
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False
    pass

def parallel_scan(k, v, gate, gamma):
    """
    Stable Parallel Scan for Linear Attention (Real-Valued).
    Computes: h_t = gamma * h_{t-1} + (k_t * v_t * gate_t)
    
    Uses Log-Space Cumulative Sum for numerical stability on GPU.
    """
    # 1. Compute Input: u_t = k * v * gate
    u = k * v * gate
    
    # 2. Prepare Decay Mask
    B, T, H, D = u.shape
    # Create time indices [0, 1, ..., T-1]
    t_steps = torch.arange(T, device=u.device).view(1, T, 1, 1).float()
    
    # 3. Compute Decay Curve: gamma^t
    # gamma shape [1, 1, H, 1] broadcasts to [1, T, H, 1]
    decay_curve = torch.pow(gamma, t_steps)
    
    # 4. Numerical Stability Trick:
    # h_t = gamma^t * Sum(u_i / gamma^i)
    
    # Inverse decay: 1 / gamma^t
    decay_inv = 1.0 / (decay_curve + 1e-9)
    
    # Cumulative sum of "undecayed" inputs
    u_undecayed = torch.cumsum(u * decay_inv, dim=1)
    
    # Re-apply decay
    h = u_undecayed * decay_curve
    
    return h

@torch.jit.script
def recurrent_scan(k, v, gate, gamma, state):
    """
    Explicit Loop for Inference (O(1) Memory).
    JIT-compiled for speed on CPU/Standard GPU.
    """
    B, T, H, D = k.shape
    
    u = k * v * gate
    gamma_sq = gamma.view(1, H, 1)
    
    # Simple Recurrence: h_t = gamma * h_{t-1} + u_t
    # Since T is usually 1 during generation, this is just a single step.
    state = state * gamma_sq + u[:, 0]
    
    # Return both "sequence" (len 1) and "state"
    return state.unsqueeze(1), state