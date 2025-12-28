import torch
import torch.nn.functional as F
import math
from typing import Tuple, Optional

def generate_random_phasors(hd_dim: int, device: torch.device = None) -> torch.Tensor:
    """
    Generates fixed orthogonal high-frequency phases for the Holographic Key.
    Replaces Standard RoPE to fix the 'Local Blur' issue.
    
    Args:
        hd_dim: The holographic dimension (must be even if using view_as_complex logic later, 
                but here we assume complex64 tensors).
    
    Returns:
        freqs: A tensor of shape (hd_dim,) containing random frequencies.
    """
    # Drawn from a wide uniform distribution to ensure orthogonality across the spectrum
    # High frequencies (> 1.0) are critical for "Needle in a Haystack" precision.
    return torch.randn(hd_dim, device=device) * 10.0

def compute_rotors(
    seq_len: int, 
    freqs: torch.Tensor, 
    offset: int = 0
) -> torch.Tensor:
    """
    Creates the Holographic Rotors (Positional Encodings) in the Complex Plane.
    Formula: Rotor_t = exp(i * t * theta)
    
    Args:
        seq_len: Length of the sequence.
        freqs: The fixed random frequencies (hd_dim,).
        offset: Starting position index (for cache/inference steps).
        
    Returns:
        rotors: Complex tensor (1, seq_len, hd_dim)
    """
    # Create position indices [0, 1, 2, ...]
    t = torch.arange(seq_len, device=freqs.device, dtype=torch.float32) + offset
    
    # Outer product: positions * frequencies
    # Shape: (seq_len, hd_dim)
    angles = torch.outer(t, freqs)
    
    # Polar to Rectangular: exp(i * theta) = cos(theta) + i*sin(theta)
    rotors = torch.polar(torch.ones_like(angles), angles)
    
    return rotors.unsqueeze(0) # Add batch dim for broadcasting

def holo_bind_and_accumulate(
    v: torch.Tensor, 
    rotors: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    The Core Holographic Memory Operation (The 'Write' Head).
    Performs Binding -> Accumulation with Mixed Precision stability.
    
    Args:
        v: Value tensor (Batch, Seq, HD_Dim) [Complex64]
        rotors: Positional Rotors (1, Seq, HD_Dim) [Complex64]
        
    Returns:
        memory_trace: The cumulative sum of bound states.
    """
    # 1. Holographic Binding (Element-wise Rotation)
    # This encodes the "Position" into the "Value"
    bound_state = v * rotors
    
    # 2. Linear Memory Accumulation (The Scan)
    # CRITICAL: Force FP32 for the cumulative sum to prevent "Swamping" 
    # (where small new tokens vanish in long contexts).
    # We cast to complex128 (Float64 real/imag) or complex64 (Float32 real/imag).
    # If input is BF16/FP16, this step must upgrade precision.
    bound_state_fp32 = bound_state.to(torch.complex64) 
    
    memory_trace = torch.cumsum(bound_state_fp32, dim=1)
    
    return memory_trace

def holo_retrieve(
    memory_trace: torch.Tensor, 
    rotors: torch.Tensor
) -> torch.Tensor:
    """
    The Holographic Retrieval Operation (The 'Read' Head).
    Performs Unbinding -> Normalization.
    
    Args:
        memory_trace: Accumulated memory (Batch, Seq, HD_Dim)
        rotors: Positional Rotors (1, Seq, HD_Dim)
        
    Returns:
        retrieved: The decoded signal (Real-valued projection ready for output).
    """
    B, T, D = memory_trace.shape
    
    # 1. Unbinding (Derotation)
    # Multiply by the CONJUGATE of the rotor. 
    # If we bound with exp(i*theta), we unbind with exp(-i*theta).
    # This cancels the phase for the target position, leaving the signal at DC (0 freq).
    raw_retrieval = memory_trace * torch.conj(rotors)
    
    # 2. Normalization (The Scaling Law Fix)
    # The magnitude of a random walk grows by sqrt(T). 
    # We divide by sqrt(T) to keep the signal variance roughly 1.0 for the MLP.
    scale = torch.sqrt(
        torch.arange(1, T + 1, device=memory_trace.device, dtype=torch.float32)
    ).view(1, T, 1)
    
    # Avoid div by zero (sanity check)
    scale = torch.clamp(scale, min=1.0)
    
    normalized_retrieval = raw_retrieval / scale
    
    # 3. Project to Real
    # The information is stored in the Magnitude/Real alignment.
    # We return the real part. The Imaginary part contains the "crosstalk noise" 
    # from other positions, which the MLP will filter out.
    return normalized_retrieval.real

def apply_spectral_decay(
    memory_trace: torch.Tensor, 
    gamma: float = 0.99
) -> torch.Tensor:
    """
    (Optional) Adds a decay factor to the memory, turning it into 
    "Leaky Holographic Memory". This helps with very long context 
    by prioritizing recent information, similar to Mamba/RWKV.
    
    Formula: h_t = h_{t-1} * gamma + x_t
    """
    # Note: This requires a parallel scan implementation (like associative_scan)
    # for efficiency, rather than simple cumsum.
    # Included here as a placeholder for the "Advanced" version of the paper.
    pass
