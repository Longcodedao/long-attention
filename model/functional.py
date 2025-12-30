import torch
import math

def generate_multiscale_phasors(num_heads: int, head_dim: int, 
                                device: torch.device = None) -> torch.Tensor:
    """
    Generates Rotors with Multi-Scale Frequencies (The "Spectrogram"
Effect).
    Instead of one uniform distribution, we assign different variance scales to different heads.
    - Low Index Heads: High Variance (Fast Rotation) -> Short-Term Precision.
    - High Index Heads: Low Variance (Slow Rotation) -> Long-Term Stability.

    Returns:
        freqs: (num_heads, head_dim)

    """

    # Generate a geometric distribution of sigmas from 10.0 (Fast) down to 0.1 (Slow)
    sigmas = torch.logspace(
        start = math.log10(10.0), 
        end = math.log10(0.1), 
        steps = num_heads,
        device = device
    )

    freqs_list = []
    for s in sigmas:
        # Generate random phases for this head with specific scale 's'
        head_freqs = torch.randn(head_dim, device = device) * s
        freqs_list.append(head_freqs)

    return torch.stack(freqs_list)
    


def generate_random_phasors(hd_dim: int, device: torch.device = None) -> torch.Tensor:
    """
    Generates fixed orthogonal high-frequency phases for the Holographic Key.
    """
    # Using std=3.0 as default for deep stability, though layers.py handles specific scaling
    return torch.randn(hd_dim, device=device) * 3.0


def compute_rotors_old(seq_len: int, freqs: torch.Tensor) -> torch.Tensor:
    """
    Creates the Holographic Rotors (Positional Encodings) in the Complex Plane.
    """
    t = torch.arange(seq_len, device=freqs.device, dtype=torch.float32)
    angles = torch.outer(t, freqs)
    rotors = torch.polar(torch.ones_like(angles), angles)
    

def compute_rotors(seq_len: int, freqs: torch.Tensor) -> torch.Tensor:
    """
    Creates Rotors. Handles Multi-Head broadcasting automatically.
    
    Args:
        seq_len: T
        freqs: (H, D) or (D,)
    
    Returns:
        rotors: (1, T, H, D) or (1, T, D)
    """
    # t: (T, 1, 1) to broadcast against H and D
    t = torch.arange(seq_len, device=freqs.device, dtype=torch.float32)
    for _ in range(freqs.ndim):
        t = t.unsqueeze(-1)

    # freqs: (1, H, D)
    f = freqs.unsqueeze(0)

    # angles: (T, H, D)
    angles = t * f

    # Polar to Rectangular 
    rotors = torch.polar(torch.ones_like(angles), angles)
    
    return rotors.unsqueeze(0)


def holo_bind_and_accumulate(v: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """
    Binding (Element-wise Mul) -> Accumulation (CumSum).
    """
    # Cast to complex64 for safety
    return torch.cumsum(v * k, dim=1)
    

def holo_retrieve(memory_trace: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """
    Unbinding (Element-wise Mul) -> Normalization.
    """
    T = memory_trace.size(1)

    # Scale by sqrt(T)
    # View shape construction to match (1, T, 1, 1) or (1, T, 1)
    view_shape = [1, T] + [1] * (memory_trace.ndim - 2)
    
    # Scale by sqrt(T) to normalize signal variance
    scale = torch.sqrt(torch.arange(1, T + 1, device=memory_trace.device, dtype=torch.float32)).view(*view_shape)
    
    scale = torch.clamp(scale, min=1.0)
    
    # Decode and Project to Real
    return (memory_trace * q).real / scale    