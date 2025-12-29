import torch
import math

def generate_random_phasors(hd_dim: int, device: torch.device = None) -> torch.Tensor:
    """
    Generates fixed orthogonal high-frequency phases for the Holographic Key.
    """
    # Using std=3.0 as default for deep stability, though layers.py handles specific scaling
    return torch.randn(hd_dim, device=device) * 3.0

def compute_rotors(seq_len: int, freqs: torch.Tensor) -> torch.Tensor:
    """
    Creates the Holographic Rotors (Positional Encodings) in the Complex Plane.
    """
    t = torch.arange(seq_len, device=freqs.device, dtype=torch.float32)
    angles = torch.outer(t, freqs)
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
    B, T, D = memory_trace.shape
    # Scale by sqrt(T) to normalize signal variance
    scale = torch.sqrt(torch.arange(1, T + 1, device=memory_trace.device, dtype=torch.float32)).view(1, T, 1)
    scale = torch.clamp(scale, min=1.0)
    
    # Decode and Project to Real
    return (memory_trace * q).real / scale