import torch
import torch.nn as nn
from .functional import (
    generate_random_phasors, 
    compute_rotors, 
    holo_bind_and_accumulate, 
    holo_retrieve
)

class HoloAttention(nn.Module):
    """
    The Holographic 'Attention' Mechanism.
    Replaces N^2 Softmax Attention with O(N) Complex-Valued Recurrence.
    """
    def __init__(self, config):
        super().__init__()
        self.d_model = config.d_model
        self.hd_dim = config.hd_dim
        
        # 1. Projections (Real -> Complex)
        # We project inputs into the Holographic "Hyper-Dimension"
        self.k_proj = nn.Linear(config.d_model, config.hd_dim, bias=False)
        self.v_proj = nn.Linear(config.d_model, config.hd_dim, bias=False)
        self.o_proj = nn.Linear(config.d_model, config.d_model, bias=False) # Output is Real
        
        # 2. Fixed Random Phasors (The "Keys")
        # Registered as buffer so they save with the model but don't update via GD
        self.register_buffer("freqs", generate_random_phasors(config.hd_dim))

    def forward(self, x):
        B, T, C = x.shape
        
        # --- Step 1: Project to Holographic Space ---
        # k, v shape: (B, T, hd_dim)
        # We cast to complex64 immediately to enable phase operations
        k_real = self.k_proj(x)
        v_real = self.v_proj(x)
        
        # In a full implementation, K determines *which* frequency to write to.
        # For this version (Linear Associative Memory), we use V as the content
        # and implicit position as the key.
        # Future improvement: Use K to modulate the frequencies (Data-Dependent).
        
        v = v_real.to(torch.complex64)
        
        # --- Step 2: Generate Positional Rotors ---
        # Rotors shape: (1, T, hd_dim)
        rotors = compute_rotors(T, self.freqs)
        
        # --- Step 3: Bind & Accumulate (The O(N) Magic) ---
        # This replaces the Attention Matrix calculation
        memory_trace = holo_bind_and_accumulate(v, rotors)
        
        # --- Step 4: Retrieve (Derotate) ---
        # This replaces the Attention * Value calculation
        output_complex = holo_retrieve(memory_trace, rotors)
        
        # --- Step 5: Project Output ---
        # We take the Real part (Magnitude/Phase alignment)
        return self.o_proj(output_complex)

class HoloBlock(nn.Module):
    """
    Standard Transformer Block structure, but swapping Self-Attention 
    for HoloAttention.
    Structure: Input -> LN -> Holo -> Add -> LN -> MLP -> Add
    """
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.d_model)
        self.attn = HoloAttention(config)
        
        self.ln2 = nn.LayerNorm(config.d_model)
        self.mlp = nn.Sequential(
            nn.Linear(config.d_model, config.d_model * config.expansion_factor),
            nn.GELU(),
            nn.Linear(config.d_model * config.expansion_factor, config.d_model)
        )

    def forward(self, x):
        # 1. Holographic Mixer Path
        residual = x
        x = self.ln1(x)
        x = residual + self.attn(x)
        
        # 2. MLP Path (The "Denoising" Step)
        residual = x
        x = self.ln2(x)
        x = residual + self.mlp(x)
        
        return x
