import torch
import torch.nn as nn
from .functional import (
    generate_multiscale_phasors, 
    compute_rotors, 
    holo_bind_and_accumulate, 
    holo_retrieve
)

class HoloAttention(nn.Module):
    """
    The Holographic 'Attention' Mechanism (v7 Production).
    Features:
    1. Dual-Path: Positional (Indexing) + Associative (Recall).
    2. Shared Q/K: Forces instant alignment for associative recall.
    3. Gated Mixing: Learnable balance between Time and Content.
    4. Phase Scaling: High-variance initialization for orthogonality.
    """
    def __init__(self, config):
        super().__init__()
        self.d_model = config.d_model
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        self.hd_dim = config.hd_dim
        self.phase_scale = config.phase_scale

        # 1. Projections (Real -> Complex)
        # We project inputs into the Holographic "Hyper-Dimension"        
        self.v_proj = nn.Linear(self.d_model, self.hd_dim, bias=False)
        self.k_proj = nn.Linear(self.d_model, self.hd_dim, bias=False) # Shared Q/K
        self.o_proj = nn.Linear(self.hd_dim, self.d_model, bias=False)
        
        # Fixed Positional Phasors
        self.register_buffer("freqs", generate_random_phasors(self.num_heads, self.head_dim))

        # 3. Multi-Head Gating
        # Each head gets its own [Positional, Associative] weighting
        # Shape: (num_heads, 2)
        # Learnable Gate (Initialize balanced)
        self.gate = nn.Parameter(
            torch.ones(self.num_heads, 2) * 0.5
        )

        self.dropout = nn.Dropout(p = config.dropout)
    
    def forward(self, x): 
        B, T, C = x.shape
        H = self.num_heads
        D = self.head_dim
        
        # --- 1. Project & Split Heads ---
        # (B, T, hd_dim) -> (B, T, H, D)
        k_real = self.k_proj(x).view(B, T, H, D)
        v_real = self.v_proj(x).view(B, T, H, D)

        v = v_real.to(torch.complex64)
        
        # Keys (Scaled for Orthogonality)
        k_angle = k_real * self.phase_scale
        k = torch.exp(1j * k_angle) 
        q = k # Shared Q/K
        
        # --- 2. Path A: Positional (Time) ---
        # Rotors are computed per-head based on their specific frequencies
        rotors = compute_rotors(T, self.freqs)
        mem_pos = holo_bind_and_accumulate(v, rotors)
        out_pos = holo_retrieve(mem_pos, torch.conj(rotors))
        
        # --- 3. Path B: Associative (Content) ---
        # Shift K along the Time axis
        k_shifted = torch.roll(k, shifts=1, dims=1)
        k_shifted[:, 0, :] = 0 # Zero out first token history
        
        mem_assoc = holo_bind_and_accumulate(v, torch.conj(k_shifted))
        out_assoc = holo_retrieve(mem_assoc, q)
        
        # --- 4. Gated Merge (Per Head) ---
        # gate[:, 0] is Positional weight, gate[:, 1] is Associative weight
        # Broadcast: (H) -> (1, 1, H, 1)        
        g_pos = self.gate[:, 0].view(1, 1, H, 1)
        g_assoc = self.gate[:, 1].view(1, 1, H, 1)
        
        out_combined = (out_pos * g_pos) + (out_assoc * g_assoc)

        # --- 5. Concatenate & Output ---
        # Flatten H and D back to hd_dim
        out_combined = out_combined.flatten(2)
        output = self.o_project(out_combined)

        return self.dropout(output)
                          

class HoloBlock(nn.Module):
    """
    Standard Transformer Block with LayerScale for deep signal propagation.
    """
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.hidden_size)
        self.attn = HoloAttention(config)
        self.ln2 = nn.LayerNorm(config.hidden_size)
        
        self.mlp = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size * config.expansion_factor),
            nn.GELU(),
            nn.Linear(config.d_model * config.expansion_factor, config.d_model),
            nn.Dropout(config.dropout)
            nn.Linear(config.hidden_size * config.expansion_factor, config.hidden_size)
        )
        
        # LayerScale: Initialize to small value (0.1) to ease optimization
        self.gamma1 = nn.Parameter(torch.ones(config.hidden_size) * 0.1)
        self.gamma2 = nn.Parameter(torch.ones(config.hidden_size) * 0.1)

    def forward(self, x):
        # Residual connection 1 (Mixer)
        res = x
        x = self.attn(self.ln1(x))
        x = res + self.gamma1 * x
        
        # Residual connection 2 (MLP)
        res = x
        x = self.mlp(self.ln2(x))
        x = res + self.gamma2 * x
        
        return x