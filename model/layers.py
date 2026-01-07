import torch
import torch.nn as nn
from .functional import (
    generate_multiscale_phasors, 
    generate_random_phasors,
    compute_rotors, 
    holo_bind_and_accumulate, 
    holo_retrieve,
    holo_fused_step
)


class HoloAttentionV2(nn.Module):
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
        self.config = config
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
        

        self.register_buffer("freqs", generate_multiscale_phasors(self.num_heads, self.head_dim))

        # 3. Multi-Head Gating
        # Each head gets its own [Positional, Associative] weighting
        # Shape: (num_heads, 2)
        # Learnable Gate (Initialize balanced)
        self.gate = nn.Parameter(
            torch.ones(self.num_heads, 2) * 0.5
        )

        self.dropout = nn.Dropout(p = config.dropout)

    # Compile this specific math heavy static function
    @torch.compile(mode="max-autotune")
    def fused_pos_step(self, v, k, q):
        # This looks simple, but torch.compile optimizes the memory access
        # better than a naive triton loop for cumulative sums
        bind = v * k
        mem = torch.cumsum(bind, dim=1)
        # Retrieve
        scale = torch.sqrt(torch.arange(1, v.size(1) + 1, device=v.device)).view(1, -1, 1, 1)
        return (mem * q).real / scale
    
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
        rotors = compute_rotors(T, self.freqs).expand(B, -1, -1, -1)
        
        # print(f"v.shape: {v.shape}")
        # print(f"rotors.shape: {rotors.shape}")
        # mem_pos = holo_bind_and_accumulate(v, rotors)
        # out_pos = holo_retrieve(mem_pos, torch.conj(rotors))
        if self.config.use_version == 1:
            mem_pos = holo_bind_and_accumulate(v, rotors)
            out_pos = holo_retrieve(mem_pos, torch.conj(rotors))
        else:
            rotors_conj = torch.conj(rotors).resolve_conj()
            out_pos = holo_fused_step(v, rotors, rotors_conj)
            # out_pos = self.fused_pos_step(v, rotors, torch.conj(rotors))
            
        # --- 3. Path B: Associative (Content) ---
        # Shift K along the Time axis
        k_shifted = torch.roll(k, shifts=1, dims=1)
        k_shifted[:, 0, :] = 0 # Zero out first token history        
        if self.config.use_version == 1:
            mem_assoc = holo_bind_and_accumulate(v, torch.conj(k_shifted))
            out_assoc = holo_retrieve(mem_assoc, q)
        else:
            k_shifted_conj = torch.conj(k_shifted).resolve_conj()
            out_assoc = holo_fused_step(v, k_shifted_conj, q)
            # out_assoc = self.fused_pos_step(v, torch.conj(k_shifted), q)
        
        # --- 4. Gated Merge (Per Head) ---
        # gate[:, 0] is Positional weight, gate[:, 1] is Associative weight
        # Broadcast: (H) -> (1, 1, H, 1)        
        g_pos = self.gate[:, 0].view(1, 1, H, 1)
        g_assoc = self.gate[:, 1].view(1, 1, H, 1)
        
        out_combined = (out_pos * g_pos) + (out_assoc * g_assoc)

        # --- 5. Concatenate & Output ---
        # Flatten H and D back to hd_dim
        out_combined = out_combined.flatten(2)
        output = self.o_proj(out_combined)

        return self.dropout(output)
                          

class HoloBlock(nn.Module):
    """
    Standard Transformer Block with LayerScale for deep signal propagation.
    """
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.d_model)
        self.attn = HoloAttentionV2(config)
        self.ln2 = nn.LayerNorm(config.d_model)
        
        self.mlp = nn.Sequential(
            nn.Linear(config.d_model, config.d_model * config.expansion_factor),
            nn.GELU(),
            nn.Linear(config.d_model * config.expansion_factor, config.d_model),     
        )
        
        # LayerScale: Initialize to small value (0.1) to ease optimization
        self.gamma1 = nn.Parameter(torch.ones(config.d_model) * 0.1)
        self.gamma2 = nn.Parameter(torch.ones(config.d_model) * 0.1)

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