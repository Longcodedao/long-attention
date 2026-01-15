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
import torch.nn.functional as F

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
        mem_pos = holo_bind_and_accumulate(v, rotors)
        out_pos = holo_retrieve(mem_pos, torch.conj(rotors))
          # print('Use Triton Kernel') 
        # rotors_conj = torch.conj(rotors).resolve_conj()
        # out_pos = holo_fused_step(v, rotors, rotors_conj)
        # out_pos = self.fused_pos_step(v, rotors, torch.conj(rotors))
            
        # --- 3. Path B: Associative (Content) ---
        # Shift K along the Time axis
        k_shifted = torch.roll(k, shifts=1, dims=1)
        k_shifted[:, 0, :] = 0 # Zero out first token history      
        
        mem_assoc = holo_bind_and_accumulate(v, torch.conj(k_shifted))
        out_assoc = holo_retrieve(mem_assoc, q)
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
        
        # === FIX IS HERE ===
        # Cast to match the weight dtype (BFloat16) before projection
        out_combined = out_combined.to(self.o_proj.weight.dtype)
        
        output = self.o_proj(out_combined)

        return self.dropout(output)


class LongAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.d_model =  config.d_model
        self.hd_dim = config.hd_dim
        self.num_heads = config.num_heads
        self.head_dim = config.hd_dim // config.num_heads
       
        # 1. LOCAL CONTEXT (Replaces Oracle Token Types)
        # Allows Gate to see "Previous Token" to decide if "Current Token" is important.
        # Kernel 3 or 4 is standard for modern SSMs (Mamba uses 4).
        self.conv1d = nn.Conv1d(
            in_channels=self.d_model,
            out_channels=self.d_model,
            kernel_size=4,
            groups=self.d_model, # Depthwise
            padding=3  # Caudel padding handled manually usually, or padding=kernel-1
        )

        # 2. PROJECTIONS
        # Shared Q/K for Associative Path
        self.qk_proj = nn.Linear(self.d_model, self.hd_dim, bias=False)
        self.v_proj = nn.Linear(self.d_model, self.hd_dim, bias=False)
        self.g_proj = nn.Linear(self.d_model, self.hd_dim, bias=True)
        self.o_proj = nn.Linear(self.hd_dim, self.d_model, bias=False)
        self.out_norm = nn.LayerNorm(self.d_model)

        # 3. SPECTROGRAM INITIALIZATION (Lesson: Phase Scaling)
        # Head 0 = Static (Needle Storage)
        # Heads 1..N = Log-Spaced (Temporal Reasoning)
        freqs = torch.zeros(self.num_heads, self.head_dim)
        if self.num_heads > 1:
            indices = torch.linspace(0, 1, self.head_dim)
            # Log-spacing for temporal heads
            for h in range(1, self.num_heads):
                freqs[h] = (10.0 ** indices) * h
        self.register_buffer("freqs", freqs)

        # 4. DECAY (Lesson: Head 0 must be Infinite)
        # Initialize Head 0 with Gamma=1.0 (No decay)
        # Initialize others with Gamma=0.999 (Slow decay)
        gamma_init = torch.ones(self.num_heads, 1, 1) * 0.999
        gamma_init[0] = 1.0
        self.gamma = nn.Parameter(gamma_init)

        # Init Tricks
        nn.init.eye_(self.v_proj.weight) # Identity Value start
        nn.init.constant_(self.g_proj.bias, 2.0) # Bias gate open

    def forward(self, x):
        B, T, C = x.shape
       
        # 1. Local Context (Conv)
        # Transpose for Conv: [B, C, T]
        x_conv = self.conv1d(x.transpose(1, 2))[:, :, :T].transpose(1, 2)
       
        # 2. Projections (Shared QK)
        qk = self.qk_proj(x_conv).view(B, T, self.num_heads, self.head_dim)
        k = F.normalize(qk, p=2, dim=-1)
        q = F.normalize(qk, p=2, dim=-1)
       
        v = self.v_proj(x_conv).view(B, T, self.num_heads, self.head_dim)
        g = torch.sigmoid(self.g_proj(x_conv)).view(B, T, self.num_heads, self.head_dim)

        # 3. Dual-Path Processing
        # Path A: Temporal (Rotors)
        t_steps = torch.arange(T, device=x.device).view(1, T, 1, 1).float()
        angles = t_steps * self.freqs.view(1, 1, self.num_heads, self.head_dim)
        rotors = torch.polar(torch.ones_like(angles), angles)
        path_a = v * rotors
       
        # Path B: Associative (Shared QK Causal Shift)
        k_shifted = torch.cat([torch.zeros_like(k[:, :1]), k[:, :-1]], dim=1)
        path_b = v * torch.conj(k_shifted)
       
        # 4. Mix & Accumulate
        # Mix logic: Head 0 (Static) uses Path B. Others might use Path A.
        # Ideally, let the model learn the mix via another gate, or sum them.
        # Simplest Production: Sum both, let Gate decide magnitude.
        state = (path_a + path_b) * g
       
        # Apply Decay
        gamma = self.gamma.view(1, 1, self.num_heads, 1)
        decay_mask = torch.pow(gamma, t_steps)
        memory = torch.cumsum(state * decay_mask, dim=1) / (decay_mask + 1e-6)
       
        # 5. Retrieve
        out = memory * q
       
        # 6. Output
        out = out.real.reshape(B, T, self.hd_dim)
        return self.out_norm(self.o_proj(out))




class HoloMLP(nn.Module): 
   def __init__(self, config): 
       super().__init__() 
       self.gate_proj = nn.Linear(
           config.d_model, 
           config.intermediate_size, 
           bias=False
       ) 
       
       self.up_proj = nn.Linear(
           config.d_model, 
           config.intermediate_size, 
           bias=False
       ) 
       
       self.down_proj = nn.Linear(
           config.intermediate_size, 
           config.d_model, 
           bias=False
       ) 
       self.act_fn = nn.SiLU() 
       self.dropout = nn.Dropout(config.resid_dropout)
       
   def forward(self, x): 
       x = self.act_fn(self.gate_proj(x)) * self.up_proj(x)
       x = self.down_proj(x)
       return self.dropout(x)
        

class HoloBlock(nn.Module):
    """
    Standard Transformer Block with LayerScale for deep signal propagation.
    """
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.d_model)
        # self.attn = HoloAttentionV2(config)
        if config.use_version == 1:
            self.attn = HoloAttentionV2(config)
        else:
            self.attn = LongAttention(config)
            
        self.ln2 = nn.LayerNorm(config.d_model)
        
        # self.mlp = nn.Sequential(
        #     nn.Linear(config.d_model, config.d_model * config.expansion_factor),
        #     nn.GELU(),
        #     nn.Linear(config.d_model * config.expansion_factor, config.d_model),     
        # )
        self.mlp = HoloMLP(config)

        
        # LayerScale: Initialize to small value (0.1) to ease optimization
        self.gamma1 = nn.Parameter(torch.ones(config.d_model) * 0.1)
        self.gamma2 = nn.Parameter(torch.ones(config.d_model) * 0.1)

    def forward(self, x):
        # Path 1: Attention + Residual Dropout (inside attn) + LayerScale
        x = x + self.gamma1 * self.attn(self.ln1(x))
        
        # Path 2: MLP + Residual Dropout (inside mlp) + LayerScale
        x = x + self.gamma2 * self.mlp(self.ln2(x))
        
        return x