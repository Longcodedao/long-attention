import torch
import torch.nn as nn
import math  # <--- Bắt buộc import math
from torch.utils.checkpoint import checkpoint
from .functional import (
    generate_random_phasors, 
    compute_rotors, 
    holo_bind_and_accumulate, 
    holo_retrieve
)

class HoloAttention(nn.Module):
    """
    The Holographic 'Attention' Mechanism.
    """
    def __init__(self, config):
        super().__init__()
        self.d_model = config.d_model
        self.hd_dim = config.hd_dim
        
        # 1. Projections
        self.k_proj = nn.Linear(config.d_model, config.hd_dim, bias=False)
        self.v_proj = nn.Linear(config.d_model, config.hd_dim, bias=False)
        
        # --- FIX 1: Input của o_proj phải là hd_dim (1024) ---
        # File cũ để là config.d_model -> Sẽ lỗi nếu hd_dim != d_model
        self.o_proj = nn.Linear(config.hd_dim, config.d_model, bias=False)
        # -----------------------------------------------------
        
        self.register_buffer("freqs", generate_random_phasors(config.hd_dim))

    def forward(self, x):
        B, T, C = x.shape
        
        k_real = self.k_proj(x)
        v_real = self.v_proj(x)
        v = v_real.to(torch.complex64)
        
        rotors = compute_rotors(T, self.freqs)
        
        memory_trace = holo_bind_and_accumulate(v, rotors)
        output_complex = holo_retrieve(memory_trace, rotors)
        
        # --- FIX 2: QUAN TRỌNG NHẤT CHO LOSS ---
        # Chia thêm cho sqrt(T) để giảm biên độ tín hiệu xuống mức an toàn.
        # Nếu không có dòng này, Loss sẽ nổ lên 60-70.
        scale_factor = 1.0 / math.sqrt(T)
        output_complex = output_complex * scale_factor
        # ---------------------------------------
        
        # Output từ holo_retrieve đã là .real, nhưng gọi lại cho chắc chắn
        return self.o_proj(output_complex.real)

class HoloBlock(nn.Module):
    """
    Standard Transformer Block structure with Gradient Checkpointing support.
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
        self.gradient_checkpointing = False

    def forward(self, x):
        if self.gradient_checkpointing and self.training:
            return checkpoint(self._forward_impl, x, use_reentrant=False)
        else:
            return self._forward_impl(x)

    def _forward_impl(self, x):
        residual = x
        x = self.ln1(x)
        x = residual + self.attn(x)
        
        residual = x
        x = self.ln2(x)
        x = residual + self.mlp(x)
        
        return x