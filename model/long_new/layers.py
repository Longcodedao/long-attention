import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel
from typing import Optional, Tuple, Union

from .config import LongConfig
from .ops.functional import parallel_scan, recurrent_scan

class LongAttention(nn.Module):
    def __init__(self, config: LongConfig):
        super().__init__()
        self.d_model = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = self.d_model // self.num_heads
        self.config = config
        
        # Projections
        self.q_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.k_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.v_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.o_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        
        # Gating & Conv
        self.conv = nn.Conv1d(self.d_model, self.d_model, config.conv_kernel, groups=self.d_model, padding=config.conv_kernel-1)
        self.gate_proj = nn.Linear(self.d_model, self.d_model, bias=True)
        self.ln = nn.LayerNorm(self.d_model)
        
        # Decay (Gamma) Init
        # Head 0 = 1.0 (Unitary/Static), Others = 0.9..0.999
        decays = torch.linspace(0.9, 0.999, self.num_heads)
        decays[0] = 1.0 
        self.gamma = nn.Parameter(decays.view(1, 1, self.num_heads, 1), requires_grad=False)
        
    def reset_parameters(self):
        # Born Open: +1.0 Bias
        nn.init.constant_(self.gate_proj.bias, self.config.gate_bias_init)

    def forward(self, x, state=None):
        B, T, C = x.shape
        
        # 1. Local Conv
        x_conv = self.conv(x.transpose(1, 2))[:, :, :T].transpose(1, 2)
        x_conv = F.silu(x_conv)
        
        # 2. Projections
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(B, T, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(B, T, self.num_heads, self.head_dim)
        
        # 3. Gating (ReLU vs Sigmoid)
        gate_logits = self.gate_proj(x_conv)
        if self.config.gate_act == 'relu':
            gate = F.relu(gate_logits).view(B, T, self.num_heads, self.head_dim)
        else:
            gate = F.sigmoid(gate_logits).view(B, T, self.num_heads, self.head_dim)
            
        k = F.normalize(k, p=2, dim=-1)
        q = F.normalize(q, p=2, dim=-1)
        
        # 4. Scan
        if state is not None:
            # Inference Mode
            mem, next_state = recurrent_scan(k, v, gate, self.gamma, state)
        else:
            # Training Mode
            mem = parallel_scan(k, v, gate, self.gamma)
            next_state = None 
            
        # 5. Output
        out = (mem * q).reshape(B, T, C)
        return self.ln(self.o_proj(out)), next_state


class RotaryEmbedding(nn.Module):
    def __init__(self, dim, base=10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, x, seq_len):
        t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb[None, :, None, :] # [1, T, 1, D]

def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin):
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)


class RoPESelfAttention(nn.Module):
    """Standard Attention with Rotary Positional Embeddings"""
    def __init__(self, config: LongConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = self.hidden_size // self.num_heads
        
        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        
        self.rotary_emb = RotaryEmbedding(self.head_dim)
        self.ln = nn.LayerNorm(self.hidden_size)

    def forward(self, x, state=None):
        B, T, C = x.shape
        
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(B, T, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(B, T, self.num_heads, self.head_dim)

        # Apply RoPE
        emb = self.rotary_emb(q, T)
        cos, sin = emb.cos(), emb.sin()
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # Scaled Dot-Product Attention
        # PyTorch 2.0+ efficient attention
        q = q.transpose(1, 2) # [B, H, T, D]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        attn_out = F.scaled_dot_product_attention(
            q, k, v, 
            is_causal=True
        )
        
        attn_out = attn_out.transpose(1, 2).reshape(B, T, C)
        return self.ln(self.o_proj(attn_out)), None

    def reset_parameters(self):
        """Called by HF _init_weights to ensure custom init"""
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.o_proj.weight)