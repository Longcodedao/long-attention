import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union
import math

from .config import LongConfig
from .ops import recurrent_scan, chunked_parallel_scan, parallel_scan

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
        
        # Local Conv 
        self.conv = nn.Conv1d(self.d_model, self.d_model, 
                              kernel_size=config.conv_kernel, 
                              groups=self.d_model, 
                              padding=config.conv_kernel - 1)

        # Gates
        self.input_gate_proj = nn.Linear(self.d_model, self.d_model, bias=True)
        self.output_gate_proj = nn.Linear(self.d_model, self.d_model, bias=True)
        self.gamma_proj = nn.Linear(self.d_model, self.num_heads, bias=True)
        self.o_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        
        # --- STABILITY: LayerNorm for V ---
        self.v_norm = nn.LayerNorm(self.head_dim)
        
        # Output Norms
        self.grp_norm = nn.GroupNorm(self.num_heads, self.d_model)
        self.mem_norm = nn.GroupNorm(self.num_heads, self.d_model, eps=1e-5)

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.o_proj.weight)

        nn.init.constant_(self.input_gate_proj.bias, self.config.gate_init_bias) 

        # Gamma Initialization
        min_decay = 0.9
        max_decay = 0.9999
        target_decays = 1 - torch.exp(
            torch.linspace(
                math.log(1 - min_decay), 
                math.log(1 - max_decay), 
                self.num_heads
            )
        )
        gamma_bias_init = torch.log(target_decays / (1 - target_decays))
        
        nn.init.zeros_(self.gamma_proj.weight)
        with torch.no_grad():
            self.gamma_proj.bias.copy_(gamma_bias_init)
            
    def forward(self, x, state=None):
        B, T, C = x.shape
        kernel_size = self.config.conv_kernel

        # --- 1. Local Convolution ---
        if state is not None:
            rnn_state, conv_cache = state
            
            x_t = x.transpose(1, 2) 
            conv_window = torch.cat([conv_cache, x_t], dim=2)
            
            x_conv = F.conv1d(conv_window, self.conv.weight, self.conv.bias, groups=self.d_model)
            x_conv = x_conv.transpose(1, 2) 
            x_conv = F.silu(x_conv)
            
            if kernel_size > 1:
                new_conv_cache = conv_window[:, :, 1:]
            else:
                new_conv_cache = conv_cache 
        else:
            x_conv_full = self.conv(x.transpose(1, 2))
            x_conv = x_conv_full[:, :, :T].transpose(1, 2) 
            x_conv = F.silu(x_conv)
            
            rnn_state = None 
            if kernel_size > 1:
                new_conv_cache = x.transpose(1, 2)[:, :, -(kernel_size - 1):]
                if new_conv_cache.shape[2] < (kernel_size - 1):
                    pad = (kernel_size - 1) - new_conv_cache.shape[2]
                    new_conv_cache = F.pad(new_conv_cache, (pad, 0))
            else:
                new_conv_cache = torch.empty(B, C, 0, device=x.device)

        # --- 2. Projections ---
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(B, T, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(B, T, self.num_heads, self.head_dim)

        # --- STABILITY NORM ---
        q = F.normalize(q, p=2, dim=-1) # L2 Norm for Matching
        k = F.normalize(k, p=2, dim=-1) # L2 Norm for Matching
        v = self.v_norm(v)              # LayerNorm for Accumulation Stability

        # --- 3. Gating ---
        input_gate = F.sigmoid(self.input_gate_proj(x_conv)).view(B, T, self.num_heads, self.head_dim)
        gamma = F.sigmoid(self.gamma_proj(x_conv)).view(B, T, self.num_heads, 1)

        # --- 4. Scan ---
        if rnn_state is None:
            mem = chunked_parallel_scan(k, v, input_gate, gamma)
            next_rnn_state = mem[:, -1].detach().clone() 
        else:
            mem, next_rnn_state = recurrent_scan(k, v, input_gate, gamma, rnn_state)

        # --- 5. Output ---
        mem_flat = mem.reshape(B * T, self.d_model) 
        mem_norm = self.mem_norm(mem_flat)
        mem = mem_norm.view(B, T, self.num_heads, self.head_dim)

        out = (mem * q).reshape(B, T, C)

        out_flat = out.reshape(B * T, C)
        out_norm = self.grp_norm(out_flat)
        out = out_norm.view(B, T, C)
        
        output_gate = F.sigmoid(self.output_gate_proj(x_conv))
        out = out * output_gate
        
        next_state = (next_rnn_state, new_conv_cache)
        
        return self.o_proj(out), next_state


# --- UPDATED: PURE SWIGLU MLP ---
# Removed: Time Mixing / Token Shifting
# Result: Standard Gated MLP (LLaMA style)
class LongMLP(nn.Module):
    def __init__(self, config: LongConfig):
        super().__init__()
        self.d_model = config.hidden_size
        self.intermediate = config.intermediate_size

        # SwiGLU projections: Gate, Value, Output
        self.w_gate = nn.Linear(self.d_model, self.intermediate, bias=False)
        self.w_val  = nn.Linear(self.d_model, self.intermediate, bias=False)
        self.w_out  = nn.Linear(self.intermediate, self.d_model, bias=False)
        
    def forward(self, x, state = None):
        """
        Forward pass for SwiGLU MLP.
        Args:
            x: Input tensor [Batch, Time, Dim]
            state: Ignored (kept for compatibility with Block signature)
        """
        # 1. Gate calculation (SiLU activation)
        gate = F.silu(self.w_gate(x))
        
        # 2. Value calculation (Linear)
        val = self.w_val(x)
        
        # 3. Element-wise multiplication (Gating) -> Output projection
        out = self.w_out(gate * val)
        
        # We return None as state because this MLP is now stateless
        return out, None


class RotaryEmbedding(nn.Module):
    def __init__(self, dim, base=10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, x, seq_len):
        t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb[None, :, None, :] 

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
        
    def forward(self, x, state=None):
        B, T, C = x.shape
        
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(B, T, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(B, T, self.num_heads, self.head_dim)

        if state is not None:
            prev_k, prev_v = state
            total_seq_len = prev_k.shape[1] + T
            
            emb = self.rotary_emb(q, total_seq_len)
            emb = emb[:, -T:, :, :] 
            cos, sin = emb.cos(), emb.sin()
            
            q, k = apply_rotary_pos_emb(q, k, cos, sin)
            
            k = torch.cat([prev_k, k], dim=1)
            v = torch.cat([prev_v, v], dim=1)
            
            next_state = (k, v)
            k_in, v_in = k, v
        else:
            emb = self.rotary_emb(q, T)
            cos, sin = emb.cos(), emb.sin()
            q, k = apply_rotary_pos_emb(q, k, cos, sin)
            k_in, v_in = k, v
            next_state = None

        q = q.transpose(1, 2) 
        k = k_in.transpose(1, 2)
        v = v_in.transpose(1, 2)
        
        out = F.scaled_dot_product_attention(
            q, k, v, 
            is_causal=True if state is None else False 
        )
        
        out = out.transpose(1, 2).contiguous().reshape(B, T, C)
        return self.o_proj(out), next_state


class LongBlock(nn.Module):
    def __init__(self, config: LongConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        
        self.is_anchor = (config.hybrid_ratio > 0) and ((layer_idx + 1) % config.hybrid_ratio == 0)
        
        if self.is_anchor:
            self.attn = RoPESelfAttention(config) 
        else:
            self.attn = LongAttention(config)

        self.mlp = LongMLP(config)
        self.ln1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.ln2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(
            self,
            hidden_states: torch.Tensor,
            past_key_value: Optional[Tuple[torch.Tensor]] = None,
            use_cache: bool = False,
        ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor]]]:
        
        attn_state = None
        mlp_state = None

        if past_key_value is not None:
            attn_state, mlp_state = past_key_value

        residual = hidden_states
        hidden_states = self.ln1(hidden_states)
        
        attn_output, next_attn_state = self.attn(hidden_states, state=attn_state)
        hidden_states = residual + attn_output

        residual = hidden_states
        hidden_states = self.ln2(hidden_states)
        
        # NOTE: mlp_state will now be None since LongMLP is stateless
        mlp_output, next_mlp_state = self.mlp(hidden_states, state=mlp_state)
        hidden_states = residual + mlp_output

        if use_cache or past_key_value is not None:
            # We still return a tuple, but the second element (MLP state) is None
            next_state = (next_attn_state, next_mlp_state)
        else:
            next_state = None

        return hidden_states, next_state