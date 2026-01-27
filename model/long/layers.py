import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union
import math

from .config import LongConfig
from .ops import recurrent_scan, chunked_parallel_scan, parallel_scan


class LongAttention(nn.Module):
    def __init__(self, config):
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
        
        # Norms
        self.v_norm = nn.LayerNorm(self.head_dim)
        self.grp_norm = nn.GroupNorm(self.num_heads, self.d_model)
        self.mem_norm = nn.LayerNorm(self.head_dim)

        self.reset_parameters()

    def reset_parameters(self):
        """
        Custom initialization logic.
        This must be called AFTER the standard model init to override defaults.
        """
        # 1. Projections (Standard Xavier/Kaiming is fine here)
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.o_proj.weight)

        # 2. Input Gate: Start CLOSED (Bias -3.0)
        # This is the most critical line for preventing Loss 300+
        nn.init.zeros_(self.input_gate_proj.weight)
        nn.init.constant_(self.input_gate_proj.bias, -3.0) 

        # 3. Output Gate: Start NEUTRAL (Bias 0.0)
        nn.init.constant_(self.output_gate_proj.bias, 0.0)

        # 4. Gamma: Geometric Decay
        # This ensures heads look at different time horizons
        with torch.no_grad():
            min_decay, max_decay = 0.9, 0.999
            target_decays = 1 - torch.exp(
                torch.linspace(
                    math.log(1 - min_decay), 
                    math.log(1 - max_decay), 
                    self.num_heads
                )
            )
            gamma_bias_init = torch.log(target_decays / (1 - target_decays))
            self.gamma_proj.bias.copy_(gamma_bias_init)
            nn.init.zeros_(self.gamma_proj.weight)

            

    def forward(self, x, state=None):
        B, T, C = x.shape
        # Convolutional State Handling
        if state is not None:
            rnn_state, conv_cache = state
            x_t = x.transpose(1, 2)
            conv_window = torch.cat([conv_cache, x_t], dim=2)
            x_conv = self.conv(conv_window)[:, :, :T].transpose(1, 2)
            new_conv_cache = conv_window[:, :, 1:] if self.config.conv_kernel > 1 else conv_cache
        else:
            x_conv = self.conv(x.transpose(1, 2))[:, :, :T].transpose(1, 2)
            rnn_state = None
            new_conv_cache = x.transpose(1, 2)[:, :, -(self.config.conv_kernel-1):] if self.config.conv_kernel > 1 else None

        x_conv = F.silu(x_conv)

        # Projections & Stability
        q = F.normalize(self.q_proj(x).view(B, T, self.num_heads, self.head_dim), p=2, dim=-1)
        k = F.normalize(self.k_proj(x).view(B, T, self.num_heads, self.head_dim), p=2, dim=-1)
        v = self.v_norm(self.v_proj(x).view(B, T, self.num_heads, self.head_dim))

        # Gating
        i_gate = torch.sigmoid(self.input_gate_proj(x_conv)).view(B, T, self.num_heads, self.head_dim)
        gamma = torch.sigmoid(self.gamma_proj(x_conv)).view(B, T, self.num_heads, 1)

        # Scan Logic (Externalized ops)
        if rnn_state is None:
            from .ops import chunked_parallel_scan
            mem = chunked_parallel_scan(k, v, i_gate, gamma)
            next_rnn_state = mem[:, -1].detach().clone()
        else:
            from .ops import recurrent_scan
            mem, next_rnn_state = recurrent_scan(k, v, i_gate, gamma, rnn_state)

        # Output Projection & Gating
        out = (self.mem_norm(mem) * q).reshape(B, T, C)
        out = self.grp_norm(out.reshape(B*T, C)).view(B, T, C)
        out = out * torch.sigmoid(self.output_gate_proj(x_conv))
        
        return self.o_proj(out), (next_rnn_state, new_conv_cache)



class LongMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Standard SwiGLU Projections
        # Gate path (for Swish activation)
        self.w_gate = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        # Value path (linear)
        self.w_val  = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        # Output projection
        self.w_out  = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x, state=None):
        """
        Standard Position-wise Feed Forward with SwiGLU.
        Args:
            x: Input tensor [Batch, Time, Hidden]
            state: Ignored (kept for compatibility with Block loop)
        """
        # SwiGLU Logic: Output = W_out ( SiLU(W_gate(x)) * W_val(x) )
        
        gate = F.silu(self.w_gate(x))
        val = self.w_val(x)
        
        # Element-wise multiplication of the gated path and value path
        out = self.w_out(gate * val)
        
        # Return None for the state, as this layer is now stateless
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
        
        self.q_proj = nn.Linear(self.hidden_size, 
                                self.hidden_size, 
                                bias=False)
        self.k_proj = nn.Linear(self.hidden_size, 
                                self.hidden_size, 
                                bias=False)
        self.v_proj = nn.Linear(self.hidden_size, 
                                self.hidden_size,
                                bias=False)
        self.o_proj = nn.Linear(self.hidden_size, 
                                self.hidden_size,
                                bias=False)
        
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


class LongBlock(nn.Module):
    """
    A single transformer block containing:
    Ln1 -> Attn -> Add -> Ln2 -> MLP -> Add
    """
    def __init__(self, config: LongConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        
        # 1. Determine Attention Type (Hybrid Logic)
        # layer_idx is 0-indexed here
        self.is_anchor = (config.hybrid_ratio > 0) and ((layer_idx + 1) % config.hybrid_ratio == 0)
        
        if self.is_anchor:
            self.attn = RoPESelfAttention(config)
        else:
            self.attn = LongAttention(config)

        # 2. MLP
        self.mlp = LongMLP(config)

        # 3. Norms (Pre-Norm architecture)
        self.ln1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.ln2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(
            self,
            hidden_states: torch.Tensor,
            past_key_value: Optional[Tuple[torch.Tensor]] = None,
            use_cache: bool = False,
        ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor]]]:
        
        # --- 0. State Unpacking ---
        # The incoming past_key_value is a Tuple: (Attention_State, MLP_State)
        attn_state = None
        mlp_state = None

        if past_key_value is not None:
            # Unpack the tuple automatically
            attn_state, mlp_state = past_key_value

        # --- 1. Attention Sub-block ---
        residual = hidden_states
        hidden_states = self.ln1(hidden_states)
        
        # We pass the specific attn_state.
        # If LongAttention: attn_state is (RNN_State, Conv_Cache)
        # If AnchorAttention: attn_state is (K_Cache, V_Cache)
        attn_output, next_attn_state = self.attn(hidden_states, state=attn_state)
        
        hidden_states = residual + attn_output

        # --- 2. MLP Sub-block ---
        residual = hidden_states
        hidden_states = self.ln2(hidden_states)
        
        # We pass the specific mlp_state (Last Token for shifting)
        mlp_output, next_mlp_state = self.mlp(hidden_states, state=mlp_state)
        
        hidden_states = residual + mlp_output

        # --- 3. State Packing ---
        # Bundle the new states together for the Model to store
        if use_cache or past_key_value is not None:
            next_state = (next_attn_state, next_mlp_state)
        else:
            next_state = None

        return hidden_states, next_state