import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union
import math

from .config import LongConfig
# Ensure you have these ops available in your .ops file
from .ops import recurrent_scan, chunked_parallel_scan

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
        # Causal padding is handled manually or via the 'padding' arg here depending on implementation
        # For standard causal conv1d in training: padding = kernel - 1
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
        
        self.grp_norm = nn.GroupNorm(self.num_heads, self.d_model)
        self.mem_norm = nn.LayerNorm(self.head_dim)

        self.reset_parameters()

    def reset_parameters(self):
        """
        Custom initialization logic.
        """
        # 1. Projections
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.o_proj.weight)

        # 2. Input Gate: Start CLOSED (Bias -3.0 or similar)
        # Prevents initial instability/explosive loss
        nn.init.zeros_(self.input_gate_proj.weight)
        nn.init.constant_(self.input_gate_proj.bias, self.config.gate_init_bias) 

        # Gamma Initialization (SSM style decay)
        with torch.no_grad():
            min_decay = 0.9
            max_decay = 0.9999
            target_decays = 1 - torch.exp(
                torch.linspace(
                    math.log(1 - min_decay), 
                    math.log(1 - max_decay), 
                    self.num_heads
                )
            )
            # Inverse sigmoid for bias init
            gamma_bias_init = torch.log(target_decays / (1 - target_decays))
            self.gamma_proj.bias.copy_(gamma_bias_init)
            nn.init.zeros_(self.gamma_proj.weight)

        # 3. Output Gate: Start NEUTRAL (Bias 0.0)
        nn.init.constant_(self.output_gate_proj.bias, 0.0)

    def forward(self, x, state=None):
        B, T, C = x.shape
        
        # --- 1. Local Convolution (Merged Logic) ---
        if state is not None:
            # Inference Mode: Use cache
            rnn_state, conv_cache = state
            
            x_t = x.transpose(1, 2).contiguous()
            # Concatenate history [old_cache, new_input]
            conv_window = torch.cat([conv_cache, x_t], dim=2)

            conv_window = conv_window.contiguous()
            # Apply convolution
            # We take the last T steps of the valid output
            # x_conv = self.conv(conv_window)[:, :, :T].transpose(1, 2).contiguous()
            x_conv = F.conv1d(
                conv_window, 
                self.conv.weight, 
                bias = self.conv.bias, 
                padding = 0, 
                groups = self.d_model
            )

            x_conv = x_conv[:, :, :T].transpose(1, 2).contiguous()
            
            # Update cache: keep the last (kernel_size - 1) elements
            if self.config.conv_kernel > 1:
                new_conv_cache = conv_window[:, :, 1:].contiguous()
            else:
                new_conv_cache = conv_cache # Should ideally be empty if kernel=1
        else:
            # ### FIX 1: Add .contiguous() here ###
            # This aligns memory layout so torch.compile doesn't crash on stride assertions
            x_input = x.transpose(1, 2).contiguous()
            
            # Training Mode: Full Sequence
            # Conv1d with padding=(k-1) results in shape L + k - 1. 
            # We slice [:, :, :T] to enforce causality (discard future leak).
            # x_conv = self.conv(x_input)[:, :, :T].transpose(1, 2).contiguous

            # FIX: Manual Padding allows torch.compile to trace layout correctly
            # Pad strictly on the left (causal) -> (Left=Kernel-1, Right=0)
            pad_amt = self.config.conv_kernel - 1
            x_padded = F.pad(x_input, (pad_amt, 0))

            x_conv = F.conv1d(
                x_padded, 
                self.conv.weight, 
                bias=self.conv.bias, 
                padding=0, 
                groups=self.d_model
            )
            x_conv = x_conv[:, :, :T].transpose(1, 2).contiguous()
            
            rnn_state = None
            
            # Prepare cache for next step if we were to switch to inference
            if self.config.conv_kernel > 1:
                # Cache the last k-1 tokens
                # Ensure conv_cache is contiguous for safety reasons
                new_conv_cache = x.transpose(1, 2)[:, :, -(self.config.conv_kernel-1):].contiguous()
            else:
                new_conv_cache = None

        x_conv = F.silu(x_conv)

        # --- 2. Projections & Stability ---
        # Normalize Q, K for stable matching (Cosine Attention style)
        # Normalize V for stable accumulation
        q = F.normalize(self.q_proj(x).view(B, T, self.num_heads, self.head_dim), p=2, dim=-1)
        k = F.normalize(self.k_proj(x).view(B, T, self.num_heads, self.head_dim), p=2, dim=-1)
        v = self.v_norm(self.v_proj(x).view(B, T, self.num_heads, self.head_dim))

        # --- 3. Gating ---
        i_gate = torch.sigmoid(self.input_gate_proj(x_conv)).view(B, T, self.num_heads, self.head_dim)
        gamma = torch.sigmoid(self.gamma_proj(x_conv)).view(B, T, self.num_heads, 1)

        
        # --- 4. Scan Logic (SSM Core) ---
        if rnn_state is None:
            # Parallel training
            mem = chunked_parallel_scan(k, v, i_gate, gamma)
            # Save the final state for potential continuation
            next_rnn_state = mem[:, -1].detach().clone()
        else:
            # Step-by-step inference
            mem, next_rnn_state = recurrent_scan(k, v, i_gate, gamma, rnn_state)

        
        # --- 5. Output Projection & Gating ---
        # Normalize memory state before combining with Query
        mem_out = self.mem_norm(mem)
        
        # Attention: Q * Memory
        out = (mem_out * q).reshape(B, T, C)
        
        # GroupNorm on the combined output
        out = self.grp_norm(out.reshape(B*T, C)).view(B, T, C)
        
        # Output gating
        out = out * torch.sigmoid(self.output_gate_proj(x_conv))
        
        return self.o_proj(out), (next_rnn_state, new_conv_cache)


class LongMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Standard SwiGLU Projections
        # Using config.intermediate_size is the standard HF naming convention
        self.w_gate = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.w_val  = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
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
        
        # Return None for the state, as this layer is stateless
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
        
        # Standard Flash/Scaled Dot Product Attention
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
        
        # Hybrid Attention Logic: Insert Standard Attention every 'ratio' layers
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

        # Attention Block
        residual = hidden_states
        hidden_states = self.ln1(hidden_states)
        
        attn_output, next_attn_state = self.attn(hidden_states, state=attn_state)
        hidden_states = residual + attn_output

        # MLP Block
        residual = hidden_states
        hidden_states = self.ln2(hidden_states)
        
        # Note: LongMLP is stateless, so next_mlp_state remains None
        mlp_output, next_mlp_state = self.mlp(hidden_states, state=mlp_state)
        hidden_states = residual + mlp_output

        if use_cache or past_key_value is not None:
            next_state = (next_attn_state, next_mlp_state)
        else:
            next_state = None

        return hidden_states, next_state