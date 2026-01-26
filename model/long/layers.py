import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union

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
        # Note: We keep this for the weights, but we might use F.conv1d manually during gen
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
        self.grp_norm = nn.GroupNorm(self.num_heads, self.d_model)
        self.mem_norm = nn.GroupNorm(self.num_heads, self.d_model, eps=1e-5)

    def reset_parameters(self):
        # Init
        nn.init.constant_(self.input_gate_proj.bias, self.config.gate_init_bias) 
        # nn.init.constant_(self.gamma_proj.bias, self.config.gate_bias_init)       

    def forward(self, x, state=None):
        B, T, C = x.shape
        kernel_size = self.config.conv_kernel

        # --- 1. Local Convolution with State Management ---
        if state is not None:
            # Generation Mode: Retrieve (Recurrent State, Conv Cache)
            rnn_state, conv_cache = state
            
            # Construct the window: [Previous Tokens] + [Current Token]
            # conv_cache shape: [B, C, Kernel-1]
            x_t = x.transpose(1, 2) # [B, C, 1]
            conv_window = torch.cat([conv_cache, x_t], dim=2)
            
            # Manually run Conv1d without padding (valid mode)
            # This ensures we use exactly the cached history + current token
            x_conv = F.conv1d(conv_window, self.conv.weight, self.conv.bias, groups=self.d_model)
            x_conv = x_conv.transpose(1, 2) # [B, 1, C]
            x_conv = F.silu(x_conv)
            
            # Update Conv Cache: Keep the last (Kernel-1) tokens
            # If Kernel=3, we keep last 2.
            if kernel_size > 1:
                new_conv_cache = conv_window[:, :, 1:]
            else:
                new_conv_cache = conv_cache # No cache needed for kernel 1
        else:
            # Parallel Mode (Prompt Processing)
            # Use standard layer with padding
            x_conv_full = self.conv(x.transpose(1, 2))
            x_conv = x_conv_full[:, :, :T].transpose(1, 2) # Slice off right padding
            x_conv = F.silu(x_conv)
            
            # Initialize States for the return
            rnn_state = None 
            # Create Conv Cache from the end of this sequence for the NEXT step
            if kernel_size > 1:
                # Take last (K-1) tokens
                new_conv_cache = x.transpose(1, 2)[:, :, -(kernel_size - 1):]
                # Handle edge case where prompt is shorter than kernel
                if new_conv_cache.shape[2] < (kernel_size - 1):
                    pad = (kernel_size - 1) - new_conv_cache.shape[2]
                    new_conv_cache = F.pad(new_conv_cache, (pad, 0))
            else:
                new_conv_cache = torch.empty(B, C, 0, device=x.device)

        # --- 2. Projections ---
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(B, T, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(B, T, self.num_heads, self.head_dim)

        # Stability
        q = F.normalize(q, p=2, dim=-1)
        k = F.normalize(k, p=2, dim=-1)

        # --- 3. Gating (Driven by x_conv) ---
        input_gate = F.sigmoid(self.input_gate_proj(x_conv)).view(B, T, self.num_heads, self.head_dim)
        gamma = F.sigmoid(self.gamma_proj(x_conv)).view(B, T, self.num_heads, 1)

        # --- 4. Scan ---
        if rnn_state is None:
            mem = chunked_parallel_scan(k, v, input_gate, gamma)
            next_rnn_state = mem[:, -1].detach().clone() 
        else:
            mem, next_rnn_state = recurrent_scan(k, v, input_gate, gamma, rnn_state)

        # --- 5. Output Norm ---
        mem_flat = mem.reshape(B * T, self.d_model) 
        mem_norm = self.mem_norm(mem_flat)
        mem = mem_norm.view(B, T, self.num_heads, self.head_dim)

        out = (mem * q).reshape(B, T, C)

        # Flatten T-> B for GroupNorm
        out_flat = out.reshape(B * T, C)
        out_norm = self.grp_norm(out_flat)
        out = out_norm.view(B, T, C)
        
        output_gate = F.sigmoid(self.output_gate_proj(x_conv))
        out = out * output_gate
        
        # Pack state as tuple
        next_state = (next_rnn_state, new_conv_cache)
        
        return self.o_proj(out), next_state


class AnchorAttention(nn.Module):
    """
    Fixed Anchor Attention with KV Caching for Generation.
    """
    def __init__(self, config: LongConfig):
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.hidden_size // config.num_heads
        
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        
        # Removed internal LayerNorm to support Pre-Norm arch in Model

    def forward(self, x, state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None):
        B, T, C = x.shape
        
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # === KV Cache Logic ===
        if state is not None:
            prev_k, prev_v = state
            # Concatenate along time dimension (dim=2)
            k = torch.cat([prev_k, k], dim=2)
            v = torch.cat([prev_v, v], dim=2)
        
        # Save new state
        next_state = (k, v)

        # Standard Flash Attention
        # Note: causal mask is handled automatically by is_causal=True 
        # but only if query length > 1 or we are training. 
        # For generation with cache, we strictly attend to all past.
        
        # If caching, q is len 1, k/v are len T_past + 1. 
        # is_causal=True works if inputs are aligned, but explicit dropout/mask might be needed 
        # for complex cases. For simplicity here:
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True if state is None else False)
        
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.o_proj(out), next_state
        

class LongMLP(nn.Module):
    def __init__(self, config: LongConfig):
        super().__init__()
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))



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
        is_anchor = (config.hybrid_ratio > 0) and ((layer_idx + 1) % config.hybrid_ratio == 0)
        
        if is_anchor:
            self.attn = AnchorAttention(config)
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
        
        # --- Attention Sub-block ---
        residual = hidden_states
        hidden_states = self.ln1(hidden_states)
        
        attn_output, next_state = self.attn(hidden_states, state=past_key_value)
        hidden_states = residual + attn_output

        # --- MLP Sub-block ---
        residual = hidden_states
        hidden_states = self.ln2(hidden_states)
        hidden_states = residual + self.mlp(hidden_states)

        return hidden_states, next_state