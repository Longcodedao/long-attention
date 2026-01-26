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
        
        self._init_weights()

    def _init_weights(self):
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


class RoPESelfAttention(nn.Module):
    """Standard Attention Wrapper for Hybrid Anchors"""
    def __init__(self, config: LongConfig):
        super().__init__()
        self.attn = nn.MultiheadAttention(config.hidden_size, config.num_heads, batch_first=True)
        self.ln = nn.LayerNorm(config.hidden_size)

    def forward(self, x, state=None):
        B, T, C = x.shape
        mask = nn.Transformer.generate_square_subsequent_mask(T, device=x.device)
        out, _ = self.attn(x, x, x, attn_mask=mask, is_causal=False)
        return self.ln(out), None
