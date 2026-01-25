import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union

from .config import LongConfig
from ops import recurrent_scan, chunked_parallel_scan

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
        self.conv = nn.Conv1d(self.d_model, 
                              self.d_model, 
                              config.conv_kernel,
                              groups=self.d_model,
                              padding=config.conv_kernel - 1 )
        self.gate_proj = nn.Linear(self.d_model, self.d_model, bias=True)
        self.ln = nn.LayerNorm(self.d_model)

        # Decay (Gamma) Init
        # Head 0 = 1.0 (Unitary/Static), Others = 0.9..0.999
        decays = torch.linspace(0.9, 0.999, self.num_heads)
        decays[0] = 1.0 
        # self.gamma = nn.Parameter(
        #     decays.view(1, 1, self.num_heads, 1), 
        #     requires_grad=False
        # )
        # Register as buffer (saved in checkpoint, not trained)
        self.register_buffer("gamma", decays.view(1, 1, self.num_heads, 1))

        self._init_weights()

        
    def _init_weights(self):
        # Born Open: +1.0 Bias
        nn.init.constant_(self.gate_proj.bias, self.config.gate_bias_init)
        
        # Optional: Initialize projections nicely
        nn.init.normal_(self.q_proj.weight, std=0.02)
        nn.init.normal_(self.k_proj.weight, std=0.02)

    def forward(self, x, state = None):
        B, T, C = x.shape

        # 1. Local Conv
        x_conv = self.conv(x.transpose(1, 2))
        x_conv = x_conv[:, :, :T].transpose(1, 2)

        # 2. Projections
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(B, T, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(B, T, self.num_heads, self.head_dim)

        # 3. Gating
        gate_logits = self.gate_proj(x_conv)
        if getattr(self.config, 'gate_act', 'sigmoid') == 'relu':
            gate = F.relu(gate_logits).view(B, T, self.num_heads, self.head_dim)
        else:
            gate = F.sigmoid(gate_logits).view(B, T, self.num_heads, self.head_dim)

        
        # Normalization (Crucial for stability)
        k = F.normalize(k, p = 2, dim = -1)
        q = F.normalize(q, p = 2, dim = -1)

        # 4, State Space Model (The Scan)
        if state is not None:
            # --- Inference Mode (Recurrent) ---
            # Uses JIT compiled kernel for speed
            mem, next_state = recurrent_scan(k, v, gate, self.gamma, state)
        else:
            # Uses Chunked Parallel Scan for numerical stability on long sequences
            mem = chunked_parallel_scan(k, v, gate, self.gamma, chunk_size=128)

            # Return last state for potential stateful chaining
            next_state = mem[:, -1, :, :]

        # 5. Output Project
        out = (mem * q).reshape(B, T, C)
        out = self.o_proj(out)

        return self.ln(out), next_state