import torch
import torch.nn as nn
import torch.nn.functional as F
from .long_attention import LongAttention

# ==========================================
# 1. CONFIGURATION
# ==========================================
class HoloConfig:
    def __init__(self, 
                 vocab_size=50257, 
                 dim=2048, 
                 n_layers=24, 
                 n_heads=32, 
                 hd_dim=8192,
                 max_seq_len=131072): # Default to 128k context
        self.vocab_size = vocab_size
        self.dim = dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.hd_dim = hd_dim
        self.max_seq_len = max_seq_len
        self.ffn_dim = int(dim * 3.5) # SwiGLU width
        # Hybrid Strategy: Sparse Attention pattern (e.g., every 8 layers)
        self.attn_layers = [n_layers // 4, n_layers // 2, 3 * n_layers // 4]

# ==========================================
# 2. ROTARY EMBEDDINGS (RoPE)
# ==========================================
class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len=131072):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, x, seq_len=None):
        if seq_len is None: seq_len = x.shape[1]
        t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos(), emb.sin()

def apply_rope(x, cos, sin):
    # x: [B, T, H, D]
    # cos, sin: [T, D] -> Reshaped for broadcast
    cos = cos[:, :x.shape[1], :].unsqueeze(0).unsqueeze(2)
    sin = sin[:, :x.shape[1], :].unsqueeze(0).unsqueeze(2)
    x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
    return (x * cos) + (torch.cat((-x2, x1), dim=-1) * sin)

# ==========================================
# 3. FLASH ATTENTION (With RoPE)
# ==========================================
class FlashAttention(nn.Module):
    """Standard FlashAttention Wrapper with RoPE"""
    def __init__(self, config):
        super().__init__()
        self.n_heads = config.n_heads
        self.head_dim = config.dim // config.n_heads
        
        self.qkv = nn.Linear(config.dim, config.dim * 3, bias=False)
        self.wo = nn.Linear(config.dim, config.dim, bias=False)
        
        # RoPE is critical for long context in standard attention
        self.rope = RotaryEmbedding(self.head_dim, config.max_seq_len)

    def forward(self, x):
        B, T, C = x.shape
        # Project
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Apply RoPE (Rotary Positional Embeddings)
        cos, sin = self.rope(v, seq_len=T)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        
        # Flash Attention (PyTorch 2.0+ optimized)
        # is_causal=True ensures autoregressive masking
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        
        # Reshape Output
        out = out.transpose(1, 2).reshape(B, T, C)
        return self.wo(out)

# ==========================================
# 4. HOLO-BLOCK & TRANSFORMER
# ==========================================
class HoloBlock(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.dim)
        self.ln2 = nn.LayerNorm(config.dim)
        
        # SwiGLU FFN
        self.w1 = nn.Linear(config.dim, config.ffn_dim, bias=False)
        self.w2 = nn.Linear(config.ffn_dim, config.dim, bias=False)
        self.w3 = nn.Linear(config.dim, config.ffn_dim, bias=False)
        
        # Hybrid Selection
        if layer_idx in config.attn_layers:
            self.mixer = FlashAttention(config)
        else:
            # Pass max_seq_len to LongAttention for buffer sizing
            self.mixer = LongAttention(config.dim, config.hd_dim, config.max_seq_len)

    def forward(self, x):
        # Mixer (LongAttention or FlashAttention)
        x = x + self.mixer(self.ln1(x))
        # FFN (SwiGLU)
        x = x + self.w2(F.silu(self.w1(self.ln2(x))) * self.w3(self.ln2(x)))
        return x

class HoloTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.emb = nn.Embedding(config.vocab_size, config.dim)
        self.layers = nn.ModuleList([HoloBlock(config, i) for i in range(config.n_layers)])
        self.ln_f = nn.LayerNorm(config.dim)
        self.head = nn.Linear(config.dim, config.vocab_size, bias=False)
        
        # Weight Tying (Standard for LLMs)
        self.emb.weight = self.head.weight
        
        # Init
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x):
        x = self.emb(x)
        for layer in self.layers:
            x = layer(x)
        return self.head(self.ln_f(x))