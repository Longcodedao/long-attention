# Holo-Transformer: $O(N)$ Long-Context Reasoning

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

**Holo-Transformer** is a hybrid Large Language Model architecture that achieves linear $O(N)$ complexity and $O(1)$ inference memory without sacrificing the reasoning capabilities of standard Transformers.

It replaces the quadratic Self-Attention matrix with **LongAttention**, a novel primitive based on **Holographic Associative Memory**. By mapping tokens to complex-valued phasors and performing algebraic binding ($Key = Content \odot Position$), it solves "Needle-in-a-Haystack" and "Reverse Sequence" tasks that baffle standard State Space Models (SSMs).

## 🚀 Key Features

* **Linear Scaling:** Training scales linearly with sequence length. Inference cache is constant size (16KB for 8k dim), regardless of context length.
* **Reasoning-Ready:** Unlike standard RNNs/SSMs, LongAttention features a **Dynamic Query Head** that solves associative recall and position-dependent reasoning tasks (e.g., Sorting, Reversal) with 100% accuracy.
* **Hardware Efficient:** Built on standard PyTorch primitives (`cumsum`, `complex_mul`). Runs natively on GPU, TPU, and Apple Silicon.
* **Leakage-Free:** Implements **Random Fourier Positional Embeddings** to ensure orthogonality and prevent neighbor signal leakage common in RoPE-based linear models.
* **Hybrid Design:** Interleaves LongAttention with sparse FlashAttention layers (e.g., every 8 layers) for maximum robustness.

## 📦 Installation

```bash
git clone [https://github.com/LONGDANG-72/long-attention.git](https://github.com/LONGDANG-72/long-attention.git)
cd long-attention
pip install -e .

⚡ Quick Start
Holo-Transformer is designed to be a drop-in replacement for standard Causal LM backends.

Important: We strongly recommend using BFloat16 (torch.bfloat16) for training. The library handles internal mixed-precision stability (casting to FP32 for the linear scan) automatically.

import torch
from long_attention import HoloTransformer, HoloConfig

# 1. Configuration (350M Scale Example)
config = HoloConfig(
    vocab_size=50257,    # GPT-2/Llama tokenizer
    dim=1024,            # Model Width
    n_layers=24,         # Depth
    n_heads=16,          # For the sparse FlashAttention layers
    hd_dim=8192,         # Holographic Dimension (Higher = Better Recall)
    max_seq_len=131072   # Context Window
)

# 2. Initialize Model (Use CUDA and BF16)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = HoloTransformer(config).to(device).to(torch.bfloat16)

print(f"Model Parameters: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")

# 3. Forward Pass (Long Context)
# Input: [Batch, SeqLen]
x = torch.randint(0, 50257, (1, 8192)).to(device)

with torch.no_grad():
    logits = model(x) # Output: [1, 8192, 50257]

print("Forward pass successful.")
