"""
Holo-Transformer Library
========================

A PyTorch library for O(N) Long-Context Reasoning using Holographic Associative Memory.

Paper: "Holo-Transformer: A Hybrid Architecture for O(N) Long-Context Reasoning"
"""

__version__ = "1.0.0"

from .long_attention import LongAttention
from .hybrid_model import HoloTransformer, HoloConfig, FlashAttention

__all__ = [
    "LongAttention", 
    "HoloTransformer", 
    "HoloConfig",
    "FlashAttention"
]