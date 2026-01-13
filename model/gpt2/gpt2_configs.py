"""
Standard configurations for GPT-2 models.
Reference: https://huggingface.co/gpt2
"""

GPT2_PRESETS = {
    # ~124 Million Parameters
    "small": {
        "n_layer": 12,
        "n_head": 12,
        "n_embd": 768
    },
    
    # ~355 Million Parameters
    "medium": {
        "n_layer": 24,
        "n_head": 16,
        "n_embd": 1024
    },
    
    # ~774 Million Parameters
    "large": {
        "n_layer": 36,
        "n_head": 20,
        "n_embd": 1280
    },
    
    # ~1.5 Billion Parameters
    "xl": {
        "n_layer": 48,
        "n_head": 25,
        "n_embd": 1600
    },
    
    # Optional: DistilGPT2 (Lightweight)
    "distil": {
        "n_layer": 6,
        "n_head": 12,
        "n_embd": 768
    }
}

def get_gpt2_config_dict(size_name):
    """Returns the configuration dictionary for a specific size."""
    size_name = size_name.lower()
    if size_name not in GPT2_PRESETS:
        raise ValueError(f"Unknown GPT-2 size: '{size_name}'. Available: {list(GPT2_PRESETS.keys())}")
    
    return GPT2_PRESETS[size_name]