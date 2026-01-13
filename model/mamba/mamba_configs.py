"""
Standard configurations for Mamba-2 models.
Reference: https://huggingface.co/gpt2
"""

MAMBA2_PRESETS = {
    # ~130 Million Parameters (Matches GPT-2 Small)
    "small": {
        "num_hidden_layers": 24, 
        "hidden_size": 768,
        "state_size": 16,
        "expand": 2,
        "conv_kernel": 4,
        "use_bias": False
    },

    # ~370 Million Parameters (Matches GPT-2 Medium)
    "medium": {
        "num_hidden_layers": 48,
        "hidden_size": 1024,
        "state_size": 16,
        "expand": 2,
        "conv_kernel": 4,
        "use_bias": False
    },
    
    # ~790 Million Parameters (Matches GPT-2 Large)
    "large": {
        "num_hidden_layers": 48,
        "hidden_size": 1536,
        "state_size": 16,
        "expand": 2,
        "conv_kernel": 4,
        "use_bias": False
    },
    
    # ~1.4 Billion Parameters (Matches GPT-2 XL)
    "xl": {
        "num_hidden_layers": 48,
        "hidden_size": 2048,
        "state_size": 16,
        "expand": 2,
        "conv_kernel": 4,
        "use_bias": False
    },
    
    # ~2.8 Billion Parameters
    "2.8b": {
        "num_hidden_layers": 64,
        "hidden_size": 2560,
        "state_size": 16,
        "expand": 2,
        "conv_kernel": 4,
        "use_bias": False
    }
}

def get_mamba_config_dict(size_name):
    """Returns the configuration dictionary for a specific size."""
    size_name = size_name.lower()
    if size_name not in MAMBA2_PRESETS:
        raise ValueError(f"Unknown Mamba size: '{size_name}'. Available: {list(MAMBA2_PRESETS.keys())}")
    
    return MAMBA2_PRESETS[size_name]