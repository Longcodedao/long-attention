"""
Standard configurations for Mamba models.
"""
"""
Standard configurations for Mamba (v1) models.
"""

MAMBA_PRESETS = {
    # ~130 Million Parameters (Matches GPT-2 Small)
    "small": {
        "num_hidden_layers": 24, 
        "hidden_size": 768,
        "state_size": 16,       # Standard for Mamba v1
        "expand": 2,            # Standard expansion factor
        "conv_kernel": 4,       # Standard local convolution width
        "use_bias": True        # Mamba v1 often uses bias (unlike v2)
    },

    # ~370 Million Parameters (Matches GPT-2 Medium)
    "medium": {
        "num_hidden_layers": 48,
        "hidden_size": 1024,
        "state_size": 16,
        "expand": 2,
        "conv_kernel": 4,
        "use_bias": True
    },
    
    # ~790 Million Parameters (Matches GPT-2 Large)
    "large": {
        "num_hidden_layers": 48,
        "hidden_size": 1536,
        "state_size": 16,
        "expand": 2,
        "conv_kernel": 4,
        "use_bias": True
    },
    
    # ~1.4 Billion Parameters (Matches GPT-2 XL)
    "xl": {
        "num_hidden_layers": 48,
        "hidden_size": 2048,
        "state_size": 16,
        "expand": 2,
        "conv_kernel": 4,
        "use_bias": True
    },
    
    # ~2.8 Billion Parameters
    "2.8b": {
        "num_hidden_layers": 64,
        "hidden_size": 2560,
        "state_size": 16,
        "expand": 2,
        "conv_kernel": 4,
        "use_bias": True
    }
}

def get_mamba_config_dict(size_name):
    """Returns the configuration dictionary for a specific size."""
    size_name = size_name.lower()
    
    # Corrected variable reference to MAMBA_PRESETS
    if size_name not in MAMBA_PRESETS:
        raise ValueError(f"Unknown Mamba size: '{size_name}'. Available: {list(MAMBA_PRESETS.keys())}")
    
    return MAMBA_PRESETS[size_name]
