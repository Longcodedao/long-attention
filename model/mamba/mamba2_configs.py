"""
Standard configurations for Mamba-2 models.
"""

MAMBA2_PRESETS = {
    # ~130 Million Parameters (Matches GPT-2 Small)
    "small": {
        "num_hidden_layers": 24, 
        "hidden_size": 768,
        "state_size": 128,      # UPDATED: Mamba-2 uses larger state size
        "expand": 2,
        "conv_kernel": 4,
        "use_bias": False,
        "head_dim": 64,         # NEW: Required for Mamba-2
        "n_groups": 1           # NEW: 1 = Standard, 8 = Grouped Query equivalent
    },

    # ~370 Million Parameters (Matches GPT-2 Medium)
    "medium": {
        "num_hidden_layers": 48,
        "hidden_size": 1024,
        "state_size": 128,
        "expand": 2,
        "conv_kernel": 4,
        "use_bias": False,
        "head_dim": 64,
        "n_groups": 1
    },
    
    # ~790 Million Parameters (Matches GPT-2 Large)
    "large": {
        "num_hidden_layers": 48,
        "hidden_size": 1536,
        "state_size": 128,
        "expand": 2,
        "conv_kernel": 4,
        "use_bias": False,
        "head_dim": 64,
        "n_groups": 1
    },
    
    # ~1.4 Billion Parameters (Matches GPT-2 XL)
    "xl": {
        "num_hidden_layers": 48,
        "hidden_size": 2048,
        "state_size": 128,
        "expand": 2,
        "conv_kernel": 4,
        "use_bias": False,
        "head_dim": 64,
        "n_groups": 1
    },
    
    # ~2.8 Billion Parameters
    "2.8b": {
        "num_hidden_layers": 64,
        "hidden_size": 2560,
        "state_size": 128,
        "expand": 2,
        "conv_kernel": 4,
        "use_bias": False,
        "head_dim": 64,
        "n_groups": 8 # Often 2.8B+ models use n_groups=8 for efficiency
    }
}

def get_mamba2_config_dict(size_name):
    """Returns the configuration dictionary for a specific size."""
    size_name = size_name.lower()
    if size_name not in MAMBA2_PRESETS:
        raise ValueError(f"Unknown Mamba size: '{size_name}'. Available: {list(MAMBA2_PRESETS.keys())}")
    
    return MAMBA2_PRESETS[size_name]