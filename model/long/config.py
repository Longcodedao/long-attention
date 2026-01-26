from transformers import PretrainedConfig

class LongConfig(PretrainedConfig):
    model_type = "long_llm"

    def __init__(
        self,
        vocab_size=50304,
        hidden_size=768,
        num_hidden_layers=12,
        num_heads=12,
        expansion_ratio=8/3, # SwiGLU standard (approx 2.67)
        layer_norm_eps=1e-5,
        max_position_embeddings=2048, 
        conv_kernel=4,
        hybrid_ratio=0, # Set > 0 (e.g., 4) to enable Anchor Layers
        initializer_range=0.02,
        tie_word_embeddings=True,
        gate_init_bias=1.0,
        use_cache=True,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_heads = num_heads
        self.layer_norm_eps = layer_norm_eps
        self.max_position_embeddings = max_position_embeddings
        self.conv_kernel = conv_kernel
        self.hybrid_ratio = hybrid_ratio
        self.initializer_range = initializer_range
        self.tie_word_embeddings = tie_word_embeddings
        self.gate_init_bias = gate_init_bias
        self.use_cache = use_cache
        
        # --- GPU-Efficient Intermediate Size Calculation ---
        # 1. Calculate raw size: 768 * (8/3) = 2048
        raw_intermediate = int(hidden_size * expansion_ratio)
        
        # 2. Round to nearest multiple of 256
        # This aligns the matrix dimensions with GPU Tensor Cores/Tiles
        # Formula: ((x + multiple - 1) // multiple) * multiple
        multiple = 256
        self.intermediate_size = ((raw_intermediate + multiple - 1) // multiple) * multiple


def get_187m_config():
    """
    Returns a configuration roughly equivalent to a 'Small' model (~180M params).
    With SwiGLU, parameters are higher than standard GELU models per layer.
    """
    return LongConfig(
        vocab_size=50304,
        hidden_size=768,
        num_hidden_layers=18, # Increased depth
        num_heads=12,
        expansion_ratio=8/3,   # Ensures intermediate_size = 2048
        conv_kernel=4,
        hybrid_ratio=4         # Pure Linear Attention (fastest)
    )

# --- Verification ---
if __name__ == "__main__":
    conf = get_187m_config()
    print(f"Hidden Size: {conf.hidden_size}")
    print(f"Intermediate Size: {conf.intermediate_size}") 
    # Output should be 2048. 
    # If it was standard expansion=4, it would be 3072 (too heavy).