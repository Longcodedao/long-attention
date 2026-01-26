from transformers import PretrainedConfig

class LongConfig(PretrainedConfig):
    model_type = "long_llm"

    def __init__(
        self,
        vocab_size=50304,
        hidden_size=768,
        num_hidden_layers=12,
        num_heads=12,
        expansion_ratio=4, 
        layer_norm_eps=1e-5,
        max_position_embeddings=2048, 
        conv_kernel=4,
        hybrid_ratio=0, # Set > 0 (e.g., 4) to enable Anchor Layers
        initializer_range=0.02,
        tie_word_embeddings=True,
        gate_init_bias = 1.0,
        use_cache = True
        **kwargs
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_heads = num_heads
        self.intermediate_size = hidden_size * expansion_ratio
        self.layer_norm_eps = layer_norm_eps
        self.max_position_embeddings = max_position_embeddings
        self.conv_kernel = conv_kernel
        self.hybrid_ratio = hybrid_ratio
        self.initializer_range = initializer_range
        self.tie_word_embeddings = tie_word_embeddings
        self.gate_init_bias = gate_init_bias
        self.use_cache = use_cache

def get_187m_config():
    return LongConfig(
        vocab_size=50304,
        hidden_size=768,
        num_hidden_layers=18, # <--- The key change
        num_heads=12
    )