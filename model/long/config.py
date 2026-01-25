from transformers import PretrainedConfig

class LongConfig(PretrainedConfig):
    model_type = "long_llm"

    def __init__(
        self,
        vocab_size = 50304,
        hidden_size = 768,
        num_hidden_layers = 12,
        num_heads = 12,
        intermediate_size = 3072,
        layer_norm_eps = 1e-5,
        max_position_embeddings = 2048, 
        conv_kernel = 4,           # Local context window
        gate_bias_init = 1.0,      # Innovation: "Born Open" (+1.0)
        gate_act = "relu",         # Innovation: Hard Gating
        hybrid_ratio=4,          # 1 Standard Attn every 4 layers
        rope_theta=10000.0,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_heads = num_heads
        self.intermediate_size = intermediate_size
        self.layer_norm_eps = layer_norm_eps
        self.max_position_embeddings = max_position_embeddings
        
        self.conv_kernel = conv_kernel
        self.gate_bias_init = gate_bias_init
        self.gate_act = gate_act
        self.hybrid_ratio = hybrid_ratio
        self.rope_theta = rope_theta