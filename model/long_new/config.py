from transformers import PretrainedConfig

class LongConfig(PretrainedConfig):
    model_type = "long_model"
    def __init__(
        self,
        vocab_size=1000,
        hidden_size=64,
        num_hidden_layers=2,
        num_heads=4,
        hybrid_ratio=2, # Every 2nd layer is RoPE
        conv_kernel=3,
        gate_bias_init=1.0,
        gate_act='relu',
        initializer_range=0.02,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_heads = num_heads
        self.hybrid_ratio = hybrid_ratio
        self.conv_kernel = conv_kernel
        self.gate_bias_init = gate_bias_init
        self.gate_act = gate_act
        self.initializer_range = initializer_range


        