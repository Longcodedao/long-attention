# long_attention/configuration_holo.py

from transformers import PretrainedConfig

class HoloConfig(PretrainedConfig):
    model_type = "holo"

    def __init__(
        self,
        vocab_size=50257,
        hidden_size=768,
        hd_dim=None,             # Changed: Default to None (Adaptive)
        holo_expansion_ratio=8,  # New: Default multiplier (8x is robust for long ctx)
        num_hidden_layers=12,
        expansion_factor=4,
        max_position_embeddings=8192,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.expansion_factor = expansion_factor
        self.max_position_embeddings = max_position_embeddings
        self.holo_expansion_ratio = holo_expansion_ratio
        
        # Adaptive Logic:
        # If hd_dim is not set, calculate it: hidden_size * ratio
        if hd_dim is None:
            self.hd_dim = hidden_size * holo_expansion_ratio
        else:
            self.hd_dim = hd_dim

        super().__init__(**kwargs)
