from transformers import PretrainedConfig

class HoloConfig(PretrainedConfig):
    model_type = "holo"

    def __init__(
        self,
        vocab_size=50257,        # Default to GPT-2 tokenizer size
        d_model=768,         # This is 'd_model'
        hd_dim=None,             # The Holographic Bus (Key/Value expansion)
        num_heads=8,             # Multi-Head Default

        holo_expansion_ratio=8,  # Default to 8x capacity
        num_hidden_layers=12,    # Depth
        expansion_factor=4,      # MLP expansion (usually 4x hidden_size)
        max_position_embeddings=8192,
        layer_norm_eps=1e-5,
        initializer_range=0.02,
        use_version = 1,
        dropout = 0.0,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        phase_scale=3.0,
        tie_word_embeddings=False, # Whether to tie input/output embeddings
        **kwargs,
    ):
        """
        Configuration class for HoloGPT.
        
        Args:
            hd_dim (int): The dimension of the holographic binding space. 
                          Ideally 2x-4x larger than hidden_size to reduce 
                          superposition noise (crosstalk).
        """
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_heads = num_heads

        self.num_hidden_layers = num_hidden_layers
        self.expansion_factor = expansion_factor
        self.max_position_embeddings = max_position_embeddings
        
        self.layer_norm_eps = layer_norm_eps
        self.initializer_range = initializer_range
        self.tie_word_embeddings = tie_word_embeddings
        
        self.dropout = dropout
        self.holo_expansion_ratio = holo_expansion_ratio
        self.phase_scale = phase_scale

        if hd_dim is None:
            self.hd_dim = d_model * holo_expansion_ratio
        else:
            self.hd_dim = d_model
            
        if self.hd_dim % num_heads != 0:
            raise ValueError(f"hd_dim {self.hd_dim} must be divisible by num_heads {num_heads}")
            
        self.head_dim = self.hd_dim // num_heads
        self.use_version = use_version
        
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            **kwargs,
        )

    @classmethod
    def from_preset(cls, size="small", **kwargs):
        """
        Factory method for standard Holo sizes.
        """
        size = size.lower()
        
        # 1. Holo-Small (~125M Params) - Comparable to GPT-2 Small
        if size == "small":
            return cls(
                d_model=768,
                num_hidden_layers=12,
                num_heads=12,
                holo_expansion_ratio=4, # Reduced slightly to keep it lightweight
                **kwargs
            )
            
        # 2. Holo-Medium (~350M Params) - Comparable to GPT-2 Medium
        elif size == "medium":
            return cls(
                d_model=1024,
                num_hidden_layers=24,
                num_heads=16,
                holo_expansion_ratio=8, # Full holographic fidelity
                **kwargs
            )

        # 3. Holo-Large (~770M Params) - Comparable to GPT-2 Large
        elif size == "large":
            return cls(
                d_model=1280,
                num_hidden_layers=36,
                num_heads=20,
                holo_expansion_ratio=8, 
                **kwargs
            )
        else:
            raise ValueError("Size must be 'small', 'medium', or 'large'")