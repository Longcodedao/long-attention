import torch
import os
from transformers import (
    GPT2Config, 
    GPT2LMHeadModel, 
    AutoTokenizer, 
    MambaConfig,        
    MambaForCausalLM,    
    Mamba2Config, 
    Mamba2ForCausalLM
)

# Adjust these imports if your folder structure is slightly different
from model.holo.configuration_holo import HoloConfig
from model.holo.modeling_holo import HoloForCausalLM 
from model.long import LongConfig, LongForCausalLM 
from model.gpt2.gpt2_configs import get_gpt2_config_dict
from model.mamba.mamba_configs import get_mamba_config_dict
from model.mamba.mamba2_configs import get_mamba2_config_dict
from transformers import AutoConfig

def get_model_and_tokenizer(model_type, 
                            model_size, 
                            vocab_size=None, # Changed to None to allow dynamic detection
                            seq_len=2048, 
                            device="cpu"):
    """
    Factory function to initialize a NEW model.
    """
    torch.set_float32_matmul_precision('high')
    
    # 1. Initialize Tokenizer first
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token 

    # 2. Sync Vocab Size
    # If vocab_size is not provided, use the tokenizer's actual count.
    # We use len(tokenizer) to ensure the embedding matrix handles all possible tokens.
    if vocab_size is None:
        vocab_size = len(tokenizer)
    
    print(f"[Model Loader] Identified Vocab Size: {vocab_size}")

    model = None
    config = None 

    # ==========================
    # 1. GPT-2
    # ==========================
    if model_type.lower() == "gpt2":
        presets = get_gpt2_config_dict(model_size)
        config = GPT2Config(
            vocab_size=vocab_size,
            n_positions=seq_len,
            n_embd=presets["n_embd"],
            n_layer=presets["n_layer"],
            n_head=presets["n_head"],
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        model = GPT2LMHeadModel(config)

    # ==========================
    # 2. Mamba (v1)
    # ==========================
    elif model_type.lower() == "mamba":
        preset = get_mamba_config_dict(model_size)
        config = MambaConfig(
            vocab_size=vocab_size,
            hidden_size=preset["hidden_size"],
            num_hidden_layers=preset["num_hidden_layers"],
            state_size=preset["state_size"],
            expand=preset["expand"],
            conv_kernel=preset["conv_kernel"],
            use_bias=preset["use_bias"],
            pad_token_id=tokenizer.eos_token_id,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        model = MambaForCausalLM(config)

    # ==========================
    # 3. Mamba (v2)
    # ==========================
    elif model_type.lower() == "mamba2":
        print(f"[Model Loader] Initializing Mamba2 ({model_size})...")
        preset = get_mamba2_config_dict(model_size) 
        
        hidden_size = preset["hidden_size"]
        expand = preset.get("expand", 2)
        head_dim = preset.get("head_dim", 64)
        expanded_dim = hidden_size * expand
        num_heads = expanded_dim // head_dim
        
        config = Mamba2Config(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_hidden_layers=preset["num_hidden_layers"],
            state_size=preset.get("state_size", 128),
            expand=expand,
            conv_kernel=preset.get("conv_kernel", 4),
            n_groups=preset.get("n_groups", 1),
            head_dim=head_dim,
            num_heads=num_heads,
            pad_token_id=tokenizer.eos_token_id,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        model = Mamba2ForCausalLM(config)
        
    # ==========================
    # 4. Holo
    # ==========================
    elif model_type.lower() == "holo":
        config = HoloConfig.from_preset(model_size, use_version=2)
        config.vocab_size = vocab_size 
        config.max_position_embeddings = seq_len
        model = HoloForCausalLM(config)

    # ==========================
    # 5. Long LLM
    # ==========================
    elif model_type.lower() == "long":
        print(f"[Model Loader] Initializing Long LLM ({model_size})...")
        config = LongConfig(
            vocab_size=vocab_size,
            max_position_embeddings=seq_len,
            hidden_size = 768,
            num_hidden_layers = 12,
            num_heads = 12,
            hybrid_ratio = 4,
            gate_init_bias = -3.0
        )

        if model_size == "small" or model_size == "187m":
            config.hidden_size = 768
            config.num_hidden_layers = 18 
            config.num_heads = 12
        elif model_size == "medium":
            config.hidden_size = 1024
            config.num_hidden_layers = 24
            config.num_heads = 16
        elif model_size == "tiny":
            config.hidden_size = 512
            config.num_hidden_layers = 6
            config.num_heads = 8

        model = LongForCausalLM(config)
            
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    model.to(device)

    if model_type.lower() == "long":
        # Try max-autotune for getting the fastest in training
        model = torch.compile(model, mode = "default")        
        
    return model, tokenizer

# ... rest of your load_model_from_path remains the same ...


def load_model_from_path(model_path, model_type=None, device="cpu"):
    """
    Loads a saved model and tokenizer from a checkpoint directory.
    
    Args:
        model_path (str): Path to the checkpoint folder (must contain config.json and pytorch_model.bin/safetensors).
        model_type (str, optional): Explicitly enforce a model architecture ('long', 'holo', 'mamba', 'gpt2').
                                    If None, attempts to infer it from config.json.
        device (str): Device to load the model onto ('cpu', 'cuda', etc.).
    """
    print(f"[Model Loader] Loading model from: {model_path}")

    # 1. Load Tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    except Exception as e:
        print(f"[Model Loader] Warning: Could not load tokenizer from path ({e}). Defaulting to GPT2.")
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. Determine Model Type
    # If the user didn't provide a type, try to find it in the config file
    if model_type is None:
        try:
            config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
            model_type = getattr(config, "model_type", "").lower()
        except Exception:
            print("[Model Loader] Warning: Could not read config.json to detect model type.")
            model_type = ""
    else:
        model_type = model_type.lower()

    print(f"[Model Loader] Architecture set to: {model_type}")

    # 3. Dispatch to the correct Model Class
    model = None

    if model_type == "long":
        model = LongForCausalLM.from_pretrained(model_path)

    elif model_type == "holo":
        model = HoloForCausalLM.from_pretrained(model_path)

    elif model_type == "mamba":
        model = MambaForCausalLM.from_pretrained(model_path)
    
    elif model_type == "mamba2":
        model = Mamba2ForCausalLM.from_pretrained(model_path)

    elif model_type == "gpt2":
        model = GPT2LMHeadModel.from_pretrained(model_path)

    else:
        # Fallback for standard HF models (Llama, Mistral, etc.)
        print(f"[Model Loader] Unrecognized custom type '{model_type}'. Attempting AutoModelForCausalLM...")
        model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)

    # 4. Move to Device
    model.to(device)
    
    return model, tokenizer