import torch
import os
from transformers import (
    GPT2Config, 
    GPT2LMHeadModel, 
    AutoTokenizer, 
    MambaConfig,        # Added missing import
    MambaForCausalLM    # Added missing import
)

# Adjust these imports if your folder structure is slightly different
from model.holo.configuration_holo import HoloConfig
from model.holo.modeling_holo import HoloForCausalLM 
from model.gpt2.gpt2_configs import get_gpt2_config_dict
from model.mamba.mamba_configs import get_mamba_config_dict

def get_model_and_tokenizer(model_type, 
                            model_size, 
                            vocab_size=50257, 
                            seq_len=1024, 
                            device="cpu"):
    """
    Factory function to initialize a NEW (untrained) model based on type and size.
    Used for Training from scratch.
    """
    
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token 

    model = None
    config = None 

    if model_type.lower() == "gpt2":
        print(f"[Model Loader] Initializing Custom GPT-2 ({model_size})...")
        presets = get_gpt2_config_dict(model_size)
        
        config = GPT2Config(
            vocab_size=vocab_size,
            n_positions=seq_len,
            n_embd=presets["n_embd"],
            n_layer=presets["n_layer"],
            n_head=presets["n_head"],
            bos_token_id=50256,
            eos_token_id=50256,
        )
        model = GPT2LMHeadModel(config)

    elif model_type.lower() == "mamba":
        print(f"[Model Loader] Initializing Mamba ({model_size})...")
        
        # Fetch preset
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
        
    elif model_type.lower() == "holo":
        print(f"[Model Loader] Initializing Holo ({model_size})...")
        config = HoloConfig.from_preset(model_size, use_version=2)
        # Ensure config matches training args
        config.vocab_size = vocab_size 
        config.max_position_embeddings = seq_len
        
        model = HoloForCausalLM(config)
            
    else:
        raise ValueError(f"Unknown model_type: {model_type}. Choose 'gpt2', 'mamba', or 'holo'.")

    model.to(device)
    return model, tokenizer
    

def load_model_from_path(model_type, model_path, device="cpu"):
    """
    Loads a PRE-TRAINED model and tokenizer from a checkpoint directory.
    Used for Evaluation or Resuming.
    """
    print(f"[Model Loader] Loading {model_type.upper()} from {model_path}...")

    # 1. Try to load Tokenizer from path, fallback to base gpt2
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    except Exception:
        print(f"[Model Loader] Tokenizer not found in {model_path}, using default 'gpt2'.")
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
    
    tokenizer.pad_token = tokenizer.eos_token

    # 2. Load Model based on type
    if model_type.lower() == "holo":
        try:
            # Holo usually needs its custom config loaded explicitly first
            config = HoloConfig.from_pretrained(model_path)
            model = HoloForCausalLM.from_pretrained(model_path, config=config)
        except Exception as e:
            raise ValueError(f"Failed to load Holo model from {model_path}. Error: {e}")

    elif model_type.lower() == "gpt2":
        try:
            model = GPT2LMHeadModel.from_pretrained(model_path)
        except Exception as e:
             raise ValueError(f"Failed to load GPT-2 model from {model_path}. Error: {e}")

    elif model_type.lower() == "mamba":
        try:
            model = MambaForCausalLM.from_pretrained(model_path)
        except Exception as e:
             raise ValueError(f"Failed to load Mamba model from {model_path}. Error: {e}")
             
    else:
        raise ValueError(f"Unknown model_type: {model_type}. Choose 'gpt2', 'mamba', or 'holo'.")

    model.to(device)
    return model, tokenizer