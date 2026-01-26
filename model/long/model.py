import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import LongAttention, AnchorAttention, LongMLP, LongBlock
from .config import LongConfig

from transformers import PreTrainedModel, PretrainedConfig, GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast, BaseModelOutputWithPast

from typing import List, Optional, Tuple, Union


class LongPreTrainedModel(PreTrainedModel):
    config_class = LongConfig
    base_model_prefix = "long_model"
    _no_split_modules = ["LongBlock"]
    supports_gradient_checkpointing = True

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)

        if hasattr(module, "reset_parameters"):
            module.reset_parameters()

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, LongModel):
            module.gradient_checkpointing = value

    
class LongModel(LongPreTrainedModel):
    def __init__(self, config: LongConfig):
        super().__init__(config)
        self.config = config
        self.vocab_size = config.vocab_size
        self.hidden_size = config.hidden_size
        self.gradient_checkpointing = False  # Initialize flag
        
        self.wte = nn.Embedding(config.vocab_size, config.hidden_size)
        # Use ModuleList of Blocks
        self.layers = nn.ModuleList([
            LongBlock(config, i) for i in range(config.num_hidden_layers)
        ])
        
        
        self.ln_f = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        self.post_init()

    # Since we use the recurrent scan in Generation,
    # by initialize it to zeros, we ensure the model 
    # starts with a "clean state" memory
    def get_initial_state(self, batch_size: int, device: torch.device):
        past_key_values = []
        
        for i, layer in enumerate(self.layers):
            # 1. MLP State (Same for all layers)
            # Just the previous token embedding [B, 1, C]
            mlp_state = torch.zeros(batch_size, 1, self.config.hidden_size, device=device)

            # 2. Attention State (Depends on layer type)
            if layer.is_anchor:
                # --- Anchor Layer (Standard Attention) ---
                # Initialize empty KV Cache
                # Dimensions: [B, H, 0, D] (Length 0 initially)
                k_cache = torch.empty(
                    batch_size, self.config.num_heads, 0, layer.attn.head_dim, device=device
                )
                v_cache = torch.empty(
                    batch_size, self.config.num_heads, 0, layer.attn.head_dim, device=device
                )
                attn_state = (k_cache, v_cache)
            else:
                # --- Long Layer (Linear Attention) ---
                # Recurrent State [B, H, D, D]
                rnn_state = torch.zeros(
                    batch_size, self.config.num_heads, layer.attn.head_dim, layer.attn.head_dim, device=device
                )
                # Conv Cache [B, C, K-1]
                conv_cache = torch.zeros(
                    batch_size, self.config.hidden_size, self.config.conv_kernel - 1, device=device
                )
                attn_state = (rnn_state, conv_cache)

            # Pack them together
            past_key_values.append((attn_state, mlp_state))

        return past_key_values
        
    def forward(
        self, 
        input_ids: torch.LongTensor, 
        past_key_values: Optional[List[torch.Tensor]] = None
    ):
        
        x = self.wte(input_ids)
        next_cache = []

        # If past_key_values is None, create a list of Nones for convenience
        # It will uses the parallel scan of the prompt at first then 
        # generate it in sequence
        if past_key_values is None:
            past_key_values = [None] * len(self.layers)

        for i, layer in enumerate(self.layers):
            past_state = past_key_values[i]

            # --- Gradient Checkpointing Logic ---
            if self.gradient_checkpointing and self.training:
                # Checkpointing requires the inputs to have requires_grad=True 
                # (usually 'x' does).
                # We define a custom lambda/function to handle the args.
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, past_state, False) # use_cache=False during training
                    return custom_forward

                # Run checkpoint
                # Note: 'use_reentrant=False' is generally recommended for newer PyTorch
                x, layer_next_state = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer),
                    x,
                    use_reentrant=False 
                )
            else:
                # Standard Forward
                x, layer_next_state = layer(x, past_key_value=past_state)

            # Save Cache
            next_cache.append(layer_next_state)

        x = self.ln_f(x)

        return BaseModelOutputWithPast(
            last_hidden_state=x,
            past_key_values=next_cache
        )

    

class LongForCausalLM(LongPreTrainedModel, GenerationMixin):
    _tie_weights_key = ["lm_head.weight"]

    def __init__(self, config: LongConfig):
        super().__init__(config)
        self.long_model = LongModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def get_input_embeddings(self):
        return self.long_model.wte

    def set_input_embeddings(self, value):
        self.long_model.wte = value

    def get_output_embeddings(self):
        return self.lm_head


    def get_initial_state(self, batch_size: int, device: Optional[torch.device] = None):
        if device is None:
            device = self.lm_head.weight.device

        return self.long_model.get_initial_state(batch_size, device)
    
    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def forward(
        self, 
        input_ids: torch.LongTensor, 
        past_key_values: Optional[List[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]]] = None, 
        labels: Optional[torch.LongTensor] = None,
        **kwargs
    ):
        outputs = self.long_model(
            input_ids=input_ids,
            past_key_values=past_key_values
        )

        hidden_states = outputs.last_hidden_state
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss(ignore_index = -100)
            loss = loss_fct(
                shift_logits.view(-1, self.config.vocab_size), 
                shift_labels.view(-1)
            )

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=hidden_states
        )

    def prepare_inputs_for_generation(
        self, 
        input_ids, 
        past_key_values=None, 
        attention_mask=None,
        **kwargs
    ):
        # 1. If we have a past state, only feed the last token to the model 
        # The model will use recurrent_scan for O(1) inference
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]

        # 2. If it's the very first step and no state exists, 
        # we can choose to pre-initialize it or let the first forward 
        # pass run in "Parallel Mode" to fill the cache.
        #
        # Note: In your LongAttention code, if past_key_values is None, 
        # it runs parallel_scan. This is usually what you want for the 
        # initial prompt, as it is much faster than recurrent steps.
        
        return {
            "input_ids": input_ids, 
            "past_key_values": past_key_values,
            "attention+mask": attention_mask,
            **kwargs
        }