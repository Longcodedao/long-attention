import torch
import torch.nn as nn
from transformers import PreTrainedModel, GenerationMixin
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from .configuration_holo import HoloConfig
from .layers import HoloBlock
from transformers.modeling_utils import ModuleUtilsMixin


class HoloPreTrainedModel(PreTrainedModel):
    config_class = HoloConfig
    base_model_prefix = "holo"

    supports_gradient_checkpointing = True
    
    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None: module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

            
class HoloModel(HoloPreTrainedModel):
    def __init__(self, config: HoloConfig):
        super().__init__(config)
        self.wte = nn.Embedding(config.vocab_size, config.d_model)
        self.drop = nn.Dropout(config.dropout)
        
        self.h = nn.ModuleList([HoloBlock(config) for _ in range(config.num_hidden_layers)])
        self.ln_f = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)

        # Initialize checkpointing flag to False by default
        self.gradient_checkpointing = False
        self.post_init()

    def get_input_embeddings(self): 
        return self.wte
        
    def set_input_embeddings(self, value): 
        self.wte = value

    def forward(self, input_ids=None, inputs_embeds=None, past_key_values=None, use_cache=None, **kwargs):
        if inputs_embeds is None:
            inputs_embeds = self.wte(input_ids)
        hidden_states = self.drop(inputs_embeds)

        # Loop through blocks purely sequentially
        for i, block in enumerate(self.h):
            if self.gradient_checkpointing and self.training:
                
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)
                    return custom_forward

                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    use_reentrant=False
                )
            else:
                # Just pass hidden_states. No cache args.
                output = block(hidden_states)
                
                # Handle cases where block returns a tuple (hidden, something_else)
                if isinstance(output, tuple):
                    hidden_states = output[0]
                else:
                    hidden_states = output
        
        hidden_states = self.ln_f(hidden_states)
        
        # ALWAYS return None for past_key_values
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=None 
        )

class HoloForCausalLM(HoloPreTrainedModel, GenerationMixin):
    def __init__(self, config: HoloConfig):
        super().__init__(config)
        self.holo = HoloModel(config)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.holo.wte.weight
        self.post_init()

    def get_input_embeddings(self):
        return self.holo.wte
        
    def set_input_embeddings(self, value): 
        self.holo.wte = value
        
    def get_output_embeddings(self): 
        return self.lm_head

    # --- CRITICAL ADDITION 1: Prepare Inputs ---
    # This method is called by .generate() at every step.
    # It slices the input_ids so we only process the NEW token, not the whole history again.
    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, **kwargs):
        
        # Forward the arguments to the model
        return {
            "input_ids": input_ids,
            "past_key_values": None,
            "use_cache": False,
            "attention_mask": kwargs.get("attention_mask"),
        }
    
    # --- CRITICAL ADDITION 2: Updated Forward ---
    def forward(self, input_ids=None, past_key_values=None, attention_mask=None, labels=None, inputs_embeds=None, **kwargs):
        outputs = self.holo(
            input_ids=input_ids, 
            inputs_embeds=inputs_embeds, 
            # We don't pass past_key_values or use_cache to inner model anymore
            **kwargs
        )
        
        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)
        
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))
            
        return CausalLMOutputWithPast(
            loss=loss, 
            logits=logits, 
            past_key_values=None, # Always None
            hidden_states=outputs.last_hidden_state
        )