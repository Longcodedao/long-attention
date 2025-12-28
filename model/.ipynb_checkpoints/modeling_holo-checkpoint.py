import torch
import torch.nn as nn
from typing import Optional, Tuple, Union
from transformers import PreTrainedModel, AutoModelForCausalLM
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast

from .configuration_holo import HoloConfig
from .layers import HoloBlock


class HoloPreTrainedModel(PreTrainedModel):
    """
    Base class for Holo-Transformer weights initialization and utilities.
    """
    config_class = HoloConfig
    base_model_prefix = "holo"
    supports_gradient_checkpointing = True
    _no_split_modules = ["HoloBlock"]

    def _init_weights(self, module):
        """
        Standard GPT-style initialization.
        """
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

class HoloModel(HoloPreTrainedModel):
    """
    The bare Holo-Transformer backbone (Embeddings + Layers).
    """
    def __init__(self, config: HoloConfig):
        super().__init__(config)
        self.config = config

        # 1. Embeddings
        # Note: We do NOT use Positional Embeddings here. 
        # The Holographic Layer handles position via complex rotation internally.
        self.wte = nn.Embedding(config.vocab_size, config.d_model)
        self.drop = nn.Dropout(0.0) # Usually 0 for LLMs, but kept for interface

        # 2. The Stack
        self.h = nn.ModuleList([HoloBlock(config) for _ in range(config.num_hidden_layers)])
        
        # 3. Final Norm
        self.ln_f = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)

        self.post_init()

    def get_input_embeddings(self):
        return self.wte

    def set_input_embeddings(self, value):
        self.wte = value

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
        **kwargs # Catch-all for past_key_values if added later
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 1. Prepare Input
        if inputs_embeds is None:
            inputs_embeds = self.wte(input_ids)
        
        hidden_states = inputs_embeds
        hidden_states = self.drop(hidden_states)

        # 2. Run Layers
        all_hidden_states = () if return_dict else None
        
        for block in self.h:
            if return_dict:
                all_hidden_states = all_hidden_states + (hidden_states,)
            
            # The Magic happens here
            hidden_states = block(hidden_states)

        # 3. Finalize
        hidden_states = self.ln_f(hidden_states)

        if not return_dict:
            return (hidden_states,)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
        )

class HoloForCausalLM(HoloPreTrainedModel):
    """
    The End-to-End Language Model (Backbone + LM Head).
    Use this for training.
    """
    def __init__(self, config: HoloConfig):
        super().__init__(config)
        self.holo = HoloModel(config)
        
        # LM Head (Projects back to Vocab)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Weight Tying (Optional but standard)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.holo.wte.weight

        self.post_init()

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        labels: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
        **kwargs
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 1. Run Backbone
        outputs = self.holo(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            return_dict=return_dict,
            **kwargs
        )
        hidden_states = outputs[0]

        # 2. Compute Logits
        logits = self.lm_head(hidden_states)

        # 3. Compute Loss (if labels provided)
        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Flatten tokens
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
        )
