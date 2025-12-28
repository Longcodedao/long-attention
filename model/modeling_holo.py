
import torch
import torch.nn as nn
from typing import Optional, Tuple, Union
from transformers import PreTrainedModel
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from .configuration_holo import HoloConfig
from .layers import HoloBlock

class HoloPreTrainedModel(PreTrainedModel):
    config_class = HoloConfig
    base_model_prefix = "holo"
    supports_gradient_checkpointing = True
    _no_split_modules = ["HoloBlock"]

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

class HoloModel(HoloPreTrainedModel):
    def __init__(self, config: HoloConfig):
        super().__init__(config)
        self.config = config
        self.wte = nn.Embedding(config.vocab_size, config.d_model)
        self.drop = nn.Dropout(0.0) 
        self.h = nn.ModuleList([HoloBlock(config) for _ in range(config.num_hidden_layers)])
        self.ln_f = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.post_init()

    def get_input_embeddings(self): return self.wte
    def set_input_embeddings(self, value): self.wte = value

    def forward(self, input_ids=None, inputs_embeds=None, return_dict=None, **kwargs):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if inputs_embeds is None: inputs_embeds = self.wte(input_ids)
        hidden_states = self.drop(inputs_embeds)
        all_hidden_states = () if return_dict else None
        for block in self.h:
            if return_dict: all_hidden_states = all_hidden_states + (hidden_states,)
            hidden_states = block(hidden_states)
        hidden_states = self.ln_f(hidden_states)
        if not return_dict: return (hidden_states,)
        return BaseModelOutputWithPast(last_hidden_state=hidden_states, hidden_states=all_hidden_states)

class HoloForCausalLM(HoloPreTrainedModel):
    def __init__(self, config: HoloConfig):
        super().__init__(config)
        self.holo = HoloModel(config)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        if config.tie_word_embeddings: self.lm_head.weight = self.holo.wte.weight
        self.post_init()

    def get_output_embeddings(self): return self.lm_head
    def set_output_embeddings(self, new_embeddings): self.lm_head = new_embeddings

    def forward(self, input_ids=None, labels=None, inputs_embeds=None, return_dict=None, **kwargs):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.holo(input_ids=input_ids, inputs_embeds=inputs_embeds, return_dict=return_dict, **kwargs)
        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))
        if not return_dict: return ((loss,) + (logits,) + outputs[1:]) if loss is not None else (logits,) + outputs[1:]
        return CausalLMOutputWithPast(loss=loss, logits=logits, hidden_states=outputs.hidden_states)
