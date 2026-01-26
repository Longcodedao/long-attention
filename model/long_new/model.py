import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast
from typing import Optional, Tuple, List, Union
from .config import LongConfig
from .layers import LongAttention, RoPESelfAttention

class LongHFModel(PreTrainedModel):
    config_class = LongConfig
    _no_split_modules = ["LongAttention", "RoPESelfAttention"]

    def __init__(self, config):
        super().__init__(config)
        self.wte = nn.Embedding(config.vocab_size, config.hidden_size)
        
        # We group layers into a ModuleList for standard HF iteration
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                "attention": RoPESelfAttention(config) if (config.hybrid_ratio > 0 and (i + 1) % config.hybrid_ratio == 0) 
                             else LongAttention(config),
                "mlp": nn.Sequential(
                    nn.Linear(config.hidden_size, 4 * config.hidden_size),
                    nn.GELU(),
                    nn.Linear(4 * config.hidden_size, config.hidden_size)
                ),
                "ln": nn.LayerNorm(config.hidden_size)
            })
            for i in range(config.num_hidden_layers)
        ])
        
        self.ln_f = nn.LayerNorm(config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        labels: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        x = self.wte(input_ids)
        next_states = []
        state_idx = 0
        
        for i, layer_block in enumerate(self.layers):
            attn_layer = layer_block["attention"]
            mlp = layer_block["mlp"]
            ln = layer_block["ln"]

            # Handle state (past_key_values) for recurrent layers
            layer_state = None
            if past_key_values is not None and isinstance(attn_layer, LongAttention):
                layer_state = past_key_values[state_idx]

            # Forward pass through Attention
            h, next_s = attn_layer(x, state=layer_state)
            
            # Residual + MLP
            x = x + h
            x = x + mlp(ln(x))
            
            if isinstance(attn_layer, LongAttention):
                next_states.append(next_s)
                state_idx += 1
        
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))

        if not return_dict:
            return (loss, logits, next_states) if loss is not None else (logits, next_states)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=next_states,
            hidden_states=None,
            attentions=None,
        )

    # Required for HF generation compatibility
    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, **kwargs):
        if past_key_values:
            input_ids = input_ids[:, -1:]
        return {"input_ids": input_ids, "past_key_values": past_key_values}