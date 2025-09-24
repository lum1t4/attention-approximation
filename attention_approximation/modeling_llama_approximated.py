from .modeling_llama import LlamaMLP, LlamaRMSNorm
from torch import nn
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.activations import ACT2FN
from transformers.modeling_utils import PreTrainedModel
from torch.nn import CrossEntropyLoss
import torch.nn.init as init
import torch
import torch.nn.functional as F


class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

class LlamaMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=config.mlp_bias)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        if self.config.pretraining_tp > 1:
            slice = self.intermediate_size // self.config.pretraining_tp
            gate_proj_slices = self.gate_proj.weight.split(slice, dim=0)
            up_proj_slices = self.up_proj.weight.split(slice, dim=0)
            down_proj_slices = self.down_proj.weight.split(slice, dim=1)

            gate_proj = torch.cat(
                [F.linear(x, gate_proj_slices[i]) for i in range(self.config.pretraining_tp)], dim=-1
            )
            up_proj = torch.cat([F.linear(x, up_proj_slices[i]) for i in range(self.config.pretraining_tp)], dim=-1)

            intermediate_states = (self.act_fn(gate_proj) * up_proj).split(slice, dim=2)
            down_proj = [
                F.linear(intermediate_states[i], down_proj_slices[i]) for i in range(self.config.pretraining_tp)
            ]
            down_proj = sum(down_proj)
        else:
            down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

        return down_proj


class CP(nn.Module):
    def __init__(self, rank: int, out_units: int = 1):
        super().__init__()
        self.rank = int(rank)
        self.out_units = int(out_units)
        self.weight = nn.Parameter(torch.ones(self.out_units, self.rank), requires_grad=False)

    def forward(self, hadamard: torch.Tensor) -> torch.Tensor:
        return hadamard @ self.weight.t()


class CPCircuitLayer(nn.Module):
    def __init__(self, config: LlamaConfig, chunk_size: int = 1_000):
        super().__init__()
        self.out_units = 1
        self.rank = config.factorization_rank
        self.chunk_size = chunk_size

        self.seq_mode_factor = nn.Linear(
            config.hidden_size, config.factorization_rank, bias=config.attention_bias
        )
        # shape: [hidden_size, rank]
        self.hidden_embeddings = nn.Parameter(
            torch.empty(config.hidden_size, config.factorization_rank)
            )
        #initilaize with a small Gaussian like Transformers embeddings
        init.normal_(self.hidden_embeddings, mean=0.0, std=0.02)

        self.cp = CP(rank=config.factorization_rank, out_units=self.out_units)

    def forward(self, hidden_states: torch.Tensor, all_indices: torch.Tensor) -> torch.Tensor:
        batch, seq_len, hidden_size = hidden_states.size()
        device = hidden_states.device

        # Move indices to correct device
        all_indices = all_indices.to(device)

        # seq_embeddings: [batch, seq_len, rank]
        seq_embeddings = self.seq_mode_factor(hidden_states)

        outputs = []
        for start in range(0, all_indices.size(0), self.chunk_size):
            end = min(start + self.chunk_size, all_indices.size(0))
            chunk = all_indices[start:end]  # [chunk_size, 2]

            seq_indices = chunk[:, 0].long()  # [chunk_size]
            hidden_indices = chunk[:, 1].long()  # [chunk_size]

            # Gather sequence embeddings for batch
            seq_emb = seq_embeddings[:, seq_indices]  # [batch, chunk_size, rank]

            # Gather hidden embeddings (shared across batch)
            hidden_emb = self.hidden_embeddings[hidden_indices]  # [chunk_size, rank]
            hidden_emb = hidden_emb.unsqueeze(0).expand(batch, -1, -1)  # [batch, chunk_size, rank]

            # Hadamard product
            hadamard = seq_emb * hidden_emb  # [batch, chunk_size, rank]

            # CP projection
            out_chunk = self.cp(hadamard)  # [batch, chunk_size, out_units]
            outputs.append(out_chunk)

        out = torch.cat(outputs, dim=1)  # [batch, total_size, out_units]

        return out.view(batch, seq_len, hidden_size, self.out_units).squeeze(3)


class LlamaApproximatedAttention(nn.Module):
    def __init__(self, config: LlamaConfig, layer_idx: int | None = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            print(f"Instantiating {self.__class__.__name__} without `layer_idx` is not recommended.")

        self.cp_circuit = CPCircuitLayer(config=config, chunk_size=10_000)

    def forward(self, hidden_states: torch.Tensor, all_indices: torch.Tensor) -> torch.Tensor:
        attn_output = self.cp_circuit(hidden_states, all_indices)
        return  attn_output


class LlamaDecoderLayer(nn.Module):

    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__()
        self.self_attn = LlamaApproximatedAttention(config=config, layer_idx=layer_idx)

        self.mlp = LlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        all_indices: torch.Tensor
    ):
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        attn_out = self.self_attn(hidden_states=hidden_states, all_indices=all_indices)

        hidden_states = residual + attn_out

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        return outputs

class LlamaPreTrainedModel(PreTrainedModel):
    config_class = LlamaConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["LlamaDecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_cache_class = True
    _supports_quantized_cache = True
    _supports_static_cache = True

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


class LlamaModel(LlamaPreTrainedModel):
    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [LlamaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.gradient_checkpointing = False

        self.layer_sharing = config.layer_sharing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        inputs_embeds: torch.FloatTensor | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
    ) -> tuple | BaseModelOutputWithPast:

        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        hidden_states = inputs_embeds

        _, seq_len, hidden_size = hidden_states.size()
        device = hidden_states.device

        grid_y, grid_x = torch.meshgrid(
            torch.arange(seq_len, dtype=torch.long, device=device),
            torch.arange(hidden_size, dtype=torch.long, device=device),
            indexing="ij"
        )
        all_indices = torch.stack([grid_y, grid_x], dim=-1).view(-1, 2)

        all_hidden_states = () if output_hidden_states else None

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    all_indices
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    all_indices
                )

            hidden_states = layer_outputs[0]

            if self.layer_sharing:
                if output_hidden_states:
                    all_hidden_states += (hidden_states,)

                if self.gradient_checkpointing and self.training:
                    layer_outputs = self._gradient_checkpointing_func(
                        decoder_layer.__call__,
                        hidden_states,
                        all_indices
                    )
                else:
                    layer_outputs = decoder_layer(
                        hidden_states,
                        all_indices
                    )

                hidden_states = layer_outputs[0]

        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states
        )


class LlamaForCausalLM(LlamaPreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = LlamaModel(config)
        self.vocab_size = config.vocab_size
        if not getattr(self.config, "share_embedding", False):
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return (
            self.lm_head
            if not getattr(self.config, "share_embedding", False)
            else self.get_input_embeddings()
        )

    def set_output_embeddings(self, new_embeddings):
        if not getattr(self.config, "share_embedding", False):
            self.lm_head = new_embeddings
        else:
            self.set_input_embeddings(new_embeddings)

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: list[torch.FloatTensor] | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        cache_position: torch.LongTensor | None = None,
    ) -> tuple | CausalLMOutputWithPast:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        if self.config.pretraining_tp > 1:
            lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
            logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
            logits = torch.cat(logits, dim=-1)
        else:
            if not getattr(self.config, "share_embedding", False):
                logits = self.lm_head(hidden_states)
            else:
                logits = F.linear(hidden_states, self.model.embed_tokens.weight)
        logits = logits.float()

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states if hasattr(outputs, 'hidden_states') else None,
        )
