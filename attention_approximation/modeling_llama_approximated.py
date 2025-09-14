from .modeling_llama import LlamaMLP, LlamaRMSNorm
from torch import nn
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.activations import ACT2FN
from transformers.modeling_utils import PreTrainedModel

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
    def __init__(self, config: LlamaConfig, chunk_size: int = 1000):
        super().__init__()
        self.out_units = 1
        self.chunk_size = chunk_size

        self.seq_mode_factor = nn.Linear(config.hidden_size, config.factorization_rank, bias=config.attention_bias),
        self.hidden_mode_factor = nn.Linear(config.seq_length, config.factorization_rank, bias=config.attention_bias)

        self.cp = CP(rank=config.factorization_rank, out_units=self.out_units)

    def forward(self, hidden_states: torch.Tensor, all_indices: torch.Tensor) -> torch.Tensor:
        batch, seq_len, hidden_size = hidden_states.size()
        num_modes = hidden_states.dim() - 1
        assert num_modes == 2, "This implementation only supports 2 modes (sequence and hidden)"

        embedding_weights = [
            self.seq_mode_factor(hidden_states),
            self.hidden_mode_factor(hidden_states.transpose(1, 2).contiguous())
        ]

        outputs = []
        for start in range(0, all_indices.size(0), self.chunk_size):
            end = start + self.chunk_size
            chunk = all_indices[start:end]
            chunk = chunk.unsqueeze(0).expand(batch, -1, -1)
            emb_list = []
            for mode_idx in range(num_modes):
                indices = chunk[:, :, mode_idx]
                w = embedding_weights[mode_idx]
                idx_expanded = indices.unsqueeze(-1).expand(-1, -1, self.rank)
                emb = torch.gather(w, dim=1, index=idx_expanded)
                emb_list.append(emb)

            stacked = torch.stack(emb_list, dim=0)
            hadamard = torch.prod(stacked, dim=0)

            out_chunk = self.cp(hadamard)
            outputs.append(out_chunk)

        out = torch.cat(outputs, dim=1)
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

        grid_y, grid_x = torch.meshgrid(
            torch.arange(seq_len, dtype=torch.long),
            torch.arange(hidden_size, dtype=torch.long),
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
