from einops import rearrange
from .modeling_llama import LlamaMLP, LlamaRMSNorm
from torch import nn
from cirkit.symbolic.initializers import ConstantTensorInitializer
from cirkit.pipeline import PipelineContext
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.activations import ACT2FN
from transformers.modeling_utils import PreTrainedModel

import torch
import torch.nn.functional as F

# Llama utilities (import these from your codebase)

# cirkit imports
from cirkit.utils.scope import Scope
from cirkit.symbolic.circuit import Circuit
from cirkit.symbolic.layers import HadamardLayer, SumLayer
from cirkit.symbolic.parameters import ConstantParameter, Parameter, TensorParameter
from cirkit.templates.utils import (
    InputLayerFactory,
    Parameterization,
    name_to_input_layer_factory,
)

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

def _input_layer_factory_builder(input_layer: str, dim: int, param: Parameterization) -> InputLayerFactory:
    """Build an InputLayerFactory using a Parameterization object (wrapping a TensorParameter).
    The Parameterization should contain a TensorParameter that wraps a ConstantTensorInitializer.
    """
    kwargs = {"num_states": dim, "weight": Parameter.from_input(param)}
    return name_to_input_layer_factory(input_layer, **kwargs)


def _build_cp_circuit_from_weights(
    shape: tuple[int, ...],
    rank: int,
    weights: list[torch.Tensor],
    input_layer: str = "embedding",
) -> Circuit:
    """
    Build a CP circuit for an N-dimensional tensor decomposition.
    
    Args:
        shape: The full tensor shape (e.g. (num_heads, seq_len, head_dim)).
        rank: CP decomposition rank.
        weights: List of factor matrices, one per axis.
                 Each must be shaped [axis_len, rank].
        input_layer: Which input layer factory to use ("embedding" by default).
    
    Returns:
        A cirkit Circuit object (not compiled).
    """
    embedding_layer_factories = []
    bs, seq_len, hidden_size = shape
    shape = (seq_len, hidden_size)
    for i, (dim, w) in enumerate(zip(shape, weights, strict=False)):
        print("dim, w shape", dim, w.shape)
        if w.shape != (dim, rank):
            raise ValueError(f"Factor {i} has shape {w.shape}, but expected {(dim, rank)}.")
        tensor_param = TensorParameter(
            bs,
            dim,
            rank,
            initializer=ConstantTensorInitializer(w.detach().cpu().numpy()),
            learnable=True,
        )
        embedding_layer_factories.append(
            _input_layer_factory_builder(input_layer, dim, tensor_param)
        )

    embedding_layers = [f(Scope([i]), rank) for i, f in enumerate(embedding_layer_factories)]

    hadamard_layer = HadamardLayer(rank, arity=len(shape))

    sum_layer = SumLayer(
        rank,
        1,
        arity=1,
        weight=Parameter.from_input(ConstantParameter(1, rank, value=1.0)),
    )

    in_layers = {sum_layer: [hadamard_layer], hadamard_layer: embedding_layers}
    outputs = [sum_layer]

    circuit = Circuit(
        layers=embedding_layers + [hadamard_layer, sum_layer],
        in_layers=in_layers,
        outputs=outputs,
    )
    return circuit

class LlamaApproximatedAttention(nn.Module):
    def __init__(self, config: LlamaConfig, layer_idx: int | None = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            print(f"Instantiating {self.__class__.__name__} without `layer_idx` is not recommended.")

        self.hidden_size = config.hidden_size
        self.seq_length = config.seq_length
        self.factorization_rank = config.factorization_rank

        self.seq_factor_linear = nn.Linear(self.seq_length, self.factorization_rank, bias=config.attention_bias)
        self.hidden_factor_linear = nn.Linear(self.hidden_size, self.factorization_rank, bias=config.attention_bias)

        self.ctx = PipelineContext(backend="torch", semiring="sum-product", fold=True, optimize=True)
        self.compiled_circuit = None

    def forward(self, hidden_states: torch.Tensor, grid_chw: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.size()
        print(hidden_states.shape, hidden_states.dtype)
        x = rearrange(hidden_states, "b s h -> (b h) s")
        seq_factors = self.seq_factor_linear(x)
        y = rearrange(hidden_states, "b s h -> (b s) h")
        hidden_factors = self.hidden_factor_linear(y)
        shape = (batch_size, seq_len, self.hidden_size)
        if self.compiled_circuit is None:
            circuit = _build_cp_circuit_from_weights(
                shape,
                self.factorization_rank,
                [seq_factors, hidden_factors],
                input_layer="embedding"
            )
            self.compiled_circuit = self.ctx.compile(circuit)

        attn_output = self.compiled_circuit(grid_chw).squeeze(dim=2).squeeze(dim=1).view(batch_size, seq_len, self.hidden_size)

        return  attn_output


class LlamaDecoderLayer(nn.Module):

    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.attn_approx = LlamaApproximatedAttention(config=config, layer_idx=layer_idx)

        self.mlp = LlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        grid_chw: torch.Tensor
    ):
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        attn_out = self.attn_approx(hidden_states=hidden_states,grid_chw=grid_chw)

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
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`LlamaDecoderLayer`]

    Args:
        config: LlamaConfig
    """

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

        grid_chw = torch.stack(torch.meshgrid(
            torch.arange(seq_len, dtype=torch.long),
            torch.arange(hidden_size, dtype=torch.long),
            indexing="ij"
        ), dim=-1).reshape(-1, 2)

        all_hidden_states = () if output_hidden_states else None

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    grid_chw
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    grid_chw
                )

            hidden_states = layer_outputs[0]

            if self.layer_sharing:
                if output_hidden_states:
                    all_hidden_states += (hidden_states,)

                if self.gradient_checkpointing and self.training:
                    layer_outputs = self._gradient_checkpointing_func(
                        decoder_layer.__call__,
                        hidden_states,
                        grid_chw
                    )
                else:
                    layer_outputs = decoder_layer(
                        hidden_states,
                        grid_chw
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
