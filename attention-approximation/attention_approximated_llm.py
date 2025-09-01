from .modeling_llama import LlamaMLP, LlamaRMSNorm
from torch import nn
from cirkit.symbolic.initializers import ConstantTensorInitializer
from cirkit.pipeline import PipelineContext
from transformers.models.llama.configuration_llama import LlamaConfig
from typing import Optional, Tuple, List, Union
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers.modeling_utils import PreTrainedModel

import torch
import torch.nn as nn
import torch.nn.functional as F

# Llama utilities (import these from your codebase)
from .modeling_llama import LlamaMLP, LlamaRMSNorm
from transformers.models.llama.configuration_llama import LlamaConfig

# cirkit imports
from cirkit.symbolic.initializers import ConstantTensorInitializer
from cirkit.pipeline import PipelineContext
from cirkit.utils.scope import Scope
from cirkit.symbolic.circuit import Circuit
from cirkit.symbolic.layers import HadamardLayer, SumLayer
from cirkit.symbolic.parameters import ConstantParameter, Parameter, TensorParameter
from cirkit.templates.utils import (
    InputLayerFactory,
    Parameterization,
    name_to_input_layer_factory,
)

import logging
_logger = logging.getLogger(__name__)


def _input_layer_factory_builder(input_layer: str, dim: int, param: Parameterization) -> InputLayerFactory:
    """Build an InputLayerFactory using a Parameterization object (wrapping a TensorParameter).
    The Parameterization should contain a TensorParameter that wraps a ConstantTensorInitializer.
    """
    kwargs = {"num_states": dim, "weight": Parameter.from_input(param)}
    return name_to_input_layer_factory(input_layer, **kwargs)


def _build_cp_circuit_from_weights(
    shape: tuple[int, ...],
    rank: int,
    weights: List[torch.Tensor],
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
    if len(shape) != len(weights):
        raise ValueError(
            f"Expected {len(shape)} factor matrices, but got {len(weights)}."
        )

    embedding_layer_factories = []
    for i, (dim, w) in enumerate(zip(shape, weights)):
        if w.shape != (dim, rank):
            raise ValueError(
                f"Factor {i} has shape {w.shape}, but expected {(dim, rank)}."
            )
        tensor_param = TensorParameter(
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


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

class LlamaApproximatedAttention(nn.Module):
    def __init__(self, config: LlamaConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            _logger.warning(f"Instantiating {self.__class__.__name__} without `layer_idx` is not recommended.")

        self.hidden_size = config.hidden_size
        self.attention_dropout = config.attention_dropout
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.factorization_rank = config.factorization_rank

        
        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )

        self.query_factor_linear = nn.Linear(self.hidden_size, self.factorization_rank, bias=config.attention_bias)
        self.key_factor_linear = nn.Linear(self.hidden_size, self.factorization_rank, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=config.attention_bias)

        self.ctx = PipelineContext(backend="torch", semiring="sum-product")
        self.compiled_ciruit = None

    def forward(self, hidden_states: torch.Tensor, 
                grid_chw: torch.Tensor, 
                attention_mask: Optional[torch.Tensor] = None, 
                output_attentions: bool = False
                ) -> torch.Tensor:
        
        batch_size, seq_len, _ = hidden_states.size()

        head_factors = torch.ones((self.num_heads, seq_len, self.factorization_rank), device=hidden_states.device, dtype=hidden_states.dtype)
        q_factors = self.query_factor_linear(hidden_states)
        k_factors = self.key_factor_linear(hidden_states)
        value_states = self.v_proj(hidden_states)

        value_states = repeat_kv(value_states, self.num_key_value_groups)

        shape = (seq_len, self.head_dim*self.num_heads)

        if self.compiled_circuit is None:
            circuit = _build_cp_circuit_from_weights(
                shape,
                self.factorization_rank,
                [head_factors, q_factors, k_factors],
                input_layer="embedding"
            )
            self.compiled_circuit = self.ctx.compile(circuit)
        
        attn_weights = self.compiled_circuit(grid_chw).squeeze(dim=2).squeeze(dim=1).view(batch_size, self.num_heads, seq_len, self.head_dim)

        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(value_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, : seq_len]
            attn_weights = attn_weights + causal_mask
        
        if attn_output.size() != (batch_size, self.num_heads, seq_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(batch_size, self.num_heads, seq_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(batch_size, seq_len, -1)
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return  attn_output, attn_weights


class LlamaDecoderLayer(nn.Module):
    """Decoder layer that uses the cirkit head approximator."""

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
        grid_chw: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool]= False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
    ):
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        attn_out = self.attn_approx(hidden_states=hidden_states,
                                    grid_chw=grid_chw,
                                    attention_mask=attention_mask,
                                    output_attentions=output_attentions)

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
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        return_legacy_cache = False
        if use_cache and not isinstance(past_key_values, Cache):  # kept for BC (non `Cache` `past_key_values` inputs)
            return_legacy_cache = True
            past_key_values = DynamicCache.from_legacy_cache(past_key_values)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        )

        # embed positions
        hidden_states = inputs_embeds

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    causal_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)
            
            # Repeat current layer if layer_sharing is enabled
            if self.layer_sharing:
                if output_hidden_states:
                    all_hidden_states += (hidden_states,)

                if self.gradient_checkpointing and self.training:
                    layer_outputs = self._gradient_checkpointing_func(
                        decoder_layer.__call__,
                        hidden_states,
                        causal_mask,
                        position_ids,
                        past_key_values,
                        output_attentions,
                        use_cache,
                        cache_position,
                    )
                else:
                    layer_outputs = decoder_layer(
                        hidden_states,
                        attention_mask=causal_mask,
                        position_ids=position_ids,
                        past_key_value=past_key_values,
                        output_attentions=output_attentions,
                        use_cache=use_cache,
                        cache_position=cache_position,
                    )

                hidden_states = layer_outputs[0]

                if use_cache:
                    next_decoder_cache = layer_outputs[2 if output_attentions else 1]

                if output_attentions:
                    all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if return_legacy_cache:
            next_cache = next_cache.to_legacy_cache()

        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

    def _update_causal_mask(
        self,
        attention_mask: torch.Tensor,
        input_tensor: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_values: Cache,
        output_attentions: bool,
    ):
        # TODO: As of torch==2.2.0, the `attention_mask` passed to the model in `generate` is 2D and of dynamic length even when the static
        # KV cache is used. This is an issue for torch.compile which then recaptures cudagraphs at each decode steps due to the dynamic shapes.
        # (`recording cudagraph tree for symint key 13`, etc.), which is VERY slow. A workaround is `@torch.compiler.disable`, but this prevents using
        # `fullgraph=True`. See more context in https://github.com/huggingface/transformers/pull/29114

        if self.config._attn_implementation == "flash_attention_2":
            if attention_mask is not None and 0.0 in attention_mask:
                return attention_mask
            return None

        # For SDPA, when possible, we will rely on its `is_causal` argument instead of its `attn_mask` argument, in
        # order to dispatch on Flash Attention 2. This feature is not compatible with static cache, as SDPA will fail
        # to infer the attention mask.
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        using_static_cache = isinstance(past_key_values, StaticCache)

        # When output attentions is True, sdpa implementation's forward method calls the eager implementation's forward
        if self.config._attn_implementation == "sdpa" and not using_static_cache and not output_attentions:
            if AttentionMaskConverter._ignore_causal_mask_sdpa(
                attention_mask,
                inputs_embeds=input_tensor,
                past_key_values_length=past_seen_tokens,
                is_training=self.training,
            ):
                return None

        dtype, device = input_tensor.dtype, input_tensor.device
        min_dtype = torch.finfo(dtype).min
        sequence_length = input_tensor.shape[1]
        if using_static_cache:
            target_length = past_key_values.get_max_length()
        else:
            target_length = (
                attention_mask.shape[-1]
                if isinstance(attention_mask, torch.Tensor)
                else past_seen_tokens + sequence_length + 1
            )

        if attention_mask is not None and attention_mask.dim() == 4:
            # in this case we assume that the mask comes already in inverted form and requires no inversion or slicing
            if attention_mask.max() != 0:
                raise ValueError("Custom 4D attention mask should be passed in inverted form with max==0`")
            causal_mask = attention_mask
        else:
            causal_mask = torch.full(
                (sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device
            )
            if sequence_length != 1:
                causal_mask = torch.triu(causal_mask, diagonal=1)
            causal_mask *= torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
            causal_mask = causal_mask[None, None, :, :].expand(input_tensor.shape[0], 1, -1, -1)
            if attention_mask is not None:
                causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
                mask_length = attention_mask.shape[-1]
                padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :]
                padding_mask = padding_mask == 0
                causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                    padding_mask, min_dtype
                )
        if (
            self.config._attn_implementation == "sdpa"
            and attention_mask is not None
            and attention_mask.device.type == "cuda"
            and not output_attentions
        ):
            # Attend to all tokens in fully masked rows in the causal_mask, for example the relevant first rows when
            # using left padding. This is required by F.scaled_dot_product_attention memory-efficient attention path.
            # Details: https://github.com/pytorch/pytorch/issues/110213
            causal_mask = AttentionMaskConverter._unmask_unattended(causal_mask, min_dtype)

        return causal_mask