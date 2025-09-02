# modeling_llama_approximated_full.py
from __future__ import annotations

import logging
from typing import Optional, Tuple, List, Union

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss

# cirkit imports (required for the approximator)
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

# transformers imports
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers.modeling_utils import PreTrainedModel

# Local imports (user's codebase). Replace with your actual implementations if different.
# For this snippet we assume `LlamaMLP` and `LlamaRMSNorm` are provided in modeling_llama.py
try:
    from .modeling_llama import LlamaMLP, LlamaRMSNorm
except Exception:
    # Minimal stand-ins so this module runs for test purposes if the real ones are not available.
    class LlamaMLP(nn.Module):
        def __init__(self, config):
            super().__init__()
            hidden = config.hidden_size
            self.fc1 = nn.Linear(hidden, hidden * 4)
            self.act = nn.GELU()
            self.fc2 = nn.Linear(hidden * 4, hidden)

        def forward(self, x):
            return self.fc2(self.act(self.fc1(x)))

    class LlamaRMSNorm(nn.Module):
        def __init__(self, hidden_size, eps=1e-6):
            super().__init__()
            self.eps = eps
            self.weight = nn.Parameter(torch.ones(hidden_size))

        def forward(self, x):
            var = x.pow(2).mean(-1, keepdim=True)
            x = x * torch.rsqrt(var + self.eps)
            return x * self.weight

# Optional Cache classes (try to import; if unavailable leave as None)
try:
    from .cache import Cache, DynamicCache, StaticCache
except Exception:
    Cache = None
    DynamicCache = None
    StaticCache = None


# ---------------------
# Logger with warning_once support
# ---------------------
class _LoggerWithOnce:
    def __init__(self, base: logging.Logger):
        self._base = base
        self._seen = set()

    def warning_once(self, msg, *args, **kwargs):
        key = (msg, args, tuple(sorted(kwargs.items())))
        if key in self._seen:
            return
        self._seen.add(key)
        self._base.warning(msg, *args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._base, name)


logger = _LoggerWithOnce(logging.getLogger(__name__))
_logger = logger


# ---------------------
# Utilities
# ---------------------
def _input_layer_factory_builder(input_layer: str, dim: int, param: Parameterization) -> InputLayerFactory:
    kwargs = {"num_states": dim, "weight": Parameter.from_input(param)}
    return name_to_input_layer_factory(input_layer, **kwargs)


def _build_cp_circuit_from_weights(
    shape: tuple[int, ...],
    rank: int,
    weights: List[torch.Tensor],
    input_layer: str = "embedding",
) -> Circuit:
    if len(shape) != len(weights):
        raise ValueError(f"Expected {len(shape)} factor matrices, but got {len(weights)}.")

    embedding_layer_factories = []
    for i, (dim, w) in enumerate(zip(shape, weights)):
        if w.shape != (dim, rank):
            raise ValueError(f"Factor {i} has shape {w.shape}, but expected {(dim, rank)}.")
        tensor_param = TensorParameter(
            dim,
            rank,
            initializer=ConstantTensorInitializer(w.detach().cpu().numpy()),
            learnable=True,
        )
        embedding_layer_factories.append(_input_layer_factory_builder(input_layer, dim, tensor_param))

    embedding_layers = [f(Scope([i]), rank) for i, f in enumerate(embedding_layer_factories)]
    hadamard_layer = HadamardLayer(rank, arity=len(shape))
    sum_layer = SumLayer(rank, 1, arity=1, weight=Parameter.from_input(ConstantParameter(1, rank, value=1.0)))
    in_layers = {sum_layer: [hadamard_layer], hadamard_layer: embedding_layers}
    outputs = [sum_layer]
    circuit = Circuit(layers=embedding_layers + [hadamard_layer, sum_layer], in_layers=in_layers, outputs=outputs)
    return circuit


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


# ---------------------
# LlamaApproximatedAttention (same implementation described earlier)
# ---------------------
class LlamaApproximatedAttention(nn.Module):
    def __init__(self, config: LlamaConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            _logger.warning(f"Instantiating {self.__class__.__name__} without `layer_idx` is not recommended.")

        self.hidden_size = config.hidden_size
        self.attention_dropout = getattr(config, "attention_dropout", 0.0)
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = getattr(config, "num_key_value_heads", self.num_heads)
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.factorization_rank = getattr(config, "factorization_rank", None)
        if self.factorization_rank is None:
            raise ValueError("config.factorization_rank must be set for LlamaApproximatedAttention")

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )

        self.query_factor_linear = nn.Linear(self.hidden_size, self.factorization_rank, bias=getattr(config, "attention_bias", False))
        self.key_factor_linear = nn.Linear(self.hidden_size, self.factorization_rank, bias=getattr(config, "attention_bias", False))

        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=getattr(config, "attention_bias", False))
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=getattr(config, "attention_bias", False))

        self.ctx = PipelineContext(backend="torch", semiring="sum-product")

        self._compiled_circuit_cache = {}
        self._grid_chw_cache = {}

    def _grid_key(self, seq_len: int, device: torch.device, dtype: torch.dtype):
        idx = device.index if getattr(device, "index", None) is not None else 0
        return (device.type, idx, str(dtype), seq_len)

    def _get_grid_chw(self, seq_len: int, device: torch.device, dtype: torch.dtype):
        key = self._grid_key(seq_len, device, dtype)
        grid = self._grid_chw_cache.get(key)
        if grid is not None:
            if grid.device != device or grid.dtype != dtype:
                grid = grid.to(device=device, dtype=dtype)
                self._grid_chw_cache[key] = grid
            return grid

        arange = torch.arange(seq_len, dtype=torch.long)
        g1, g2 = torch.meshgrid(arange, arange, indexing="ij")
        grid = torch.stack((g1.reshape(-1), g2.reshape(-1)), dim=-1).to(device=device, dtype=torch.long)
        self._grid_chw_cache[key] = grid
        return grid

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        past_values: Optional[torch.Tensor] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        batch_size, cur_seq_len, _ = hidden_states.size()

        q_factors = self.query_factor_linear(hidden_states)
        k_factors = self.key_factor_linear(hidden_states)
        v_states = self.v_proj(hidden_states)

        if past_values is not None:
            if past_values.device != v_states.device:
                past_values = past_values.to(v_states.device)
            v_states = torch.cat([past_values, v_states], dim=1)

        seq_len = v_states.shape[1]

        v_states_reshaped = v_states.view(batch_size, seq_len, self.num_key_value_heads, self.head_dim)
        value_states = repeat_kv(v_states_reshaped, self.num_key_value_groups)

        compiled_key = seq_len
        compiled_circuit = self._compiled_circuit_cache.get(compiled_key)

        head_factors = torch.ones((self.num_heads, seq_len, self.factorization_rank), device=hidden_states.device, dtype=hidden_states.dtype)

        if compiled_circuit is None:
            placeholder_q = torch.zeros((seq_len, self.factorization_rank), dtype=torch.float32)
            placeholder_k = torch.zeros((seq_len, self.factorization_rank), dtype=torch.float32)
            try:
                circuit = _build_cp_circuit_from_weights(
                    shape=(seq_len, seq_len),
                    rank=self.factorization_rank,
                    weights=[head_factors.mean(dim=0), placeholder_q, placeholder_k],
                    input_layer="embedding",
                )
                compiled_circuit = self.ctx.compile(circuit)
                self._compiled_circuit_cache[compiled_key] = compiled_circuit
            except Exception as e:
                _logger.debug("Template compile failed, compiling per-forward with actual factors: %s", e)
                q_for_compile = q_factors.detach().cpu().view(-1, self.factorization_rank)[:seq_len]
                k_for_compile = k_factors.detach().cpu().view(-1, self.factorization_rank)[:seq_len]
                circuit = _build_cp_circuit_from_weights(
                    shape=(seq_len, seq_len),
                    rank=self.factorization_rank,
                    weights=[head_factors.reshape(seq_len, -1)[:seq_len], q_for_compile, k_for_compile],
                    input_layer="embedding",
                )
                compiled_circuit = self.ctx.compile(circuit)
                self._compiled_circuit_cache[compiled_key] = compiled_circuit

        grid_chw = self._get_grid_chw(seq_len, device=v_states.device, dtype=v_states.dtype)

        attn_flat = compiled_circuit(grid_chw)

        attn_scores = None
        if attn_flat.dim() == 1:
            attn_scores = attn_flat.view(seq_len, seq_len).unsqueeze(0).unsqueeze(0).expand(batch_size, self.num_heads, seq_len, seq_len)
        elif attn_flat.dim() == 2:
            n0, n1 = attn_flat.shape
            if n1 == 1:
                attn_scores = attn_flat.view(seq_len, seq_len).unsqueeze(0).unsqueeze(0).expand(batch_size, self.num_heads, seq_len, seq_len)
            elif n1 == (batch_size * self.num_heads):
                attn_scores = attn_flat.view(seq_len, seq_len, batch_size, self.num_heads).permute(2, 3, 0, 1).contiguous()
            else:
                raise RuntimeError(f"Unexpected compiled circuit second dim: {n1}; expected 1 or batch*heads ({batch_size*self.num_heads})")
        elif attn_flat.dim() == 3:
            n0, n1, n2 = attn_flat.shape
            if (n1 == batch_size and n2 == self.num_heads) or (n1 == batch_size and n2 == 1):
                attn_scores = attn_flat.view(seq_len, seq_len, n1, n2).permute(2, 3, 0, 1).contiguous()
                if attn_scores.shape[1] == 1:
                    attn_scores = attn_scores.expand(batch_size, self.num_heads, seq_len, seq_len)
            else:
                squeezed = attn_flat.squeeze()
                if squeezed.numel() == seq_len * seq_len:
                    attn_scores = squeezed.view(seq_len, seq_len).unsqueeze(0).unsqueeze(0).expand(batch_size, self.num_heads, seq_len, seq_len)
                else:
                    raise RuntimeError("Unable to interpret compiled circuit output shape: %s" % (attn_flat.shape,))
        else:
            raise RuntimeError("Unable to interpret compiled circuit output shape: %s" % (attn_flat.shape,))

        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, : seq_len]
            attn_scores = attn_scores + causal_mask

        attn_probs = nn.functional.softmax(attn_scores, dim=-1, dtype=torch.float32).to(value_states.dtype)
        attn_probs = nn.functional.dropout(attn_probs, p=self.attention_dropout, training=self.training)

        attn_output = torch.matmul(attn_probs, value_states)

        if attn_output.size() != (batch_size, self.num_heads, seq_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(batch_size, self.num_heads, seq_len, self.head_dim)}, but is {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous().reshape(batch_size, seq_len, -1)
        attn_output = self.o_proj(attn_output)

        new_past_values = v_states.detach() if use_cache else None
        attn_weights_to_return = attn_probs if output_attentions else None

        return attn_output, attn_weights_to_return, new_past_values


# ---------------------
# Decoder layer & model classes
# ---------------------
class LlamaDecoderLayer(nn.Module):
    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.attn_approx = LlamaApproximatedAttention(config=config, layer_idx=layer_idx)
        self.mlp = LlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=getattr(config, "rms_norm_eps", 1e-6))
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=getattr(config, "rms_norm_eps", 1e-6))

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[torch.Tensor] = None,  # treated as past_values
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
    ):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        attn_output, attn_weights, present_values = self.attn_approx(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            past_values=past_key_value,
            use_cache=use_cache,
        )

        if attn_output.shape[1] != residual.shape[1]:
            attn_output = attn_output[:, -residual.shape[1] :, :]

        hidden_states = residual + attn_output

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        if use_cache:
            outputs += (present_values,)

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
        std = getattr(self.config, "initializer_range", 0.02)
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
        self.padding_idx = getattr(config, "pad_token_id", 0)
        self.vocab_size = getattr(config, "vocab_size", 32000)

        self.embed_tokens = nn.Embedding(self.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList([LlamaDecoderLayer(config, layer_idx) for layer_idx in range(getattr(config, "num_hidden_layers", 2))])
        self.norm = LlamaRMSNorm(config.hidden_size, eps=getattr(config, "rms_norm_eps", 1e-6))
        self.gradient_checkpointing = False

        self.layer_sharing = getattr(config, "layer_sharing", False)
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
        past_key_values: Optional[Union[torch.Tensor, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else getattr(self.config, "output_attentions", False)
        output_hidden_states = output_hidden_states if output_hidden_states is not None else getattr(self.config, "output_hidden_states", False)
        use_cache = use_cache if use_cache is not None else getattr(self.config, "use_cache", False)
        return_dict = return_dict if return_dict is not None else getattr(self.config, "use_return_dict", False)

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one")

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once("`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`.")
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        return_legacy_cache = False
        if use_cache and not isinstance(past_key_values, Cache):
            return_legacy_cache = True
            if DynamicCache is not None:
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device)
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = None
        try:
            causal_mask = self._update_causal_mask(attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions)
        except Exception:
            causal_mask = attention_mask

        hidden_states = inputs_embeds

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

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if return_legacy_cache and hasattr(next_cache, "to_legacy_cache"):
            next_cache = next_cache.to_legacy_cache()

        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(last_hidden_state=hidden_states, past_key_values=next_cache, hidden_states=all_hidden_states, attentions=all_self_attns)

    def _update_causal_mask(
        self,
        attention_mask: torch.Tensor,
        input_tensor: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_values: Cache,
        output_attentions: bool,
    ):
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        using_static_cache = isinstance(past_key_values, StaticCache)

        dtype, device = input_tensor.dtype, input_tensor.device
        min_dtype = torch.finfo(dtype).min
        sequence_length = input_tensor.shape[1]
        if using_static_cache:
            target_length = past_key_values.get_max_length()
        else:
            target_length = (
                attention_mask.shape[-1] if isinstance(attention_mask, torch.Tensor) else past_seen_tokens + sequence_length + 1
            )

        if attention_mask is not None and attention_mask.dim() == 4:
            if attention_mask.max() != 0:
                raise ValueError("Custom 4D attention mask should be passed in inverted form with max==0`")
            causal_mask = attention_mask
        else:
            causal_mask = torch.full((sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device)
            if sequence_length != 1:
                causal_mask = torch.triu(causal_mask, diagonal=1)
            causal_mask *= torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
            causal_mask = causal_mask[None, None, :, :].expand(input_tensor.shape[0], 1, -1, -1)
            if attention_mask is not None:
                causal_mask = causal_mask.clone()
                mask_length = attention_mask.shape[-1]
                padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :]
                padding_mask = padding_mask == 0
                causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(padding_mask, min_dtype)

        return causal_mask


# ---------------------
# LlamaForCausalLM
# ---------------------
class LlamaForCausalLM(LlamaPreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.model = LlamaModel(config)
        self.vocab_size = getattr(config, "vocab_size", 32000)
        if not getattr(config, "share_embedding", False):
            self.lm_head = nn.Linear(config.hidden_size, self.vocab_size, bias=False)
        else:
            self.lm_head = None

        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        if not getattr(self.config, "share_embedding", False):
            return self.lm_head
        else:
            return self.get_input_embeddings()

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
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[torch.Tensor, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else getattr(self.config, "output_attentions", False)
        output_hidden_states = output_hidden_states if output_hidden_states is not None else getattr(self.config, "output_hidden_states", False)
        return_dict = return_dict if return_dict is not None else getattr(self.config, "use_return_dict", False)
        use_cache = use_cache if use_cache is not None else getattr(self.config, "use_cache", False)

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        hidden_states = outputs[0]

        if getattr(self.config, "pretraining_tp", 1) > 1:
            # parallel lm_head splitting -- not required for test; leave a simple concat flow
            lm_head_slices = self.lm_head.weight.split(self.vocab_size // getattr(self.config, "pretraining_tp", 1), dim=0)
            logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(getattr(self.config, "pretraining_tp", 1))]
            logits = torch.cat(logits, dim=-1)
        else:
            if not getattr(self.config, "share_embedding", False):
                logits = self.lm_head(hidden_states)
            else:
                logits = F.linear(hidden_states, self.model.embed_tokens.weight)

        logits = logits.float()

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.vocab_size)
            shift_labels = shift_labels.view(-1)
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        past = outputs[1] if len(outputs) > 1 else None

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(loss=loss, logits=logits, past_key_values=past, hidden_states=outputs.hidden_states if hasattr(outputs, "hidden_states") else None, attentions=outputs.attentions if hasattr(outputs, "attentions") else None)


# ---------------------
# Simple test
# ---------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    device = torch.device("cpu")

    # Build a tiny config for testing
    cfg = LlamaConfig(
        vocab_size=128,
        hidden_size=32,          # must be divisible by num_attention_heads
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
    )
    # add our custom param
    setattr(cfg, "factorization_rank", 8)
    setattr(cfg, "attention_dropout", 0.0)
    setattr(cfg, "rms_norm_eps", 1e-6)
    setattr(cfg, "initializer_range", 0.02)
    setattr(cfg, "use_cache", False)
    setattr(cfg, "share_embedding", False)

    model = LlamaForCausalLM(cfg).to(device)
    model.eval()

    batch_size = 1
    seq_len = 6
    input_ids = torch.randint(0, cfg.vocab_size, (batch_size, seq_len), dtype=torch.long, device=device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, use_cache=False)
        if isinstance(outputs, tuple):
            logits = outputs[0]
        else:
            logits = outputs.logits
        print("logits.shape:", logits.shape)  # expected (batch, seq_len, vocab_size)

        # optional loss test
        labels = input_ids.clone()
        out = model(input_ids=input_ids, labels=labels)
        loss = out.loss if hasattr(out, "loss") else out[0]
        print("loss:", float(loss))