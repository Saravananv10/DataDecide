import math
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn

from .aliases import PathOrStr
from .config import ModelConfig
from .exceptions import OLMoConfigurationError
from .initialization import ModuleType, initialize_parameters
from .torch_util import Activation, LayerNormBase, LowPrecisionLayerNorm, RMSNorm


class OLMoBlock(nn.Module):
    """
    A standard OLMo block.
    """

    def __init__(self, config: ModelConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        hidden_size = config.d_model
        self.attn_out_proj = nn.Linear(hidden_size, hidden_size, bias=config.include_bias)
        self.attn_out_proj._is_residual = True  # type: ignore
        self.block_out_proj = nn.Linear(hidden_size, hidden_size, bias=config.include_bias)
        self.block_out_proj._is_residual = True  # type: ignore

        self.attn_norm = self._build_norm(hidden_size)
        self.ff_norm = self._build_norm(hidden_size)

        self.dropout1 = nn.Dropout(config.residual_dropout)
        self.dropout2 = nn.Dropout(config.residual_dropout)

        self.attention = self._build_attention(hidden_size)
        self.ff_layer = self._build_ff_layer(hidden_size)

    def _build_norm(self, hidden_size: int) -> LayerNormBase:
        if self.config.layer_norm_type == "default":
            return nn.LayerNorm(
                hidden_size,
                eps=self.config.low_precision_layer_norm_epsilon,  # this is intentional
                elementwise_affine=self.config.layer_norm_with_affine,
            )
        elif self.config.layer_norm_type == "low_precision":
            return LowPrecisionLayerNorm(
                hidden_size,
                eps=self.config.low_precision_layer_norm_epsilon,
                elementwise_affine=self.config.layer_norm_with_affine,
            )
        elif self.config.layer_norm_type == "rms":
            return RMSNorm(
                hidden_size,
                eps=self.config.rms_norm_epsilon,
                elementwise_affine=self.config.layer_norm_with_affine,
            )
        else:
            raise OLMoConfigurationError(f"Unknown layer norm type: {self.config.layer_norm_type}")

    def _build_attention(self, hidden_size: int):
        # TODO: add support for different attention types, e.g. flash attention.
        # For now, we just use a basic self attention implementation.
        return SelfAttention(self.config, layer_idx=self.layer_idx)

    def _build_ff_layer(self, hidden_size: int):
        # TODO: add support for different ff layer types, e.g. fused.
        # For now, we just use a basic ff layer implementation.
        return FeedForwardLayer(self.config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
    ) -> Union[
        Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]], Optional[torch.Tensor]],
        Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]],
    ]:
        # Self attention.
        residual = hidden_states
        hidden_states = self.attn_norm(hidden_states)
        attn_outputs = self.attention(
            hidden_states,
            attention_mask=attention_mask,
            layer_past=layer_past,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        attn_output = attn_outputs[0]  # output_attn: a, present, (attentions)
        attn_output = self.attn_out_proj(attn_output)
        hidden_states = residual + self.dropout1(attn_output)

        # Feed forward.
        residual = hidden_states
        hidden_states = self.ff_norm(hidden_states)
        feed_forward_hidden_states = self.ff_layer(hidden_states)
        feed_forward_hidden_states = self.block_out_proj(feed_forward_hidden_states)
        hidden_states = residual + self.dropout2(feed_forward_hidden_states)

        outputs = (hidden_states,)

        if use_cache:
            outputs = outputs + (attn_outputs[1],)

        if output_attentions:
            outputs = outputs + (attn_outputs[2],)

        return outputs  # type: ignore


class SelfAttention(nn.Module):
    """
    A basic self attention implementation.
    """

    def __init__(self, config: ModelConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.d_model
        self.num_heads = config.n_heads
        self.head_dim = self.hidden_size // self.num_heads
        if self.head_dim * self.num_heads != self.hidden_size:
            raise OLMoConfigurationError(
                f"hidden_size ({self.hidden_size}) must be divisible by num_heads ({self.num_heads})"
            )

        if config.n_kv_heads is None:
            self.num_kv_heads = self.num_heads
        else:
            self.num_kv_heads = config.n_kv_heads
        if self.num_heads % self.num_kv_heads != 0:
            raise OLMoConfigurationError(
                f"num_heads ({self.num_heads}) must be divisible by num_kv_heads ({self.num_kv_heads})"
            )
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads

        self.q_proj = nn.Linear(
            self.hidden_size, self.num_heads * self.head_dim, bias=config.include_bias
        )
        self.k_proj = nn.Linear(
            self.hidden_size, self.num_kv_heads * self.head_dim, bias=config.include_bias
        )
        self.v_proj = nn.Linear(
            self.hidden_size, self.num_kv_heads * self.head_dim, bias=config.include_bias
        )

        self.attn_dropout = nn.Dropout(config.attention_dropout)

        self.rotary_emb = None
        if config.rope:
            self.rotary_emb = RotaryEmbedding(
                self.head_dim,
                max_position_embeddings=config.max_sequence_length,
                base=10000,  # TODO: make this configurable
            )

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def _shape_kv(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return (
            tensor.view(bsz, seq_len, self.num_kv_heads, self.head_dim)
            .transpose(1, 2)
            .contiguous()
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]], Optional[torch.Tensor]]:
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = self._shape(query_states, q_len, bsz)
        key_states = self._shape_kv(key_states, q_len, bsz)
        value_states = self._shape_kv(value_states, q_len, bsz)

        if self.rotary_emb is not None:
            # Apply rotary embeddings to query and key.
            # TODO: Handle past_key_values for rotary embeddings.
            # For now, we assume that if layer_past is not None, then we are in generation mode.
            # In this case, we only need to apply rotary embeddings to the new query and key states.
            # This is because the past key and value states have already been rotated.
            # This is not strictly correct, but it's a common approximation.
            # The correct way to do this is to rotate the past key and value states by the new
            # positions, but this is more complicated.
            # See https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py#L325
            # for an example of how to do this correctly.
            kv_seq_len = key_states.shape[-2]
            if layer_past is not None:
                kv_seq_len += layer_past[0].shape[-2]
            cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
            query_states, key_states = apply_rotary_pos_emb(
                query_states, key_states, cos, sin, position_ids=None
            )  # TODO: pass position_ids

        if layer_past is not None:
            # reuse k, v, self_attention
            key_states = torch.cat([layer_past[0], key_states], dim=2)
            value_states = torch.cat([layer_past[1], value_states], dim=2)

        past_key_value = (key_states, value_states) if use_cache else None

        # repeat k/v heads if n_kv_heads < n_heads
        if self.num_queries_per_kv > 1:
            key_states = key_states.repeat_interleave(self.num_queries_per_kv, dim=1)
            value_states = value_states.repeat_interleave(self.num_queries_per_kv, dim=1)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(
            self.head_dim
        )

        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights + attention_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
            query_states.dtype
        )
        attn_weights = self.attn_dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        if output_attentions:
            # this operation is a bit awkward, but it's required to
            # make sure that attn_weights keeps its gradient.
            # In order to do so, attn_weights have to be reshaped
            # twice and have to be reused in the following
            attn_weights = attn_weights.view(bsz, self.num_heads, q_len, kv_seq_len)
        else:
            attn_weights = None

        return attn_output, past_key_value, attn_weights


class FeedForwardLayer(nn.Module):
    """
    A basic feed forward layer implementation.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.d_model
        self.intermediate_size = config.mlp_hidden_size
        self.activation_fn = Activation(config.activation_type)

        self.up_proj = nn.Linear(
            self.hidden_size, self.intermediate_size, bias=config.include_bias
        )
        self.down_proj = nn.Linear(
            self.intermediate_size, self.hidden_size, bias=config.include_bias
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.up_proj(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.down_proj(hidden_states)
        return hidden_states


class RotaryEmbedding(torch.nn.Module):
    # From https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py#L94
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=torch.get_default_dtype()
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)

        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :].to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :].to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        return (
            self.cos_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
            self.sin_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
        )


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    # The first two dimensions of cos and sin are always 1, so we can `squeeze` them.
    cos = cos.squeeze(1).squeeze(0)  # [seq_len, dim]
    sin = sin.squeeze(1).squeeze(0)  # [seq_len, dim]
    cos = cos[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    sin = sin[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class OLMo(nn.Module):
    """
    The OLMo model.
    """

    def __init__(self, config: ModelConfig, init_params: bool = True):
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model, self.padding_idx)
        self.embed_dropout = nn.Dropout(config.embedding_dropout)

        self.layers = nn.ModuleList(
            [OLMoBlock(config, layer_idx=i) for i in range(config.n_layers)]
        )
        self.norm = self._build_norm(config.d_model)

        if not config.weight_tying:
            self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        if init_params:
            self.reset_parameters()

    def _build_norm(self, hidden_size: int) -> LayerNormBase:
        if self.config.layer_norm_type == "default":
            return nn.LayerNorm(
                hidden_size,
                eps=self.config.low_precision_layer_norm_epsilon,  # this is intentional
                elementwise_affine=self.config.layer_norm_with_affine,
            )
        elif self.config.layer_norm_type == "low_precision":
            return LowPrecisionLayerNorm(
                hidden_size,
                eps=self.config.low_precision_layer_norm_epsilon,
                elementwise_affine=self.config.layer_norm_with_affine,
            )
        elif self.config.layer_norm_type == "rms":
            return RMSNorm(
                hidden_size,
                eps=self.config.rms_norm_epsilon,
                elementwise_affine=self.config.layer_norm_with_affine,
            )
        else:
            raise OLMoConfigurationError(f"Unknown layer norm type: {self.config.layer_norm_type}")

    def reset_parameters(self):
        # Call this to initialize parameters that have been added directly (i.e. not part of other modules)
        # or to re-initialize all parameters.
        initialize_parameters(self, self.config.init_fn)  # type: ignore

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def get_output_embeddings(self):
        if self.config.weight_tying:
            return self.embed_tokens
        else:
            return self.lm_head

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        # TODO: add more arguments here, e.g. token_type_ids, position_ids, etc.
    ) -> Dict[str, Any]:
        output_attentions = (
            output_attentions if output_attentions is not None else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        # retrieve input_ids and inputs_embeds
        if input_ids is not None:
            batch_size, seq_length = input_ids.shape
        else:
            raise ValueError("You have to specify either input_ids")

        seq_length_with_past = seq_length
        past_key_values_length = 0

        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2]
            seq_length_with_past = seq_length_with_past + past_key_values_length

        # TODO: add position_ids and token_type_ids
        # For now, we just assume that position_ids are [0, 1, ..., seq_length - 1]
        # and token_type_ids are all 0s.
        # This is not correct for all use cases, e.g. if padding is used.
        # See https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py#L770
        # for an example of how to do this correctly.
        position_ids = torch.arange(
            past_key_values_length,
            seq_length + past_key_values_length,
            dtype=torch.long,
            device=input_ids.device,
        )
        position_ids = position_ids.unsqueeze(0).view(-1, seq_length)

        inputs_embeds = self.embed_tokens(input_ids)
        hidden_states = self.embed_dropout(inputs_embeds)

        if attention_mask is None:
            # Create a causal mask.
            attention_mask = torch.ones(
                (batch_size, seq_length_with_past),
                dtype=torch.bool,
                device=hidden_states.device,
            )
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask, (batch_size, seq_length), hidden_states, past_key_values_length
        )

        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)  # type: ignore

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                layer_past=past_key_value,
                use_cache=use_cache,
                output_attentions=output_attentions,
            )
            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[1],)  # type: ignore
            if output_attentions:
                all_self_attns += (layer_outputs[2],)  # type: ignore

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)  # type: ignore

        next_cache = next_decoder_cache if use_cache else None

        if self.config.weight_tying:
            logits = F.linear(hidden_states, self.embed_tokens.weight)
        else:
            logits = self.lm_head(hidden_states)

        if self.config.scale_logits:
            logits = logits / math.sqrt(self.config.d_model)

        # TODO: add loss calculation
        loss = None

        return {
            "loss": loss,
            "logits": logits,
            "past_key_values": next_cache,
            "hidden_states": all_hidden_states,
            "attentions": all_self_attns,
        }

    def _prepare_decoder_attention_mask(
        self, attention_mask, input_shape, inputs_embeds, past_key_values_length
    ):
        # create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        combined_attention_mask = None
        if input_shape[-1] > 1:
            combined_attention_mask = _make_causal_mask(
                input_shape,
                inputs_embeds.dtype,
                device=inputs_embeds.device,
                past_key_values_length=past_key_values_length,
            )

        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            expanded_attn_mask = _expand_mask(
                attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]
            ).to(inputs_embeds.device)
            combined_attention_mask = (
                expanded_attn_mask
                if combined_attention_mask is None
                else expanded_attn_mask + combined_attention_mask
            )

        return combined_attention_mask

    @classmethod
    def from_pretrained(cls, model_type: str):
        # This is a placeholder. Actual weight loading needs to be implemented.
        print(f"Loading pretrained OLMo model: {model_type}")
        # This is where you would load the actual OLMo weights.
        # For now, we'll just create a model with the specified config.
        if model_type == "olmo-1b":
            config = ModelConfig(
                d_model=2048,
                n_layers=16,
                n_heads=16,
                vocab_size=50304, # Closest to OLMo's 50257 but divisible by 128
                embedding_size=50304,
                max_sequence_length=2048,
                activation_type="swiglu",
                block_type="sequential",
                rope=True,
            )
        elif model_type == "olmo-7b":
            config = ModelConfig(
                d_model=4096,
                n_layers=32,
                n_heads=32,
                vocab_size=50304,
                embedding_size=50304,
                max_sequence_length=2048,
                activation_type="swiglu",
                block_type="sequential",
                rope=True,
            )
        else:
            raise ValueError(f"Unknown OLMo model type: {model_type}")

        # Add a note that actual weight loading is not implemented
        print("NOTE: This is a placeholder `from_pretrained` method. "
              "Actual weight loading is not implemented.")

        return cls(config)


# Copied from transformers.models.bart.modeling_bart._make_causal_mask
def _make_causal_mask(
    input_ids_shape: torch.Size, dtype: torch.dtype, device: torch.device, past_key_values_length: int = 0
):
    """
    Make causal mask used for bi-directional self-attention.
    """
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), torch.tensor(torch.finfo(dtype).min, device=device), device=device)
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)

    if past_key_values_length > 0:
        mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=dtype, device=device), mask], dim=-1)
    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)


# Copied from transformers.models.bart.modeling_bart._expand_mask
def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)
