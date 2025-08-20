# Copyright The FMS Model Optimizer Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""FMS registration of attention BMM operation using torch-registered scaled BMM."""

# Standard
from typing import NotRequired, Optional, Unpack
import math

# Third Party
import torch

# Local
from fms_mo.aiu_addons.fp8.fp8_utils import ScaledTensor
from fms_mo.prep import available_packages
import fms_mo.aiu_addons.fp8.fp8_spyre_op  # pylint: disable=unused-import

if available_packages["fms"]:
    # Third Party
    from fms.modules.attention import (
        AttentionKwargs,
        _sdpa_compute_op,
        _sdpa_update_attn_kwargs,
        register_attention_op,
    )
    from fms.utils.spyre.paged import (
        SpyrePagedAttentionKwargs,
        __spyre_paged_validate_attn_kwargs_op,
    )

    class MathFP8AttentionKwargs(AttentionKwargs):
        """TypedDict for FP8 attention."""

        mask: NotRequired[torch.Tensor]
        do_scale_q: bool
        do_scaled_bmm: bool
        is_causal_mask: bool

    # TODO: Figure out better scales for AIU? These come from vLLM
    Q_RANGE = 200.0
    K_RANGE = 200.0
    V_RANGE = 100.0

    def _construct_fp8_cache(tensor: torch.Tensor, scale: torch.Tensor) -> ScaledTensor:
        """Construct the custom object to save KV cache with its scales."""
        return ScaledTensor(tensor, scale, True)

    def _math_fp8_store_op(
        keys: torch.Tensor,  # pylint: disable=unused-argument
        values: torch.Tensor,
        key_cache: torch.Tensor | None,
        value_cache: torch.Tensor | None,
        **attn_kwargs: Unpack[MathFP8AttentionKwargs],
    ) -> tuple[ScaledTensor, ScaledTensor, ScaledTensor, ScaledTensor]:
        """Implement math of KV cache storing."""

        # Grab scale from kv-cache if already there, compute dynamically otherwise
        if isinstance(key_cache, ScaledTensor) and isinstance(
            value_cache, ScaledTensor
        ):
            k_scale = key_cache._scale
            v_scale = value_cache._scale
        else:
            k_scale = (
                (torch.abs(keys).amax(dim=(1, 2, 3)) / K_RANGE)
                .clamp(min=1e-5)
                .to(dtype=torch.float32)
            )
            v_scale = (
                (torch.abs(values).amax(dim=(1, 2, 3)) / V_RANGE)
                .clamp(min=1e-5)
                .to(dtype=torch.float32)
            )

        # Scale kv tensors for storage
        keys = (
            (keys / k_scale.view(-1, 1, 1, 1)).to(torch.float8_e4m3fn).transpose(2, 1)
        )
        values = (
            (values / v_scale.view(-1, 1, 1, 1)).to(torch.float8_e4m3fn).transpose(2, 1)
        )

        if (
            isinstance(key_cache, ScaledTensor)
            and isinstance(value_cache, ScaledTensor)
            and value_cache.numel() > 0
        ):
            key_cache = torch.cat((key_cache._data, keys), dim=2)
            value_cache = torch.cat((value_cache._data, values), dim=2)
            key_cache = _construct_fp8_cache(key_cache, k_scale)
            value_cache = _construct_fp8_cache(value_cache, v_scale)
            return (
                key_cache,
                value_cache,
                key_cache,
                value_cache,
            )
        # If it's a new kv cache, ensure it's contiguous for spyre use cases
        keys = _construct_fp8_cache(keys.contiguous(), k_scale)
        values = _construct_fp8_cache(values.contiguous(), v_scale)
        return (keys, values, keys, values)

    def _math_fp8_compute_op(
        query: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        nheads: int,
        kvheads: int,
        p_dropout: float,
        scale_factor: float | None,
        **attn_kwargs: Unpack[MathFP8AttentionKwargs],
    ) -> torch.Tensor:
        """Implement computation of scaled dot product attention, leveraging
        the custom scaled BMM op that was pre-registered for torch.compile."""

        orig_dtype = query.dtype
        do_scaled_bmm = attn_kwargs.get("do_scaled_bmm", False)

        if do_scaled_bmm:
            # Scaling the Q tensor is optional
            q_scale = torch.tensor(1.0, dtype=torch.float32, device=query.device)
            if attn_kwargs.get("do_scale_q", False):
                q_scale.copy_(torch.abs(query).max() / Q_RANGE)
                query = query / q_scale

            query = query.to(torch.float8_e4m3fn)
        query = query.transpose(2, 1)

        # Grab kv cache and deal with cases where no store op was run
        if isinstance(key_cache, ScaledTensor) and isinstance(
            value_cache, ScaledTensor
        ):
            # Store op was run
            k_scale = key_cache._scale
            v_scale = value_cache._scale
            key_cache = key_cache._data
            value_cache = value_cache._data
        else:
            # Store op wasn't run (e.g. encoders, use_cache=False)
            k_scale = (
                (torch.abs(key_cache).amax(dim=(1, 2, 3)) / K_RANGE)
                .clamp(min=1e-5)
                .to(dtype=torch.float32)
            )
            v_scale = (
                (torch.abs(value_cache).amax(dim=(1, 2, 3)) / V_RANGE)
                .clamp(min=1e-5)
                .to(dtype=torch.float32)
            )
            key_cache = (key_cache / k_scale.view(-1, 1, 1, 1)).to(torch.float8_e4m3fn)
            value_cache = (value_cache / v_scale.view(-1, 1, 1, 1)).to(
                torch.float8_e4m3fn
            )

        # If store wasn't run, we need to transpose the tensors here
        # TODO: Refactor FMS to avoid edge cases where this fails; add use_cache param here
        if key_cache.shape[1] != kvheads and key_cache.shape[2] == kvheads:
            key_cache = key_cache.transpose(2, 1)
            value_cache = value_cache.transpose(2, 1)

        # Most of the code that follows is a copy of Pytorch SDPA, with fp8 additions
        mask = attn_kwargs.get("mask", None)
        if mask is not None:
            # Our expected mask format is bs x q_len x k_len, so to make it broadcastable
            # we need to create the nheads dimension
            while len(mask.size()) != 4:  # expects bs (x nheads) x q_len x kv_len
                mask = mask.unsqueeze(1)

        L, S = query.size(-2), key_cache.size(-2)
        scale_factor = (
            1 / math.sqrt(query.size(-1)) if scale_factor is None else scale_factor
        )
        attn_bias = torch.zeros(L, S, dtype=orig_dtype, device=query.device)
        if attn_kwargs.get("is_causal_mask", False):
            assert mask is None
            temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
            attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
            attn_bias.to(torch.float32)

        if mask is not None:
            if mask.dtype == torch.bool:
                attn_bias.masked_fill_(mask.logical_not(), float("-inf"))
            else:
                attn_bias = mask + attn_bias

        expansion = nheads // kvheads
        if expansion > 1:
            key_cache = key_cache.repeat_interleave(
                query.size(-3) // key_cache.size(-3), -3
            )
            value_cache = value_cache.repeat_interleave(
                query.size(-3) // value_cache.size(-3), -3
            )

        if do_scaled_bmm:
            attn_weight = (
                torch.ops.spyre.scaled_bmm(
                    query,
                    key_cache.transpose(-2, -1),
                    q_scale,
                    k_scale,
                    out_dtype=orig_dtype,
                    use_fast_accum=True,
                )
                * scale_factor
            )
        else:
            key_t = (
                (key_cache.to(dtype=orig_dtype) * k_scale.view(-1, 1, 1, 1))
                .to(dtype=orig_dtype)
                .transpose(-2, -1)
            )
            attn_weight = (query * math.sqrt(scale_factor)) @ (
                key_t * math.sqrt(scale_factor)
            )
        attn_weight += attn_bias
        attn_weight = torch.softmax(attn_weight, dim=-1)
        attn_weight = torch.dropout(attn_weight, p_dropout, train=True)
        # Do matmul in orig_dtype
        attn = attn_weight @ (
            value_cache.to(dtype=orig_dtype) * v_scale.view(-1, 1, 1, 1)
        ).to(dtype=orig_dtype)

        attn = attn.to(orig_dtype).transpose(2, 1).contiguous()
        return attn

    register_attention_op(
        "math_fp8",
        _math_fp8_store_op,
        _math_fp8_compute_op,
        update_attn_kwargs_op=_sdpa_update_attn_kwargs,
    )

    def _spyre_scaled_paged_store_op(
        keys: torch.Tensor,
        values: torch.Tensor,
        key_cache: Optional[torch.Tensor],
        value_cache: Optional[torch.Tensor],
        **attn_kwargs: Unpack[SpyrePagedAttentionKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # For paged store, we must have pre-allocated the kv-cache
        assert key_cache is not None and isinstance(
            key_cache, ScaledTensor
        ), "kv cache must be preallocated"
        assert value_cache is not None and isinstance(
            value_cache, ScaledTensor
        ), "kv cache must be preallocated"
        if not key_cache._scaled:
            key_cache._scale = (
                (torch.abs(keys).amax(dim=(1, 2, 3)) / K_RANGE)
                .clamp(min=1e-5)
                .to(dtype=torch.float32)
            )
            value_cache._scale = (
                (torch.abs(values).amax(dim=(1, 2, 3)) / V_RANGE)
                .clamp(min=1e-5)
                .to(dtype=torch.float32)
            )

        result_key_cache_data, result_value_cache_data = (
            torch.ops.spyre.scaled_paged_attn_store(
                keys,
                values,
                key_cache._data,
                value_cache._data,
                key_cache._scale,
                value_cache._scale,
                attn_kwargs["slot_mapping"],
            )
        )

        result_key_cache = _construct_fp8_cache(result_key_cache_data, key_cache._scale)
        result_value_cache = _construct_fp8_cache(
            result_value_cache_data, value_cache._scale
        )

        # for prefill, we want to return the original keys/values
        if attn_kwargs.get("block_table", None) is None:
            return keys, values, result_key_cache, result_value_cache
        return (
            result_key_cache,
            result_value_cache,
            result_key_cache,
            result_value_cache,
        )

    def _spyre_scaled_paged_compute_op(
        query: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        nheads: int,  # pylint: disable=unused-argument
        kvheads: int,  # pylint: disable=unused-argument
        p_dropout: float,  # pylint: disable=unused-argument
        scale_factor: Optional[float],
        **attn_kwargs,
    ) -> torch.Tensor:
        assert isinstance(key_cache, ScaledTensor), "kv cache must be scaled"
        assert isinstance(value_cache, ScaledTensor), "kv cache must be scaled"
        if scale_factor is None:
            scale_factor = 1 / math.sqrt(query.shape[-1])
        return torch.ops.spyre.scaled_paged_attn_compute(
            query,
            key_cache._data,
            value_cache._data,
            key_cache._scale,
            value_cache._scale,
            scale_factor,
            attn_kwargs["current_tkv_mask"],
            attn_kwargs["left_padded_prompt_mask"],
            attn_kwargs["block_table"],
        )

    def __spyre_scaled_paged_validate_attn_kwargs_op(
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        past_key_value_states: Optional[list[tuple[torch.Tensor, torch.Tensor]]] = None,
        **attn_kwargs,
    ):
        __spyre_paged_validate_attn_kwargs_op(
            input_ids, position_ids, past_key_value_states, **attn_kwargs
        )

        if past_key_value_states is not None:
            for k, v in past_key_value_states:
                assert isinstance(k, ScaledTensor)
                assert isinstance(v, ScaledTensor)

                # assert that for each layer, the scales are per-sequence
                assert k._scale.shape[0] == input_ids.shape[0]
                assert v._scale.shape[0] == input_ids.shape[0]

    register_attention_op(
        "spyre_paged_attn_fp8",
        _spyre_scaled_paged_store_op,
        compute_op=_sdpa_compute_op,
        is_prefill_op=lambda **attn_kwargs: attn_kwargs.get("block_table", None)
        is None,
        compute_decode_op=_spyre_scaled_paged_compute_op,
        validate_attn_kwargs_op=__spyre_scaled_paged_validate_attn_kwargs_op,
    )
