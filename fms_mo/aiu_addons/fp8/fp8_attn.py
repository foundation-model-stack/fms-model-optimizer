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
from typing import NotRequired, Unpack
import math

# Third Party
from fms.modules.attention import (
    AttentionKwargs,
    _sdpa_update_attn_kwargs,
    register_attention_op,
)
import torch

# Local
from fms_mo.aiu_addons.fp8.fp8_utils import ScaledTensor
import fms_mo.aiu_addons.fp8.fp8_spyre_op  # pylint: disable=unused-import


class MathFP8AttentionKwargs(AttentionKwargs):
    """TypedDict for FP8 attention."""

    mask: NotRequired[torch.Tensor]
    do_scale_q: bool
    is_causal_mask: bool


# TODO: Figure out better scales for AIU? These come from vLLM
Q_RANGE = 200.0
K_RANGE = 200.0
V_RANGE = 100.0


def _construct_fp8_cache(tensor: torch.Tensor, scale: torch.Tensor) -> ScaledTensor:
    """Construct the custom object to save KV cache with its scales."""
    return ScaledTensor(tensor, scale)


def _math_fp8_store_op(
    keys: torch.Tensor,  # pylint: disable=unused-argument
    values: torch.Tensor,
    key_cache: torch.Tensor | None,
    value_cache: torch.Tensor | None,
    **attn_kwargs: Unpack[MathFP8AttentionKwargs],
) -> tuple[ScaledTensor, ScaledTensor, ScaledTensor, ScaledTensor]:
    """Implement math of KV cache storing."""

    if isinstance(key_cache, ScaledTensor) and isinstance(value_cache, ScaledTensor):
        k_scale = key_cache._scale
        v_scale = value_cache._scale
    else:
        k_scale = (torch.abs(keys).max() / K_RANGE).to(dtype=torch.float32)
        v_scale = (torch.abs(values).max() / V_RANGE).to(dtype=torch.float32)

    keys = (keys / k_scale).to(torch.float8_e4m3fn).transpose(2, 1)
    values = (values / v_scale).to(torch.float8_e4m3fn).transpose(2, 1)

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
    """Implement computation of attention BMM, leveraging the custom scaled attention
    BMM op that was pre-registered for torch.compile."""

    orig_dtype = query.dtype

    q_scale = torch.tensor(1.0, dtype=torch.float32, device=query.device)
    if attn_kwargs.get("do_scale_q", False):
        q_scale.copy_(torch.abs(query).max() / Q_RANGE)
        query = query / q_scale

    query = query.to(torch.float8_e4m3fn).transpose(2, 1)

    if isinstance(key_cache, ScaledTensor) and isinstance(value_cache, ScaledTensor):
        k_scale = key_cache._scale
        v_scale = value_cache._scale
        key_cache = key_cache._data
        value_cache = value_cache._data
    else:
        k_scale = (torch.abs(key_cache).max() / K_RANGE).to(dtype=torch.float32)
        v_scale = (torch.abs(value_cache).max() / V_RANGE).to(dtype=torch.float32)
        key_cache = (key_cache / k_scale).to(torch.float8_e4m3fn)
        value_cache = (value_cache / v_scale).to(torch.float8_e4m3fn)

    # no longer transposing prior to store, so need to check this in case of no cache
    # TODO: Refactor FMS to avoid edge cases where this fails; add use_cache param here
    if key_cache.shape[1] != kvheads and key_cache.shape[2] == kvheads:
        key_cache = key_cache.transpose(2, 1)
        value_cache = value_cache.transpose(2, 1)

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

    attn_weight = (
        torch.ops.sendnn.scaled_bmm(
            query,
            key_cache.transpose(-2, -1),
            q_scale,
            k_scale,
            out_dtype=orig_dtype,
            use_fast_accum=True,
        )
        * scale_factor
    )
    attn_weight += attn_bias
    attn_weight = torch.softmax(attn_weight, dim=-1)
    attn_weight = torch.dropout(attn_weight, p_dropout, train=True)
    # Do matmul in orig_dtype
    attn = attn_weight @ (value_cache.to(dtype=orig_dtype) * v_scale)

    attn = attn.to(orig_dtype).transpose(2, 1).contiguous()
    return attn


register_attention_op(
    "math_fp8",
    _math_fp8_store_op,
    _math_fp8_compute_op,
    update_attn_kwargs_op=_sdpa_update_attn_kwargs,
)
