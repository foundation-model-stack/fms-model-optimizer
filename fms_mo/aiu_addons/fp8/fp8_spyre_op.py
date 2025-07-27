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
"""Torch registration of FP8xFP8 operation for attention BMMs."""

# Standard
from typing import Optional

# Third Party
from torch import Tensor
import torch
import torch.nn.functional as F

# pylint: disable=unused-argument
# abstract op must be registered with specific I/O, even if not in use by the op function

# pylint: disable=not-callable
# torch.nn.functional.scaled_dot_product_attention not recognized as callable
# open issue in PyLint: https://github.com/pytorch/pytorch/issues/119482


def _scaled_mm_cpu_out(
    mat1: Tensor,
    mat2: Tensor,
    scale1: Tensor,
    scale2: Tensor,
    bias: Optional[Tensor] = None,
    scale_result: Optional[Tensor] = None,
    out_dtype: Optional[torch.dtype] = None,
    use_fast_accum: bool = False,
    *,
    out: Optional[Tensor] = None,
) -> Tensor:
    if out_dtype is None:
        out_dtype = torch.float32
    mat1 = (mat1.to(dtype=out_dtype) * scale1).to(dtype=out_dtype)
    mat2 = (mat2.to(dtype=out_dtype) * scale2).to(dtype=out_dtype)

    if bias is not None:
        ret = torch.addmm(bias, mat1, mat2).to(dtype=out_dtype)
    else:
        ret = torch.mm(mat1, mat2).to(dtype=out_dtype)

    if out is not None:
        out.copy_(ret)
        return out
    return ret


torch.library.register_kernel(torch.ops.aten._scaled_mm.out, "cpu", _scaled_mm_cpu_out)


@torch.library.register_kernel("aten::_scaled_mm", "cpu")
def _scaled_mm_cpu(
    mat1: Tensor,
    mat2: Tensor,
    scale1: Tensor,
    scale2: Tensor,
    bias: Optional[Tensor] = None,
    scale_result: Optional[Tensor] = None,
    out_dtype: Optional[torch.dtype] = None,
    use_fast_accum: bool = False,
) -> Tensor:
    return _scaled_mm_cpu_out(
        mat1,
        mat2,
        scale1,
        scale2,
        bias,
        scale_result,
        out_dtype,
        use_fast_accum,
        out=None,
    )


@torch.library.custom_op("spyre::scaled_bmm", mutates_args=())
def spyre_scaled_bmm(
    mat1: Tensor,
    mat2: Tensor,
    scale1: Tensor,
    scale2: Tensor,
    out_dtype: torch.dtype | None = None,
    use_fast_accum: bool = False,
) -> Tensor:
    """Implement a custom scaled attention BMM op: a batched version of _scaled_mm.
    The operations that are part of this function are not exposed to the computational
    graph, but are invoked when running on non-Spyre devices.
    """

    assert (
        mat1.shape[:-2] == mat2.shape[:-2]
    ), "batch dimensions must match for mat1 and mat2"
    assert scale1.numel() == 1, "only per-tensor scales supported"
    assert scale2.numel() == 1, "only per-tensor scales supported"
    mat1 = mat1.view(-1, *mat1.shape[-2:])
    mat2 = mat2.view(-1, *mat2.shape[-2:])
    out = torch.empty(
        (mat1.shape[0], mat1.shape[1], mat2.shape[2]),
        dtype=out_dtype,
        device=mat1.device,
    )
    for b_idx in range(mat1.shape[0]):
        out[b_idx] = torch._scaled_mm(
            mat1[b_idx],
            mat2[b_idx],
            scale1,
            scale2,
            out_dtype=out_dtype,
            use_fast_accum=use_fast_accum,
        )
    return out.view(*mat1.shape[:-2], mat1.shape[1], mat2.shape[2])


@spyre_scaled_bmm.register_fake
def _(
    mat1: Tensor,
    mat2: Tensor,
    scale1: Tensor,
    scale2: Tensor,
    out_dtype: torch.dtype | None = None,
    use_fast_accum: bool = False,
) -> Tensor:
    """Template for scaled attention BMM operation. I/O retain the expected size."""

    return torch.empty(
        (*mat1.shape[:-2], mat1.shape[-2], mat2.shape[-1]),
        dtype=out_dtype,
        device=mat1.device,
    )


@torch.library.custom_op(
    "spyre::scaled_paged_attn_store", mutates_args=(), device_types="cpu"
)
def scaled_paged_attn_store(
    key: Tensor,
    value: Tensor,
    key_cache: Tensor,
    value_cache: Tensor,
    key_scale: Tensor,
    value_scale: Tensor,
    slot_mapping: Tensor,
) -> tuple[Tensor, Tensor]:
    """
    FP8 CPU implementation of the Paged Attn store operation.
    Scales key and value tensors, and stores them to the paged KV cache
    using the same schema as vLLM.
    """
    result_key_cache = key_cache.clone()
    result_value_cache = value_cache.clone()
    for seq_i, slot_mapping_seq in enumerate(slot_mapping):
        for tok_i, slot in enumerate(slot_mapping_seq):
            block_number = slot.item() // 64
            position = slot.item() % 64

            result_key_cache[block_number, position, :, :] = (
                key[seq_i, tok_i, :, :] / key_scale[seq_i]
            ).to(dtype=torch.float8_e4m3fn)
            result_value_cache[block_number, position, :, :] = (
                value[seq_i, tok_i, :, :] / value_scale[seq_i]
            ).to(dtype=torch.float8_e4m3fn)
    return result_key_cache, result_value_cache


@scaled_paged_attn_store.register_fake
def scaled_paged_attn_store_meta(
    key: Tensor,
    value: Tensor,
    key_cache: Tensor,
    value_cache: Tensor,
    key_scale: Tensor,
    value_scale: Tensor,
    slot_mapping: Tensor,
) -> tuple[Tensor, Tensor]:
    """
    Fake tensor implementation of the FP8 Paged Attn store operation.
    """
    return key_cache, value_cache


@torch.library.custom_op(
    "spyre::scaled_paged_attn_compute", mutates_args={}, device_types="cpu"
)
def scaled_paged_attn_compute(
    query: Tensor,
    key_cache: Tensor,
    value_cache: Tensor,
    key_scale: Tensor,
    value_scale: Tensor,
    scale: float,
    current_tkv_mask: Tensor,
    left_padded_prompt_mask: Tensor,
    block_table: Tensor,
) -> Tensor:
    """
    FP8 CPU implementation of the Paged Attn compute operation.
    Implements a CPU fallback to run the kernel that has been confirmed
    to match the vLLM fused kernel.
    """
    # torch.zeros(NUM_BLOCKS, BLOCK_SIZE, kvheads, head_size, dtype=model_dtype),
    output = torch.zeros_like(query)
    num_query_heads = query.shape[2]
    num_kv_heads = value_cache.shape[2]
    head_size = value_cache.shape[3]
    block_size = value_cache.shape[1]
    num_seqs = query.shape[0]

    block_tables_lst = block_table.cpu().tolist()

    seq_lens_lst = current_tkv_mask.cpu().tolist()
    for i in range(num_seqs):
        q = query[i]
        block_table = block_tables_lst[i]
        start_pos = int(left_padded_prompt_mask[i].item())
        seq_len = int(seq_lens_lst[i])

        keys_lst: list[torch.Tensor] = []
        values_lst: list[torch.Tensor] = []
        for j in range(start_pos, seq_len):
            block_number = int(block_table[j // block_size])
            block_offset = j % block_size

            k = key_cache[block_number, block_offset, :, :]
            k = k.reshape(num_kv_heads, head_size)
            keys_lst.append(k)

            v = value_cache[block_number, block_offset, :, :]
            values_lst.append(v)
        keys = torch.stack(keys_lst, dim=0)
        values = torch.stack(values_lst, dim=0)
        if num_kv_heads > 1:
            # Handle MQA and GQA
            keys = torch.repeat_interleave(keys, num_query_heads // num_kv_heads, dim=1)
            values = torch.repeat_interleave(
                values, num_query_heads // num_kv_heads, dim=1
            )

        out = F.scaled_dot_product_attention(  # noqa: E1102
            q.transpose(0, 1).unsqueeze(0),  # format for sdpa
            (keys.transpose(0, 1).unsqueeze(0).to(dtype=q.dtype) * key_scale[i]).to(
                dtype=q.dtype
            ),  # format for sdpa
            (values.transpose(0, 1).unsqueeze(0).to(dtype=q.dtype) * value_scale[i]).to(
                dtype=q.dtype
            ),  # format for sdpa
            is_causal=False,  # decode assumes no causal mask
            scale=scale,
        )

        out = out.view(num_query_heads, head_size)
        output[i].copy_(out, non_blocking=True)
    return output


@scaled_paged_attn_compute.register_fake
def scaled_paged_attn_compute_meta(
    query: Tensor,
    key_cache: Tensor,
    value_cache: Tensor,
    key_scale: Tensor,
    value_scale: Tensor,
    scale: float,
    current_tkv_mask: Tensor,
    left_padded_prompt_mask: Tensor,
    block_table: Tensor,
) -> Tensor:
    """
    Fake tensor implementation of the FP8 Paged Attn compute operation.
    """
    return torch.zeros_like(query)
