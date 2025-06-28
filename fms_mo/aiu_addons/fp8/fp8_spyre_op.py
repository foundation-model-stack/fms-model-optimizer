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

# Third Party
from torch import Tensor
import torch

# pylint: disable=unused-argument
# abstract op must be registered with specific I/O, even if not in use by the op function


@torch.library.custom_op("spyre::scaled_bmm", mutates_args=())
def sendnn_scaled_bmm(
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


@sendnn_scaled_bmm.register_fake
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
