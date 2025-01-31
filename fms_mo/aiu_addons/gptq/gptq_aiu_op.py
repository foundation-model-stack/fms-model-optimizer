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
"""Registration of GPTQ W4A16 node compatible with AIU compiler"""

# Standard
import logging

# Third Party
import torch

# pylint: disable=unused-argument
# gptq op must be registered with specific I/O, even if not in use by the op function

logger = logging.getLogger(__name__)


def register_aiu_gptq_op():
    """Register AIU-specific op to enable torch compile without graph break.
    The op preserves I/O shapes of a `X @ W^T` matmul but performs no operation.
    Quantization parameters are taken as arguments, so that they end up attached to
    the computational graph.
    """
    if hasattr(torch.ops, "gptq_gemm") and hasattr(
        torch.ops.gptq_gemm, "i4f16_fxinputs_aiu"
    ):
        logger.warning("AIU op has already been registered")
        return

    op_namespace_id = "gptq_gemm::i4f16_fxinputs_aiu"
    torch.library.define(
        op_namespace_id,
        "(Tensor x, Tensor qw, Tensor qzeros, Tensor scales, Tensor g_idx) -> Tensor",
    )

    # Add implementations for the operator
    @torch.library.impl(op_namespace_id, "default")
    def i4f16_fxinputs_aiu(x, qw, qzeros, scales, g_idx):
        # on AIU, GPTQ qw is [out_feat, in_feat]
        outshape = x.shape[:-1] + (qw.shape[0],)
        x = x.view(-1, x.shape[-1])
        output = torch.zeros(
            (x.shape[0], qw.shape[0]),
            dtype=torch.float16,
            device=x.device,
        )
        return output.view(outshape)

    @torch.library.impl_abstract(op_namespace_id)
    def i4f16_fxinputs_aiu_abstract(x, qw, qzeros, scales, g_idx):
        outshape = x.shape[:-1] + (qw.shape[0],)
        return torch.empty(
            outshape,
            dtype=torch.float16,
            device=x.device,
            requires_grad=False,
        )

    logger.info("GPTQ op 'i4f16_fxinputs_aiu' has been registered")
    return
