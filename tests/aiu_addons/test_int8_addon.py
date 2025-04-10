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
"""Test suite for FMS addon for AIU, introducing INT8xINT8 functionalities"""

# Third Party
import torch

# Local
from fms_mo.aiu_addons.i8i8.i8i8_aiu_op import register_aiu_i8i8_op


def test_i8i8_registration() -> None:
    """Call the registration function of INT8xINT8 operation, adding the op to torch
    namespace.
    Note: registration must be called before other INT8 tests that use this op.
    """

    register_aiu_i8i8_op()
    assert hasattr(torch.ops, "fms_mo")
    assert hasattr(torch.ops.fms_mo, "i8i8_aiu")


def test_i8i8_op(
    get_i8i8_gemm_inputs: tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        str,
        str,
        bool,
        torch.Tensor,
    ],
) -> None:
    """Validate output shapes and content of INT8xINT8 matmul.
    Computations are simulated, using quantized/dequantized tensors.
    """

    (
        x,
        weight,
        bias,
        qdata,
        weight_quant_type,
        activ_quant_type,
        smoothquant,
        reference_out,
    ) = get_i8i8_gemm_inputs

    # enforce fp16 dtype on all fp parameters for this test
    x = x.to(torch.float16)
    qdata = qdata.to(torch.float16)

    out = torch.ops.fms_mo.i8i8_aiu(
        x,
        weight,
        bias,
        qdata,
        weight_quant_type,
        activ_quant_type,
        smoothquant,
    )

    error_tolerance = 1e-4  # TODO: this needs adjusting
    assert out.size() == x.size()[:-1] + (weight.size(0),)
    assert torch.all((out - reference_out).abs() < error_tolerance)
    # assert torch.linalg.norm(out - reference_out) < error_tolerance  # alternative check
