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

"""
Test LSQ quantizers
"""

# Third Party
from test_quantizer_utils import quantizer_error, set_base_options
import torch


# Tests LSQQuantization
def test_lsq_single_sided(
    tensor_single_sided: torch.Tensor,
    lsq_quantizer_single_sided: torch.autograd.Function,
    torch_quantizer_single_sided: torch.nn.Module,
    base_options: dict,
    other_options: dict = None,
):
    """
    Test LSQ w/ single-sided tensors

    Args:
        tensor_single_sided (torch.Tensor): Tensor to quantize.
        lsq_quantizer_single_sided (torch.autograd.Function): LSQ Quantizer
        torch_quantizer_single_sided (torch.nn.Module): Torch Quantizer
        base_options (dict): Base options for quantization.
        other_options (dict, optional): Other Options for quantization. Defaults to None.
    """
    # Set base quantizer and other options
    native_pt = base_options["nativePT"]
    base_options["nativePT"] = False  # Override: not supported in LSQ
    set_base_options(
        lsq_quantizer_single_sided, torch_quantizer_single_sided, base_options
    )

    # Create quantized tensors from FMS Model Optimizer + torch
    qtensor_fms_mo = lsq_quantizer_single_sided(tensor_single_sided).detach()
    qtensor_torch = torch_quantizer_single_sided(tensor_single_sided).detach()

    setup = torch_quantizer_single_sided.get_setup()

    quantizer_error(
        tensor_single_sided,
        qtensor_fms_mo,
        qtensor_torch,
        setup,
        base_options,
        other_options,
    )
    base_options["nativePT"] = native_pt


def test_lsq_rc_single_sided(
    tensor_single_sided: torch.Tensor,
    lsq_rc_quantizer_single_sided: torch.autograd.Function,
    torch_quantizer_single_sided: torch.nn.Module,
    base_options: dict,
    other_options: dict = None,
):
    """
    Test LSQ_rc w/ single-sided tensors

    Args:
        tensor_single_sided (torch.Tensor): Tensor to quantize.
        lsq_rc_quantizer_single_sided (torch.autograd.Function): LSQ Quantizer
        torch_quantizer_single_sided (torch.nn.Module): Torch Quantizer
        base_options (dict): Base options for quantization.
        other_options (dict, optional): Other Options for quantization. Defaults to None.
    """
    # Set base quantizer and other options
    native_pt = base_options["nativePT"]
    base_options["nativePT"] = False  # Override: not supported in LSQ
    set_base_options(
        lsq_rc_quantizer_single_sided, torch_quantizer_single_sided, base_options
    )

    # Create quantized tensors from FMS Model Optimizer + torch
    qtensor_fms_mo = lsq_rc_quantizer_single_sided(tensor_single_sided).detach()
    qtensor_torch = torch_quantizer_single_sided(tensor_single_sided).detach()

    setup = torch_quantizer_single_sided.get_setup()

    quantizer_error(
        tensor_single_sided,
        qtensor_fms_mo,
        qtensor_torch,
        setup,
        base_options,
        other_options,
    )
    base_options["nativePT"] = native_pt
