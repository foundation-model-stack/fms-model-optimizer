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
Test PACT2Sym quantizers
"""

# Third Party
from test_quantizer_utils import quantizer_error, set_base_options
import torch

######################################
# Custom Options for PACT2Sym family #
######################################

# PACT2Sym does not support pact_plus, align_zero


# Must override this method for each test file
def set_other_options(torch_quantizer):
    """
    Set other options for Torch quantizer

    Args:
        torch_quantizer (torch.nn.Module):  Torch quantizer
    """
    torch_quantizer.set_qlevel_lowering(True)


##################
# PACT2Sym tests #
##################


def test_pact2sym_symmetric(
    tensor: torch.Tensor,
    pact2sym_quantizer_symmetric: torch.autograd.Function,
    torch_quantizer_symmetric: torch.nn.Module,
    base_options: dict,
    other_options: dict = None,
):
    """
    Test PACT2Sym w/ aymmetric tensors

    Args:
        tensor (torch.Tensor): Tensor to quantize.
        pact2sym_quantizer_symmetric (torch.autograd.Function): PACT2Sym Quantizer
        torch_quantizer_symmetric (torch.nn.Module): Torch Quantizer
        base_options (dict): Base options for quantization.
        other_options (dict, optional): Other Options for quantization. Defaults to None.
    """
    # Set base quantizer and other options
    # Override: not supported in PACT2Sym, but restore after
    native_pt = base_options["nativePT"]
    base_options["nativePT"] = False
    set_base_options(
        pact2sym_quantizer_symmetric, torch_quantizer_symmetric, base_options
    )
    set_other_options(torch_quantizer_symmetric)

    # Create quantized tensors from FMS Model Optimizer + torch
    qtensor_fms_mo = pact2sym_quantizer_symmetric(tensor).detach()
    qtensor_torch = torch_quantizer_symmetric(tensor).detach()

    setup = torch_quantizer_symmetric.get_setup()

    # There should be no differences between these two tensors
    quantizer_error(
        tensor, qtensor_fms_mo, qtensor_torch, setup, base_options, other_options
    )
    base_options["nativePT"] = native_pt  # reset value


def test_pact2sym_rc_symmetric(
    tensor,
    pact2sym_rc_quantizer_symmetric,
    torch_quantizer_symmetric,
    base_options: dict,
    other_options: dict = None,
):
    """
    Test PACT2Sym_rc w/ ymmetric tensors

    Args:
        tensor (torch.Tensor): Tensor to quantize.
        pact2sym_rc_quantizer_symmetric (torch.autograd.Function): PACT2Sym_rc Quantizer
        torch_quantizer_symmetric (torch.nn.Module): Torch Quantizer
        base_options (dict): Base options for quantization.
        other_options (dict, optional): Other Options for quantization. Defaults to None.
    """
    # Set base quantizer and other options
    set_base_options(
        pact2sym_rc_quantizer_symmetric, torch_quantizer_symmetric, base_options
    )
    set_other_options(torch_quantizer_symmetric)

    # Create quantized tensors from FMS Model Optimizer + torch
    qtensor_fms_mo = pact2sym_rc_quantizer_symmetric(tensor).detach()
    qtensor_torch = torch_quantizer_symmetric(tensor).detach()

    setup = torch_quantizer_symmetric.get_setup()

    # There should be no differences between these two tensors
    quantizer_error(
        tensor, qtensor_fms_mo, qtensor_torch, setup, base_options, other_options
    )
