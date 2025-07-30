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
Test PACT+Sym quantizers
"""

# Third Party
from test_quantizer_utils import quantizer_error, set_base_options
import pytest
import torch

#########################################
# Custom Options for PACTplusSym family #
#########################################

# PACTplusSym does not support pact_plus, align_zero

other_options_params = []
for extend_act_range in [True, False]:
    other_options_params.append({"extend_act_range": extend_act_range})


@pytest.fixture(params=other_options_params)
def other_options(request):
    """
    Fixture for other options available for PACT+Sym

    Args:
        request (dict): Single Other Option param

    Returns:
        dict: Single Other Option param
    """
    return request.param


# Must override this method for each test file
def set_other_options(
    fms_mo_quantizer: torch.autograd.Function,
    torch_quantizer: torch.nn.Module,
    other_option: dict,
):
    """
    Set other options for FMS and Torch quantizer

    Args:
        fms_mo_quantizer (torch.autograd.Function):  FMS quantizer
        torch_quantizer_single_sided (torch.nn.Module): Torch Quantizer
        other_option (dict): Other Option params
    """
    fms_mo_quantizer.extend_act_range = other_option["extend_act_range"]
    fms_mo_quantizer.set_quantizer()

    # Set clip_valn for torch quantizer in extended range
    if other_option["extend_act_range"]:
        torch_quantizer.qscheme.qlevel_lowering = False  # uses all qlevels

        n_half = 2 ** (torch_quantizer.num_bits - 1)
        torch_quantizer.n_levels = 2**torch_quantizer.num_bits - 1
        torch_quantizer.clip_low = -torch_quantizer.clip_high * n_half / (n_half - 1)
        torch_quantizer.scale = torch_quantizer.clip_high / (n_half - 1)
        torch_quantizer.zero_point = torch.tensor(0)
        torch_quantizer.quant_min = -n_half
        torch_quantizer.quant_max = n_half - 1
    else:
        torch_quantizer.qscheme.qlevel_lowering = True  # reset
        torch_quantizer.set_quant_bounds()


#####################
# PACTplusSym tests #
#####################


def test_pactplussym_symmetric(
    tensor: torch.Tensor,
    pactplussym_quantizer_symmetric: torch.autograd.Function,
    torch_quantizer_symmetric: torch.nn.Module,
    base_options: dict,
    other_options: dict,
):
    """
    Test PACT+Sym w/ symmetric tensors

    Args:
        tensor (torch.Tensor): Tensor to quantize.
        pactplussym_quantizer_symmetric (torch.autograd.Function): PACT+Sym Quantizer
        torch_quantizer_symmetric (torch.nn.Module): Torch Quantizer
        base_options (dict): Base options for quantization.
        other_options (dict): Other Options for quantization.
    """
    # Set base quantizer and other options
    # Override: not supported in PACTplusSym, but restore after
    native_pt = base_options["nativePT"]
    base_options["nativePT"] = False
    set_base_options(
        pactplussym_quantizer_symmetric, torch_quantizer_symmetric, base_options
    )
    set_other_options(
        pactplussym_quantizer_symmetric, torch_quantizer_symmetric, other_options
    )

    # Create quantized tensors from FMS Model Optimizer + torch
    qtensor_fms_mo = pactplussym_quantizer_symmetric(tensor).detach()
    qtensor_torch = torch_quantizer_symmetric(tensor).detach()

    setup = torch_quantizer_symmetric.get_setup()

    # There should be no differences between these two tensors
    quantizer_error(
        tensor, qtensor_fms_mo, qtensor_torch, setup, base_options, other_options
    )
    base_options["nativePT"] = native_pt  # reset value


def test_pactplussym_rc_symmetric(
    tensor: torch.Tensor,
    pactplussym_rc_quantizer_symmetric: torch.autograd.Function,
    torch_quantizer_symmetric: torch.nn.Module,
    base_options: dict,
    other_options: dict,
):
    """
    Test PACT+Sym w/ symmetric tensors

    Args:
        tensor (torch.Tensor): Tensor to quantize.
        pactplussym_rc_quantizer_symmetric (torch.autograd.Function): PACT+Sym Quantizer
        torch_quantizer_symmetric (torch.nn.Module): Torch Quantizer
        base_options (dict): Base options for quantization.
        other_options (dict, optional): Other Options for quantization. Defaults to None.
    """
    # Set base quantizer and other options
    set_base_options(
        pactplussym_rc_quantizer_symmetric, torch_quantizer_symmetric, base_options
    )
    set_other_options(
        pactplussym_rc_quantizer_symmetric, torch_quantizer_symmetric, other_options
    )

    # Create quantized tensors from FMS Model Optimizer + torch
    qtensor_fms_mo = pactplussym_rc_quantizer_symmetric(tensor).detach()
    qtensor_torch = torch_quantizer_symmetric(tensor).detach()

    setup_torch = torch_quantizer_symmetric.get_setup()

    # There should be no differences between these two tensors
    quantizer_error(
        tensor, qtensor_fms_mo, qtensor_torch, setup_torch, base_options, other_options
    )
