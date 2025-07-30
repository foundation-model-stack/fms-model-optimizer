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
Test PACT quantizers
"""

# Third Party
from test_quantizer_utils import quantizer_error, set_base_options
import pytest
import torch

##################################
# Custom Options for PACT family #
##################################

other_options_params = []
for pact_plus in [True, False]:
    for align_zero in [True]:
        other_options_params.append({"pact_plus": pact_plus, "align_zero": align_zero})


@pytest.fixture(params=other_options_params)
def other_options(request):
    """
    Fixture for other options available for PACT

    Args:
        request (dict): Single Other Option param

    Returns:
        dict: Single Other Option param
    """
    return request.param


# Must override this method for each test file
def set_other_options(fms_mo_quantizer, other_option):
    """
    Set other options for FMS and Torch quantizer

    Args:
        fms_mo_quantizer (torch.autograd.Function):  FMS quantizer
        other_option (dict): Other Option params
    """
    fms_mo_quantizer.pact_plus = other_option["pact_plus"]
    fms_mo_quantizer.align_zero = other_option["align_zero"]
    fms_mo_quantizer.set_quantizer()


##############
# PACT tests #
##############


def test_pact_single_sided(
    tensor_single_sided: torch.Tensor,
    pact_quantizer_single_sided: torch.autograd.Function,
    torch_quantizer_single_sided: torch.nn.Module,
    base_options: dict,
    other_options: dict,
):
    """
    Test PACT w/ single-sided tensors

    Args:
        tensor_single_sided (torch.Tensor): Tensor to quantize.
        pact_quantizer_single_sided (torch.autograd.Function): PACT Quantizer
        torch_quantizer_single_sided (torch.nn.Module): Torch Quantizer
        base_options (dict): Base options for quantization.
        other_options (dict, optional): Other Options for quantization. Defaults to None.
    """
    # Set base quantizer and other options
    # Override: not supported in PACT, but restore after
    native_pt = base_options["nativePT"]
    base_options["nativePT"] = False
    set_base_options(
        pact_quantizer_single_sided, torch_quantizer_single_sided, base_options
    )
    set_other_options(pact_quantizer_single_sided, other_options)

    # Create quantized tensors from FMS Model Optimizer + torch
    qtensor_fms_mo = pact_quantizer_single_sided(tensor_single_sided).detach()
    qtensor_torch = torch_quantizer_single_sided(tensor_single_sided).detach()

    setup = torch_quantizer_single_sided.get_setup()

    # There should be no differences between these two tensors
    quantizer_error(
        tensor_single_sided,
        qtensor_fms_mo,
        qtensor_torch,
        setup,
        base_options,
        other_options,
    )
    base_options["nativePT"] = native_pt  # reset value


def test_pact_rc_single_sided(
    tensor_single_sided: torch.Tensor,
    pact_rc_quantizer_single_sided: torch.autograd.Function,
    torch_quantizer_single_sided: torch.nn.Module,
    base_options: dict,
    other_options: dict,
):
    """
    Test PACT_rc w/ single-sided tensors

    Args:
        tensor_single_sided (torch.Tensor): Tensor to quantize.
        pact_rc_quantizer_single_sided (torch.autograd.Function): PACT Quantizer
        torch_quantizer_single_sided (torch.nn.Module): Torch Quantizer
        base_options (dict): Base options for quantization.
        other_options (dict, optional): Other Options for quantization. Defaults to None.
    """
    # Set base quantizer and other options
    set_base_options(
        pact_rc_quantizer_single_sided, torch_quantizer_single_sided, base_options
    )
    set_other_options(pact_rc_quantizer_single_sided, other_options)

    # Create quantized tensors from FMS Model Optimizer + torch
    qtensor_fms_mo = pact_rc_quantizer_single_sided(tensor_single_sided).detach()
    qtensor_torch = torch_quantizer_single_sided(tensor_single_sided).detach()

    setup = torch_quantizer_single_sided.get_setup()

    # There should be no differences between these two tensors
    quantizer_error(
        tensor_single_sided,
        qtensor_fms_mo,
        qtensor_torch,
        setup,
        base_options,
        other_options,
    )
