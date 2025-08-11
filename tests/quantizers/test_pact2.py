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
Test PACT2 quantizers
"""

# Third Party
from test_quantizer_utils import quantizer_error, set_base_options
import pytest
import torch

###################################
# Custom Options for PACT2 family #
###################################

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


# Use to override conftest.py fixtures with parametrize
custom_option_params = []

###############
# PACT2 tests #
###############


def test_pact2_asymmetric(
    tensor: torch.Tensor,
    pact2_quantizer_asymmetric: torch.autograd.Function,
    torch_quantizer_asymmetric: torch.nn.Module,
    base_options: dict,
    other_options: dict,
):
    """
    Test PACT2 w/ asymmetric tensors

    Args:
        tensor (torch.Tensor): Tensor to quantize.
        pact2_quantizer_asymmetric (torch.autograd.Function): PACT2 Quantizer
        torch_quantizer_asymmetric (torch.nn.Module): Torch Quantizer
        base_options (dict): Base options for quantization.
        other_options (dict, optional): Other Options for quantization. Defaults to None.
    """
    # Set base quantizer and other options
    set_base_options(
        pact2_quantizer_asymmetric, torch_quantizer_asymmetric, base_options
    )
    set_other_options(pact2_quantizer_asymmetric, other_options)

    # Create quantized tensors from PACT + torch
    qtensor_pact2 = pact2_quantizer_asymmetric(tensor).detach()
    qtensor_torch = torch_quantizer_asymmetric(tensor).detach()

    setup = torch_quantizer_asymmetric.get_setup()

    # There should be no differences between these two tensors
    quantizer_error(
        tensor, qtensor_pact2, qtensor_torch, setup, base_options, other_options
    )


def test_pact2_rc_asymmetric(
    tensor: torch.Tensor,
    pact2_rc_quantizer_asymmetric: torch.autograd.Function,
    torch_quantizer_asymmetric: torch.nn.Module,
    base_options: dict,
    other_options: dict,
):
    """
    Test PACT2_rc w/ asymmetric tensors

    Args:
        tensor (torch.Tensor): Tensor to quantize.
        pact2_rc_quantizer_asymmetric (torch.autograd.Function): PACT2 Quantizer
        torch_quantizer_asymmetric (torch.nn.Module): Torch Quantizer
        base_options (dict): Base options for quantization.
        other_options (dict, optional): Other Options for quantization. Defaults to None.
    """
    # Set base quantizer and other options
    set_base_options(
        pact2_rc_quantizer_asymmetric, torch_quantizer_asymmetric, base_options
    )
    set_other_options(pact2_rc_quantizer_asymmetric, other_options)

    # Create quantized tensors from PACT + torch
    qtensor_pact2 = pact2_rc_quantizer_asymmetric(tensor).detach()
    qtensor_torch = torch_quantizer_asymmetric(tensor).detach()

    setup = torch_quantizer_asymmetric.get_setup()

    # There should be no differences between these two tensors
    quantizer_error(
        tensor, qtensor_pact2, qtensor_torch, setup, base_options, other_options
    )
