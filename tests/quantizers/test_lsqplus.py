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
Test LSQ+ quantizers
"""

# Third Party
from test_quantizer_utils import quantizer_error, set_base_options
import torch


def set_other_options(
    tensor: torch.Tensor,
    torch_quantizer: torch.nn.Module,
):
    """
    Set other options for FMS and Torch quantizers

    Args:
        tensor (torch.Tensor): Tensor to be quantized.
        torch_quantizer (torch.nn.Module): Torch Quantizer.
    """
    tensor_min, tensor_max = tensor.min(), tensor.max()
    torch_quantizer.n_levels = (
        2 ** (torch_quantizer.num_bits) - 1
    )  # override qlevel_lowering
    torch_quantizer.scale = (tensor_max - tensor_min) / torch_quantizer.n_levels

    # Torch requires integer zero_points ; this is NOT compatible with LSQPlus w/o variances
    torch_quantizer.zero_point = torch.round(
        tensor_min + 2 ** (torch_quantizer.num_bits - 1) * torch_quantizer.scale
    ).to(torch.int)
    torch_quantizer.symmetric_nlevel = 0
    torch_quantizer.is_symmetric = True  # force symmetric even if using asymmetric data
    torch_quantizer.set_quant_range()


def test_lsqplus_symmetric(
    tensor: torch.Tensor,
    lsqplus_quantizer_symmetric: torch.autograd.Function,
    torch_quantizer_symmetric: torch.nn.Module,
    base_options: dict,
    other_options: dict = None,
):
    """
    Test LSQ+ w/ symmetric tensors

    Args:
        tensor (torch.Tensor): Tensor to quantize.
        lsqplus_quantizer_symmetric (torch.autograd.Function): LSQ+ Quantizer
        torch_quantizer_symmetric (torch.nn.Module): Torch Quantizer
        base_options (dict): Base options for quantization.
        other_options (dict, optional): Other Options for quantization. Defaults to None.
    """
    # Set base quantizer and other options
    native_pt = base_options["nativePT"]
    base_options["nativePT"] = False  # Override: not supported in LSQ
    set_base_options(
        lsqplus_quantizer_symmetric, torch_quantizer_symmetric, base_options
    )
    set_other_options(tensor, torch_quantizer_symmetric)

    # Create quantized tensors from FMS Model Optimizer + torch
    qtensor_fms_mo = lsqplus_quantizer_symmetric(tensor).detach()
    qtensor_torch = torch_quantizer_symmetric(tensor).detach()

    setup = torch_quantizer_symmetric.get_setup()

    quantizer_error(
        tensor, qtensor_fms_mo, qtensor_torch, setup, base_options, other_options
    )
    base_options["nativePT"] = native_pt


def test_lsqplus_asymmetric(
    tensor: torch.Tensor,
    lsqplus_quantizer_asymmetric: torch.autograd.Function,
    torch_quantizer_asymmetric: torch.nn.Module,
    base_options: dict,
    other_options: dict = None,
):
    """
    Test LSQ+ w/ asymmetric tensors

    Args:
        tensor (torch.Tensor): Tensor to quantize.
        lsqplus_quantizer_asymmetric (torch.autograd.Function): LSQ+ Quantizer
        torch_quantizer_asymmetric (torch.nn.Module): Torch Quantizer
        base_options (dict): Base options for quantization.
        other_options (dict, optional): Other Options for quantization. Defaults to None.
    """
    # Set base quantizer and other options
    native_pt = base_options["nativePT"]
    base_options["nativePT"] = False  # Override: not supported in LSQ
    set_base_options(
        lsqplus_quantizer_asymmetric, torch_quantizer_asymmetric, base_options
    )
    set_other_options(tensor, torch_quantizer_asymmetric)

    # Create quantized tensors from FMS Model Optimizer + torch
    qtensor_fms_mo = lsqplus_quantizer_asymmetric(tensor).detach()
    qtensor_torch = torch_quantizer_asymmetric(tensor).detach()

    setup = torch_quantizer_asymmetric.get_setup()

    quantizer_error(
        tensor, qtensor_fms_mo, qtensor_torch, setup, base_options, other_options
    )
    base_options["nativePT"] = native_pt


def test_lsqplus_rc_symmetric(
    tensor: torch.Tensor,
    lsqplus_rc_quantizer_symmetric: torch.autograd.Function,
    torch_quantizer_symmetric: torch.nn.Module,
    base_options,
    other_options=None,
):
    """
    Test LSQ+_rc w/ symmetric tensors

    Args:
        tensor (torch.Tensor): Tensor to quantize.
        lsqplus_rc_quantizer_symmetric (torch.autograd.Function): LSQ+ Quantizer
        torch_quantizer_symmetric (torch.nn.Module): Torch Quantizer
        base_options (dict): Base options for quantization.
        other_options (dict, optional): Other Options for quantization. Defaults to None.
    """
    # Set base quantizer and other options
    native_pt = base_options["nativePT"]
    base_options["nativePT"] = False  # Override: not supported in LSQ
    set_base_options(
        lsqplus_rc_quantizer_symmetric, torch_quantizer_symmetric, base_options
    )
    set_other_options(tensor, torch_quantizer_symmetric)

    # Create quantized tensors from FMS Model Optimizer + torch
    qtensor_fms_mo = lsqplus_rc_quantizer_symmetric(tensor).detach()
    qtensor_torch = torch_quantizer_symmetric(tensor).detach()

    setup = torch_quantizer_symmetric.get_setup()

    quantizer_error(
        tensor, qtensor_fms_mo, qtensor_torch, setup, base_options, other_options
    )
    base_options["nativePT"] = native_pt


def test_lsqplus_rc_asymmetric(
    tensor,
    lsqplus_rc_quantizer_asymmetric: torch.autograd.Function,
    torch_quantizer_asymmetric: torch.nn.Module,
    base_options: dict,
    other_options: dict = None,
):
    """
    Test LSQ+_rc w/ symmetric tensors

    Args:
        tensor (torch.Tensor): Tensor to quantize.
        lsqplus_rc_quantizer_asymmetric (torch.autograd.Function): LSQ+ Quantizer
        torch_quantizer_asymmetric (torch.nn.Module): Torch Quantizer
        base_options (dict): Base options for quantization.
        other_options (dict, optional): Other Options for quantization. Defaults to None.
    """
    # Set base quantizer and other options
    native_pt = base_options["nativePT"]
    base_options["nativePT"] = False  # Override: not supported in LSQ
    set_base_options(
        lsqplus_rc_quantizer_asymmetric, torch_quantizer_asymmetric, base_options
    )
    set_other_options(
        tensor,
        torch_quantizer_asymmetric,
    )

    # Create quantized tensors from FMS Model Optimizer + torch
    qtensor_fms_mo = lsqplus_rc_quantizer_asymmetric(tensor).detach()
    qtensor_torch = torch_quantizer_asymmetric(tensor).detach()

    setup = torch_quantizer_asymmetric.get_setup()

    quantizer_error(
        tensor, qtensor_fms_mo, qtensor_torch, setup, base_options, other_options
    )
    base_options["nativePT"] = native_pt
