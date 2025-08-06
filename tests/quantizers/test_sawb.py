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
Test SAWB quantizers
"""

# Third Party
from test_quantizer_utils import quantizer_error, set_base_options
import pytest
import torch

# Local
from fms_mo.quant_refactor.quantizers_new import SAWB
from fms_mo.quant_refactor.sawb_rc import SAWB_rc
from fms_mo.quant_refactor.torch_quantizer import TorchQuantizer

other_option_params = []
for clipSTE in [True, False]:
    for align_zero in [True, False]:
        for use16bins in [False]:
            other_option_params.append(
                {"clipSTE": clipSTE, "align_zero": align_zero, "use16bins": use16bins}
            )


@pytest.fixture(params=other_option_params)
def other_options(request):
    """
    Fixture for other options available for SAWB

    Args:
        request (dict): Single Other Option param

    Returns:
        dict: Single Other Option param
    """
    return request.param


# Must override this method for each test file
def set_other_options(
    tensor: torch.Tensor,
    fms_mo_quantizer: torch.autograd.Function,
    torch_quantizer: torch.nn.Module,
    other_option: dict,
    axis: int = 0,
):
    """
    Set other options for FMS and Torch quantizer

    Args:
        tensor (torch.Tensor): Tensor to quantize.
        fms_mo_quantizer (torch.autograd.Function):  FMS quantizer
        torch_quantizer (torch.nn.Module): Torch Quantizer
        other_option (dict): Other Option params
        axis (int, optional): Per channel axis dimension. Defaults to 0.
    """
    fms_mo_quantizer.clipSTE = other_option["clipSTE"]
    fms_mo_quantizer.align_zero = other_option["align_zero"]
    fms_mo_quantizer.extended_range = other_option["use16bins"]
    fms_mo_quantizer.set_quantizer()

    # Set n_levels, scale, zero_point for torch_quantizer
    num_bits = torch_quantizer.num_bits

    # For SAWB Zero STEs
    if other_option["align_zero"]:
        # SAWBPlus16ZeroSTE - no sawb_params
        if (
            torch_quantizer.qscheme.q_unit == "perT"
            and other_option["clipSTE"]
            and other_option["use16bins"]
            and num_bits == 4
        ):
            # Set num_bits and [clip_low,clip_high]
            torch_quantizer.n_levels = 2**num_bits - 1
            torch_quantizer.quant_min = -8
            torch_quantizer.quant_max = 7
            torch_quantizer.set_sawb_clip_code(tensor)

            # Scale uses clip_high
            torch_quantizer.scale = (
                torch_quantizer.clip_high * (8.0 / 7.0 + 1.0) / torch_quantizer.n_levels
            )
            torch_quantizer.zero_point = torch.tensor(0)
            torch_quantizer.set_shift_sawb(0)

            # Do not call set_quant_range() ; overriden w/ fixed [qint_min, qint_max]
            return

        # SAWBPlusZeroPerChSTE - TODO: perCh test not functional yet
        elif other_option["clipSTE"] and torch_quantizer.qscheme.q_unit == "perCh":
            Nch = tensor.shape[axis]

            torch_quantizer.qscheme.q_unit = "perCh"
            torch_quantizer.qscheme.Nch = Nch
            torch_quantizer.qscheme.qlevel_lowering = True
            torch_quantizer.set_sawb_clip_code(tensor, perCh=True)  # sets clip vals

        else:  # SAWBZeroSTE, SAWBPlusZeroSTE - sawb_params_code
            # Set num_bits and [clip_low,clip_high]
            torch_quantizer.n_levels = 2**num_bits - 2
            torch_quantizer.set_sawb_clip_code(tensor)

            torch_quantizer.scale = 1.0 / torch_quantizer.n_levels
            torch_quantizer.zero_point = torch.tensor(0)
            torch_quantizer.set_shift_sawb(torch_quantizer.n_levels / 2)
            torch_quantizer.is_single_sided = True
            torch_quantizer.is_symmetric = False

    else:  # SAWBSTE, SAWBPlusSTE ; standard scale,zero_point for [0,1] squashing
        # Set num_bits and [clip_low,clip_high]
        torch_quantizer.n_levels = 2**num_bits - 1
        torch_quantizer.set_sawb_clip(tensor)

        torch_quantizer.scale = 1.0 / torch_quantizer.n_levels
        torch_quantizer.zero_point = torch.tensor(0)
        torch_quantizer.set_shift_sawb(2 ** (num_bits - 1))
        torch_quantizer.is_single_sided = True
        torch_quantizer.is_symmetric = False

    # Set quant_min,quant_max based on n_levels, scale, zero_point
    torch_quantizer.set_quant_range()


def use_tensor_squashing(
    other_option: dict,
    num_bits: int,
    is_perCh: bool,
):
    """
    Specify if tensor squashing is needed (not SAWBPlus16ZeroSTE, SAWBPlusZeroPerChSTE)

    Args:
        other_option (dict): Other Option params
        num_bits (int): Number of bits for quantization
        is_perCh (bool): If qscheme.q_unit == "perCh"

    Returns:
        bool: Specify if tensor needs to be squashed
    """
    is_plus_zero = other_option["clipSTE"] and other_option["align_zero"]
    is_SAWBPlus16ZeroSTE = is_plus_zero and other_option["use16bins"] and num_bits == 4
    is_SAWBPlusZeroPerChSTE = is_plus_zero and is_perCh
    return not (is_SAWBPlus16ZeroSTE or is_SAWBPlusZeroPerChSTE)


# SAWB_new settings for TorchQuantizer ; refactored SAWB does not use tensor squashing
def set_other_options_rc(
    tensor: torch.Tensor,
    fms_mo_quantizer: torch.autograd.Function,
    torch_quantizer: torch.nn.Module,
    other_option: dict,
    axis: int = 0,
):
    """
    Set other options for _rc FMS and Torch quantizer

    Args:
        tensor (torch.Tensor): Tensor to quantize.
        fms_mo_quantizer (torch.autograd.Function):  FMS quantizer
        torch_quantizer (torch.nn.Module): Torch Quantizer
        other_option (dict): Other Option params
        axis (int, optional): Per channel axis dimension. Defaults to 0.
    """
    fms_mo_quantizer.clipSTE = other_option["clipSTE"]
    fms_mo_quantizer.align_zero = other_option["align_zero"]
    fms_mo_quantizer.use16bins = other_option["use16bins"]

    fms_mo_quantizer.recompute_clips = True  # Allow computing clips in forward
    fms_mo_quantizer.eval()  # set nn.Module to eval mode
    fms_mo_quantizer.set_quantizer()

    # Set n_levels, scale, zero_point for torch_quantizer
    num_bits = torch_quantizer.num_bits

    # For SAWB Zero STEs
    if other_option["align_zero"]:
        # SAWBPlus16ZeroSTE_rc - no sawb_params
        if (
            torch_quantizer.qscheme.q_unit == "perT"
            and other_option["clipSTE"]
            and other_option["use16bins"]
            and num_bits == 4
        ):
            # Set num_bits and [clip_low,clip_high]
            torch_quantizer.n_levels = 2**num_bits - 1
            torch_quantizer.set_sawb_clip_code(tensor, code=403)  # sets clip_high
            torch_quantizer.scale = (
                torch_quantizer.clip_high * (8.0 / 7.0 + 1.0) / torch_quantizer.n_levels
            )
            torch_quantizer.zero_point = torch.tensor(0)

            # override quant_min calculation w/o resetting qlevel_lowering in qscheme
            torch_quantizer.symmetric_nlevel = 0
            torch_quantizer.set_quant_range()

        # SAWBPlusZeroPerChSTE_rc - TODO: perCh test not functional yet
        elif (
            torch_quantizer.qscheme.q_unit == "perCh"
            and other_option["clipSTE"]
            and not other_option["use16bins"]
        ):
            Nch = tensor.shape[axis]

            torch_quantizer.qscheme.q_unit = "perCh"
            torch_quantizer.qscheme.Nch = Nch
            torch_quantizer.qscheme.qlevel_lowering = True
            torch_quantizer.set_sawb_clip_code(tensor, perCh=True)  # sets clip vals
            torch_quantizer.set_quant_bounds()

        else:  # SAWBZeroSTE_rc, SAWBPlusZeroSTE_rc - sawb_params_code
            torch_quantizer.set_sawb_clip_code(tensor)
            torch_quantizer.set_quant_bounds()

    else:  # SAWBSTE_rc, SAWBPlusSTE_rc
        # Set num_bits and [clip_low,clip_high]
        torch_quantizer.n_levels = 2**num_bits - 1
        torch_quantizer.set_sawb_clip(tensor)

        torch_quantizer.set_quant_bounds()


def set_per_channel(
    tensor: torch.Tensor,
    fms_mo_quantizer: torch.autograd.Function,
    torch_quantizer: torch.nn.Module,
    axis: int = 0,
):
    """
    Setup quantizers to use per channel SAWB

    Args:
        tensor (torch.Tensor): Tensor to quantize.
        fms_mo_quantizer (torch.autograd.Function): FMS quantizer.
        torch_quantizer (torch.nn.Module): Torch Quantizer
        axis (int, optional): Per channel axis dimension. Defaults to 0.
    """
    Nch = tensor.shape[axis]

    # Setup quantizer to use SAWBPlusZeroPerChSTE
    fms_mo_quantizer.clipSTE = True
    fms_mo_quantizer.align_zero = True
    fms_mo_quantizer.recompute_clips = True
    fms_mo_quantizer.set_quantizer()

    torch_quantizer.qscheme.q_unit = "perCh"
    torch_quantizer.qscheme.Nch = Nch
    torch_quantizer.qscheme.qlevel_lowering = True
    torch_quantizer.set_sawb_clip_code(tensor, perCh=True)  # sets clip vals
    torch_quantizer.set_quant_bounds()


##############
# SAWB tests #
##############


def test_sawb_symmetric(
    tensor: torch.Tensor,
    sawb_quantizer_symmetric: torch.autograd.Function,
    torch_quantizer_symmetric: torch.nn.Module,
    base_options: dict,
    other_options: dict,
):
    """
    Test SAWB w/ symmetric tensors

    Args:
        tensor (torch.Tensor): Tensor to quantize.
        sawb_quantizer_symmetric (torch.autograd.Function): Qmax Quantizer
        torch_quantizer_symmetric (torch.nn.Module): Torch Quantizer
        base_options (dict): Base options for quantization.
        other_options (dict): Other Options for quantization.
    """

    # Set base quantizer and SAWB options ; save nativePT
    native_pt = base_options["nativePT"]
    base_options["nativePT"] = False  # Not supported for SAWB
    set_base_options(sawb_quantizer_symmetric, torch_quantizer_symmetric, base_options)
    # SAWB requires tensor to set parameters for TorchQuantizer

    use16bins = other_options["use16bins"]
    other_options["use16bins"] = False  # Not implemented for quantizer_rc.SAWB
    set_other_options(
        tensor, sawb_quantizer_symmetric, torch_quantizer_symmetric, other_options
    )

    # Create quantized tensors from FMS Model Optimizer + torch
    qtensor_fms_mo = sawb_quantizer_symmetric(tensor).detach()

    # Squash tensor to [0,1] before forward if needed
    use_squashing = use_tensor_squashing(
        other_options,
        torch_quantizer_symmetric.num_bits,
        torch_quantizer_symmetric.qscheme.q_unit == "perCh",
    )
    if use_squashing:
        tensor2 = torch_quantizer_symmetric.squash_tensor_sawb(tensor)
    else:
        tensor2 = tensor

    qtensor_torch = torch_quantizer_symmetric(tensor2).detach()

    if base_options["dequantize"]:
        # Unsquash tensor to [-clip,clip] after forward for floats and symmetric qlevels for ints
        if use_squashing:
            qtensor_torch = torch_quantizer_symmetric.unsquash_tensor_sawb(
                qtensor_torch
            )
    else:
        qtensor_torch = torch_quantizer_symmetric.shift_qtensor_sawb(qtensor_torch)

    setup = torch_quantizer_symmetric.get_setup()

    # There should be no differences between these two tensors
    quantizer_error(
        tensor, qtensor_fms_mo, qtensor_torch, setup, base_options, other_options
    )
    base_options["nativePT"] = native_pt  # reset value
    other_options["use16bins"] = use16bins


def test_sawb_rc_symmetric(
    tensor: torch.Tensor,
    sawb_rc_quantizer_symmetric: torch.autograd.Function,
    torch_quantizer_symmetric: torch.nn.Module,
    base_options: dict,
    other_options: dict,
):
    """
    Test SAWB_rc w/ symmetric tensors

    Args:
        tensor (torch.Tensor): Tensor to quantize.
        sawb_rc_quantizer_symmetric (torch.autograd.Function): Qmax Quantizer
        torch_quantizer_symmetric (torch.nn.Module): Torch Quantizer
        base_options (dict): Base options for quantization.
        other_options (dict): Other Options for quantization.
    """
    # Set base quantizer and SAWB options
    set_base_options(
        sawb_rc_quantizer_symmetric, torch_quantizer_symmetric, base_options
    )
    # SAWB requires tensor to set parameters for TorchQuantizer

    # Set TorchQuantizer qlevel_lowering based on align_zero as temp
    qlevel_lowering = torch_quantizer_symmetric.qscheme.qlevel_lowering
    torch_quantizer_symmetric.qscheme.qlevel_lowering = other_options[
        "align_zero"
    ]  # use legacy setting for qlevel_lowering
    set_other_options_rc(
        tensor, sawb_rc_quantizer_symmetric, torch_quantizer_symmetric, other_options
    )

    # Create quantized tensors from FMS Model Optimizer + torch
    qtensor_fms_mo = sawb_rc_quantizer_symmetric(tensor).detach()
    qtensor_torch = torch_quantizer_symmetric(tensor).detach()

    setup = torch_quantizer_symmetric.get_setup()

    # There should be no differences between these two tensors
    quantizer_error(
        tensor, qtensor_fms_mo, qtensor_torch, setup, base_options, other_options
    )
    torch_quantizer_symmetric.qscheme.qlevel_lowering = qlevel_lowering  # reset value


def test_sawb_symmetric_perCh(
    tensor: torch.Tensor,
    quantizer_symmetric_perCh: dict,
    base_options: dict,
):
    """
    Test SAWB w/ symmetric tensors for per channel

    Args:
        tensor (torch.Tensor): Tensor to quantize.
        quantizer_symmetric_perCh (dict): Symmetric quantizer settings for per channel.
        base_options (dict): Base options for quantization.
    """

    Nch = tensor.shape[0]
    clip_val = torch.rand(Nch) + 2.5  # [2.5,3.5]

    # SAWB computes clip_val_vec in forward()
    sawb_quantizer_symmetric_perCh = SAWB(
        num_bits=quantizer_symmetric_perCh["num_bits"],
        perCh=Nch,
    )

    # Clip val is not optional, but gets overriden in set_per_channel
    torch_quantizer_symmetric_perCh = TorchQuantizer(
        num_bits=quantizer_symmetric_perCh["num_bits"],
        clip_low=-clip_val,
        clip_high=clip_val,
        qscheme=quantizer_symmetric_perCh["scheme"],
    )

    # Set base quantizer and SAWB options ; save nativePT
    native_pt = base_options["nativePT"]
    base_options["nativePT"] = False  # Not supported for SAWB

    # Create only set of other options for SAWB perCh STEs
    other_options = {"clipSTE": True, "align_zero": True, "use16bins": False}

    set_base_options(
        sawb_quantizer_symmetric_perCh, torch_quantizer_symmetric_perCh, base_options
    )
    set_per_channel(
        tensor, sawb_quantizer_symmetric_perCh, torch_quantizer_symmetric_perCh
    )

    # Create quantized tensors from FMS Model Optimizer + torch
    qtensor_fms_mo = sawb_quantizer_symmetric_perCh(tensor).detach()
    qtensor_torch = torch_quantizer_symmetric_perCh(tensor).detach()

    setup = torch_quantizer_symmetric_perCh.get_setup()

    # There should be no differences between these two tensors
    # SAWB uses torch functions, so zero out errors
    quantizer_error(
        tensor,
        qtensor_fms_mo,
        qtensor_torch,
        setup,
        base_options,
        other_options,
        max_norm_tol=0.0,
        l2_norm_tol=0.0,
        nonzero_tol=0.0,
    )
    base_options["nativePT"] = native_pt  # reset value


def test_sawb_rc_symmetric_perCh(
    tensor: torch.Tensor,
    quantizer_symmetric_perCh: dict,
    base_options: dict,
):
    """
    Test SAWB_rc w/ symmetric tensors for perCh

    Args:
        tensor (torch.Tensor): Tensor to quantize.
        base_options (dict): Base options for quantization.
        other_options (dict): Other Options for quantization.
    """
    Nch = tensor.shape[0]
    clip_val = torch.rand(Nch) + 2.5  # [2.5,3.5]

    # Need to set proper Nch; registered parameters can't change shape (Quantizer.init())
    qscheme = quantizer_symmetric_perCh["scheme"]
    qscheme.Nch = Nch

    # SAWB computes clip_val_vec in forward()
    sawb_rc_quantizer_symmetric_perCh = SAWB_rc(
        num_bits=quantizer_symmetric_perCh["num_bits"],
        init_clip_valn=-clip_val,
        init_clip_val=clip_val,
        qscheme=qscheme,
    )

    # Clip val is not optional, but gets overriden in set_per_channel
    torch_quantizer_symmetric_perCh = TorchQuantizer(
        num_bits=quantizer_symmetric_perCh["num_bits"],
        clip_low=-clip_val,
        clip_high=clip_val,
        qscheme=qscheme,
    )

    # Create only set of other options for SAWB perCh STEs
    other_options = {"clipSTE": True, "align_zero": True, "use16bins": False}

    # Set base quantizer and SAWB options
    set_base_options(
        sawb_rc_quantizer_symmetric_perCh, torch_quantizer_symmetric_perCh, base_options
    )
    # SAWB requires tensor to set parameters for TorchQuantizer
    set_per_channel(
        tensor, sawb_rc_quantizer_symmetric_perCh, torch_quantizer_symmetric_perCh
    )

    # Create quantized tensors from FMS Model Optimizer + torch
    qtensor_fms_mo = sawb_rc_quantizer_symmetric_perCh(tensor).detach()
    qtensor_torch = torch_quantizer_symmetric_perCh(tensor).detach()

    setup = torch_quantizer_symmetric_perCh.get_setup()

    quantizer_error(
        tensor,
        qtensor_fms_mo,
        qtensor_torch,
        setup,
        base_options,
        other_options,
    )
