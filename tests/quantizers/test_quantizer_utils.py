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
Quantizer Testing Utility Functions
"""

# Standard
from typing import Union
import logging

# Third Party
import torch
import torch.linalg

logger = logging.getLogger(__name__)


def quantizer_setup(
    fms_mo_quantizer_setup: tuple,
    torch_quantizer_setup: tuple,
):
    """
    Check quantization parameter equality for FMS and Torch quantizers
    n_level, clip_valn, clip_val, scale, zero_point

    Args:
        fms_mo_quantizer_setup (tuple): FMS quantizer parameters
        torch_quantizer_setup (tuple): Torch quantizer parameters
    """
    (
        n_levels_fms_mo,
        clip_valn_fms_mo,
        clip_val_fms_mo,
        scale_fms_mo,
        zero_point_fms_mo,
    ) = fms_mo_quantizer_setup
    (
        _,
        clip_low_torch,
        clip_high_torch,
        scale_torch,
        zero_point_torch,
        n_level_torch,
        _,
        _,
        _,
    ) = torch_quantizer_setup

    assert torch.equal(n_levels_fms_mo, n_level_torch), "n_levels are not equal"
    assert torch.isclose(
        clip_valn_fms_mo, clip_low_torch
    ), "clip_low values are not close"
    assert torch.isclose(
        clip_val_fms_mo, clip_high_torch
    ), "clip_high values are not close"
    assert torch.isclose(scale_fms_mo, scale_torch), "scales values are not close"
    assert torch.equal(zero_point_fms_mo, zero_point_torch), "zero_points are not equal"


def quantizer_error(
    tensor: torch.Tensor,
    qtensor_fms_mo: Union[torch.Tensor, torch.Tensor],
    qtensor_torch: Union[torch.Tensor, torch.Tensor],
    setup,
    base_options,
    other_options,
    max_norm_tol=1e-5,
    l2_norm_tol=1e-2,
    nonzero_tol=1e-2,
    error_override=False,
):
    """
    Check various types of quantizer numerical errors for FMS and Torch quantizied tensors

    Args:
        tensor (torch.Tensor): Tensor to quantize.
        qtensor_fms_mo (Union[torch.Tensor, torch.Tensor]): Quantizied FMS tensor
        qtensor_torch (Union[torch.Tensor, torch.Tensor]): Quantizied Torch tensor
        setup (tuple): Quantized parameters
        base_options (dict): Base options dictionary
        other_options (dict): Other options dictionary
        max_norm_tol (float, optional): Max Norm tolerance. Defaults to 1e-5.
        l2_norm_tol (float, optional): L2 Norm tolerance. Defaults to 1e-2.
        nonzero_tol (float, optional): Non-zero tolerance. Defaults to 1e-2.

    Raises:
        e_dtype: Qtensor datatypes are not equal
        e_nan_inf: Qtensor(s) contain non-numbers
        e_constant_qtensor: Qtensor(s) are constant from non-constant tensor
        e_value: Qtensors differ in numerical value
    """

    # If using PyTorch functions, set error tolerances to zero
    if base_options["nativePT"] and not error_override:
        max_norm_tol = 0.0
        l2_norm_tol = 0.0
        nonzero_tol = 0.0

    # Check if qtensors are same datatype before checking value differnces
    with torch.no_grad():
        try:
            assert (
                qtensor_fms_mo.dtype == qtensor_torch.dtype
            ), "Data types of the tensors are not equal"
        except AssertionError as e_dtype:
            # Debugging info
            logger.error("\nTensor Datatype Error:")
            logger.error("Quantizer Tuple = %s", setup)
            logger.error("Base Options = %s", base_options)
            logger.error("Other Options = %s", other_options)
            logger.error("FMS Model Optimizer datatype = %s", qtensor_fms_mo.dtype)
            logger.error("Torch datatype = %s", qtensor_torch.dtype)
            raise e_dtype  # Reraise exception

    # Check if there any non-numbers in FMS Model Optimizer qtensor
    with torch.no_grad():
        # Get indicies of NaN/INF
        nan_indices_fms_mo = torch.isnan(qtensor_fms_mo)
        inf_indices_fms_mo = torch.isinf(qtensor_fms_mo)
        nan_indices_torch = torch.isnan(qtensor_torch)
        inf_indices_torch = torch.isinf(qtensor_torch)

        # Count number of occurances of NaN/INF
        num_nan_fms_mo = torch.sum(nan_indices_fms_mo).item()
        num_inf_fms_mo = torch.sum(inf_indices_fms_mo).item()
        num_nan_torch = torch.sum(nan_indices_torch).item()
        num_inf_torch = torch.sum(inf_indices_torch).item()

        try:
            assert (
                num_nan_fms_mo == 0
            ), "Found NaN entries in FMS Model Optimizer Tensor"
            assert (
                num_inf_fms_mo == 0
            ), "Found INF entries in FMS Model Optimizer Tensor"
            assert num_nan_torch == 0, "Found NaN entries in Torch Tensor"
            assert num_inf_torch == 0, "Found INF entries in Torch Tensor"
        except AssertionError as e_nan_inf:
            # Debugging info
            logger.error("\nTensor Non-Number Error:")
            logger.error("Quantizer Tuple = %s", setup)
            logger.error("Base Options = %s", base_options)
            logger.error("Other Options = %s", other_options)
            if num_nan_fms_mo > 0:
                logger.error("FMS Model Optimizer # of NaN = %s", num_nan_fms_mo)
                logger.error(
                    "Original Tensor =\n%s", tensor[nan_indices_fms_mo].detach()
                )
                logger.error(
                    "FMS Model Optimizer Tensor =\n%s",
                    qtensor_fms_mo[nan_indices_fms_mo].detach(),
                )
                logger.error(
                    "Torch Tensor =\n%s", qtensor_torch[nan_indices_fms_mo].detach()
                )
            if num_inf_fms_mo > 0:
                logger.error("FMS Model Optimizer # of INF = %s", num_inf_fms_mo)
                logger.error(
                    "Original Tensor =\n%s", tensor[inf_indices_fms_mo].detach()
                )
                logger.error(
                    "FMS Model Optimizer Tensor =\n%s",
                    qtensor_fms_mo[inf_indices_fms_mo].detach(),
                )
                logger.error(
                    "Torch Tensor =\n%s", qtensor_torch[inf_indices_fms_mo].detach()
                )
            if num_nan_torch > 0:
                logger.error("Torch # of NaN = %s", num_nan_torch)
                logger.error(
                    "Original Tensor =\n%s", tensor[nan_indices_torch].detach()
                )
                logger.error(
                    "FMS Model Optimizer Tensor =\n%s",
                    qtensor_fms_mo[nan_indices_torch].detach(),
                )
                logger.error(
                    "Torch Tensor =\n%s", qtensor_torch[nan_indices_torch].detach()
                )
            if num_inf_torch > 0:
                logger.error("Torch # of INF = %s", num_inf_torch)
                logger.error(
                    "Original Tensor =\n%s", tensor[inf_indices_torch].detach()
                )
                logger.error(
                    "FMS Model Optimizer Tensor =\n%s",
                    qtensor_fms_mo[inf_indices_torch].detach(),
                )
                logger.error(
                    "Torch Tensor =\n%s", qtensor_torch[inf_indices_torch].detach()
                )
            raise e_nan_inf  # Reraise exception

    # Get value setup for analysis
    (
        num_bits,
        clip_low,
        clip_high,
        _n_level,
        scale,
        _zero_point,
        _quant_min,
        _quant_max,
        _qscheme,
    ) = setup

    # Check if qtensors are constant for non-constant tensor with appropriate spacing of elements
    if tensor.unique().numel() > 1 and (tensor.max() - tensor.min()) > scale.min():
        fms_mo_unique_vals = qtensor_fms_mo.unique()
        torch_unique_vals = qtensor_torch.unique()

        try:
            assert (
                fms_mo_unique_vals.numel() > 1
            ), "FMS Model Optimizer Tensor has constant value w/ non-constant input"
            assert (
                torch_unique_vals.numel() > 1
            ), "Torch Tensor has constant value w/ non-constant input"
        except AssertionError as e_constant_qtensor:
            # Debugging info
            logger.error("\nTensor Constant Tensor Error:")
            logger.error("Quantizer Tuple = %s", setup)
            logger.error("Base Options = %s", base_options)
            logger.error("Other Options = %s", other_options)
            logger.error("Original Tensor =\n%s", tensor.detach())
            logger.error(
                "FMS Model Optimizer Tensor unique vals =\n%s",
                fms_mo_unique_vals.detach(),
            )
            logger.error("Torch Tensor unique vals =\n%s", torch_unique_vals.detach())
            raise e_constant_qtensor  # reraise exception

    # In rare instances due to banker's rounding (ie torch.round), tensors can differ by 1 qlevel
    # To avoid this, do not count qlevel as error ; |abs(diff) - scale|~=0 --> qlevel off by 1
    if base_options["dequantize"]:
        diff = qtensor_fms_mo - qtensor_torch
        abs_diff = abs(diff)
        nonzero_diff_indices = abs_diff > max_norm_tol
        # float epsilon distance required
        scale_diff_indices = abs(abs_diff - scale) < 1e-3
        nonscale_nonzero_diff_indices = nonzero_diff_indices.logical_and(
            torch.logical_not(scale_diff_indices)
        )
        dtype_range = clip_high.max() - clip_low.min()
    else:
        # Cast uints to int32 to avoid negative number overflow
        diff = qtensor_fms_mo.to(torch.int32) - qtensor_torch.to(torch.int32)
        abs_diff = abs(diff)
        nonzero_diff_indices = abs_diff > 0
        scale_diff_indices = abs_diff == 1
        nonscale_nonzero_diff_indices = nonzero_diff_indices.logical_and(
            torch.logical_not(scale_diff_indices)
        )
        dtype_range = 2**num_bits

    # Count total number of these indices
    total_nonzero_indices = torch.sum(nonzero_diff_indices).item()
    total_scale_indices = torch.sum(scale_diff_indices).item()
    total_nonscale_nonzero_indices = torch.sum(nonscale_nonzero_diff_indices).item()

    assert total_nonscale_nonzero_indices == total_nonzero_indices - total_scale_indices

    # At this point, we don't want to count any potential problems from banker's rounding

    with torch.no_grad():
        try:
            # Check for large difference in values for current dtype (ie underflow/overflow)
            e_string = "Large difference in magnitudes for dtype"
            assert torch.max(abs_diff) <= dtype_range / 2.0, e_string

            # Compute L2 matrix norm on non-scale indices for accumlated errors
            #   need to cast to float for ints
            # Check if number of differences are large for tensor size excluding qlevel
            e_string = "Number of tensor element differences over tolerance"
            assert total_nonzero_indices / diff.numel() <= nonzero_tol, e_string
            # assert total_nonscale_nonzero_indices/diff.numel() <= nonzero_tol, e_string

            # Do not include scale indices since they are a known value that we can't avoid
            e_string = "Tensor L2 difference norm exceeds allowed tolerance"
            norm_tensor = diff.float()
            # norm_tensor = diff[nonscale_nonzero_diff_indices].float()
            assert (
                torch.linalg.norm(norm_tensor).item()  # pylint: disable=not-callable
                / diff.numel()
                <= l2_norm_tol
            ), e_string

        except AssertionError as e_value:
            # Debugging info
            logger.error("\nTensor Value Error: %s", e_string)
            logger.error("Quantizer Tuple = %s", setup)
            logger.error("Base Options = %s", base_options)
            logger.error("Other Options = %s", other_options)
            logger.error(
                "Total Non-zero Indices = %s/%s", total_nonzero_indices, diff.numel()
            )
            logger.error(
                "Total Non-scale Indices = %s/%s", total_scale_indices, diff.numel()
            )
            logger.error(
                "Total Non-scale, Non-zero Indices = %s/%s",
                total_nonscale_nonzero_indices,
                diff.numel(),
            )
            logger.error("Nonzero Indicies =\n%s", nonzero_diff_indices)
            logger.error("Original Tensor =\n%s", tensor[nonzero_diff_indices].detach())
            logger.error(
                "FMS Model Optimizer Tensor =\n%s",
                qtensor_fms_mo[nonzero_diff_indices].detach(),
            )
            logger.error(
                "Torch Tensor =\n%s", qtensor_torch[nonzero_diff_indices].detach()
            )
            logger.error("Total Diff vals =%s", diff.unique().numel())
            logger.error("Diff unique vals =\n%s", diff.unique().detach())

            logger.error("input tensor =\n%s", tensor)
            logger.error("torch_scale =\n%s", scale)
            logger.error("torch_zero_point =\n%s", _zero_point)
            raise e_value  # Reraise exception


def set_base_options(
    fms_mo_quantizer: torch.autograd.Function,
    torch_quantizer: torch.nn.Module,
    base_options: dict,
):
    """
    Set base options for FMS and Torch quantizer

    base_options defined in conftest.py

    Args:
        fms_mo_quantizer (torch.autograd.Function):  FMS quantizer
        torch_quantizer (torch.nn.Module): Torch Quantizer
        base_options (dict): Other Option params
    """
    # Native PT forces FMS Model Optimizer to call PT
    fms_mo_quantizer.use_PT_native_Qfunc = base_options["nativePT"]

    # Dequantize to floats
    fms_mo_quantizer.dequantize = base_options["dequantize"]
    torch_quantizer.dequantize = base_options["dequantize"]

    fms_mo_quantizer.set_quantizer()
