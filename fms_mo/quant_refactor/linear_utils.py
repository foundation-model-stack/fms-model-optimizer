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
Linear Quantization Utility functions

Raises:
    ValueError: Lower clip value is less than 0 for symmetric quantization
"""

# Standard
from typing import Tuple

# Third Party
import torch


def transform_clips(
    input_tensor_dtype: torch.dtype,
    clip_valn: torch.Tensor,
    clip_val: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Transform clip values to input datatype

    Args:
        input_tensor_dtype (torch.dtype): Input dtype.
        clip_valn (torch.Tensor): Lower clip value.
        clip_val (torch.Tensor): Upper clip value.

    Returns:
        [torch.Tensor, torch.Tensor]: Casted clip values
    """
    return clip_valn.to(input_tensor_dtype), clip_val.to(input_tensor_dtype)


def qint_bounds(
    num_bits: torch.Tensor,
    zero_point: torch.Tensor,
    symmetric: bool = False,
    qlevel_lowering: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.dtype]:
    """
    Compute qint clamp bounds based on qparams.

    Args:
        num_bits (torch.Tensor): Number of bits for quantization.
        zero_point (torch.Tensor): Quantized int bin mapping to fp 0.0.
        symmetric (bool, optional): Specify if quantized bounds are symmetric. Defaults to False.
        qlevel_lowering (bool, optional): Specify if quantized levels is reduced. Defaults to False.

    Returns:
        [torch.Tensor, torch.Tensor, torch.dtype]: qint_bounds and torch.qint_dtype
    """
    # Set quantization bounds and datatype based on zero_point
    if symmetric and torch.sum(zero_point) == 0:
        qlevel_symmetric = torch.tensor(1) if qlevel_lowering else torch.tensor(0)
        qlevel_min, qlevel_max = (
            -(2 ** (num_bits - 1)) + qlevel_symmetric,
            2 ** (num_bits - 1) - 1,
        )  # eg (-128,127) or (-8,7)
        int_dtype = torch.int8
    else:  # single_sided or zero_point!=0
        qlevel_min, qlevel_max = (
            torch.tensor(0),
            2**num_bits - 1,
        )  # eg (0, 255) or (0,15)
        int_dtype = torch.uint8
    return qlevel_min, qlevel_max, int_dtype


def linear_quantize(
    input_tensor: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor,
) -> torch.Tensor:
    """
    Quantize a tensor to quantized int space

    Args:
        input_tensor (torch.Tensor): Tensor to quantize.
        scale (torch.Tensor): Dequantized range of a quantized integer bin.
        zero_point (torch.Tensor): Quantized integer bin mapping to fp 0.0.

    Returns:
        torch.Tensor: Quantized int tensor
    """
    return torch.round(
        input_tensor / scale.to(input_tensor.device)
        + zero_point.to(input_tensor.device)
    )


def linear_quantize_residual(
    input_tensor: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize a tensor to quantized int space and compute residual

    Args:
        input_tensor (torch.Tensor): Tensor to quantize.
        scale (torch.Tensor): Dequantized range of a quantized integer bin.
        zero_point (torch.Tensor): Quantized integer bin mapping to fp 0.0.

    Returns:
        [torch.Tensor, torch.Tensor]: Quantized int tensor and residual
    """
    unrounded = input_tensor / scale.to(input_tensor.device) + zero_point.to(
        input_tensor.device
    )
    rounded = torch.round(unrounded)
    with torch.no_grad():
        residual = torch.round(unrounded)
    return rounded, residual


def linear_quantize_LSQresidual(
    input_tensor: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize a tensor to quantized int space and compute LSQ residual

    Args:
        input_tensor (torch.Tensor): Tensor to quantize.
        scale (torch.Tensor): Dequantized range of a quantized integer bin.
        zero_point (torch.Tensor): Quantized integer bin mapping to fp 0.0.

    Returns:
        [torch.Tensor, torch.Tensor]: Quantized int tensor and residual
    """
    unrounded = input_tensor / scale.to(input_tensor.device) + zero_point.to(
        input_tensor.device
    )
    rounded = torch.round(unrounded)
    with torch.no_grad():
        residual = rounded - unrounded
    return rounded, residual


def linear_dequantize(
    input_tensor: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor,
) -> torch.Tensor:
    """
    Dequantize a tensor to dequantized fp space

    Args:
        input_tensor (torch.Tensor): Tensor to dequantize.
        scale (torch.Tensor): Dequantized range of a quantized integer bin.
        zero_point (torch.Tensor): Quantized integer bin mapping to fp 0.0.

    Returns:
        torch.Tensor: Dequantized fp tensor
    """
    return (input_tensor - zero_point.to(input_tensor.device)) * scale.to(
        input_tensor.device
    )


def linear_quantize_zp(
    input_tensor: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor,
    num_bits: torch.Tensor,
) -> torch.Tensor:
    """
    Quantize a tensor to dequantized fp space and zero out upper bound values

    Args:
        input_tensor (torch.Tensor): Tensor to quantize.
        scale (torch.Tensor): Dequantized range of a quantized integer bin.
        zero_point (torch.Tensor): Quantized integer bin mapping to fp 0.0.
        num_bits (torch.Tensor): Number of bits for quantization.

    Returns:
        torch.Tensor: Quantized fp tensor
    """
    Qp = 2 ** (num_bits - 1) - 1
    out = torch.round(
        input_tensor / scale.to(input_tensor.device)
        + zero_point.to(input_tensor.device)
    )
    out = torch.where(out > Qp, torch.ones_like(out) * Qp, out)
    return out


def linear_dequantize_zp(
    input_tensor: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor,
) -> torch.Tensor:
    """
    Dequantize a tensor to dequantized fp space

    Args:
        input_tensor (torch.Tensor): Tensor to dequantize.
        scale (torch.Tensor): Dequantized range of a quantized integer bin.
        zero_point (torch.Tensor): Quantized integer bin mapping to fp 0.0.

    Returns:
        torch.Tensor: Dequantized fp tensor
    """
    return (input_tensor - zero_point.to(input_tensor.device)) * scale.to(
        input_tensor.device
    )


def linear_quantization(
    input_tensor: torch.Tensor,
    num_bits: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor,
    dequantize: bool = True,
    symmetric: bool = False,
    qlevel_lowering: bool = False,
) -> torch.Tensor:
    """
    Perform quantization and dequantization on tensor:

    # Reference PT implementation:
    pytorch.org/docs/stable/generated/torch.ao.quantization.fake_quantize.FakeQuantize.html
    x_out = (clamp(round(x/scale + zero_point), quant_min, quant_max) - zero_point) * scale

    Args:
        input_tensor (torch.Tensor): Tensor to quantize.
        num_bits (torch.Tensor): Number of bits for quantization.
        scale (torch.Tensor): Dequantized range of a quantized integer bin.
        zero_point (torch.Tensor): Quantized integer bin mapping to fp 0.0.
        dequantize (bool, optional): Return dequantized or int tensor. Defaults to True.
        symmetric (bool, optional): Specify if clip values are symmetric. Defaults to False.
        qlevel_lowering (bool, optional): Specify lowering of quantized levels. Defaults to False.

    Returns:
        torch.Tensor: Quantized or Dequantized tensor
    """
    qint_min, qint_max, int_dtype = qint_bounds(
        num_bits, zero_point, symmetric, qlevel_lowering
    )
    output = linear_quantize(input_tensor, scale, zero_point)
    output = output.clamp(qint_min, qint_max)
    if dequantize:
        output = linear_dequantize(output, scale, zero_point).to(input_tensor.dtype)
    else:
        output = output.to(int_dtype)
    return output


def asymmetric_linear_quantization_params(
    num_bits: torch.Tensor,
    sat_min: torch.Tensor,
    sat_max: torch.Tensor,
    integral_zero_point: bool = True,
    signed: bool = False,
    qlevel_lowering: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Default quantization setup for asymmetric quantization

    Args:
        num_bits (torch.Tensor): Number of bits for quantization.
        sat_min (torch.Tensor): Lower clip value.
        sat_max (torch.Tensor): Upper clip value.
        integral_zero_point (bool, optional): Specify using int zero point. Defaults to True.
        signed (bool, optional): Specify if clip values are . Defaults to False.
        qlevel_lowering (bool, optional): Specify lowering of quantized levels. Defaults to False.

    Returns:
        [torch.Tensor, torch.Tensor, torch.Tensor]:
            # of quantized levels, scale, zero point
    """
    with torch.no_grad():
        n_levels = 2**num_bits - 2 if qlevel_lowering else 2**num_bits - 1
        diff = sat_max - sat_min
        # If float values are all 0, we just want the quantized values to be 0 as well.
        # So overriding the saturation value to 'n', so the scale becomes 1
        diff[diff == 0.0] = n_levels
        scale = diff / n_levels
        zero_point = -sat_min / scale
        if integral_zero_point:
            zero_point = zero_point.round().to(torch.int)
        if signed:
            zero_point += 2 ** (num_bits - 1)

        return n_levels, scale, zero_point


def symmetric_linear_quantization_params(
    num_bits: torch.Tensor,
    sat_val: torch.Tensor,
    qlevel_lowering: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Default quantization setup for symmetric quantization

    Args:
        num_bits (torch.Tensor): Number of bits for quantization.
        sat_max (torch.Tensor): Upper clip value.  Can be multi-valued (perCh/perGp).
        qlevel_lowering (bool, optional): Specify lowering of quantized levels. Defaults to False.
        Ngp_or_ch (int, optional):

    Returns:
        [torch.Tensor, torch.Tensor, torch.Tensor]:
            # of quantized levels, scale, zero point
    """
    if torch.any(sat_val < 0.0):
        raise ValueError("Saturation value must be >= 0")

    with torch.no_grad():
        n_levels = (
            2**num_bits - 2 if qlevel_lowering else 2**num_bits - 1
        )  # Always use qlevel_lowering
        # If float values are all 0, we just want the quantized values to be 0 as well.
        # So overriding the saturationvalue to '2n', so the scale becomes 1
        diff = 2 * sat_val
        diff[diff == 0.0] = n_levels
        scale = diff / n_levels
        zero_point = torch.zeros_like(scale)
        return n_levels, scale, zero_point


def per_channel_axis(
    scale: torch.Tensor,
    zero_point: torch.Tensor,
    tensor_shape: torch.Size,
    axis: int = 0,
):
    """
    Change scale and zero_point to target axis dimension of input tensor.
    Axis values: 0 -> Nx1, 1 -> 1xN

    Note: for Transformers, axis = 0 is desired

    Args:
        scale (torch.Tensor): Dequantized range of a quantized integer bin.
        zero_point (torch.Tensor): Quantized int bin mapping to fp 0.0.
        tensor_shape (torch.Size): Shape of quantized tensor

    Returns:
        scale, zero_point:
    """
    if axis == 0:
        scale = scale.unsqueeze(1)
        zero_point = zero_point.unsqueeze(1)
    elif axis == 1:
        scale = scale.unsqueeze(0)
        zero_point = zero_point.unsqueeze(0)
    else:
        raise ValueError("Axis must be 0 or 1")

    # Check that tensor shape axis is same as scale/zp broadcast
    assert tensor_shape[axis] == scale.shape[axis]
    assert tensor_shape[axis] == zero_point.shape[axis]

    return scale, zero_point
