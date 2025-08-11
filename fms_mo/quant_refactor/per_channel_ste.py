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
Base Per Channel Quantizer Class
"""

# Standard
from typing import Tuple

# Third Party
import torch

# Local
from fms_mo.quant_refactor.linear_utils import (
    asymmetric_linear_quantization_params,
    linear_quantization,
    per_channel_axis,
    symmetric_linear_quantization_params,
    transform_clips,
)

default_axis = int(0)


class PerChannelSTE(torch.autograd.Function):
    """Base class for customized forward/backward functions that is NOT using PT native func.
    There's a family of non-learnable quantizers, such as SAWB, MinMax,
    whose forward can be the same quant functions and backward is simply STE.
    We just need to calculate scale in the upper level quantizer class then those quantizers
    could all be using the same base "STE function"

    math should be consistent with pytorch: https://pytorch.org/docs/stable/quantization.html
        x_int = round(x/scale + zp)
        x_dq  = (x_int - zp) * scale # TODO check clamp before or after dQ
    """

    @staticmethod
    def forward(
        ctx,
        input_tensor: torch.Tensor,
        num_bits: torch.Tensor,
        clip_valn: torch.Tensor,
        clip_val: torch.Tensor,
        dequantize: bool = True,
        symmetric: bool = False,
        qlevel_lowering: bool = False,
        axis: int = default_axis,
    ):
        """
        General forward method:
            Set clip values to dtype of input_tensor tensor
            Compute # of quantized levels, scale, and zero point
            Save data for backward()
            Perform linear quantization on input_tensor tensor
            return output

        Args:
            ctx (torch.autograd.Function): Forward/Backward context object.
            input_tensor (torch.Tensor): Tensor to be quantized.
            num_bits (torch.Tensor): Number of bit for quantization.
            clip_valn (torch.Tensor): Lower clip value bound.
            clip_val (torch.Tensor): Upper clip value bound.
            dequantize (bool, optional): Return dequantized or int tensor. Defaults to True.
            symmetric (bool, optional): Specify if clip values are symmetric. Defaults to False.
            qlevel_lowering (bool, optional): Specify lowering of quantized levels.
                Defaults to True.
            axis (int, optional): Specify which tensor dimension to quantize indiviually.
                Defaults to 0.

        Returns:
            torch.Tensor: Dequantized or Quantized output tensor.
        """
        clip_valn, clip_val = transform_clips(input_tensor.dtype, clip_valn, clip_val)
        n_levels, scale, zero_point = PerChannelSTE.calc_qparams(
            num_bits, clip_valn, clip_val, qlevel_lowering, axis, input_tensor.shape
        )
        PerChannelSTE.save_tensors(
            ctx,
            tensors=(input_tensor, n_levels, clip_valn, clip_val, scale, zero_point),
        )
        output = linear_quantization(
            input_tensor,
            num_bits,
            scale,
            zero_point,
            dequantize,
            symmetric,
            qlevel_lowering,
        )
        return output

    @classmethod
    def calc_qparams(
        cls,
        num_bits: torch.Tensor,
        clip_valn: torch.Tensor,
        clip_val: torch.Tensor,
        symmetric: bool = False,
        qlevel_lowering: bool = True,
        axis: int = default_axis,
        tensor_shape: torch.Size = None,
    ):
        """
        Compute the scale and zero_point from num_bits and clip values

        Args:
            num_bits (torch.Tensor): Number of bit for quantization.
            clip_valn (torch.Tensor): Lower clip value.
            clip_val (torch.Tensor): Upper clip value.
            symmetric (bool, optional): Specify if clip values are symmetric. Defaults to False.
            qlevel_lowering (bool, optional): Specify lowering of quantized levels.
                Defaults to True.
            axis (int, optional): Specify which tensor dimension to quantize indiviually.
                Defaults to 0.

        Returns:
            torch.Tensor, torch.Tensor, torch.Tensor: Quantized parameters
        """
        if symmetric:
            n_levels, scale, zero_point = symmetric_linear_quantization_params(
                num_bits,
                clip_val,
                qlevel_lowering,
            )
        else:
            n_levels, scale, zero_point = asymmetric_linear_quantization_params(
                num_bits,
                clip_valn,
                clip_val,
                integral_zero_point=True,
                signed=False,
            )

        # Broadcast scale, zero_point based on axis
        scale, zero_point = per_channel_axis(scale, zero_point, tensor_shape, axis)

        return n_levels, scale, zero_point

    # The save_tensors and backward unpacking must be synced
    @classmethod
    def save_tensors(cls, ctx, tensors) -> None:
        """
        Save computed data to ctx for backward()

        Args:
            ctx (torch.autograd.Function): Forward/Backward context object.
            tensors (list(torch.Tensor)): List of tensors to save.
        """
        ctx.save_for_backward(*tensors)

    @staticmethod
    def backward(ctx, grad_output):
        """
        General STE backward method:
            Return grad_output + None args to match forward input_tensors

        Args:
            ctx (torch.autograd.Function): Forward/Backward context object.
            grad_output (torch.Tensor): Gradient tensor

        Returns:
            torch.Tensor, None,...,None: STE Gradient
        """
        return grad_output, None, None, None, None, None


class PerChannelSTE_PTnative(torch.autograd.Function):
    """Base class for customized forward/backward functions.
    There's a family of non-learnable quantizers, such as SAWB, MinMax,
    whose forward can leverage PT native functions and backward is simply STE.
    We just need to calculate scale in the upper level quantizer class then those quantizers
    could all be using the same base "STE function"

    math should be consistent with pytorch: https://pytorch.org/docs/stable/quantization.html
        x_int = round(x/scale + zp)
        x_dq  = (x_int - zp) * scale

    This type of class will be used by Quantizer.forward(), e.g.
    """

    @staticmethod
    def forward(
        ctx,
        input_tensor: torch.Tensor,
        num_bits: torch.Tensor,
        clip_valn: torch.Tensor,
        clip_val: torch.Tensor,
        dequantize: bool = True,
        symmetric: bool = False,
        qlevel_lowering: bool = False,
        axis: int = default_axis,
    ):
        """
        General forward method:
            Set clip values to dtype of input_tensor tensor
            Compute # of quantized levels, scale, and zero point
            Perform PTnative linear quantization on input_tensor tensor
            return output

        Args:
            ctx (torch.autograd.Function): Forward/Backward context object.
            input_tensor (torch.Tensor): Tensor to be quantized.
            num_bits (torch.Tensor): Number of bit for quantization.
            clip_valn (torch.Tensor): Lower clip value bound.
            clip_val (torch.Tensor): Upper clip value bound.
            dequantize (bool, optional): Return dequantized or int tensor. Defaults to True.
            symmetric (bool, optional): Specify if clip values are symmetric. Defaults to False.
            qlevel_lowering (bool, optional): Specify lowering of quantized levels.
                Defaults to True.
            axis (int, optional): Specify which tensor dimension to quantize indiviually.
                Defaults to 0.

        Returns:
            torch.Tensor: Dequantized or Quantized output tensor.
        """
        clip_valn, clip_val = transform_clips(
            input_tensor.dtype,
            clip_valn,
            clip_val,
        )
        (
            _,
            scale,
            zero_point,
            qint_l,
            qint_h,
            qint_dtype,
        ) = PerChannelSTE_PTnative.calc_qparams(
            num_bits,
            clip_valn,
            clip_val,
            symmetric,
            qlevel_lowering,
        )
        output = PerChannelSTE_PTnative.linear_quantization(
            input_tensor,
            scale,
            zero_point,
            qint_l,
            qint_h,
            qint_dtype,
            dequantize,
            axis,
        )
        return output

    @classmethod
    def calc_qparams(
        cls,
        num_bits: torch.Tensor,
        clip_valn: torch.Tensor,
        clip_val: torch.Tensor,
        symmetric: bool = False,
        qlevel_lowering: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
        """
        Compute the scale and zero_point from num_bits and clip values.
        Also, compute qint bounds for PT clamping.

        Args:
            num_bits (torch.Tensor): Number of bit for quantization.
            clip_valn (torch.Tensor): Lower clip value.
            clip_val (torch.Tensor): Upper clip value.
            symmetric (bool, optional): Specify if clip values are symmetric. Defaults to False.
            qlevel_lowering (bool, optional): Specify lowering of quantized levels.
                Defaults to True.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Quantized parameters
        """
        n_levels = 2**num_bits - 2 if qlevel_lowering else 2**num_bits - 1
        scale = (clip_val - clip_valn) / n_levels
        zero_point = (
            torch.zeros_like(scale).to(torch.int)
            if symmetric
            else torch.round(-clip_valn / scale).to(torch.int)
        )
        qint_l, qint_h, qint_dtype = PerChannelSTE_PTnative.qint_bounds(
            num_bits, zero_point, symmetric, qlevel_lowering
        )
        # Note: fake_quantize_per_channel_affine does not need matching dimensions for scale/zp to tensor
        return n_levels, scale, zero_point, qint_l, qint_h, qint_dtype

    @classmethod
    def qint_bounds(
        cls,
        num_bits: torch.Tensor,
        zero_point: torch.Tensor,
        symmetric: bool = False,
        qlevel_lowering: bool = True,
    ) -> Tuple[int, int, torch.dtype]:
        """
        qlevel_symmetric: shift qlevel from [-2**(b-1), 2**(b-1)-1] to [-2**(b-1)+1, 2**(b-1)-1]
        For int8: [-127,127] ; For int4 [-7,7]
        qint bounds must be ints, not tensors
        """
        num_bits_int = (
            num_bits.item() if isinstance(num_bits, torch.Tensor) else num_bits
        )
        if symmetric and torch.sum(zero_point) == 0:
            qlevel_symmetric = 1 if qlevel_lowering else 0
            qint_l, qint_h = (
                -(2 ** (num_bits_int - 1)) + qlevel_symmetric,
                2 ** (num_bits_int - 1) - 1,
            )
            qint_dtype = torch.qint8
        else:  # single_sided or zero_point != 0
            qint_l, qint_h = 0, 2**num_bits_int - 1
            qint_dtype = torch.quint8
        return qint_l, qint_h, qint_dtype

    @classmethod
    def linear_quantization(
        cls,
        input_tensor: torch.Tensor,
        scale: torch.Tensor,
        zero_point: torch.Tensor,
        qint_l: int,
        qint_h: int,
        qint_dtype: torch.dtype,
        dequantize: bool = True,
        axis: int = default_axis,
    ) -> torch.Tensor:
        """
        Linear quantization for PTnative STE

        Args:
            input_tensor (torch.Tensor): Tensor to be quantized
            scale (torch.Tensor): Quantized bin ranges per channel.
            zero_point (torch.Tensor): Quantized integer bin mapping to fp 0.0 per channel.
            qint_l (int): Quantized integer lower clip value.
            qint_h (int): Quantized integer upper clip value.
            qint_dtype (torch.dtype): Quantized integer dtype.
            dequantize (bool, optional): Specify to return fp or quantized int. Defaults to True.
            axis (int, optional): Specify which tensor dimension to quantize indiviually.
                Defaults to 0.

        Returns:
            torch.Tensor: PTnative quantized or dequantized tensor.
        """
        if dequantize:
            # Note: scale + zero_point are per channel tensors
            output = torch.fake_quantize_per_channel_affine(
                input_tensor.float(),
                scale.float(),
                zero_point,
                axis=axis,
                quant_min=qint_l,
                quant_max=qint_h,
            ).to(input_tensor.dtype)
        else:
            output = (
                torch.quantize_per_channel(
                    input_tensor.float(),
                    scale.float(),
                    zero_point,
                    axis=axis,
                    dtype=qint_dtype,
                )
                .int_repr()
                .clamp(qint_l, qint_h)
            )
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        General STE backward method:
            Return grad_output + None args to match forward input_tensor

        Args:
            ctx (torch.autograd.Function): Forward/Backward context object.
            grad_output (torch.Tensor): Gradient tensor

        Returns:
            torch.Tensor, None,...,None: STE Gradient
        """
        return grad_output, None, None, None, None, None, None


class PerChannelSTESAWB(PerChannelSTE):
    """
    PerChannelSTE Base for SAWB

    Extends:
        PerChannelSTE
    """

    @staticmethod
    def forward(
        ctx,
        input_tensor: torch.Tensor,
        num_bits: torch.Tensor,
        clip_valn: torch.Tensor,
        clip_val: torch.Tensor,
        dequantize: bool = True,
        symmetric: bool = False,
        qlevel_lowering: bool = False,
        use_code: bool = False,
        axis: int = default_axis,
    ):
        """
        General forward method:
            Set clip values to dtype of input_tensor tensor
            Compute # of quantized levels, scale, and zero point
            Save data for backward()
            Perform linear quantization on input_tensor tensor
            return output

        Args:
            ctx (torch.autograd.Function): Forward/Backward context object.
            input_tensor (torch.Tensor): Tensor to be quantized.
            num_bits (torch.Tensor): Number of bit for quantization.
            clip_valn (torch.Tensor): Lower clip value bound.
            clip_val (torch.Tensor): Upper clip value bound.
            dequantize (bool, optional): Return dequantized or int tensor. Defaults to True.
            symmetric (bool, optional): Specify if clip values are symmetric. Defaults to False.
            qlevel_lowering (bool, optional): Specify lowering of quantized levels.
                Defaults to True.
            use_code (bool, optional): Specify a specific SAWB code for quantization.
                Defaults to False.
            axis (int, optional): Specify an axis to quantize.  Defaults to default_axis.

        Returns:
            torch.Tensor: Dequantized or Quantized output tensor.
        """
        clip_valn, clip_val = transform_clips(input_tensor.dtype, clip_valn, clip_val)
        n_levels, scale, zero_point = PerChannelSTESAWB.calc_qparams(
            num_bits,
            clip_valn,
            clip_val,
            symmetric,
            qlevel_lowering,
            axis,
            input_tensor.shape,
            use_code,
        )
        PerChannelSTE.save_tensors(
            ctx,
            tensors=(input_tensor, n_levels, clip_valn, clip_val, scale, zero_point),
        )
        output = linear_quantization(
            input_tensor,
            num_bits,
            scale,
            zero_point,
            dequantize,
            symmetric,
            qlevel_lowering,
        )
        return output

    @classmethod
    def calc_qparams(
        cls,
        num_bits: torch.Tensor,
        _clip_valn: torch.Tensor,
        clip_val: torch.Tensor,
        symmetric: bool = False,
        qlevel_lowering: bool = True,
        axis: int = default_axis,
        tensor_shape: torch.Size = None,
        use_code: bool = False,
    ):
        """
        Compute the scale and zero_point from num_bits and clip values

        Args:
            num_bits (torch.Tensor): Number of bit for quantization.
            clip_valn (torch.Tensor): Lower clip value.
            clip_val (torch.Tensor): Upper clip value.
            symmetric (bool, optional): Specify if clip values are symmetric. Defaults to False.
            qlevel_lowering (bool, optional): Specify lowering of quantized levels.
                Defaults to True.

        Returns:
            torch.Tensor, torch.Tensor, torch.Tensor: Quantized parameters
        """
        # SAWB is always symmetric
        output = None
        if symmetric:
            n_levels = (
                2 ** (num_bits) - 2
                if ((use_code and num_bits.item() in [2, 4, 8]) or qlevel_lowering)
                else 2 ** (num_bits) - 1
            )
            _, scale, zero_point = symmetric_linear_quantization_params(
                num_bits, clip_val, qlevel_lowering
            )

            # Broadcast scale, zero_point to tensor shape
            scale, zero_point = per_channel_axis(scale, zero_point, tensor_shape, axis)

            output = n_levels, scale, zero_point
        else:
            raise ValueError("SAWB has non-symmetric Qscheme")
        return output


class PerChannelSTESAWB_PTnative(PerChannelSTE_PTnative):
    """Base class for customized forward/backward functions.
    There's a family of non-learnable quantizers, such as SAWB, MinMax,
    whose forward can leverage PT native functions and backward is simply STE.
    We just need to calculate scale in the upper level quantizer class then those quantizers
    could all be using the same base "STE function"

    math should be consistent with pytorch: https://pytorch.org/docs/stable/quantization.html
        x_int = round(x/scale + zp)
        x_dq  = (x_int - zp) * scale

    This type of class will be used by Quantizer.forward(), e.g.
    """

    @staticmethod
    def forward(
        ctx,
        input_tensor: torch.Tensor,
        num_bits: torch.Tensor,
        clip_valn: torch.Tensor,
        clip_val: torch.Tensor,
        dequantize: bool = True,
        symmetric: bool = False,
        qlevel_lowering: bool = False,
        use_code: bool = False,
        axis: int = default_axis,
    ):
        """
        General forward method:
            Set clip values to dtype of input_tensor tensor
            Compute # of quantized levels, scale, and zero point
            Perform PTnative linear quantization on input_tensor tensor
            return output

        Args:
            ctx (torch.autograd.Function): Forward/Backward context object.
            input_tensor (torch.Tensor): Tensor to be quantized.
            num_bits (torch.Tensor): Number of bit for quantization.
            clip_valn (torch.Tensor): Lower clip value bound.
            clip_val (torch.Tensor): Upper clip value bound.
            dequantize (bool, optional): Return dequantized or int tensor. Defaults to True.
            symmetric (bool, optional): Specify if clip values are symmetric. Defaults to False.
            qlevel_lowering (bool, optional): Specify lowering of quantized levels.
                Defaults to True.
            use_code (bool, optional): Specify a specific SAWB code for quantization.
            axis (int, optional): Specify which tensor dimension to quantize indiviually.
                Defaults to 0.

        Returns:
            torch.Tensor: Dequantized or Quantized output tensor.
        """
        clip_valn, clip_val = transform_clips(
            input_tensor.dtype,
            clip_valn,
            clip_val,
        )
        (
            _,
            scale,
            zero_point,
            qint_l,
            qint_h,
            qint_dtype,
        ) = PerChannelSTESAWB_PTnative.calc_qparams(
            num_bits,
            clip_valn,
            clip_val,
            symmetric,
            qlevel_lowering,
        )
        output = PerChannelSTE_PTnative.linear_quantization(
            input_tensor,
            scale,
            zero_point,
            qint_l,
            qint_h,
            qint_dtype,
            dequantize,
            axis,
        )
        return output

    @classmethod
    def calc_qparams(
        cls,
        num_bits: torch.Tensor,
        clip_valn: torch.Tensor,
        clip_val: torch.Tensor,
        symmetric: bool = False,
        qlevel_lowering: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
        """
        Compute the scale and zero_point from num_bits and clip values.
        Also, compute qint bounds for PT clamping.

        Args:
            num_bits (torch.Tensor): Number of bit for quantization.
            clip_valn (torch.Tensor): Lower clip value.
            clip_val (torch.Tensor): Upper clip value.
            symmetric (bool, optional): Specify if clip values are symmetric. Defaults to False.
            qlevel_lowering (bool, optional): Specify lowering of quantized levels.
                Defaults to True.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Quantized parameters
        """
        n_levels = 2**num_bits - 2 if qlevel_lowering else 2**num_bits - 1
        scale = (clip_val - clip_valn) / n_levels
        zero_point = (
            torch.zeros_like(scale).to(torch.int)
            if symmetric
            else torch.round(-clip_valn / scale).to(torch.int)
        )
        qint_l, qint_h, qint_dtype = PerChannelSTESAWB_PTnative.qint_bounds(
            num_bits, zero_point, symmetric, qlevel_lowering
        )
        # Note: fake_quantize_per_channel_affine does not need matching dimensions for scale/zp to tensor
        return n_levels, scale, zero_point, qint_l, qint_h, qint_dtype

    @classmethod
    def qint_bounds(
        cls,
        num_bits: torch.Tensor,
        zero_point: torch.Tensor,
        symmetric: bool = False,
        qlevel_lowering: bool = True,
    ) -> Tuple[int, int, torch.dtype]:
        """
        qlevel_symmetric: shift qlevel from [-2**(b-1), 2**(b-1)-1] to [-2**(b-1)+1, 2**(b-1)-1]
        For int8: [-127,127] ; For int4 [-7,7]
        qint bounds must be ints, not tensors
        """
        num_bits_int = (
            num_bits.item() if isinstance(num_bits, torch.Tensor) else num_bits
        )
        if symmetric and torch.sum(zero_point) == 0:
            qlevel_symmetric = 1 if qlevel_lowering else 0
            qint_l, qint_h = (
                -(2 ** (num_bits_int - 1)) + qlevel_symmetric,
                2 ** (num_bits_int - 1) - 1,
            )
            qint_dtype = torch.qint8
        else:  # single_sided or zero_point != 0
            qint_l, qint_h = 0, 2**num_bits_int - 1
            qint_dtype = torch.quint8
        return qint_l, qint_h, qint_dtype

    @staticmethod
    def backward(ctx, grad_output):
        """
        General STE backward method:
            Return grad_output + None args to match forward input_tensor

        Args:
            ctx (torch.autograd.Function): Forward/Backward context object.
            grad_output (torch.Tensor): Gradient tensor

        Returns:
            torch.Tensor, None,...,None: STE Gradient
        """
        return grad_output, None, None, None, None, None, None


class PerChannelSTEQmax(PerChannelSTE):
    """
    PerChannelSTE Base for Qmax

    Extends:
        PerChannelSTE
    """

    @staticmethod
    def forward(
        ctx,
        input_tensor: torch.Tensor,
        num_bits: torch.Tensor,
        clip_valn: torch.Tensor,
        clip_val: torch.Tensor,
        dequantize: bool = True,
        symmetric: bool = False,
        qlevel_lowering: bool = False,
        use_minmax: bool = False,
        axis: int = default_axis,
    ):
        """
        General forward method:
            Set clip values to dtype of input_tensor tensor
            Compute # of quantized levels, scale, and zero point
            Save data for backward()
            Perform linear quantization on input_tensor tensor
            return output

        Args:
            ctx (torch.autograd.Function): Forward/Backward context object.
            input_tensor (torch.Tensor): Tensor to be quantized.
            num_bits (torch.Tensor): Number of bit for quantization.
            clip_valn (torch.Tensor): Lower clip value bound.
            clip_val (torch.Tensor): Upper clip value bound.
            dequantize (bool, optional): Return dequantized or int tensor. Defaults to True.
            symmetric (bool, optional): Specify if clip values are symmetric. Defaults to False.
            qlevel_lowering (bool, optional): Specify lowering of quantized levels.
                Defaults to True.
            axis (int, optional): Specify which tensor dimension to quantize indiviually.
                Defaults to 0.

        Returns:
            torch.Tensor: Dequantized or Quantized output tensor.
        """
        clip_valn, clip_val = transform_clips(input_tensor.dtype, clip_valn, clip_val)
        n_levels, scale, zero_point = PerChannelSTEQmax.calc_qparams(
            num_bits,
            clip_valn,
            clip_val,
            symmetric,
            qlevel_lowering,
            axis,
            input_tensor.shape,
            use_minmax,
        )
        PerChannelSTE.save_tensors(
            ctx,
            tensors=(input_tensor, n_levels, clip_valn, clip_val, scale, zero_point),
        )
        output = linear_quantization(
            input_tensor,
            num_bits,
            scale,
            zero_point,
            dequantize,
            symmetric,
            qlevel_lowering,
        )
        return output

    @classmethod
    def calc_qparams(
        cls,
        num_bits: torch.Tensor,
        clip_valn: torch.Tensor,
        clip_val: torch.Tensor,
        symmetric: bool = False,
        qlevel_lowering: bool = True,
        axis: int = default_axis,
        tensor_shape: torch.Size = None,
        use_minmax: bool = False,
    ):
        """
        Compute the scale and zero_point from num_bits and clip values

        Args:
            num_bits (torch.Tensor): Number of bit for quantization.
            clip_valn (torch.Tensor): Lower clip value.
            clip_val (torch.Tensor): Upper clip value.
            symmetric (bool, optional): Specify if clip values are symmetric. Defaults to False.
            qlevel_lowering (bool, optional): Specify lowering of quantized levels.
                Defaults to True.
            axis (int, optional): Specify which tensor dimension to quantize indiviually.
                Defaults to 0.
            use_minmax (bool, optional): Specify to use Qminmax.  Defaults to False.

        Returns:
            torch.Tensor, torch.Tensor, torch.Tensor: Quantized parameters
        """
        if use_minmax:  # asymmetric case
            n_levels = 2**num_bits - 1
            _, scale, zero_point = asymmetric_linear_quantization_params(
                num_bits, clip_valn, clip_val, qlevel_lowering=False
            )
        else:
            n_levels = 2**num_bits - 2 if qlevel_lowering else 2**num_bits - 1
            _, scale, zero_point = symmetric_linear_quantization_params(
                num_bits, clip_val, qlevel_lowering
            )

        # Broadcast scale, zero_point to tensor shape
        scale, zero_point = per_channel_axis(scale, zero_point, tensor_shape, axis)

        return n_levels, scale, zero_point


class PerChannelSTEQmax_PTnative(PerChannelSTE_PTnative):
    """
    PerChannelSTEQmax_PTnative Base for Qmax

    Extends:
        PerChannelSTE_PTnative
    """

    @staticmethod
    def forward(
        ctx,
        input_tensor: torch.Tensor,
        num_bits: torch.Tensor,
        clip_valn: torch.Tensor,
        clip_val: torch.Tensor,
        dequantize: bool = True,
        symmetric: bool = False,
        qlevel_lowering: bool = False,
        use_minmax: bool = False,
        axis: int = default_axis,
    ):
        """
        General forward method:
            Set clip values to dtype of input_tensor tensor
            Compute # of quantized levels, scale, and zero point
            Save data for backward()
            Perform linear quantization on input_tensor tensor
            return output

        Args:
            ctx (torch.autograd.Function): Forward/Backward context object.
            input_tensor (torch.Tensor): Tensor to be quantized.
            num_bits (torch.Tensor): Number of bit for quantization.
            clip_valn (torch.Tensor): Lower clip value bound.
            clip_val (torch.Tensor): Upper clip value bound.
            dequantize (bool, optional): Return dequantized or int tensor. Defaults to True.
            symmetric (bool, optional): Specify if clip values are symmetric. Defaults to False.
            qlevel_lowering (bool, optional): Specify lowering of quantized levels.
                Defaults to True.
            axis (int, optional): Specify which tensor dimension to quantize indiviually.
                Defaults to 0.

        Returns:
            torch.Tensor: Dequantized or Quantized output tensor.
        """
        clip_valn, clip_val = transform_clips(
            input_tensor.dtype,
            clip_valn,
            clip_val,
        )
        (
            _,
            scale,
            zero_point,
            qint_l,
            qint_h,
            qint_dtype,
        ) = PerChannelSTEQmax_PTnative.calc_qparams(
            num_bits,
            clip_valn,
            clip_val,
            symmetric,
            qlevel_lowering,
            axis=axis,
            tensor_shape=input_tensor.shape,
            use_minmax=use_minmax,
        )
        output = PerChannelSTE_PTnative.linear_quantization(
            input_tensor,
            scale,
            zero_point,
            qint_l,
            qint_h,
            qint_dtype,
            dequantize,
            axis,
        )
        return output

    @classmethod
    def calc_qparams(
        cls,
        num_bits: torch.Tensor,
        clip_valn: torch.Tensor,
        clip_val: torch.Tensor,
        symmetric: bool = False,
        qlevel_lowering: bool = True,
        axis: int = default_axis,
        tensor_shape: torch.Size = None,
        use_minmax: bool = False,
    ):
        """
        Compute the scale and zero_point from num_bits and clip values

        Args:
            num_bits (torch.Tensor): Number of bit for quantization.
            clip_valn (torch.Tensor): Lower clip value.
            clip_val (torch.Tensor): Upper clip value.
            symmetric (bool, optional): Specify if clip values are symmetric. Defaults to False.
            qlevel_lowering (bool, optional): Specify lowering of quantized levels.
                Defaults to True.
            axis (int, optional): Specify which tensor dimension to quantize indiviually.
                Defaults to 0.
            use_minmax (bool, optional): Specify to use Qminmax.  Defaults to False.

        Returns:
            torch.Tensor, torch.Tensor, torch.Tensor: Quantized parameters
        """
        if use_minmax:  # asymmetric case
            n_levels = 2**num_bits - 1
            _, scale, zero_point = asymmetric_linear_quantization_params(
                num_bits, clip_valn, clip_val, qlevel_lowering=False
            )
        else:
            n_levels = 2**num_bits - 2 if qlevel_lowering else 2**num_bits - 1
            _, scale, zero_point = symmetric_linear_quantization_params(
                num_bits, clip_val, qlevel_lowering
            )

        qint_min, qint_max, qint_dtype = PerChannelSTE_PTnative.qint_bounds(
            num_bits, zero_point, symmetric, qlevel_lowering
        )

        # Note: PTnative doesn't require broadcast for scale/zero_point
        return n_levels, scale, zero_point, qint_min, qint_max, qint_dtype

    @staticmethod
    def backward(ctx, grad_output):
        """
        General STE backward method:
            Return grad_output + None args to match forward input_tensor

        Args:
            ctx (torch.autograd.Function): Forward/Backward context object.
            grad_output (torch.Tensor): Gradient tensor

        Returns:
            torch.Tensor, None,...,None: STE Gradient
        """
        return grad_output, None, None, None, None, None, None
