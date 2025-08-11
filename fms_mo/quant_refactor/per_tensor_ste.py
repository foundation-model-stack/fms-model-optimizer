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
Base Tensor Quantizer classes.
The PACT family of quantizers implements PerTensorSTE.
The SAWB and Qmax families implements their respective PerTensorSTE<family> quantizers.
Each group has a PyTorch native implementation that uses the fake_quantize_per_tensor_affine.
Each STE implements the torch.autograd.Function forward() and backward() functions.
"""

# Standard
from typing import Tuple

# Third Party
import torch

# Local
from fms_mo.quant_refactor.linear_utils import (
    asymmetric_linear_quantization_params,
    linear_quantization,
    symmetric_linear_quantization_params,
    transform_clips,
)


class PerTensorSTE(torch.autograd.Function):
    """
    Base class for customized forward/backward functions that is NOT using PT native func.
    There's a family of non-learnable quantizers, such as SAWB, MinMax,
    whose forward can be the same quant functions and backward is simply STE.
    We just need to calculate scales in the upper level quantizer class then those quantizers
    could all be using the same base "STE function"

    Math should be consistent with pytorch: https://pytorch.org/docs/stable/quantization.html
        x_int = round(x/scale + zp).clamp(qint_l, qint_h)
        x_dq  = (x_int - zp) * scale
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
        qlevel_lowering: bool = True,
    ) -> torch.Tensor:
        """
        General forward method:
            Set clip values to dtype of input tensor
            Compute # of quantized levels, scale, and zero point
            Save data for backward()
            Perform linear quantization on input tensor
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

        Returns:
            torch.Tensor: Dequantized or Quantized output tensor.
        """

        clip_valn, clip_val = transform_clips(
            input_tensor.dtype,
            clip_valn,
            clip_val,
        )
        n_levels, scale, zero_point = PerTensorSTE.calc_qparams(
            num_bits, clip_valn, clip_val, symmetric, qlevel_lowering
        )
        PerTensorSTE.save_tensors(
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

    @classmethod
    def calc_qparams(
        cls,
        num_bits: torch.Tensor,
        clip_valn: torch.Tensor,
        clip_val: torch.Tensor,
        symmetric: bool = False,
        qlevel_lowering: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Quantized parameters
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
        return n_levels, scale, zero_point

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        """
        General STE backward method:
            Return grad_output + None args to match forward inputs

        Args:
            ctx (torch.autograd.Function): Forward/Backward context object.
            grad_output (torch.Tensor): Gradient tensor

        Returns:
            torch.Tensor, None,...,None: STE Gradient
        """
        return grad_output, None, None, None, None, None, None


class PerTensorSTE_PTnative(torch.autograd.Function):
    """
    Base class for customized forward/backward functions that IS using PT native func.
    There's a family of non-learnable quantizers, such as SAWB, MinMax,
    whose forward can be the same quant functions and backward is simply STE.
    We just need to calculate scales in the upper level quantizer class then those quantizers
    could all be using the same base "STE function"

    Math should be consistent with pytorch: https://pytorch.org/docs/stable/quantization.html
        x_int = round(x/scale + zp).clamp(qint_l, qint_h)
        x_dq  = (x_int - zp) * scale
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
        qlevel_lowering: bool = True,
    ) -> torch.Tensor:
        """
        General forward method:
            Set clip values to dtype of input tensor
            Compute # of quantized levels, scale, and zero point
            Perform PTnative linear quantization on input tensor
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
        ) = PerTensorSTE_PTnative.calc_qparams(
            num_bits, clip_valn, clip_val, symmetric, qlevel_lowering
        )
        output = PerTensorSTE_PTnative.linear_quantization(
            input_tensor, scale, zero_point, qint_l, qint_h, qint_dtype, dequantize
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
            torch.tensor(0)
            if symmetric
            else torch.round(-clip_valn / scale).to(torch.int)
        )
        qint_l, qint_h, qint_dtype = PerTensorSTE_PTnative.qint_bounds(
            num_bits, zero_point, symmetric, qlevel_lowering
        )
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
        if symmetric and zero_point == 0:
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
    ) -> torch.Tensor:
        """
        Linear quantization for PTnative STE

        Args:
            input_tensor (torch.Tensor): Tensor to be quantized
            scale (torch.Tensor): Quantized bin range.
            zero_point (torch.Tensor): Quantized integer bin mapping to fp 0.0.
            qint_l (int): Quantized integer lower clip value.
            qint_h (int): Quantized integer upper clip value.
            qint_dtype (torch.dtype): Quantized integer dtype.
            dequantize (bool, optional): Specify to return fp or quantized int. Defaults to True.

        Returns:
            torch.Tensor: PTnative quantized or dequantized tensor.
        """
        if dequantize:
            output = torch.fake_quantize_per_tensor_affine(
                input_tensor.float(),
                scale.float(),
                zero_point,
                quant_min=qint_l,
                quant_max=qint_h,
            ).to(input_tensor.dtype)
        else:
            output = (
                torch.quantize_per_tensor(input_tensor, scale, zero_point, qint_dtype)
                .int_repr()
                .clamp(qint_l, qint_h)
            )
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        General STE backward method:
            Return grad_output + None args to match forward inputs

        Args:
            ctx (torch.autograd.Function): Forward/Backward context object.
            grad_output (torch.Tensor): Gradient tensor

        Returns:
            [torch.Tensor, None,...,None]: STE Gradient
        """
        return grad_output, None, None, None, None, None, None


class PerTensorSTESAWB(PerTensorSTE):
    """
    PerTensorSTE Base for SAWB

    Extends:
        PerTensorSTE
    """

    @staticmethod
    def forward(
        ctx,
        input_tensor: torch.Tensor,
        num_bits: torch.Tensor,
        clip_valn: torch.Tensor,
        clip_val: torch.Tensor,
        dequantize: bool = True,
        symmetric: bool = True,
        qlevel_lowering: bool = True,
        use_code: bool = False,
    ) -> torch.Tensor:
        """
        General forward method:
            Set clip values to dtype of input tensor
            Compute # of quantized levels, scale, and zero point
            Save data for backward()
            Perform linear quantization on input tensor
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

        Returns:
            torch.Tensor: Dequantized or Quantized output tensor.
        """
        clip_valn, clip_val = transform_clips(input_tensor.dtype, clip_valn, clip_val)
        n_levels, scale, zero_point = PerTensorSTESAWB.calc_qparams(
            num_bits, clip_valn, clip_val, symmetric, qlevel_lowering, use_code
        )
        PerTensorSTE.save_tensors(
            ctx, tensors=(input_tensor, n_levels, clip_val, scale, zero_point)
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
        symmetric: bool = True,
        qlevel_lowering: bool = False,
        use_code: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute the scale and zero_point from num_bits and clip values

        Args:
            num_bits (torch.Tensor): Number of bit for quantization.
            clip_valn (torch.Tensor): Lower clip value.
            clip_val (torch.Tensor): Upper clip value.
            symmetric (bool, optional): Specify if clip values are symmetric. Defaults to False.
            qlevel_lowering (bool, optional): Specify lowering of quantized levels.
                Defaults to True.
            use_code (bool, optional): Specify using SAWB code. Defaults to False.

        Returns:
            [torch.Tensor, torch.Tensor, torch.Tensor]: Quantized parameters
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

            output = n_levels, scale, zero_point
        else:
            raise ValueError("SAWB has non-symmetric Qscheme")
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        General STE backward method:
            Return grad_output + None args to match forward inputs

        Args:
            ctx (torch.autograd.Function): Forward/Backward context object.
            grad_output (torch.Tensor): Gradient tensor

        Returns:
            [torch.Tensor, None,...,None]: STE Gradient
        """
        return grad_output, None, None, None, None, None, None, None


class PerTensorSTESAWB_PTnative(PerTensorSTE_PTnative):
    """
    PerTensorSTE Base for SAWB PTnative

    Extends:
        PerTensorSTE_PTnative
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
        qlevel_lowering: bool = True,
        use_code: bool = False,
    ) -> torch.Tensor:
        """
        General forward method:
            Set clip values to dtype of input tensor
            Compute # of quantized levels, scale, and zero point
            Perform linear quantization on input tensor
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
            use_code (bool, optional): Use SAWB code. Defaults to False.

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
            _,
            scale,
            zero_point,
            qint_l,
            qint_h,
            qint_dtype,
        ) = PerTensorSTESAWB_PTnative.calc_qparams(
            num_bits, clip_valn, clip_val, symmetric, qlevel_lowering, use_code
        )
        output = PerTensorSTE_PTnative.linear_quantization(
            input_tensor, scale, zero_point, qint_l, qint_h, qint_dtype, dequantize
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
        use_code: bool = False,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        int,
        int,
        torch.dtype,
    ]:
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
            use_code (bool, optional): Specify using SAWB code. Defaults to False.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor,
            torch.Tensor, torch.Tensor, torch.Tensor,
            torch.dtype]: Quantized PTnative parameters
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
            qint_min, qint_max, qint_dtype = PerTensorSTE_PTnative.qint_bounds(
                num_bits, zero_point, symmetric, qlevel_lowering
            )

            output = (
                n_levels,
                clip_val,
                scale,
                zero_point.to(torch.int),
                qint_min,
                qint_max,
                qint_dtype,
            )
        else:
            raise ValueError("SAWB PTnative has non-symmetric Qscheme")
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        General STE backward method:
            Return grad_output + None args to match forward inputs

        Args:
            ctx (torch.autograd.Function): Forward/Backward context object.
            grad_output (torch.Tensor): Gradient tensor

        Returns:
            [torch.Tensor, None,...,None]: STE Gradient
        """
        return grad_output, None, None, None, None, None, None, None


class PerTensorSTEQmax(PerTensorSTE):
    """
    PerTensorSTE Base for Qmax

    Extends:
        PerTensorSTE
    """

    @staticmethod
    def forward(
        ctx,
        input_tensor: torch.Tensor,
        num_bits: torch.Tensor,
        clip_valn: torch.Tensor,
        clip_val: torch.Tensor,
        dequantize: bool = True,
        symmetric: bool = True,
        qlevel_lowering: bool = True,
        minmax: bool = False,
    ) -> torch.Tensor:
        """
        General forward method:
            Set clip values to dtype of input tensor
            Compute # of quantized levels, scale, and zero point
            Save data for backward()
            Perform linear quantization on input tensor
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
            minmax (bool, optional): Specify to use Qminmax STES. Defaults to False.

        Returns:
            torch.Tensor: Dequantized or Quantized output tensor.
        """
        clip_valn, clip_val = transform_clips(
            input_tensor.dtype,
            clip_valn,
            clip_val,
        )
        n_levels, scale, zero_point = PerTensorSTEQmax.calc_qparams(
            num_bits, clip_valn, clip_val, symmetric, qlevel_lowering, minmax
        )
        PerTensorSTE.save_tensors(
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
        qlevel_lowering: bool = False,
        use_minmax: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the scale and zero_point from num_bits and clip values

        Args:
            num_bits (torch.Tensor): Number of bit for quantization.
            clip_valn (torch.Tensor): Lower clip value.
            clip_val (torch.Tensor): Upper clip value.
            symmetric (bool, optional): Specify if clip values are symmetric. Defaults to False.
            qlevel_lowering (bool, optional): Specify lowering of quantized levels.
                Defaults to True.
            use_minmax (bool, optional): Specify using Qminmax. Defaults to False.

        Returns:
            [torch.Tensor, torch.Tensor, torch.Tensor]: Quantized PTnative parameters
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

        return n_levels, scale, zero_point

    @staticmethod
    def backward(ctx, grad_output):
        """
        General STE backward method:
            Return grad_output + None args to match forward inputs

        Args:
            ctx (torch.autograd.Function): Forward/Backward context object.
            grad_output (torch.Tensor): Gradient tensor

        Returns:
            [torch.Tensor, None,...,None]: STE Gradient
        """
        return grad_output, None, None, None, None, None, None, None


class PerTensorSTEQmax_PTnative(PerTensorSTE_PTnative):
    """
    PerTensorSTE Base for QMax PTnative

    Extends:
        PerTensorSTE_PTnative
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
        qlevel_lowering: bool = True,
        use_minmax: bool = False,
    ) -> torch.Tensor:
        """
        General forward method:
            Set clip values to dtype of input tensor
            Compute # of quantized levels, scale, and zero point
            Perform linear quantization on input tensor
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
            _,
            scale,
            zero_point,
            qint_l,
            qint_h,
            qint_dtype,
        ) = PerTensorSTEQmax_PTnative.calc_qparams(
            num_bits, clip_valn, clip_val, symmetric, qlevel_lowering, use_minmax
        )
        output = PerTensorSTE_PTnative.linear_quantization(
            input_tensor, scale, zero_point, qint_l, qint_h, qint_dtype, dequantize
        )
        return output

    @classmethod
    def calc_qparams(
        cls,
        num_bits: torch.Tensor,
        clip_valn: torch.Tensor,
        clip_val: torch.Tensor,
        symmetric: bool = True,
        qlevel_lowering: bool = False,
        use_minmax: bool = False,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        int,
        int,
        torch.dtype,
    ]:
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
            use_minmax (bool, optional): Specify using Qminmax. Defaults to False.

        Returns:
            [torch.Tensor, torch.Tensor, torch.Tensor,
            torch.Tensor, torch.Tensor, torch.Tensor,
            torch.dtype]: Quantized PTnative parameters
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

        qint_min, qint_max, qint_dtype = PerTensorSTE_PTnative.qint_bounds(
            num_bits, zero_point, symmetric, qlevel_lowering
        )
        return n_levels, clip_val, scale, zero_point, qint_min, qint_max, qint_dtype

    @staticmethod
    def backward(ctx, grad_output):
        """
        General STE backward method:
            Return grad_output + None args to match forward inputs

        Args:
            ctx (torch.autograd.Function): Forward/Backward context object.
            grad_output (torch.Tensor): Gradient tensor

        Returns:
            [torch.Tensor, None,...,None]: STE Gradient
        """
        return grad_output, None, None, None, None, None, None, None
