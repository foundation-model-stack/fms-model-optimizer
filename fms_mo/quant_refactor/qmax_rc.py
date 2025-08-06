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
Qmax Quantizers and STEs
"""

# Standard
from typing import Tuple

# Third Party
import torch

# Local
from fms_mo.quant_refactor.base_quant import Qscheme, Quantizer
from fms_mo.quant_refactor.per_channel_ste import (
    PerChannelSTE_PTnative,
    PerChannelSTEQmax,
    PerChannelSTEQmax_PTnative,
)
from fms_mo.quant_refactor.per_tensor_ste import (
    PerTensorSTE_PTnative,
    PerTensorSTEQmax,
    PerTensorSTEQmax_PTnative,
)

clip_valn_default = torch.Tensor([-8.0])
clip_val_default = torch.Tensor([8.0])
qscheme_per_tensor = Qscheme(
    unit="perT",
    symmetric=False,
    Nch=None,
    Ngrp=None,
    single_sided=False,
    qlevel_lowering=False,
)


class Qmax_rc(Quantizer):
    """
    SAWB with custom backward (gradient pass through for clip function)
    if align_zero: quantizer = SAWBSTE() for coded sawb such as 103, 403, 803
    if not align_zero: quantizer = SAWBZeroSTE() for normal precision setting such as 2, 4, 8
    Qmax can quantize both weights and activations

    Extends:
        Quantizer
    """

    def __init__(
        self,
        num_bits: torch.Tensor,
        init_clip_valn: torch.Tensor = clip_valn_default,
        init_clip_val: torch.Tensor = clip_val_default,
        qscheme: Qscheme = qscheme_per_tensor,
        dequantize: bool = True,
        align_zero: bool = False,
        clipSTE: bool = True,
        minmax: bool = False,
        extend_act_range: bool = False,
        **kwargs,
    ):
        """
        Init SAWB Quantizer

        Args:
            num_bits (torch.Tensor): Number of bits for quantization.
            init_clip_valn (torch.Tensor, optional): Lower clip value bound. Defaults to -8.0.
            init_clip_val (torch.Tensor, optional): Upper clip value bound. Defaults to 8.0.
            qscheme (Qscheme, optional): Quantization scheme.
                Defaults to Qscheme( unit="perT", symmetric=True, Nch=None, Ngrp=None,
                                       single_sided=False, qlevel_lowering=False, ).
            dequantize (bool, optional): Return dequantized or int tensor. Defaults to True.
            align_zero (bool, optional): Specify using qlevel_lowering. Defaults to False.
            clipSTE (bool, optional): Specify to not clip backward() input tensor.
                Defaults to False.
            extend_act_range (bool, optional): Specify using full quantization range.
                Defaults to False.
            use16bins (bool, optional): Specify using special 16bin quantization. Defaults to False.
        """
        super().__init__(
            num_bits,
            dequantize,
            qscheme=qscheme,
            use_PT_native_Qfunc=kwargs.get("use_PT_native_Qfunc", False),
        )

        if not self.training:
            with torch.no_grad():
                self.clip_valn.data *= init_clip_valn
                self.clip_val.data *= init_clip_val

        self.align_zero = align_zero
        self.clipSTE = clipSTE
        self.minmax = minmax
        self.extend_act_range = extend_act_range

        # Training
        self.movAvgFac = 0.1
        self.Niter = 0
        # copy new clip vals in forward without training from nn.Module
        self.recompute_clips = False

        self.set_quantizer()

    def copy_legacy_vars(self):
        """
        Copy legacy variables to newer versions.
        Allows for better backwards compatability
        """
        self.qscheme.qlevel_lowering = self.align_zero

    def set_quantizer(self):
        """
        Set quantizer STE based on current member variables
        """
        self.copy_legacy_vars()  # copy align_zero

        if self.use_PT_native_Qfunc:
            if self.minmax:
                self.quantizer_name = "Qminmax"
                if self.perGrp:
                    # self.quantizer = QmaxPerGrpSTE_PTnative
                    pass
                else:
                    self.quantizer = (
                        QmaxPerChSTE_PTnative
                        if self.perCh
                        else PerTensorSTEQmax_PTnative
                    )
            else:
                if self.extend_act_range:
                    self.quantizer_name = "QmaxExtend"
                    self.set_clip_ratio()
                    self.quantizer = QmaxExtendRangeSTE_PTnative
                else:
                    self.quantizer_name = "Qminmax" if self.minmax else "Qmax"
                    self.quantizer = (
                        QmaxPerChSTE_PTnative
                        if self.perCh
                        else PerTensorSTEQmax_PTnative
                    )
        else:
            if self.minmax:
                self.quantizer_name = "Qminmax"
                if self.perGrp:
                    self.quantizer = QminmaxPerGpSTE_rc
                else:
                    self.quantizer = QminmaxPerChSTE_rc if self.perCh else QminmaxSTE_rc
            elif self.extend_act_range:
                self.quantizer = QmaxExtendRangeSTE_rc
                self.quantizer_name = "QmaxExtend"
                self.set_clip_ratio()
            else:
                self.quantizer_name = "Qmax"
                if self.perGrp:
                    self.quantizer = QmaxPerGpSTE_rc
                else:
                    self.quantizer = QmaxPerChSTE_rc if self.perCh else QmaxSTE_rc

    def set_clip_ratio(self):
        """
        Compute clip_ratio used in Extended Range STEs
        """
        n_half = 2 ** (self.num_bits - 1)
        self.clip_ratio = -n_half / (n_half - 1)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Qmax forward() function.
        Has codepath for saving previous runs computed clip values.

        Args:
            input_tensor (torch.Tensor): Tensor to be quantized.

        Returns:
            torch.Tensor: Quantized or dequantized tensor.
        """
        if self.perCh:
            if self.minmax:
                clip_val_temp = torch.max(
                    input_tensor.reshape([self.qscheme.Nch, -1]), dim=1
                ).values
                clip_valn_temp = torch.min(
                    input_tensor.reshape([self.qscheme.Nch, -1]), dim=1
                ).values
            else:
                clip_val_temp = torch.max(
                    input_tensor.abs().reshape([self.qscheme.Nch, -1]), dim=1
                ).values
                clip_valn_temp = -clip_val_temp
            assert (
                len(clip_val_temp) == input_tensor.shape[0]
            ), f"dimension error, input{input_tensor.shape}, clipval{clip_val_temp.shape}"
        elif self.perGrp:
            if self.minmax:
                clip_val_temp = torch.max(
                    input_tensor.reshape(self.perGrp), dim=1
                ).values
                clip_valn_temp = torch.min(
                    input_tensor.reshape(self.perGrp), dim=1
                ).values
            else:
                clip_val_temp = torch.max(
                    input_tensor.abs().reshape(self.perGrp), dim=1
                ).values
                clip_valn_temp = -clip_val_temp
            assert (
                len(clip_val_temp)
                == (input_tensor.shape[0] * input_tensor.shape[1] // self.perGrp[1])
            ), f"dimension error, input{input_tensor.shape}, clip_val{clip_val_temp.shape}"
        elif self.extend_act_range:
            if input_tensor.max() >= input_tensor.min().abs():
                clip_val_temp = input_tensor.max()
                clip_valn_temp = clip_val_temp * self.clip_ratio
            else:
                clip_valn_temp = input_tensor.min()
                clip_val_temp = clip_valn_temp / self.clip_ratio
        else:
            if self.minmax:  # asymmetric
                clip_val_temp = input_tensor.max()
                clip_valn_temp = input_tensor.min()
            else:  # symmetric
                clip_val_temp = input_tensor.abs().max()
                clip_valn_temp = -clip_val_temp

        if len(clip_val_temp.shape) == 0:
            clip_val_temp = clip_val_temp.unsqueeze(dim=0)
        if len(clip_valn_temp.shape) == 0:
            clip_valn_temp = clip_valn_temp.unsqueeze(dim=0)

        if (self.Niter == 0 and self.training) or self.recompute_clips:
            # to avoid unintended bwd ops added to the graph, cause memory leak sometimes
            with torch.no_grad():
                self.clip_val.copy_(clip_val_temp)
                self.clip_valn.copy_(clip_valn_temp)

        if self.training:
            output = self.quantizer.apply(
                input_tensor,
                self.num_bits,
                clip_valn_temp,
                clip_val_temp,  # use new clip_vals first, then do moving average
                self.dequantize,
                self.qscheme.symmetric,
                self.qscheme.qlevel_lowering,
                self.minmax,
            )
            with torch.no_grad():
                # to avoid unintended bwd ops added to the graph, cause memory leak sometimes
                self.clip_val.copy_(
                    self.clip_val * (1.0 - self.movAvgFac)
                    + clip_val_temp * self.movAvgFac
                )
                if self.extend_act_range:
                    self.clip_valn.copy_(self.clip_val * self.clip_ratio)
                else:
                    self.clip_valn.copy_(
                        self.clip_valn * (1.0 - self.movAvgFac)
                        + clip_valn_temp * self.movAvgFac
                    )
        else:
            output = self.quantizer.apply(
                input_tensor,
                self.num_bits,
                self.clip_valn,
                self.clip_val,
                self.dequantize,
                self.qscheme.symmetric,
                self.qscheme.qlevel_lowering,
                self.minmax,
            )

        self.Niter += 1
        return output


class QmaxSTE_rc(PerTensorSTEQmax):
    """
    QMax with zero alignment (symmetric)

    Extends:
        PerTensorSTEQmax: Uses PerTensorSTEQmax.forward()
    """

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward function for Qmax STE

        Args:
            ctx (torch.autograd.Function): Context object.
            grad_output (torch.Tensor): Gradient to clip

        Returns:
            [torch.Tensor, None,...,None]: Gradients
        """
        return grad_output, None, None, None, None, None, None, None


class QminmaxSTE_rc(PerTensorSTEQmax):
    """minmax with zero alignment (asymmetric)
    Dequantization always enabled (cannot be turned off, at this time)
    """

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward function for Qminmax STE

        Args:
            ctx (torch.autograd.Function): Context object.
            grad_output (torch.Tensor): Gradient to clip

        Returns:
            [torch.Tensor, None,...,None]: Gradients
        """
        return grad_output, None, None, None, None, None, None, None


class QmaxExtendRangeSTE_rc(torch.autograd.Function):
    """
    2-sided Max quantizer using a single clip

    Assume negative clip (clip_valn) was derived from positive clip (clip_val)
    using extended range, such that:
        clip_valn / clip_val = -2^(b-1)/(2^(b-1)-1)
    Example:
        b = 8 --> [-128/127 * clip_val, clip_val]
        b = 4 --> [-8/7 * clip_val, clip_val]

    This quantizer has NO zero_point, and it is therefore suitable for BMM
    Zero alignment (FP=0 --> INT=0 --> FP=0) is guaranteed (align_zero is ignored)
    No in_place functionalities
    Dequantization is functional
    """

    @staticmethod
    def forward(
        ctx,
        input_tensor: torch.Tensor,
        num_bits: torch.Tensor,
        clip_valn: torch.Tensor,
        clip_val: torch.Tensor,
        dequantize: bool = True,
        _symmetric: bool = False,
        _qlevel_lowering: bool = True,
        _use_minmax: bool = False,
    ):
        """
        Forward function for QmaxExtendRangeSTE

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
            _use_minmax (bool, optional): Specify using Qminmax. Defaults to False.

        Returns:
            torch.Tensor: Dequantized or Quantized output tensor.
        """
        clip_val, clip_valn = (
            clip_val.to(input_tensor.dtype),
            clip_valn.to(input_tensor.dtype),
        )

        scale = clip_val / (2 ** (num_bits - 1) - 1)
        qint_min, qint_max = -(2 ** (num_bits - 1)), 2 ** (num_bits - 1) - 1

        # quantize to range [-2^(b-1), 2^(b-1)-1]
        output = torch.round(input_tensor / scale)
        output = output.clamp(qint_min, qint_max)
        if dequantize:
            output = output * scale
        else:
            output = output.to(torch.int8)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward function for Qmax Extended Range STE

        Args:
            ctx (torch.autograd.Function): Context object.
            grad_output (torch.Tensor): Gradient to clip

        Returns:
            [torch.Tensor, None,...,None]: Gradients
        """
        return grad_output, None, None, None, None, None, None, None


class QmaxExtendRangeSTE_PTnative(PerTensorSTEQmax_PTnative):
    """
    Qmax Extended Range STE w/ PT native kernel

    Extends:
        PerTensorSTEQmax_PTnative
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
        """ "
        Forward function for QmaxExtendRangeSTE w/ PTnative kernel

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
            use_minmax (bool, optional): Specify using Qminmax. Defaults to False.

        Returns:
            torch.Tensor: Dequantized or Quantized output tensor.
        """
        (
            _,
            _,
            scale,
            zero_point,
            qint_l,
            qint_h,
            qint_dtype,
        ) = QmaxExtendRangeSTE_PTnative.calc_qparams(
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
        symmetric: bool = False,
        qlevel_lowering: bool = True,
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
            [torch.Tensor, torch.Tensor, torch.Tensor
            torch.Tensor, torch.Tensor, torch.Tensor, torch.dtype]:
                Quantized parameters for PTnative kernels
        """
        n_levels = 2**num_bits - 2
        scale = 2 * clip_val / n_levels
        zero_point = torch.tensor(0)
        qint_min, qint_max = -(2 ** (num_bits - 1)), 2 ** (num_bits - 1) - 1
        qint_dtype = torch.qint8

        return n_levels, clip_val, scale, zero_point, qint_min, qint_max, qint_dtype


class QmaxPerChSTE_rc(PerChannelSTEQmax):
    """
    Max with zero alignment (symmetric)
    "dequantize=False" option is functional
    """

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward function for Qmax Per Channel STE

        Args:
            ctx (torch.autograd.Function): Context object.
            grad_output (torch.Tensor): Gradient to clip

        Returns:
            [torch.Tensor, None,...,None]: Gradients
        """
        return grad_output, None, None, None, None, None, None


class QmaxPerChSTE_PTnative(PerChannelSTEQmax_PTnative):
    """
    Max with zero alignment (symmetric)
    "dequantize=False" option is functional
    """

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward function for Qmax Per Channel STE

        Args:
            ctx (torch.autograd.Function): Context object.
            grad_output (torch.Tensor): Gradient to clip

        Returns:
            [torch.Tensor, None,...,None]: Gradients
        """
        return grad_output, None, None, None, None, None, None


class QmaxPerGpSTE_rc(torch.autograd.Function):
    """
    Max with zero alignment (symmetric)
    per group quantization
    """

    @staticmethod
    def forward(
        ctx, input_tensor, num_bits, _dequantize, inplace, cv, _cvn, align_zero
    ) -> torch.Tensor:
        """
        TODO (bmgroth): docstring
        """
        if inplace:
            ctx.mark_dirty(input_tensor)
        _clip_val = cv
        input_shape = input_tensor.shape
        clip_val_shape = cv.shape
        # use clip_val shape to reshape input
        input_tensor = input_tensor.reshape(clip_val_shape[0], -1)
        scale = (2**num_bits - 2) if align_zero else (2**num_bits - 1)
        zero_point = 0.0

        # here use symmetric similar to sawbperCh code
        _nspace = 2**num_bits - 2  # lose one level
        int_l = -(2 ** (num_bits - 1)) + 1
        int_u = -int_l  # symmetric
        scale = (
            cv * 2 / (2**num_bits - 2)
        )  # original SAWB assumes odd number of bins when calc clip_val
        zero_point = torch.zeros_like(scale)  # centers around 0 and align 0
        # FIXME, fake quantize function only support float.
        output = torch.fake_quantize_per_channel_affine(
            input_tensor.float(),
            scale.float(),
            zero_point.float(),
            axis=0,
            quant_min=int_l,
            quant_max=int_u,
        ).to(input_tensor.dtype)
        # reshape back to original shape
        output = output.reshape(input_shape)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward function for Qmax Per Group STE

        Args:
            ctx (torch.autograd.Function): Context object.
            grad_output (torch.Tensor): Gradient to clip

        Returns:
            [torch.Tensor, None,...,None]: Gradients
        """
        return grad_output, None, None, None, None, None, None


class QminmaxPerChSTE_rc(PerChannelSTEQmax):
    """
    per channel minmax with zero alignment (asymmetric)
    """

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward function for Qminmax Per Channel STE

        Args:
            ctx (torch.autograd.Function): Context object.
            grad_output (torch.Tensor): Gradient to clip

        Returns:
            [torch.Tensor, None,...,None]: Gradients
        """
        return grad_output, None, None, None, None, None, None


class QminmaxPerGpSTE_rc(torch.autograd.Function):
    """
    per group minmax with zero alignment (asymmetric)
    """

    @staticmethod
    def forward(
        ctx, input_tensor, num_bits, _dequantize, inplace, cv, cvn, align_zero
    ) -> torch.Tensor:
        """
        TODO (bmgroth): docstring
        """
        if inplace:
            ctx.mark_dirty(input_tensor)
        cv, cvn = cv.to(input_tensor.dtype), cvn.to(input_tensor.dtype)
        input_shape = input_tensor.shape
        clip_val_shape = cv.shape
        input_tensor = input_tensor.reshape(clip_val_shape[0], -1)
        scale = (2**num_bits - 1) / (cv - cvn)
        zero_point = cvn * scale
        if align_zero:
            zero_point = torch.round(zero_point)
        output = (input_tensor.clamp(cvn[:, None], cv[:, None]) - cvn[:, None]) * scale[
            :, None
        ]
        output = (torch.round(output) + zero_point[:, None]) / scale[:, None]
        # reshape back to original shape
        output = output.reshape(input_shape)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward function for Qminmax Per Group STE

        Args:
            ctx (torch.autograd.Function): Context object.
            grad_output (torch.Tensor): Gradient to clip

        Returns:
            [torch.Tensor, None,...,None]: Gradients
        """
        return grad_output, None, None, None, None, None, None
