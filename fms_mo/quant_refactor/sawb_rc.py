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
SAWB Quantizer Rewrite
"""

# Standard
from typing import Tuple

# Third Party
import torch

# Local
from fms_mo.quant_refactor.base_quant import Qscheme, Quantizer
from fms_mo.quant_refactor.per_channel_ste import (
    PerChannelSTESAWB,
    PerChannelSTESAWB_PTnative,
)
from fms_mo.quant_refactor.per_tensor_ste import (
    PerTensorSTESAWB,
    PerTensorSTESAWB_PTnative,
)

# from fms_mo.quant_refactor.linear_utils import linear_dequantize, linear_quantize
from fms_mo.quant_refactor.sawb_utils import sawb_params, sawb_params_code

clip_valn_default = torch.tensor([-8.0])
clip_val_default = torch.tensor([8.0])
qscheme_per_tensor = Qscheme(
    unit="perT",
    symmetric=False,
    single_sided=False,
    qlevel_lowering=False,
)


class SAWB_rc(Quantizer):
    """
    SAWB with custom backward (gradient pass through for clip function)
    if align_zero: quantizer = SAWBSTE() for coded sawb such as 103, 403, 803
    if not align_zero: quantizer = SAWBZeroSTE() for normal precision setting such as 2, 4, 8

    SAWB is only used to quantize weights

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
        clipSTE: bool = False,
        align_zero: bool = False,
        use16bins: bool = False,
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
            clipSTE (bool, optional): Specify to not clip backward() input tensor.
                Defaults to False.
            align_zero (bool, optional): Specify using qlevel_lowering. Defaults to False.
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

        self.clipSTE = clipSTE
        self.align_zero = align_zero
        self.use16bins = use16bins

        self.set_quantizer()

        # Training variables
        self.movAvgFac = 0.1
        self.Niter = 0

        # copy new clip vals in forward without training from nn.Module
        self.recompute_clips = False

    def copy_legacy_vars(self):
        """
        Copy legacy variables to newer versions.
        Allows for better backwards compatability
        """
        self.qscheme.qlevel_lowering = self.align_zero
        self.extended_ranged = self.use16bins

    def set_quantizer(self):
        """
        Set quantizer STE based on current member variables
        """
        self.copy_legacy_vars()
        self.use_extended_range_4bits = (
            self.clipSTE
            and self.qscheme.qlevel_lowering
            and self.extended_ranged
            and self.num_bits == 4
        )

        # PTnative quantizers
        if self.use_PT_native_Qfunc:
            if self.perCh:
                self.use_code = self.qscheme.qlevel_lowering
                self.quantizer = PerChannelSTESAWB_PTnative
            else:
                self.use_code = self.qscheme.qlevel_lowering
                self.quantizer = PerTensorSTESAWB_PTnative

        else:  # Non-PTnative quantizers
            self.use_code = self.qscheme.qlevel_lowering
            if self.clipSTE:
                if self.qscheme.qlevel_lowering:
                    self.quantizer = (
                        SAWBPlusZeroPerChSTE_rc
                        if self.perCh and self.num_bits in [2, 4, 8]
                        else SAWBPlusZeroSTE_rc
                    )
                else:
                    self.quantizer = SAWBPlusSTE_rc
            else:
                # if perCh but no sawb+ (e.g. `sawb_perCh`) will use a per-tensor
                # clip copied over each channel
                if self.qscheme.qlevel_lowering:
                    self.quantizer = SAWBZeroSTE_rc
                else:
                    self.quantizer = SAWBSTE_rc

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        SAWB forward() function.
        Has codepath for saving previous runs computed clip values.

        Args:
            input_tensor (torch.Tensor): Tensor to be quantized.

        Returns:
            torch.Tensor: Quantized or dequantized tensor.
        """
        # Set new clip vals based on options
        if self.perCh:
            bits2codeDict = {2: 103, 4: 403, 8: 803}
            num_bits_int = (
                self.num_bits.item()
                if isinstance(self.num_bits, torch.Tensor)
                else self.num_bits
            )
            code = bits2codeDict[num_bits_int]
            _, clip_val_temp = sawb_params_code(
                input_tensor, self.num_bits, code, perCh=True
            )
        elif self.use_extended_range_4bits:
            _, clip_val_temp = sawb_params_code(
                input_tensor, self.num_bits, 403, perCh=False
            )
        else:
            if self.use_code:
                bits2codeDict = {2: 103, 4: 403, 8: 803}
                num_bits_int = (
                    self.num_bits.item()
                    if isinstance(self.num_bits, torch.Tensor)
                    else self.num_bits
                )
                code = bits2codeDict[num_bits_int]
                _, clip_val_temp = sawb_params_code(
                    input_tensor, self.num_bits, code, perCh=False
                )
            else:
                _, clip_val_temp = sawb_params(
                    input_tensor, self.num_bits, self.qscheme.qlevel_lowering
                )
        clip_val_temp = clip_val_temp.to(input_tensor.dtype)  # keep dtype of input
        clip_valn_temp = -clip_val_temp

        # If shape == 0, unsqueese clip vals
        if len(clip_val_temp.shape) == 0:
            clip_val_temp = clip_val_temp.unsqueeze(dim=0)
        if len(clip_valn_temp.shape) == 0:
            clip_valn_temp = clip_valn_temp.unsqueeze(dim=0)

        if (self.Niter == 0 and self.training) or self.recompute_clips:
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
                self.use_code,
            )

            # Copy new clip vals to parameter clip vals with moving average calc
            with torch.no_grad():
                self.clip_val.copy_(
                    self.clip_val * (1.0 - self.movAvgFac)
                    + clip_val_temp * self.movAvgFac
                )
                # Special case for extended range
                if self.use_extended_range_4bits:
                    clip_ratio = -15.0 / 7.0
                    self.clip_valn.copy_(self.clip_val * clip_ratio)
                else:
                    self.clip_valn.copy_(
                        self.clip_valn * (1.0 - self.movAvgFac)
                        + clip_valn_temp * self.movAvgFac
                    )

        else:  # Inference case
            output = self.quantizer.apply(
                input_tensor,
                self.num_bits,
                self.clip_valn,
                self.clip_val,
                self.dequantize,
                self.qscheme.symmetric,
                self.qscheme.qlevel_lowering,
                self.use_code,
            )
        self.Niter += 1
        return output


class SAWBSTE_rc(PerTensorSTESAWB):
    """
    SAWB without zero alignment

    Extends:
        PerTensorSTESAWB: Uses PerTensorSTESAWB.forward()
    """

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward function for SAWB STE

        Args:
            ctx (torch.autograd.Function): Context object.
            grad_output (torch.Tensor): Gradient to clip

        Returns:
            [torch.Tensor, None,...,None]: Gradients
        """
        input_tensor, _, _, clip_val, _, _ = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input = torch.where(
            input_tensor < -clip_val, torch.zeros_like(grad_input), grad_input
        )
        grad_input = torch.where(
            input_tensor > clip_val, torch.zeros_like(grad_input), grad_input
        )
        return grad_input, None, None, None, None, None, None, None


class SAWBZeroSTE_rc(PerTensorSTESAWB):
    """
    SAWB with zero alignment (symmetric) and gradient clipping
    Supported bits: 2, 4, 8
    Other bits requests: runs x.abs().max(), not SAWB
    "dequantize=False" option is functional

    Extends:
        PerTensorSTESAWB: Uses PerTensorSTESAWB.forward()
    """

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward function for SAWBZero STE

        Args:
            ctx (torch.autograd.Function): Context object.
            grad_output (torch.Tensor): Gradient to clip

        Returns:
            [torch.Tensor, None,...,None]: Gradients
        """
        input_tensor, _, _, clip_val, _, _ = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input = torch.where(
            input_tensor < -clip_val, torch.zeros_like(grad_input), grad_input
        )
        grad_input = torch.where(
            input_tensor > clip_val, torch.zeros_like(grad_input), grad_input
        )
        return grad_input, None, None, None, None, None, None, None


class SAWBPlusSTE_rc(PerTensorSTESAWB):
    """
    SAWB+: no zero alignment and no gradient clipping
    Incorrect behavior for "dequantize=False" - do not use

    Extends:
        PerTensorSTESAWB: Uses PerTensorSTESAWB.forward()
    """

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward function for SAWB+ STE

        Args:
            ctx (torch.autograd.Function): Context object.
            grad_output (torch.Tensor): Gradient to clip

        Returns:
            [torch.Tensor, None,...,None]: Gradients
        """
        grad_input = grad_output.clone()
        return grad_input, None, None, None, None, None, None, None


class SAWBPlusZeroSTE_rc(PerTensorSTESAWB):
    """SAWB+ with zero alignment (symmetric) and no gradient clipping
    Supported bits: 2, 4, 7, 8
    Other bits requests: runs x.abs().max(), not SAWB
    "dequantize=False" option is functional

    Extends:
        PerTensorSTESAWB: Uses PerTensorSTESAWB.forward()
    """

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward function for SAWB+ Zero STE

        Args:
            ctx (torch.autograd.Function): Context object.
            grad_output (torch.Tensor): Gradient to clip

        Returns:
            [torch.Tensor, None,...,None]: Gradients
        """
        return grad_output, None, None, None, None, None, None, None


# class SAWBPlus16ZeroSTE_rc(torch.autograd.Function):
#     """
#     SAWB with zero alignment but use 16 bins instead of 15, i.e. asymmetric and 4 bit only
#     Uses code=403

#     Extended Range STE doesn't inherit PerTensorSTE_SAWB - too many changes to functions...
#     """

#     @staticmethod
#     def forward(
#         ctx,
#         input_tensor: torch.Tensor,
#         num_bits: torch.Tensor,
#         _clip_valn: torch.Tensor,
#         clip_val: torch.Tensor,
#         dequantize: bool = True,
#         _symmetric: bool = True,
#         _qlevel_lowering: bool = True,
#         _code: bool = True,
#     ) -> torch.Tensor:
#         """
#         Forward function for SAWBPlus16ZeroSTE

#         Args:
#             ctx (torch.autograd.Function): Forward/Backward context object.
#             input_tensor (torch.Tensor): Tensor to be quantized.
#             num_bits (torch.Tensor): Number of bit for quantization.
#             clip_valn (torch.Tensor): Lower clip value bound.
#             clip_val (torch.Tensor): Upper clip value bound.
#             dequantize (bool, optional): Return dequantized or int tensor. Defaults to True.
#             symmetric (bool, optional): Specify if clip values are symmetric. Defaults to False.
#             qlevel_lowering (bool, optional): Specify lowering of quantized levels.
#                 Defaults to True.
#             use_code (bool, optional): Specify using SAWB code. Defaults to False.

#         Returns:
#             torch.Tensor: Dequantized or Quantized output tensor.
#         """
#         assert num_bits == 4, "only implemented for 4bit"

#         # Do not take sawb_params_code n_levels definition for extended range
#         _, clip_val = sawb_params_code(input_tensor, num_bits, 403, perCh=False)
#         n_levels = 2**num_bits - 1

#         scale = clip_val * (8.0 / 7.0 + 1.0) / n_levels
#         zero_point = torch.tensor(0)

#         if len(clip_val.shape) == 0:
#             clip_val = clip_val.unsqueeze(dim=0)

#         output = linear_quantize(input_tensor, scale, zero_point)
#         output = output.clamp(-8, 7)
#         if dequantize:
#             output = linear_dequantize(output, scale, zero_point)
#         else:
#             output = output.to(torch.int8)
#         return output

#     @staticmethod
#     def backward(ctx, grad_output):
#         """
#         Backward function for SAWB+ 16bin Zero STE

#         Args:
#             ctx (torch.autograd.Function): Context object.
#             grad_output (torch.Tensor): Gradient to clip

#         Returns:
#             [torch.Tensor, None,...,None]: Gradients
#         """
#         grad_input = grad_output.clone()
#         return grad_input, None, None, None, None, None, None, None


# class SAWBPlus16ZeroSTE_PTnative(PerTensorSTESAWB_PTnative):
#     """
#     SAWB with zero alignment but use 16 bins instead of 15, i.e. asymmetric and 4 bit only.
#     Uses PyTorch native kernels.

#     Extends:
#         PerTensorSTESAWB_PTnative
#     """

#     @staticmethod
#     def forward(
#         ctx,
#         input_tensor: torch.Tensor,
#         num_bits: torch.Tensor,
#         clip_valn: torch.Tensor,
#         clip_val: torch.Tensor,
#         dequantize: bool = True,
#         symmetric: bool = False,
#         qlevel_lowering: bool = True,
#         use_code: bool = False,
#     ) -> torch.Tensor:
#         """
#         Forward function for SAWBPlus16ZeroSTE_PTnative

#         Args:
#             ctx (torch.autograd.Function): Forward/Backward context object.
#             num_bits (torch.Tensor): Number of bit for quantization.
#             clip_valn (torch.Tensor): Lower clip value bound.
#             clip_val (torch.Tensor): Upper clip value bound.
#             input_tensor (torch.Tensor): Tensor to be quantized.
#             dequantize (bool, optional): Return dequantized or int tensor. Defaults to True.
#             symmetric (bool, optional): Specify if clip values are symmetric. Defaults to False.
#             qlevel_lowering (bool, optional): Specify lowering of quantized levels.
#                 Defaults to True.
#             use_code (bool, optional): Specify using SAWB code. Defaults to False.

#         Returns:
#             torch.Tensor: Dequantized or Quantized output tensor.
#         """
#         (
#             _,
#             _,
#             scale,
#             zero_point,
#             qint_l,
#             qint_h,
#             qint_dtype,
#         ) = SAWBPlus16ZeroSTE_PTnative.calc_qparams(
#             num_bits,
#             clip_valn,
#             clip_val,
#             symmetric,
#             qlevel_lowering,
#             use_code,
#             input_tensor,
#         )
#         output = PerTensorSTESAWB_PTnative.linear_quantization(
#             input_tensor, scale, zero_point, qint_l, qint_h, qint_dtype, dequantize
#         )
#         return output

#     @classmethod
#     def calc_qparams(
#         cls,
#         num_bits: torch.Tensor,
#         clip_valn: torch.Tensor,
#         clip_val: torch.Tensor,
#         symmetric: bool = False,
#         qlevel_lowering: bool = False,
#         use_code: bool = False,
#         input_tensor: torch.Tensor = None,
#     ) -> Tuple[
#         torch.Tensor,
#         torch.Tensor,
#         torch.Tensor,
#         torch.Tensor,
#         int,
#         int,
#     ]:
#         """
#         Compute the scale and zero_point from num_bits and clip values

#         Args:
#             num_bits (torch.Tensor): Number of bit for quantization.
#             clip_valn (torch.Tensor): Lower clip value.
#             clip_val (torch.Tensor): Upper clip value.
#             symmetric (bool, optional): Specify if clip values are symmetric. Defaults to False.
#             qlevel_lowering (bool, optional): Specify lowering of quantized levels.
#                 Defaults to True.
#             use_code (bool, optional): Specify using SAWB code. Defaults to False.
#             input_tensor (torch.Tensor): Tensor to be quantized. Defaults to None.

#         Returns:
#             [torch.Tensor, torch.Tensor, torch.Tensor
#             torch.Tensor, torch.Tensor, torch.Tensor, torch.dtype]:
#                 Quantized parameters for PTnative kernels
#         """

#         # Do not take sawb_params_code n_levels definition for extended range
#         _, clip_val = sawb_params_code(input_tensor, num_bits, 403, perCh=False)
#         n_levels = 2**num_bits - 1
#         scale = clip_val * (8.0 / 7.0 + 1.0) / n_levels
#         zero_point = torch.tensor(0)
#         qint_l, qint_h, qint_dtype = -8, 7, torch.qint8
#         return n_levels, clip_val, scale, zero_point, qint_l, qint_h, qint_dtype


class SAWBPlusZeroPerChSTE_rc(PerChannelSTESAWB):
    """
    per-channel SAWB with zero alignment, ca,8n use 15 or 16 bins, i.e. [-7,7] or [-7]
    """

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward function for SAWBPlusZeroPerChSTE

        Args:
            ctx (torch.autograd.Function): Context object.
            grad_output (torch.Tensor): Gradient to clip

        Returns:
            [torch.Tensor, None,...,None]: Gradients
        """
        grad_input = grad_output.clone()
        return grad_input, None, None, None, None, None, None, None
