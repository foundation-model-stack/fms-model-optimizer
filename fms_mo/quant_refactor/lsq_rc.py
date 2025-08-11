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
LSQ Quantizer

PTnative STEs not possible for LSQ since we need to compute residual before dequantize.
PT doesn't have a "dequantize" function.
      output = linear_quantize(..)
      rounded = output.round()
      residual = rounded - output
"""

# Standard
import math

# Third Party
import torch

# Local
from fms_mo.quant_refactor.base_quant import Qscheme, Quantizer
from fms_mo.quant_refactor.linear_utils import (
    asymmetric_linear_quantization_params,
    linear_dequantize,
    linear_quantize_LSQresidual,
    qint_bounds,
)
from fms_mo.quant_refactor.per_tensor_ste import PerTensorSTE

clip_valn_default = torch.tensor(-8.0)
clip_val_default = torch.tensor(8.0)
qscheme_per_tensor = Qscheme(
    unit="perT",
    symmetric=False,
    Nch=None,
    Ngrp=None,
    single_sided=True,
    qlevel_lowering=False,
)


class LSQQuantization_rc(Quantizer):
    """
    LSQ Quantizer

    Extends:
        Quantizer
    """

    def __init__(
        self,
        num_bits: torch.Tensor,
        init_clip_valn: torch.Tensor = clip_valn_default,
        init_clip_val: torch.Tensor = clip_val_default,
        qscheme=qscheme_per_tensor,
        dequantize: bool = True,
        **kwargs,
    ):
        """
        Init LSQ Quantizer

        Args:
            num_bits (torch.Tensor): Number of bits for quantization.
            init_clip_valn (torch.Tensor, optional): Lower clip value bound. Defaults to -8.0.
            init_clip_val (torch.Tensor, optional): Upper clip value bound. Defaults to 8.0.
            qscheme (Qscheme, optional): Quantization scheme.
                Defaults to Qscheme( unit="perT", symmetric=False, Nch=None, Ngrp=None,
                                       single_sided=True, qlevel_lowering=False, ).
            dequantize (bool, optional): Return dequantized or int tensor. Defaults to True.
            kwargs.use_PT_native_Qfunc (bool): Use native PT quantizer.  Defaults to False.
        """
        super().__init__(
            num_bits,
            dequantize,
            qscheme=qscheme,
            use_PT_native_Qfunc=kwargs.get("use_PT_native_Qfunc", False),
        )
        self.set_quantizer()

        with torch.no_grad():
            self.clip_valn.data *= init_clip_valn  # always 0.0
            self.clip_val.data *= init_clip_val

    def set_quantizer(self):
        """
        Set quantizer STE - use LSQQuantizationSTE_rc
        """
        self.quantizer = LSQQuantizationSTE_rc


class LSQQuantizationSTE_rc(PerTensorSTE):
    """
    1-sided LSQ quantization STE

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
        symmetric: bool = False,
        qlevel_lowering: bool = False,
    ) -> torch.Tensor:
        """
        LSQ Quantization STE forward() function.

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

        clip_valn, clip_val = (
            clip_valn.to(input_tensor.dtype),
            clip_val.to(input_tensor.dtype),
        )

        n_levels, scale, zero_point = asymmetric_linear_quantization_params(
            num_bits, clip_valn.data, clip_val.data, qlevel_lowering=qlevel_lowering
        )
        qint_min, qint_max, int_dtype = qint_bounds(
            num_bits, zero_point, symmetric, qlevel_lowering
        )

        output, residual = linear_quantize_LSQresidual(input_tensor, scale, zero_point)
        output = output.clamp(qint_min, qint_max)

        with torch.no_grad():
            residual /= n_levels
        ctx.save_for_backward(input_tensor, clip_val, residual)

        if dequantize:
            output = linear_dequantize(output, scale, zero_point)
        else:
            output = output.to(int_dtype)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward function for LSQ Quantization STE

        Args:
            ctx (torch.autograd.Function): Context object.
            grad_output (torch.Tensor): Gradient to clip

        Returns:
            [torch.Tensor, torch.Tensor, None,...,None]: Gradients
        """
        input_tensor, clip_val, residual = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input = torch.where(
            input_tensor < 0, torch.zeros_like(grad_input), grad_input
        )
        grad_input = torch.where(
            input_tensor > clip_val, torch.zeros_like(grad_input), grad_input
        )

        grad_alpha = grad_output.clone()
        grad_alpha = torch.where(
            input_tensor < clip_val, grad_alpha * residual, grad_alpha
        )
        grad_alpha = grad_alpha.sum().expand_as(clip_val)
        ndim = float(sum(list(grad_output.shape)))
        grad_scale = math.sqrt(1 / ndim / 3.0)  # 3 is for 2bit, should be n
        return grad_input, grad_alpha * grad_scale, None, None, None, None, None


class LSQPlus_rc(Quantizer):
    """
    LSQ+ Quantizater

    Extends:
        Quantizer
    """

    def __init__(
        self,
        num_bits: torch.Tensor,
        init_clip_valn: torch.Tensor = clip_valn_default,
        init_clip_val: torch.Tensor = clip_val_default,
        qscheme=qscheme_per_tensor,
        dequantize: bool = True,
        **kwargs,
    ):
        """
        Init LSQ+ Quantizer

        Args:
            num_bits (torch.Tensor): Number of bits for quantization.
            init_clip_valn (torch.Tensor, optional): Lower clip value bound. Defaults to -8.0.
            init_clip_val (torch.Tensor, optional): Upper clip value bound. Defaults to 8.0.
            qscheme (Qscheme, optional): Quantization scheme.
                Defaults to Qscheme( unit="perT", symmetric=False, Nch= None, Ngrp= None,
                                       single_sided=True, qlevel_lowering=False, ).
            dequantize (bool, optional): Return dequantized or int tensor. Defaults to True.
            kwargs.use_PT_native_Qfunc (bool): Use native PT quantizer.  Defaults to False.
        """
        super().__init__(
            num_bits,
            dequantize,
            qscheme=qscheme,
            use_PT_native_Qfunc=kwargs.get("use_PT_native_Qfunc", True),
        )
        self.set_quantizer()

        with torch.no_grad():
            self.clip_valn.data *= init_clip_valn
            self.clip_val.data *= init_clip_val

        # clip_val, clip_valn are not training paramaters
        # scale, zero_point are trained instead
        self.scale = torch.nn.Parameter(self.clip_val)
        self.zero_point = torch.nn.Parameter(self.clip_valn)
        self.counter = torch.nn.Parameter(torch.zeros_like(self.clip_val))

    def set_quantizer(self):
        """
        Set quantizer STE - use LSQPlus_func_rc
        """
        self.quantizer = LSQPlus_func_rc

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        LSQ+ Quantizer forward function
        Sets scale, zero_point on first call, but uses cached values later

        Args:
            input_tensor (torch.Tensor): Tensor to be quantized

        Returns:
            torch.Tensor: Dequantized or Quantized output tensor.
        """
        with torch.no_grad():
            if self.counter == 0:
                xmin, xmax = input_tensor.min(), input_tensor.max()
                self.scale.data = (
                    torch.ones_like(self.clip_val)
                    * (xmax - xmin)
                    / (2.0**self.num_bits - 1)
                )
                self.zero_point.data = torch.ones_like(self.zero_point) * (
                    xmin + 2 ** (self.num_bits - 1) * self.scale
                )
                self.zero_point.data = self.zero_point.data.round()
                self.counter += 1

        output = self.quantizer.apply(
            input_tensor,
            self.num_bits,
            self.scale,
            self.zero_point,
            self.dequantize,
            self.qscheme.symmetric,
            self.qscheme.qlevel_lowering,
        )
        return output


class LSQPlus_func_rc(torch.autograd.Function):
    """2-side LSQ+ from CVPR workshop paper"""

    @staticmethod
    def forward(
        ctx,
        input_tensor: torch.Tensor,
        num_bits: torch.Tensor,
        scale: torch.Tensor,
        zero_point: torch.Tensor,  # clip vals are not passed to forward
        dequantize: bool = True,
        _symmetric: bool = False,
        _qlevel_lowering: bool = False,
    ) -> torch.Tensor:
        """
        LSQ+ Quantization STE forward() function.

        Args:
            ctx (torch.autograd.Function): Forward/Backward context object.
            input_tensor (torch.Tensor): Tensor to be quantized.
            num_bits (torch.Tensor): Number of bit for quantization.
            scale (torch.Tensor): Dequantized range of a quantized integer bin.
            zero_point (torch.Tensor): Quantized integer bin mapping to fp 0.0.
            dequantize (bool, optional): Return dequantized or int tensor. Defaults to True.
            _symmetric (bool, optional): Specify if clip values are symmetric. Defaults to False.
            _qlevel_lowering (bool, optional): Specify lowering of quantized levels.
                Defaults to True.

        Returns:
            torch.Tensor: Dequantized or Quantized output tensor.
        """

        ctx.save_for_backward(input_tensor, scale.clone(), zero_point.clone())
        n_half = 2 ** (num_bits - 1)
        clip_valn, clip_val = -n_half, n_half - 1  # [-128,127], [-8,7]

        # Manual linear_quantize + clamp to quantized clip_vals + round
        output = (input_tensor - zero_point) / scale
        output = output.clamp(clip_valn, clip_val)
        rounded = output.round()
        ctx.residual = rounded - output

        with torch.no_grad():
            # Save clip vals
            ctx.clip_valn = clip_valn * scale + zero_point
            ctx.clip_val = clip_val * scale + zero_point
            ctx.num_bits = num_bits

        if dequantize:
            output = rounded * scale + zero_point
        else:
            output = rounded.to(torch.int8)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward function for LSQ+ Quantization STE

        Args:
            ctx (torch.autograd.Function): Context object.
            grad_output (torch.Tensor): Gradient to clip

        Returns:
            [torch.Tensor, torch.Tensor, torch.Tensor, None,...,None]: Gradients
        """
        input_tensor, clip_vals, clip_valb = ctx.saved_tensors
        p = 2.0 ** (ctx.num_bits - 1) - 1
        n = -(2.0 ** (ctx.num_bits - 1))
        grad_input = grad_output.clone()
        grad_input = torch.where(
            input_tensor <= ctx.clip_valn, torch.zeros_like(grad_input), grad_input
        )
        grad_input = torch.where(
            input_tensor >= ctx.clip_val, torch.zeros_like(grad_input), grad_input
        )

        grad_s = grad_output.clone()
        grad_s = torch.where(
            ((input_tensor < ctx.clip_val) & (input_tensor > ctx.clip_valn)),
            grad_s * ctx.residual,
            grad_s,
        )
        grad_s = torch.where(input_tensor <= ctx.clip_valn, n * grad_s, grad_s)
        grad_s = torch.where(input_tensor >= ctx.clip_val, p * grad_s, grad_s)
        grad_s = grad_s.sum().expand_as(clip_vals)

        grad_beta = grad_output.clone()
        grad_beta = torch.where(
            ((input_tensor < ctx.clip_val) & (input_tensor > ctx.clip_valn)),
            torch.zeros_like(grad_beta),
            grad_beta,
        )
        grad_beta = grad_beta.sum().expand_as(clip_valb)

        return grad_input, grad_s, grad_beta, None, None, None, None
