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
PACT+ Symmetric Quantizer
"""

# Third Party
import torch

# Local
from fms_mo.quant_refactor.base_quant import Qscheme, Quantizer
from fms_mo.quant_refactor.per_tensor_ste import PerTensorSTE, PerTensorSTE_PTnative

clip_valn_default = torch.tensor([-8.0])
clip_val_default = torch.tensor([8.0])
qscheme_per_tensor = Qscheme(
    unit="perT",
    symmetric=True,
    Nch=None,
    Ngrp=None,
    single_sided=False,
    qlevel_lowering=False,
)


class PACTplusSym_rc(Quantizer):
    """
    Two-sided symmetric PACT+
    PACTplusSym can be used to quantize both weights and activations

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
        extend_act_range: bool = False,
        **kwargs,
    ):
        """
        Init PACT+Sym quantizer

        Args:
            num_bits (torch.Tensor): Number of bits for quantization.
            init_clip_valn (torch.Tensor, optional): Lower clip value bound. Defaults to -8.0.
            init_clip_val (torch.Tensor, optional): Upper clip value bound. Defaults to 8.0.
            qscheme (Qscheme, optional): Quantization scheme.
                Defaults to Qscheme( unit="perT", symmetric=False, Nch=None, Ngrp=None,
                                       single_sided=False, qlevel_lowering=False, ).
            dequantize (bool, optional): Return dequantized or int tensor. Defaults to True.
            extend_act_range (bool, optional): Use full quantization range. Defaults to False.
            kwargs.use_PT_native_Qfunc (bool, optional): Use native PT quantizer.
                Defaults to False.
        """
        super().__init__(
            num_bits,
            dequantize,
            qscheme=qscheme,
            use_PT_native_Qfunc=kwargs.get("use_PT_native_Qfunc", False),
        )

        with torch.no_grad():
            self.clip_valn.data *= init_clip_valn
            self.clip_val.data *= init_clip_val

        self.extend_act_range = extend_act_range
        self.set_quantizer()

    def set_quantizer(self):
        """
        Set quantizer STE based on current member variables
        """
        # PTnative overrides all other options except extended range (custom autograd func)
        if self.use_PT_native_Qfunc:
            if self.extend_act_range:
                self.quantizer = PACTplusExtendRangeSTE_PTnative
            else:
                self.quantizer = PerTensorSTE_PTnative
        else:
            if self.extend_act_range:
                self.quantizer = PACTplusExtendRangeSTE_rc
                self.quantizer_name = "PACT+extend"
            else:
                self.quantizer = PACTplusSymSTE_rc
                self.quantizer_name = "PACT+sym"

    def set_extend_act_range(self, extend_act_range: bool):
        """
        Setter function for extend_act_range

        Args:
            extend_act_range (bool): Use full quantization range.
        """
        self.extend_act_range = extend_act_range


class PACTplusSymSTE_rc(PerTensorSTE):
    """
    Symmetric 2-sided PACT+

    Extends:
        PerTensorSTE: Uses PerTensorSTE.forward()
    """

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward function for PACTSym

        Args:
            ctx (torch.autograd.Function): Context object.
            grad_output (torch.Tensor): Gradient to clip

        Returns:
            [torch.Tensor, torch.Tensor, None,...,None]: Gradients
        """
        input_tensor, n_levels, _, clip_val, scale, _ = ctx.saved_tensors
        z = (input_tensor + clip_val) / scale
        delz = 2 * (torch.round(z) - z) / n_levels

        grad_input = grad_output.clone()
        grad_input = torch.where(
            input_tensor <= (-clip_val), torch.zeros_like(grad_input), grad_input
        )
        grad_input = torch.where(
            input_tensor >= clip_val, torch.zeros_like(grad_input), grad_input
        )

        grad_alpha = torch.ones_like(grad_output) * delz
        grad_alpha = torch.where(
            input_tensor <= -clip_val, -torch.ones_like(grad_input), grad_alpha
        )
        grad_alpha = torch.where(
            input_tensor >= clip_val, torch.ones_like(grad_input), grad_alpha
        )
        grad_alpha *= grad_output

        grad_alpha = grad_alpha.sum().expand_as(clip_val)

        return grad_input, grad_alpha, None, None, None, None, None


class PACTplusExtendRangeSTE_rc(torch.autograd.Function):
    """
    2-sided PACT+ using a single clip

    Negative clip (clip_valn) is derived from positive clip (clip_val),
    it is not an separate parameter and has no gradient
    This quantizer is equivalent to one WITHOUT zero_point, and it is
    therefore suitable for BMM

    Extended range [-2^(b-1)/(2^(b-1)-1) * clip_val, clip_val] is used for
    quantization, instead of the symmetric [-clip_val, clip_val]
    Example:
        b = 8 --> [-128/127 * clip_val, clip_val]
        b = 4 --> [-8/7 * clip_val, clip_val]

    Zero alignment (FP=0 --> INT=0 --> FP=0) is guaranteed
    Out Of Range input grad clipping is always enabled
    No in_place functionalities
    No Automatic Mixed Precision (AMP) functionalities
    Dequantization is functional
    """

    @staticmethod
    def forward(
        ctx,
        input_tensor: torch.Tensor,
        num_bits: torch.Tensor,
        clip_valn: torch.Tensor = clip_valn_default,
        clip_val: torch.Tensor = clip_val_default,
        dequantize: bool = True,
        _symmetric: bool = True,
        _qlevel_lowering: bool = False,
    ) -> torch.Tensor:
        """
        Forward function for PACT+Sym Extended Range

        Args:
            ctx (torch.autograd.Function): Forward/Backward context object.
            input_tensor (torch.Tensor): Tensor to be quantized.
            num_bits (torch.Tensor): Number of bit for quantization.
            clip_valn (torch.Tensor): Lower clip value bound.
            clip_val (torch.Tensor): Upper clip value bound.
            dequantize (bool, optional): Return dequantized or int tensor. Defaults to True.
            _symmetric (bool, optional): Specify if clip values are symmetric. Defaults to False.
            _qlevel_lowering (bool, optional): Specify lowering of quantized levels.
                Defaults to True.

        Returns:
            torch.Tensor: Dequantized or Quantized output tensor.
        """
        n_half = 2 ** (num_bits - 1)  # levels in half range: 8b: 128; 4b: 8; 2b: 2
        clip_valn = -clip_val * n_half / (n_half - 1)  # 8b: -128/127 * clip_val
        scale = clip_val / (n_half - 1)
        zero_point = torch.tensor(0)

        ctx.save_for_backward(
            input_tensor, num_bits, clip_valn, clip_val, scale, zero_point, n_half
        )

        # quantize to range [-2^(b-1), 2^(b-1)-1]
        output = torch.round(input_tensor / scale)
        output = output.clamp(-n_half, n_half - 1)
        if dequantize:
            output = output * scale
        else:
            output = output.to(torch.int8)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward function for PACTSym

        Args:
            ctx (torch.autograd.Function): Context object.
            grad_output (torch.Tensor): Gradient to clip

        Returns:
            [torch.Tensor, torch.Tensor, None,...,None]: Gradients
        """
        input_tensor, _, clip_valn, clip_val, scale, _, n_half = ctx.saved_tensors

        grad_input = grad_output.clone()
        grad_input = torch.where(
            input_tensor <= clip_valn, torch.zeros_like(grad_input), grad_input
        )
        grad_input = torch.where(
            input_tensor >= clip_val, torch.zeros_like(grad_input), grad_input
        )

        z = (input_tensor - clip_valn) / scale
        grad_alpha = (torch.round(z) - z) / (n_half - 1)
        grad_alpha = torch.where(
            input_tensor <= clip_valn,
            clip_valn / clip_val * torch.ones_like(grad_input),
            grad_alpha,
        )
        grad_alpha = torch.where(
            input_tensor >= clip_val, torch.ones_like(grad_input), grad_alpha
        )
        grad_alpha *= grad_output

        grad_alpha = grad_alpha.sum().expand_as(clip_val)

        return grad_input, grad_alpha, None, None, None, None, None


class PACTplusExtendRangeSTE_PTnative(torch.autograd.Function):
    """
    2-sided PACT+ using Extended Range w/ PTnative kernels
    """

    @staticmethod
    def forward(
        ctx,
        input_tensor: torch.Tensor,
        num_bits: torch.Tensor,
        _clip_valn: torch.Tensor = clip_valn_default,
        clip_val: torch.Tensor = clip_val_default,
        dequantize: bool = True,
        _symmetric: bool = True,
        _qlevel_lowering: bool = False,
    ) -> torch.Tensor:
        """
        Forward function for PACT+Sym Extended Range w/ PTnative

        Args:
            ctx (torch.autograd.Function): Forward/Backward context object.
            input_tensor (torch.Tensor): Tensor to be quantized.
            num_bits (torch.Tensor): Number of bit for quantization.
            _clip_valn (torch.Tensor): Lower clip value bound.
            clip_val (torch.Tensor): Upper clip value bound.
            dequantize (bool, optional): Return dequantized or int tensor. Defaults to True.
            _symmetric (bool, optional): Specify if clip values are symmetric. Defaults to False.
            _qlevel_lowering (bool, optional): Specify lowering of quantized levels.
                Defaults to True.

        Returns:
            torch.Tensor: Dequantized or Quantized output tensor.
        """
        n_half = 2 ** (num_bits - 1)  # levels in half range: 8b: 128; 4b: 8; 2b: 2
        _clip_valn = -clip_val * n_half / (n_half - 1)  # 8b: -128/127 * clip_val
        scale = clip_val / (n_half - 1)
        zero_point = torch.tensor(0)
        qint_l, qint_h = -n_half, n_half - 1

        if dequantize:
            output = torch.fake_quantize_per_tensor_affine(
                input_tensor.float(),
                scale.float(),
                zero_point.float(),
                quant_min=qint_l,
                quant_max=qint_h,
            ).to(input_tensor.dtype)
        else:
            qint_dtype = torch.qint8
            output = (
                torch.quantize_per_tensor(input_tensor, scale, zero_point, qint_dtype)
                .int_repr()
                .clamp(qint_l, qint_h)
            )
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward function for PACTSym

        Args:
            ctx (torch.autograd.Function): Context object.
            grad_output (torch.Tensor): Gradient to clip

        Returns:
            [torch.Tensor, None,...,None]: Gradients
        """
        return grad_output, None, None, None, None, None, None
