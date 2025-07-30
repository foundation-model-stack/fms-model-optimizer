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
PACT2Symmetric Quantizer
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


class PACT2Sym_rc(Quantizer):
    """
    Two-sided PACT with symmetric clip values

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
        **kwargs,
    ):
        """
        Init PACT2Sym quantizer

        Args:
            num_bits (torch.Tensor): Number of bits for quantization.
            init_clip_valn (torch.Tensor, optional): Lower clip value bound. Defaults to -8.0.
            init_clip_val (torch.Tensor, optional): Upper clip value bound. Defaults to 8.0.
            qscheme (Qscheme, optional): Quantization scheme.
                Defaults to Qscheme( unit="perT", symmetric=False, Nch=None, Ngrp=None,
                                       single_sided=False, qlevel_lowering=False, ).
            dequantize (bool, optional): Return dequantized or int tensor. Defaults to True.
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

        self.set_quantizer()

    def set_quantizer(self):
        """
        Set quantizer STE based on current member variables
        """
        if self.use_PT_native_Qfunc:
            self.quantizer = PerTensorSTE_PTnative
        else:
            self.quantizer = PACT2Sym_STE_rc


class PACT2Sym_STE_rc(PerTensorSTE):
    """
    Symmetric with zero in the center. For example, 4bit -- > [-7, 7] with FP0 align to INT0

    Extends:
        PerTensorSTE: Uses PerTensorSTE.forward()
    """

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward function for PACT2Sym

        Args:
            ctx (torch.autograd.Function): Context object.
            grad_output (torch.Tensor): Gradient to clip

        Returns:
            [torch.Tensor, torch.Tensor, None,...,None]: Gradients
        """
        input_tensor, _, _, clip_val, _, _ = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input = torch.where(
            input_tensor <= -clip_val, torch.zeros_like(grad_input), grad_input
        )
        grad_input = torch.where(
            input_tensor >= clip_val, torch.zeros_like(grad_input), grad_input
        )

        grad_alpha = grad_output.clone()
        grad_alpha = torch.where(
            torch.logical_and(input_tensor < clip_val, input_tensor > -clip_val),
            torch.zeros_like(grad_alpha),
            grad_alpha,
        )
        grad_alpha = grad_alpha.sum().expand_as(clip_val)

        return grad_input, grad_alpha, None, None, None, None, None
