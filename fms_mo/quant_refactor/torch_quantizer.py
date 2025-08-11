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
Torch Quantizer that can mimic FMS quantizer functionality.

Used for testing as the reference result.
Tests implement FMS functional in set_base_options() and set_other_options().
"""

# Standard
import logging

# Third Party
import torch

# Local
from fms_mo.quant_refactor.base_quant import Qscheme
from fms_mo.quant_refactor.sawb_utils import sawb_params, sawb_params_code

logger = logging.getLogger(__name__)

qscheme_per_tensor = Qscheme(
    unit="perT",
    symmetric=False,
    Nch=None,
    Ngrp=None,
    single_sided=False,
    qlevel_lowering=False,
)


# Create a Torch Quanitizer class that returns  float + int quantized tensors
class TorchQuantizer(torch.nn.Module):
    """
    Torch Quantizer Class
    """

    def __init__(
        self,
        num_bits: torch.Tensor,
        clip_low: torch.Tensor,
        clip_high: torch.Tensor,
        dequantize: bool = True,
        qscheme: Qscheme = qscheme_per_tensor,
    ) -> None:
        """
        Init TorchQuantizer Class

        Args:
            num_bits (torch.Tensor): Number of bits for quantization.
            clip_low (torch.Tensor): Lower clip value bound.
            clip_high (torch.Tensor): Upper clip value bound.
            dequantize (bool, optional): Return dequantized or int tensor. Defaults to True.
            qscheme (Qscheme, optional): Quantization scheme.
                Defaults to Qscheme( unit="perT", symmetric=True, Nch=None, Ngrp=None,
                                       single_sided=False, qlevel_lowering=False, ).
        """
        super().__init__()
        self.num_bits = num_bits
        self.num_bits_int = (
            num_bits.item() if isinstance(num_bits, torch.Tensor) else num_bits
        )
        # turn clips into tensors (from python float)
        self.clip_low = (
            torch.Tensor([clip_low])
            if not isinstance(clip_low, torch.Tensor)
            else clip_low
        )
        self.clip_high = (
            torch.Tensor([clip_high])
            if not isinstance(clip_high, torch.Tensor)
            else clip_high
        )
        self.symmetric_zp0 = False
        self.qscheme = qscheme
        self.set_quant_bounds()
        self.dequantize = dequantize
        # if self.qscheme == torch.per_channel_affine: # TODO: How to use this?
        #     # (scales, zero_points, axis)
        #     self.ch_affine_parms = (scales, zero_points, axis)
        # else:
        #     self.ch_affine_parms = None
        # available dtypes https://pytorch.org/docs/stable/tensors.html
        self.dtype_dict = {
            (32, True): torch.qint32,
            (8, True): torch.qint8,
            (8, False): torch.quint8,
            (4, True): torch.qint8,
            (4, False): torch.quint8,
        }

    def get_setup(self):
        """
        Get TorchQuantizer quantization for debugging

        Returns:
            [int, float, float, int, float, int, int, int, qscheme]: Quantization parameters
        """
        return (
            self.num_bits_int,
            self.clip_low,
            self.clip_high,
            self.n_levels.item(),
            self.scale,
            self.zero_point,
            self.quant_min,
            self.quant_max,
            self.qscheme,
        )

    def set_quant_bounds(self):
        """
        Set quantization parameters based on current member variables
        """
        self.isQscheme = isinstance(self.qscheme, Qscheme)
        self.is_single_sided = self.isQscheme and self.qscheme.single_sided
        self.is_symmetric = self.isQscheme and self.qscheme.symmetric

        self.symmetric_nlevel = (
            1 if (self.qscheme.qlevel_lowering) else 0
        )  # lower qlevels by 1 and add 1 to quant_min
        self.n_levels = 2**self.num_bits - 1 - self.symmetric_nlevel
        self.scale = (self.clip_high - self.clip_low) / (self.n_levels)
        # this "ZP" will map the float value we choose (clip_low in this case) to the 0 bin
        self.zero_point = (
            torch.zeros(self.scale.shape, dtype=torch.int)
            if (self.is_symmetric)
            else torch.round(-self.clip_low / self.scale).to(torch.int)
        )

        self.set_quant_range()

    def set_quant_range(self):
        """
        Set quantization integer range based on member variables
        """
        if self.is_symmetric and torch.sum(self.zero_point) == 0:
            # Either [-8,7];[-128,127] for non-symmetric or [-7,7];[-127,127] for qlevel_lowering
            self.quant_min, self.quant_max = (
                -(2 ** (self.num_bits - 1)) + self.symmetric_nlevel,
                2 ** (self.num_bits - 1) - 1,
            )
        else:  # single_sided or zero_point != 0
            self.quant_min, self.quant_max = 0, self.n_levels  # eg (0, 255) or (0,15)

    def set_qscheme(self, qscheme: Qscheme):
        """
        Setter function for qscheme

        Args:
            qscheme (Qscheme): Quantization scheme.
        """
        self.qscheme = qscheme
        self.set_quant_bounds()  # reset quant_min,quant_max

    def set_single_sided(self, single_sided: bool):
        """
        Setter function for qscheme.single_sided

        Args:
            single_sided (bool): Specify if tensor is single-sided.
        """
        self.qscheme.single_sided = single_sided
        self.set_quant_bounds()

    def set_qlevel_lowering(self, qlevel_lowering: bool):
        """
        Setter function for qscheme.qlevel_lowering

        Args:
            qlevel_lowering (bool): Specify lowering of quantized levels. Defaults to True.
        """
        self.qscheme.qlevel_lowering = qlevel_lowering
        self.set_quant_bounds()

    # SAWB STEs has many definitions for
    def set_sawb_clip(self, tensor: torch.Tensor, qlevel_lowering: bool = False):
        """
        Setter for clip values using SAWB

        Args:
            tensor (torch.Tensor): Tensor to be quantized.
            qlevel_lowering (bool, optional): Specify lowering of quantized levels.
                Defaults to False.
        """
        _, self.clip_high = sawb_params(tensor, self.num_bits_int, qlevel_lowering)
        self.clip_low = -self.clip_high

    def set_sawb_clip_code(self, tensor: torch.Tensor, code=None, perCh=False):
        """
        Setter for clip values using SAWB codes

        Args:
            tensor (torch.Tensor): Tensor to be quantized.
            code (int, optional): Specify using SAWB code. Defaults to None.
            perCh (bool, optional): Specify if using perCh quantization. Defaults to False.
        """
        bits2code = {2: 103, 4: 403, 7: 703, 8: 803}
        if self.num_bits_int in bits2code:
            if code is None:  # only change code if not provided
                code = bits2code[self.num_bits_int]
            _, self.clip_high = sawb_params_code(tensor, self.num_bits, code, perCh)
        else:
            self.clip_high = tensor.abs().max()
        self.clip_low = -self.clip_high

    def squash_tensor_sawb(self, tensor: torch.Tensor):
        """
        Legacy tensors squash tensor [-clip,clip] -> [-1,1] -> [-.5,.5] -> [0,1]

        Args:
            tensor (torch.Tensor): Tensor to be squashed.

        Returns:
            torch.Tensor: Squashed tensor.
        """
        return tensor.div(self.clip_high).clamp(-1, 1).mul(0.5).add(0.5)

    def unsquash_tensor_sawb(self, tensor):
        """
        Legacy tensors unsquashed [0,1] -> [0,2] -> [-1,1] -> [-clip,clip]

        Args:
            tensor (torch.Tensor): Tensor to be unsquashed.

        Returns:
            torch.Tensor: Unsquashed tensor.
        """
        return tensor.mul(2.0).sub(1.0).mul(self.clip_high)

    def set_shift_sawb(self, shift_sawb: int):
        """
        Setter function for sawb_shift.  Some STEs "shift" quantized int range.

        Args:
            shift_sawb (int): Shift tensor by shift_sawb.
        """
        self.shift_sawb = shift_sawb

    # Shift qtensor to be symmetric about zero
    def shift_qtensor_sawb(self, tensor: torch.Tensor):
        """
        Shift a quantized int tensor by shift_sawb

        Args:
            tensor (torch.Tensor): Tensor to be quantized.

        Returns:
            torch.Tensor: Shifted quantized int tensor
        """
        return (tensor - self.shift_sawb).to(torch.int8)

    def get_torch_dtype(self):
        """
        Get torch dtype based on num_bits, signed

        Returns:
            torch.dtype: Tensor dtype
        """
        if self.is_single_sided:
            signed = False
        else:
            signed = (torch.sum(self.zero_point) == 0).item()
        return self.dtype_dict.get(
            (self.num_bits_int, signed)
        )  # NOTE .item() won't work for perCh

    def forward(self, tensor: torch.Tensor):
        """
        TorchQuantizer forward() function w/ PT kernels.

        Args:
            tensor (torch.Tensor): Tensor to be quantized

        Raises:
            RuntimeError: Unknown dtype based on num_bits and zero_point

        Returns:
            torch.Tensor: Quantized or dequantized tensor.
        """

        if self.dequantize:
            if self.qscheme.Nch:  # Per Channel
                output = torch.fake_quantize_per_channel_affine(
                    tensor,
                    self.scale.float(),
                    self.zero_point.float(),
                    self.qscheme.axis,
                    self.quant_min,
                    self.quant_max,
                )
            elif self.qscheme.Ngrp:  # Per Group
                pass
            else:  # Per Tensor
                output = torch.fake_quantize_per_tensor_affine(
                    tensor,
                    self.scale,
                    self.zero_point,
                    self.quant_min,
                    self.quant_max,
                )
        else:
            dtype = self.get_torch_dtype()
            if dtype:
                if self.qscheme.q_unit == "perCh":
                    output = torch.quantize_per_channel(
                        tensor,
                        self.scale,
                        self.zero_point,
                        self.qscheme.axis,
                        dtype,
                    )
                elif self.qscheme.q_unit == "perGrp":
                    raise RuntimeError(
                        "TorchQuantizer forward not implemented for perGrp"
                    )
                else:  # Per Tensor
                    output = torch.quantize_per_tensor(
                        tensor,
                        self.scale,
                        self.zero_point,
                        dtype,
                    )
                # Clamp required if storing int4 into int8 tensor (no PT support for int4)
                output = output.int_repr().clamp(self.quant_min, self.quant_max)
            else:
                raise RuntimeError(
                    f"num_bits {self.num_bits} and sign {(self.zero_point==0).item()}"
                    "combination results in unavailable dtype."
                )

        return output
