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

"""Util function transformers_prepare_input() is borrowed from huggingface transformers/trainer.py
Trainer class method _prepare_input().
see https://github.com/huggingface/transformers/blob/v4.46.3/src/transformers/trainer.py#L3497

Class/function MSEObserver, ObserverBase, fake_quantize_per_channel_affine,
fake_quantize_per_tensor_affine, _transform_to_ch_axis, CyclicTempDecay, LinearTempDecay,
AdaRoundSTE, AdaRoundQuantizerare are modified from BRECQ's repo: https://github.com/yhhhli/BRECQ

"""

# pylint: disable=too-many-return-statements

# Standard
from collections.abc import Mapping
from typing import Any, Union
import logging
import math
import os
import random

# Third Party
from packaging.version import Version
import numpy as np
import torch
import torch.fx
import torch.nn as nn  # pylint: disable=consider-using-from-import
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def get_activation_quantizer(
    qa_mode="PACT",
    nbits=32,
    clip_val=None,
    clip_valn=None,
    non_neg=False,
    align_zero=True,
    extend_act_range=False,
    use_swcap=False,
    use_PT_native_Qfunc=False,
    use_subnormal=False,
):
    """Return a quantizer for activation quantization
    Regular quantizers:
    - pact, pact2 (non_neg, cgpact, pact+)
    - pactsym/pactsym+
    - max, minmax, maxsym
    - lsq+, lsq (inactive), qil, qsilu, dorefa, fix
    - brecq (PTQ)
    SWCAP quantizers (do not dequantize):
    - pact/pact+/pactsym
    - sawb/sawb+
    - max
    """

    if not use_swcap:
        QPACTLUT = {
            "pact_uni": PACT,
            "pact_bi": PACT2,
            "cgpact_uni": PACT,
            "cgpact_bi": PACT2,
            "pact+_uni": PACT,
            "pact+_bi": PACT2,
        }
        if "pact" in qa_mode and "sym" not in qa_mode:
            keyQact = qa_mode + "_uni" if non_neg else qa_mode + "_bi"
            cggrad = "cgpact" in qa_mode
            pact_plus = "pact+" in qa_mode
            act_quantizer = (
                QPACTLUT[keyQact](
                    nbits,
                    init_clip_val=clip_val,
                    init_clip_valn=clip_valn,
                    dequantize=True,
                    inplace=False,
                    cggrad=cggrad,
                    pact_plus=pact_plus,
                )
                if non_neg
                else QPACTLUT[keyQact](
                    nbits,
                    init_clip_val=clip_val,
                    init_clip_valn=clip_valn,
                    dequantize=True,
                    inplace=False,
                    cggrad=cggrad,
                    pact_plus=pact_plus,
                    align_zero=align_zero,
                    # only implemented in pact2ste and pactplus2ste
                    use_PT_native_Qfunc=use_PT_native_Qfunc,
                )
            )
        elif qa_mode == "lsq+":
            act_quantizer = LSQPlus(
                nbits,
                init_clip_vals=clip_val,
                init_clip_valb=clip_valn,
                dequantize=True,
                inplace=False,
            )
        elif qa_mode == "qsilu":
            act_quantizer = QSILU(
                nbits,
                init_clip_val=clip_val,
                init_clip_valn=-0.28746,
                dequantize=True,
                inplace=False,
            )
        elif qa_mode == "dorefa":
            act_quantizer = dorefa_quantize_activation
        elif (
            qa_mode == "max"
        ):  # NOTE Need to be careful using this for activation, particular to 1 sided.
            act_quantizer = Qmax(nbits, align_zero=align_zero, minmax=False)
        elif qa_mode == "minmax":
            act_quantizer = Qmax(nbits, align_zero=align_zero, minmax=True)
        elif qa_mode == "fix":
            act_quantizer = QFixSymmetric(
                nbits, init_clip_val=clip_val, align_zero=align_zero
            )
        elif qa_mode == "maxsym":
            act_quantizer = Qmax(
                nbits,
                align_zero=True,
                minmax=False,
                extend_act_range=extend_act_range,
            )
        elif qa_mode == "pactsym":
            act_quantizer = PACT2Sym(
                nbits,
                init_clip_val=clip_val,
                dequantize=True,
                inplace=False,
            )
        elif qa_mode == "pactsym+":
            act_quantizer = PACTplusSym(
                nbits,
                init_clip_val=clip_val,
                dequantize=True,
                inplace=False,
                intg_zp=align_zero,
                OORgradnoclip=False,
                extend_act_range=extend_act_range,
            )
        elif qa_mode == "brecq":
            act_quantizer = UniformAffineQuantizer(nbits, inited=True)
        elif "fp8" in qa_mode:
            if "custom" in qa_mode:
                act_quantizer = to_custom_fp8(
                    bits=nbits,
                    q_mode=qa_mode,
                    use_subnormal=use_subnormal,
                    scale_to_max="scale" in qa_mode,
                )
            else:
                # qa_mode should be one of:
                # [fp8_e4m3_sat, fp8_e5m2_sat, fp8_e4m3_scale, fp8_e5m2_scale]
                # by default, emulate = True, unless using a GPU that support FP8 computation
                # NOTE: emulate will be similar to dequantize.
                perToken = "perToken" in qa_mode
                act_quantizer = to_fp8(
                    nbits,
                    q_mode=qa_mode,
                    perToken=perToken,
                    emulate=True,
                )
        elif qa_mode == "pertokenmax":
            act_quantizer = PerTokenMax(nbits)
        else:
            raise ValueError(f"unrecognized activation quantization mode {qa_mode}")
    else:  # swcap-compatible activation quantizers
        if qa_mode in ("pact", "pact+"):
            if non_neg:
                assert qa_mode == "pact", "pact+ not yet supported on single side PACT"
                act_quantizer = PACT_sw(
                    nbits,
                    init_clip_val=clip_val,
                    dequantize=False,
                    align_zero=align_zero,
                )
            else:
                pact_plus = qa_mode == "pact+"
                act_quantizer = PACT2_sw(
                    nbits,
                    init_clip_val=clip_val,
                    init_clip_valn=clip_valn,
                    dequantize=False,
                    align_zero=align_zero,
                    pact_plus=pact_plus,
                )
        elif qa_mode == "pactsym":
            act_quantizer = PACT2sym_sw(nbits, init_clip_val=clip_val, dequantize=False)
        elif qa_mode == "sawb":
            act_quantizer = SAWB_sw(
                nbits, dequantize=False, clipSTE=False, recompute=False
            )
        elif qa_mode == "sawb+":
            act_quantizer = SAWB_sw(
                nbits, dequantize=False, clipSTE=True, recompute=False
            )
        elif qa_mode == "max":
            act_quantizer = Qmax_sw(nbits, dequantize=False)
        else:
            raise ValueError(
                f"activation quantization mode {qa_mode} is incompatible with swcap"
            )

    return act_quantizer


def get_weight_quantizer(
    qw_mode="SAWB+",
    nbits=32,
    clip_val=None,
    clip_valn=None,
    align_zero=True,
    w_shape=None,
    use_swcap=False,
    recompute=False,
    perGp=None,
    use_subnormal=False,
):
    """Return a quantizer for weight quantization
    Regular quantizers:
    - sawb (16, perCh, +, interp)
    - max, minmax
    - pact, cgpact, pact+
    - lsq+, fix, dorefa
    - brecq, adaround
    SWCAP quantizers:
    - sawb/sawb+
    - max
    """
    weight_quantizer = None
    if not use_swcap:
        cggrad = "cgpact" in qw_mode

        if "sawb" in qw_mode:
            Nch = w_shape[0] if w_shape is not None and "perCh" in qw_mode else False
            clipSTE = "+" in qw_mode
            intp = "interp" in qw_mode
            weight_quantizer = SAWB(
                nbits,
                dequantize=True,
                inplace=False,
                align_zero=True,
                clipSTE=clipSTE,
                perCh=Nch,
                interp=intp,
            )
        elif "max" in qw_mode:
            Nch = w_shape[0] if w_shape is not None and "perCh" in qw_mode else False
            Ngp = (
                [w_shape[0] * w_shape[1] // perGp, perGp]
                if "perGp" in qw_mode
                else False
            )  # store clip_val size and group size
            weight_quantizer = Qmax(
                nbits,
                align_zero=align_zero,
                minmax="min" in qw_mode,
                perCh=Nch,
                perGp=Ngp,
            )
        elif qw_mode == "pact":
            weight_quantizer = PACT2(
                nbits,
                init_clip_val=clip_val,
                init_clip_valn=clip_valn,
                cggrad=cggrad,
                dequantize=True,
                inplace=False,
            )
        elif qw_mode == "cgpact":
            ...
            # TODO check implementation
        elif qw_mode == "pact+":
            weight_quantizer = PACTplusSym(
                nbits,
                init_clip_val=clip_val,
                dequantize=True,
                inplace=False,
                intg_zp=align_zero,
                OORgradnoclip=False,
            )
        elif qw_mode == "lsq+":
            weight_quantizer = LSQPlus(
                nbits,
                init_clip_vals=clip_val,
                init_clip_valb=clip_valn,
                dequantize=True,
                inplace=False,
            )
        elif qw_mode == "fix":
            weight_quantizer = QFixSymmetric(
                nbits, init_clip_val=clip_val, align_zero=align_zero
            )
        elif qw_mode == "brecq":
            weight_quantizer = UniformAffineQuantizer(nbits, inited=True)
        elif "adaround" in qw_mode:
            useSAWB = (
                "SAWB" in qw_mode
            )  # use SAWB to determine delta, also allow grad/update for weights
            weight_quantizer = AdaRoundQuantizer(
                nbits,
                round_mode="learned_hard_sigmoid" if not useSAWB else "weight_STE",
                useSAWB=useSAWB,
                perCh="perCh" in qw_mode,
                multimodal="multimodal" in qw_mode,
                scalebyoptim="optim" in qw_mode,
            )
        elif "fp8" in qw_mode:
            if "custom" in qw_mode:
                weight_quantizer = to_custom_fp8(
                    bits=nbits,
                    q_mode=qw_mode,
                    use_subnormal=use_subnormal,
                    scale_to_max="scale" in qw_mode,
                )
            else:
                # qw_mode should be one of:
                # [fp8_e4m3_sat, fp8_e5m2_sat, fp8_e4m3_scale, fp8_e5m2_scale] + 'perCh'
                # by default, emulate = True, unless using a GPU that support FP8 computation
                # NOTE: emulate will be similar to dequantize.
                Nch = (
                    w_shape[0] if w_shape is not None and "perCh" in qw_mode else False
                )
                weight_quantizer = to_fp8(
                    nbits,
                    q_mode=qw_mode,
                    emulate=True,
                    perCh=Nch,
                )
        else:
            raise ValueError(f"unrecognized weight quantized mode {qw_mode}")
    else:  # swcap-compatible weight quantizers
        assert (
            align_zero
        ), "Error during weight quantizer selection: swcap requires zero alignment"
        if qw_mode == "sawb":
            weight_quantizer = SAWB_sw(
                nbits, dequantize=False, clipSTE=False, recompute=recompute
            )
        elif qw_mode == "sawb+":
            weight_quantizer = SAWB_sw(
                nbits, dequantize=False, clipSTE=True, recompute=recompute
            )
        elif qw_mode == "max":
            weight_quantizer = Qmax_sw(nbits, dequantize=False, recompute=recompute)
        else:
            raise ValueError(
                f"activation quantized mode {qw_mode} is incompatible with swcap"
            )

    return weight_quantizer


######SAWB Quantizers#######
class SAWB(nn.Module):
    """SAWB with custom backward (gradient pass through for clip function)
    if align_zero: quantizer = SAWBSTE() for coded sawb such as 103, 403, 803
    if not align_zero: quantizer = SAWBZeroSTE() for normal precision setting such as 2, 4, 8

    SAWB is only used to quantize weights
    """

    def __init__(
        self,
        num_bits,
        dequantize=True,
        inplace=False,
        align_zero=False,
        clipSTE=True,
        perCh=False,
        interp=False,
    ):
        super().__init__()
        if num_bits in [2, 4, 8]:
            self.num_bits = num_bits
        else:
            raise ValueError("FMS: SAWB supports 2, 4, and 8-bit quantization only.")
        self.dequantize = dequantize
        self.inplace = inplace
        self.align_zero = align_zero
        self.clipSTE = clipSTE
        self.perCh = perCh  # if perCh, this will be the number of ch_out
        self.interp = interp

        self.set_quantizer()

        # self.register_buffer(
        #     "sawb_clip", torch.zeros(perCh) if perCh else torch.Tensor([0.0])
        # )  # will obsolete soon
        self.register_buffer(
            "clip_val", torch.zeros(perCh) if perCh else torch.Tensor([0.0])
        )  # make it consistent with other quantizers

    def set_quantizer(self):
        if self.clipSTE:
            if self.align_zero:
                self.quantizer = (
                    SAWBPlusZeroPerChSTE
                    if self.perCh and self.num_bits in [2, 4, 8]
                    else SAWBPlusZeroSTE
                )
            else:
                self.quantizer = SAWBPlusSTE
        else:
            # if perCh but no sawb+ (e.g. `sawb_perCh`) will use a per-tensor clip
            # copied over each channel
            if self.align_zero:
                self.quantizer = SAWBZeroSTE
            else:
                self.quantizer = SAWBSTE

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        input_tensor = self.quantizer.apply(
            input_tensor,
            self.num_bits,
            self.dequantize,
            self.inplace,
            self.clip_val,
            self.training,
        )
        # NOTE: in the past, SAWB didn't check eval/training and recalc clipvals no matter what,
        #       now we should pass self.training to avoid confusion
        return input_tensor

    def __repr__(self):
        inplace_str = ", inplace" if self.inplace else ""
        return (
            f"{self.__class__.__name__}(num_bits={self.num_bits}, "
            f"quantizer={self.quantizer}{inplace_str})"
        )


class SAWBPlusZeroSTE(torch.autograd.Function):
    """SAWB+ with zero alignment (symmetric) and no gradient clipping
    Supported bits: 2, 4, 7, 8
    Other bits requests: runs x.abs().max(), not SAWB
    "dequantize=False" option is functional
    """

    @staticmethod
    def forward(
        ctx, input_tensor, num_bits, dequantize, inplace, objSAWB_clip_val, istraining
    ):
        if inplace:
            ctx.mark_dirty(input_tensor)
        scale = 2**num_bits - 2
        zero_point = 0.0
        if istraining:
            bits2code = {2: 103, 4: 403, 7: 703, 8: 803}
            if num_bits in bits2code:
                clip_val, _ = sawb_params_code(
                    num_bits, bits2code[num_bits], input_tensor
                )
            else:
                clip_val = input_tensor.abs().max()
        else:
            # do not recalc clipval when under eval mode
            clip_val = objSAWB_clip_val

        # Sometimes sawb returns negative clipvals, add a safety check
        if clip_val <= 0:
            clip_val = input_tensor.abs().max()

        if len(clip_val.shape) == 0:
            clip_val = clip_val.unsqueeze(dim=0)
        objSAWB_clip_val.copy_(clip_val)
        output = input_tensor.mul(1 / clip_val).clamp(-1, 1).mul(0.5).add(0.5)
        output = linear_quantize(output, scale, zero_point, inplace)
        if dequantize:
            output = linear_dequantize(output, scale, zero_point, inplace)
            output = (2 * output - 1) * clip_val
        else:
            output -= scale / 2
        return output

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None, None, None, None


class SAWBPlusZeroPerChSTE(torch.autograd.Function):
    """per-channel SAWB with zero alignment, can use 15 bins, i.e. [-7,7]"""

    @staticmethod
    def forward(
        ctx, input_tensor, num_bits, dequantize, inplace, objSAWB_clip_val, istraining
    ):
        # assert num_bits in [4, 8], "only implemented for 4bit and 8bit"
        if inplace:
            ctx.mark_dirty(input_tensor)

        if istraining:
            # only recalc clipvals under training mode
            SAWBcode_mapping = {8: 803, 4: 403, 2: 103}
            if num_bits in [2, 4, 8]:
                sawb_code = SAWBcode_mapping[num_bits]
                clip_val, _ = sawb_params_code(
                    num_bits, sawb_code, input_tensor, perCh=True
                )
            else:
                # use min/max for 8bit sawb for now.
                clip_val = torch.max(
                    input_tensor.abs().reshape([input_tensor.shape[0], -1]), dim=1
                ).values
                assert (
                    len(clip_val) == input_tensor.shape[0]
                ), f"dimension error, input_tensor{input_tensor.shape}, clipval{clip_val.shape}"
        else:
            # do not recalc clipval when under eval mode
            clip_val = objSAWB_clip_val

        objSAWB_clip_val.copy_(clip_val)

        int_l = -(2 ** (num_bits - 1)) + 1
        int_u = -int_l

        scale = clip_val * 2 / (2**num_bits - 2)
        # original SAWB assumes odd number of bins when calc clip_val
        zero_point = torch.zeros_like(scale)  # SAWB always centers around 0 and align 0

        if dequantize:
            output = torch.fake_quantize_per_channel_affine(
                input_tensor.float(),
                scale.float(),
                zero_point.float(),
                axis=0,
                quant_min=int_l,
                quant_max=int_u,
            ).to(
                clip_val.dtype
            )  # NOTE return will be a fp32 tensor; function only support float()
        else:
            output = torch.quantize_per_channel(
                input_tensor, scale, zero_point, 0, torch.qint8
            ).int_repr()
            # NOTE return will be a torch.int8 tensor

        return output

    @staticmethod
    def backward(ctx, grad_output):
        grad_input_tensor = grad_output.clone()
        return grad_input_tensor, None, None, None, None, None


class SAWBZeroSTE(torch.autograd.Function):
    """SAWB with zero alignment (symmetric) and gradient clipping
    Supported bits: 2, 4, 7, 8
    Other bits requests: runs x.abs().max(), not SAWB
    "dequantize=False" option is functional
    """

    @staticmethod
    def forward(
        ctx, input_tensor, num_bits, dequantize, inplace, objSAWB_clip_val, istraining
    ):
        if inplace:
            ctx.mark_dirty(input_tensor)
        scale = 2**num_bits - 2
        zero_point = 0.0

        if istraining:
            bits2code = {2: 103, 4: 403, 7: 703, 8: 803}
            if num_bits in bits2code:
                clip_val, _ = sawb_params_code(
                    num_bits, bits2code[num_bits], input_tensor
                )
            else:
                clip_val = input_tensor.abs().max()
        else:
            # do not recalc clipval when under eval mode
            clip_val = objSAWB_clip_val

        if len(clip_val.shape) == 0:
            clip_val = clip_val.unsqueeze(dim=0)
        objSAWB_clip_val.copy_(clip_val)
        output = input_tensor.mul(1 / clip_val).clamp(-1, 1).mul(0.5).add(0.5)
        output = linear_quantize(output, scale, zero_point, inplace)
        if dequantize:
            output = linear_dequantize(output, scale, zero_point, inplace)
            output = (2 * output - 1) * clip_val
        else:
            output -= scale / 2
        ctx.save_for_backward(input_tensor, clip_val)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input_tensor, clip_val = ctx.saved_tensors
        grad_input_tensor = grad_output.clone()
        grad_input_tensor = torch.where(
            input_tensor < -clip_val,
            torch.zeros_like(grad_input_tensor),
            grad_input_tensor,
        )
        grad_input_tensor = torch.where(
            input_tensor > clip_val,
            torch.zeros_like(grad_input_tensor),
            grad_input_tensor,
        )
        return grad_input_tensor, None, None, None, None, None


def sawb_params_code(num_bits, code, out, perCh=False):
    with torch.no_grad():
        coeff_dict = {
            102: (3.12, -2.064),  # [-a, -a/3, a/3, a] equivalent to 2 bits
            103: (2.6, -1.71),  # [-a, 0, a]
            403: (12.035, -12.03),  # [-a, -6/7a, ..., 0, ..., 6/7a, a]
            703: (28.24, -30.81),
            803: (31.76, -35.04),
        }

        if not coeff_dict.get(code) is None:
            coeff = coeff_dict[code]
        else:
            raise ValueError(f"SAWB not implemented for code={code}")

        if perCh:
            # per-channel
            reduce_dim = list(range(1, len(out.shape)))
            # conv W=[ch_o, ch_i, ki, ij], linear W=[ch_o, ch_i], reduce all dim but ch_out
            mu = torch.mean(out.abs(), dim=reduce_dim)
            std = torch.mean(out**2, dim=reduce_dim).sqrt()
            clip_val_vec = coeff[1] * mu + coeff[0] * std
            return clip_val_vec, None

        # per-tensor
        x = out.flatten()
        mu = x.abs().mean()
        std = x.mul(x).mean().sqrt()

        clip_val = coeff[1] * mu + coeff[0] * std

        if code in [102]:
            nspace = 2**num_bits - 1
        elif code in [403, 103, 703, 803]:
            nspace = 2**num_bits - 2
        else:
            raise ValueError(f"SAWB not implemented for code={code}")

        return clip_val, nspace


class SAWBPlusSTE(torch.autograd.Function):
    """
    SAWB+: no zero alignment and no gradient clipping
    Incorrect behavior for "dequantize=False" - do not use
    """

    @staticmethod
    def forward(
        ctx, input_tensor, num_bits, dequantize, inplace, objSAWB_clip_val, istraining
    ):
        if inplace:
            ctx.mark_dirty(input_tensor)
        scale, zero_point = asymmetric_linear_quantization_params(
            num_bits, saturation_min=0, saturation_max=1, signed=False
        )  # returns scale = 2^bits-1, zero_point = 0

        if istraining:
            # only recalc clipval under training mode
            if num_bits in [2, 3, 4, 5]:  # 8
                clip_val = sawb_params(num_bits, input_tensor)
            else:
                clip_val = input_tensor.abs().max()
        else:
            # do not recalc clipval when under eval mode
            clip_val = objSAWB_clip_val

        if len(clip_val.shape) == 0:
            clip_val = clip_val.unsqueeze(dim=0)
        objSAWB_clip_val.copy_(clip_val)
        output = input_tensor.mul(1 / clip_val).clamp(-1, 1).mul(0.5).add(0.5)
        output = linear_quantize(output, scale, zero_point, inplace)
        if dequantize:
            output = linear_dequantize(output, scale, zero_point, inplace)
        output = (2 * output - 1) * clip_val
        return output

    @staticmethod
    def backward(ctx, grad_output):
        grad_input_tensor = grad_output.clone()
        return grad_input_tensor, None, None, None, None, None


class SAWBSTE(torch.autograd.Function):
    """
    SAWB without zero alignment
    Incorrect behavior for "dequantize=False" - do not use
    """

    @staticmethod
    def forward(
        ctx, input_tensor, num_bits, dequantize, inplace, objSAWB_clip_val, istraining
    ):
        if inplace:
            ctx.mark_dirty(input_tensor)
        scale, zero_point = asymmetric_linear_quantization_params(
            num_bits, saturation_min=0, saturation_max=1, signed=False
        )
        if istraining:
            if num_bits in [2, 3, 4, 5]:
                clip_val = sawb_params(num_bits, input_tensor)
            else:
                clip_val = input_tensor.abs().max()
        else:
            # do not recalc clipval when under eval mode
            clip_val = objSAWB_clip_val

        if len(clip_val.shape) == 0:
            clip_val = clip_val.unsqueeze(dim=0)
        objSAWB_clip_val.copy_(clip_val)
        output = input_tensor.mul(1 / clip_val).clamp(-1, 1).mul(0.5).add(0.5)
        output = linear_quantize(output, scale, zero_point, inplace)
        if dequantize:
            output = linear_dequantize(output, scale, zero_point, inplace)
        output = (2 * output - 1) * clip_val
        ctx.save_for_backward(input_tensor, clip_val)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input_tensor, clip_val = ctx.saved_tensors
        grad_input_tensor = grad_output.clone()
        grad_input_tensor = torch.where(
            input_tensor < -clip_val,
            torch.zeros_like(grad_input_tensor),
            grad_input_tensor,
        )
        grad_input_tensor = torch.where(
            input_tensor > clip_val,
            torch.zeros_like(grad_input_tensor),
            grad_input_tensor,
        )
        return grad_input_tensor, None, None, None, None, None


def sawb_params(num_bits, out):
    with torch.no_grad():
        x = out.flatten()
        mu = x.abs().mean()
        std = x.mul(x).mean().sqrt()

        dic_coeff = {
            2: (3.12, -2.064),
            3: (7.509, -6.892),
            4: (12.68, -12.80),
            5: (17.74, -18.64),
            8: (31.76, -35.04),
        }
        if num_bits > 8:
            raise ValueError(f"SAWB not implemented for num_bits={num_bits}")
        coeff = dic_coeff[num_bits]
        clip_val = coeff[1] * mu + coeff[0] * std

        return clip_val


#####################################
##############1-side PACT###############
class PACT(nn.Module):
    """1-sided original PACT
    PACT is only used to quantize activations
    """

    def __init__(
        self,
        num_bits,
        init_clip_val,
        init_clip_valn=0,  # pylint: disable=unused-argument
        dequantize=True,
        inplace=False,
        cggrad=False,
        grad_scale=False,
        pact_plus=False,
    ):
        super().__init__()
        self.num_bits = num_bits
        if isinstance(init_clip_val, torch.Tensor):
            self.clip_val = nn.Parameter(init_clip_val)
        else:
            self.clip_val = nn.Parameter(torch.Tensor([init_clip_val]))
        self.dequantize = dequantize
        self.inplace = inplace
        self.cggrad = cggrad
        self.grad_scale = grad_scale

        self.quantizer = (
            CGPACT_STE
            if self.cggrad
            else PACTplusSTE
            if pact_plus
            else CGPACT_gScale_STE
            if self.grad_scale
            else PACT_STE
        )

    def forward(self, input_tensor):
        input_tensor = self.quantizer.apply(
            input_tensor,
            self.clip_val,
            self.num_bits,
            self.dequantize,
            self.inplace,
        )
        return input_tensor

    def __repr__(self):
        inplace_str = ", inplace" if self.inplace else ""
        return (
            f"{self.__class__.__name__}(num_bits={self.num_bits}, clip_val={self.clip_val[0]}, "
            f"cggrad={self.cggrad}, grad_scale={self.grad_scale}, quantizer={self.quantizer}, "
            f"{inplace_str})"
        )


class PACT_STE(torch.autograd.Function):
    """1-sided original PACT"""

    @staticmethod
    def forward(ctx, input_tensor, clip_val, num_bits, dequantize, inplace):
        clip_val = clip_val.to(input_tensor.dtype)
        ctx.save_for_backward(input_tensor, clip_val)
        if inplace:
            ctx.mark_dirty(input_tensor)
        scale, zero_point = asymmetric_linear_quantization_params(
            num_bits, saturation_min=0, saturation_max=clip_val.data, signed=False
        )
        if isinstance(clip_val, torch.Tensor):
            if input_tensor.min() < 0:
                raise ValueError(
                    "FMS: input_tensor to single_side PACT should be non-negative."
                )
            output = torch.where(
                input_tensor > clip_val,
                torch.ones_like(input_tensor) * clip_val,
                input_tensor,
            )
        else:
            output = clamp(input_tensor, 0, clip_val.data, inplace=inplace)
        output = linear_quantize(output, scale, zero_point, inplace)
        if dequantize:
            output = linear_dequantize(output, scale, zero_point, inplace)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input_tensor, clip_val = ctx.saved_tensors
        grad_input_tensor = grad_output.clone()
        grad_input_tensor = torch.where(
            input_tensor < 0, torch.zeros_like(grad_input_tensor), grad_input_tensor
        )
        grad_input_tensor = torch.where(
            input_tensor > clip_val,
            torch.zeros_like(grad_input_tensor),
            grad_input_tensor,
        )

        grad_alpha = grad_output.clone()
        grad_alpha = torch.where(
            input_tensor < clip_val, torch.zeros_like(grad_alpha), grad_alpha
        )
        grad_alpha = grad_alpha.sum().expand_as(clip_val)
        return grad_input_tensor, grad_alpha, None, None, None, None


class CGPACT_STE(torch.autograd.Function):
    """1-sided CGPACT
    use calibrated clip_val gradient to update clip_val
    """

    @staticmethod
    def forward(ctx, input_tensor, clip_val, num_bits, dequantize, inplace):
        ctx.save_for_backward(input_tensor, clip_val)
        if inplace:
            ctx.mark_dirty(input_tensor)
        scale, zero_point = asymmetric_linear_quantization_params(
            num_bits, 0, clip_val.data, signed=False
        )
        if isinstance(clip_val, torch.Tensor):
            if input_tensor.min() < 0:
                raise ValueError(
                    "FMS: input_tensor to ClippedLinearQuantization should be non-negative."
                )
            output = torch.where(
                input_tensor > clip_val,
                torch.ones_like(input_tensor) * clip_val,
                input_tensor,
            )
        else:
            output = clamp(input_tensor, 0, clip_val.data, inplace=inplace)
        output, ctx.residual = linear_quantize_residual(
            output, scale, zero_point, inplace
        )
        with torch.no_grad():
            n = 2**num_bits - 1
            ctx.residual /= n

        if dequantize:
            output = linear_dequantize(output, scale, zero_point, inplace)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input_tensor, clip_val = ctx.saved_tensors
        grad_input_tensor = grad_output.clone()

        grad_input_tensor = torch.where(
            input_tensor < 0, torch.zeros_like(grad_input_tensor), grad_input_tensor
        )
        grad_input_tensor = torch.where(
            input_tensor > clip_val,
            torch.zeros_like(grad_input_tensor),
            grad_input_tensor,
        )

        grad_alpha = grad_output.clone()
        grad_alpha = torch.where(
            input_tensor < clip_val, grad_alpha * ctx.residual, grad_alpha
        )
        grad_alpha = grad_alpha.sum().expand_as(clip_val)
        return grad_input_tensor, grad_alpha, None, None, None


class CGPACT_gScale_STE(torch.autograd.Function):
    """1-sided CGPACT
    use calibrated clip_val gradient to update clip_val with scaled gradient
    """

    @staticmethod
    def forward(ctx, input_tensor, clip_val, num_bits, dequantize, inplace):
        ctx.save_for_backward(input_tensor, clip_val)
        if inplace:
            ctx.mark_dirty(input_tensor)
        scale, zero_point = asymmetric_linear_quantization_params(
            num_bits, saturation_min=0, saturation_max=clip_val.data, signed=False
        )
        if isinstance(clip_val, torch.Tensor):
            if input_tensor.min() < 0:
                raise ValueError(
                    "FMS: input_tensor to ClippedLinearQuantization should be non-negative."
                )
            output = torch.where(
                input_tensor > clip_val,
                torch.ones_like(input_tensor) * clip_val,
                input_tensor,
            )
        else:
            output = clamp(input_tensor, 0, clip_val.data, inplace=inplace)
        output, ctx.residual = linear_quantize_residual(
            output, scale, zero_point, inplace
        )
        with torch.no_grad():
            n = 2**num_bits - 1
            ctx.residual /= n

        if dequantize:
            output = linear_dequantize(output, scale, zero_point, inplace)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input_tensor, clip_val = ctx.saved_tensors
        grad_input_tensor = grad_output.clone()

        grad_input_tensor = torch.where(
            input_tensor < 0, torch.zeros_like(grad_input_tensor), grad_input_tensor
        )
        grad_input_tensor = torch.where(
            input_tensor > clip_val,
            torch.zeros_like(grad_input_tensor),
            grad_input_tensor,
        )

        grad_alpha = grad_output.clone()
        grad_alpha = torch.where(
            input_tensor < clip_val, grad_alpha * ctx.residual, grad_alpha
        )
        grad_alpha = grad_alpha.sum().expand_as(clip_val)
        ndim = float(sum(list(grad_output.shape)))
        grad_scale = math.sqrt(1 / ndim / 3.0)  # 3 is for 2bit, should be n
        return grad_input_tensor, grad_alpha * grad_scale, None, None, None


# Single-sided PACT+, simplified version, no LUT support
class PACTplusSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_tensor, clip_val, num_bits, dequantize, inplace):
        clip_val = clip_val.to(input_tensor.dtype)
        n_levels = 2**num_bits - 1
        scale = n_levels / clip_val
        stepsize = 1.0 / scale
        # --- zp is not needed for single-sided PACT
        ctx.save_for_backward(input_tensor, clip_val)

        if inplace:
            ctx.mark_dirty(input_tensor)
        ctx.n_levels = n_levels
        ctx.n_bits = num_bits
        ctx.stepsize = stepsize

        if isinstance(clip_val, torch.Tensor):
            output = torch.clamp(input_tensor, torch.zeros_like(clip_val), clip_val)
        else:
            output = clamp(input_tensor, 0, clip_val.data, inplace=inplace)

        output = torch.round(output / stepsize.to(output.device))
        if dequantize:
            output = output * stepsize.to(output.device)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input_tensor, clip_val = ctx.saved_tensors
        n_levels = ctx.n_levels
        stepsize = ctx.stepsize
        z = input_tensor / stepsize

        # direct compute grad_alpha
        grad_input_tensor = grad_output.clone()
        delz = (z - torch.round(z)) / n_levels
        grad_alpha = -grad_output.clone() * delz

        grad_input_tensor = torch.where(
            input_tensor <= 0, torch.zeros_like(grad_input_tensor), grad_input_tensor
        )
        grad_input_tensor = torch.where(
            input_tensor >= clip_val,
            torch.zeros_like(grad_input_tensor),
            grad_input_tensor,
        )

        grad_alpha = torch.where(
            input_tensor <= 0, torch.zeros_like(grad_alpha), grad_alpha
        )
        grad_alpha = torch.where(input_tensor >= clip_val, grad_output, grad_alpha)

        grad_alpha = grad_alpha.sum().expand_as(clip_val)

        return grad_input_tensor, grad_alpha, None, None, None, None


class PACTplusSymSTE_rev1(torch.autograd.Function):
    """Symmetric 2-sided PACT+"""

    @staticmethod
    def forward(
        ctx,
        input_tensor,
        clip_val,
        num_bits,
        dequantize,
        inplace,
        intg_zp,
        OORgradnoclip,
    ):
        clip_val = clip_val.to(input_tensor.dtype)
        n_levels = 2**num_bits - 2 if intg_zp else 2**num_bits - 1
        scale = n_levels / (2 * clip_val)
        stepsize = 1.0 / scale

        ctx.save_for_backward(input_tensor, clip_val)
        if inplace:
            ctx.mark_dirty(input_tensor)
        ctx.n_levels = n_levels
        ctx.n_bits = num_bits
        ctx.stepsize = stepsize
        ctx.OORgradnoclip = OORgradnoclip
        if isinstance(clip_val, torch.Tensor):
            output = torch.where(
                input_tensor > clip_val,
                torch.ones_like(input_tensor) * clip_val,
                input_tensor,
            )
            output = torch.where(
                output < -clip_val, torch.ones_like(input_tensor) * -clip_val, output
            )
        else:
            output = clamp(input_tensor, -clip_val.data, clip_val.data, inplace=inplace)

        output = torch.round((output + clip_val) / stepsize.to(output.device))
        if dequantize:
            output = output * stepsize.to(output.device) - clip_val
        else:
            # TODO: fix inconsistency:
            # SAWB and Qmax use a different definition for scale, where
            # "scale = 2^b-2", not divided by 2*clip_val
            output -= torch.round(clip_val * scale)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input_tensor, clip_val = ctx.saved_tensors
        n_levels = ctx.n_levels
        stepsize = ctx.stepsize
        OORgradclip = not ctx.OORgradnoclip
        z = (input_tensor + clip_val) / stepsize
        delz = 2 * (torch.round(z) - z) / n_levels

        grad_input_tensor = grad_output.clone()
        if OORgradclip:
            grad_input_tensor = torch.where(
                input_tensor <= (-clip_val),
                torch.zeros_like(grad_input_tensor),
                grad_input_tensor,
            )
            grad_input_tensor = torch.where(
                input_tensor >= clip_val,
                torch.zeros_like(grad_input_tensor),
                grad_input_tensor,
            )

        grad_alpha = torch.ones_like(grad_output) * delz
        grad_alpha = torch.where(
            input_tensor <= -clip_val, -torch.ones_like(grad_input_tensor), grad_alpha
        )
        grad_alpha = torch.where(
            input_tensor >= clip_val, torch.ones_like(grad_input_tensor), grad_alpha
        )
        grad_alpha *= grad_output

        grad_alpha = grad_alpha.sum().expand_as(clip_val)

        return grad_input_tensor, grad_alpha, None, None, None, None, None, None


class PACTplusExtendRangeSTE(torch.autograd.Function):
    """2-sided PACT+ using a single clip

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
    Out Of Range input_tensor grad clipping is always enabled
    No in_place functionalities
    No Automatic Mixed Precision (AMP) functionalities
    Dequantization is functional
    """

    @staticmethod
    def forward(
        ctx,
        input_tensor,
        clip_val,
        num_bits,
        dequantize,
        _inplace,  # not in use
        _intg_zp,  # not in use
        _OORgradnoclip,  # not in use
    ):
        n_half = 2 ** (num_bits - 1)  # levels in half range: 8b: 128; 4b: 8; 2b: 2
        clip_valn = -clip_val * n_half / (n_half - 1)  # 8b: -128/127 * clip_val
        stepsize = clip_val / (n_half - 1)

        ctx.save_for_backward(input_tensor, clip_val, clip_valn, stepsize)
        ctx.n_half = n_half

        # clip input_tensor distribution on both sides
        output = torch.where(
            input_tensor > clip_val,
            torch.ones_like(input_tensor) * clip_val,
            input_tensor,
        )
        output = torch.where(
            output < clip_valn, torch.ones_like(input_tensor) * clip_valn, output
        )

        # quantize to range [-2^(b-1), 2^(b-1)-1]
        output = torch.round(output / stepsize)

        if dequantize:
            output = output * stepsize
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input_tensor, clip_val, clip_valn, stepsize = ctx.saved_tensors
        n_half = ctx.n_half

        grad_input_tensor = grad_output.clone()
        grad_input_tensor = torch.where(
            input_tensor <= clip_valn,
            torch.zeros_like(grad_input_tensor),
            grad_input_tensor,
        )
        grad_input_tensor = torch.where(
            input_tensor >= clip_val,
            torch.zeros_like(grad_input_tensor),
            grad_input_tensor,
        )

        z = (input_tensor - clip_valn) / stepsize
        grad_alpha = (torch.round(z) - z) / (n_half - 1)
        grad_alpha = torch.where(
            input_tensor <= clip_valn,
            clip_valn / clip_val * torch.ones_like(grad_input_tensor),
            grad_alpha,
        )
        grad_alpha = torch.where(
            input_tensor >= clip_val, torch.ones_like(grad_input_tensor), grad_alpha
        )
        grad_alpha *= grad_output

        grad_alpha = grad_alpha.sum().expand_as(clip_val)

        return grad_input_tensor, grad_alpha, None, None, None, None, None, None


class PACTplusSym(nn.Module):
    """Two-sided symmetric PACT+
    PACTplusSym can be used to quantize both weights and activations
    """

    def __init__(
        self,
        num_bits,
        init_clip_val,
        dequantize=True,
        inplace=False,
        intg_zp=False,
        OORgradnoclip=False,
        extend_act_range=False,
    ):
        super().__init__()
        self.num_bits = num_bits

        if isinstance(init_clip_val, torch.Tensor):
            self.clip_val = nn.Parameter(init_clip_val)
        elif not isinstance(init_clip_val, torch.Tensor):
            self.clip_val = nn.Parameter(torch.Tensor([init_clip_val]))

        self.dequantize = dequantize
        self.inplace = inplace
        self.intg_zp = intg_zp
        self.OORgradnoclip = (
            OORgradnoclip  # disable Out Of Range input_tensor grad clipping
        )
        if not extend_act_range:
            self.quantizer = PACTplusSymSTE_rev1
            self.quantizer_name = "PACT+sym"
        else:
            self.quantizer = PACTplusExtendRangeSTE
            self.quantizer_name = "PACT+extend"
        # TODO why is there duplicated codes? clean up needed
        self.quantizer = (
            PACTplusSymSTE_rev1 if not extend_act_range else PACTplusExtendRangeSTE
        )

    def forward(self, input_tensor):
        input_tensor = self.quantizer.apply(
            input_tensor,
            self.clip_val,
            self.num_bits,
            self.dequantize,
            self.inplace,
            self.intg_zp,
            self.OORgradnoclip,
        )
        return input_tensor

    def __repr__(self):
        inplace_str = ", inplace" if self.inplace else ""
        return (
            f"{self.quantizer_name}(num_bits={self.num_bits}, clip_val={self.clip_val[0]:.2f}, "
            f"OORgradnoclip={self.OORgradnoclip}, class={self.__class__.__name__}{inplace_str})"
        )


class PACT2Sym_STE(torch.autograd.Function):
    """Symmetric with zero in the center. For example, 4bit -- > [-7, 7] with FP0 align to INT0"""

    @staticmethod
    def forward(ctx, input_tensor, clip_val, num_bits, dequantize, inplace):
        clip_val = clip_val.to(input_tensor.dtype)
        ctx.save_for_backward(input_tensor, clip_val)
        if inplace:
            ctx.mark_dirty(input_tensor)
        scale, zero_point = symmetric_linear_quantization_params(
            num_bits, clip_val.data
        )
        if isinstance(clip_val, torch.Tensor):
            output = torch.where(
                input_tensor > clip_val,
                torch.ones_like(input_tensor) * clip_val,
                input_tensor,
            )
            output = torch.where(
                output < -clip_val, torch.ones_like(input_tensor) * (-clip_val), output
            )
        else:
            output = clamp(input_tensor, -clip_val.data, clip_val.data, inplace)
        output = linear_quantize(output, scale, zero_point, inplace)
        if dequantize:
            output = linear_dequantize(output, scale, zero_point, inplace)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input_tensor, clip_val = ctx.saved_tensors
        grad_input_tensor = grad_output.clone()
        grad_input_tensor = torch.where(
            input_tensor <= -clip_val,
            torch.zeros_like(grad_input_tensor),
            grad_input_tensor,
        )
        grad_input_tensor = torch.where(
            input_tensor >= clip_val,
            torch.zeros_like(grad_input_tensor),
            grad_input_tensor,
        )

        grad_alpha = grad_output.clone()
        grad_alpha = torch.where(
            torch.logical_and(input_tensor < clip_val, input_tensor > -clip_val),
            torch.zeros_like(grad_alpha),
            grad_alpha,
        )
        grad_alpha = grad_alpha.sum().expand_as(clip_val)

        return grad_input_tensor, grad_alpha, None, None, None, None


# for symmetric pact2 mainly for QBMM, can
class PACT2Sym(nn.Module):
    """Two-sided PACT with symmetric clip values"""

    def __init__(self, num_bits, init_clip_val=8.0, dequantize=True, inplace=False):
        super().__init__()
        self.num_bits = num_bits
        if isinstance(init_clip_val, torch.Tensor):
            self.clip_val = nn.Parameter(init_clip_val)
        elif not isinstance(init_clip_val, torch.Tensor):
            self.clip_val = nn.Parameter(torch.Tensor([init_clip_val]))

        self.dequantize = dequantize
        self.inplace = inplace
        self.quantizer = PACT2Sym_STE

    def forward(self, input_tensor):
        input_tensor = self.quantizer.apply(
            input_tensor,
            self.clip_val,
            self.num_bits,
            self.dequantize,
            self.inplace,
        )
        return input_tensor

    def __repr__(self):
        inplace_str = ", inplace" if self.inplace else ""
        return (
            f"PACT2Sym(num_bits={self.num_bits}, clip_val={self.clip_val[0]}, "
            f"{self.__class__}{inplace_str})"
        )


#####################################
##############2-side PACT###############
class PACT2(nn.Module):
    """Two-sided original PACT
    PACT2 can be used to quantize both weights and activations
    """

    def __init__(
        self,
        num_bits,
        init_clip_valn=-8.0,
        init_clip_val=8.0,
        dequantize=True,
        inplace=False,
        cggrad=False,
        pcq_w=False,
        grad_scale=False,
        pact_plus=False,
        **kwargs,
    ):
        super().__init__()
        self.num_bits = num_bits
        if isinstance(init_clip_val, torch.Tensor) and isinstance(
            init_clip_valn, torch.Tensor
        ):
            self.clip_val = nn.Parameter(init_clip_val)
            self.clip_valn = nn.Parameter(init_clip_valn)
        elif not isinstance(init_clip_val, torch.Tensor) and not isinstance(
            init_clip_valn, torch.Tensor
        ):
            self.clip_val = nn.Parameter(torch.Tensor([init_clip_val]))
            self.clip_valn = nn.Parameter(torch.Tensor([init_clip_valn]))
        else:
            raise ValueError(
                "FMS: init_clip_val and init_clip_valn  should be the same instance type."
            )
        self.dequantize = dequantize
        self.inplace = inplace
        self.cggrad = cggrad
        self.pcq_w = pcq_w
        self.grad_scale = grad_scale
        self.pact_plus = pact_plus
        self.align_zero = kwargs.get("align_zero", False)
        self.use_PT_native_Qfunc = kwargs.get("use_PT_native_Qfunc", False)

        self.quantizer = (
            CGPACT2_STE
            if self.cggrad
            else CGPACT2_perChannel_STE
            if self.pcq_w
            else CGPACT2_gScale_STE
            if self.grad_scale
            else PACTplus2STE
            if self.pact_plus
            else PACT2_STE
        )

    def set_quantizer(self):
        self.quantizer = (
            CGPACT2_STE
            if self.cggrad
            else CGPACT2_perChannel_STE
            if self.pcq_w
            else CGPACT2_gScale_STE
            if self.grad_scale
            else PACTplus2STE
            if self.pact_plus
            else PACT2_STE
        )

    def forward(self, input_tensor):
        input_tensor = self.quantizer.apply(
            input_tensor,
            self.clip_val,
            self.clip_valn,
            self.num_bits,
            self.dequantize,
            self.inplace,
            self.align_zero,
            self.use_PT_native_Qfunc,
        )
        return input_tensor

    def __repr__(self):
        clip_str = (
            f", pos-clip={self.clip_val[0]:.4f}, neg-clip={self.clip_valn[0]:.4f},"
            f"quantizer={self.quantizer}"
        )
        inplace_str = ", inplace" if self.inplace else ""
        return f"{self.__class__.__name__}(num_bits={self.num_bits}{clip_str}{inplace_str})"


class PACT2_STE(torch.autograd.Function):
    """two-sided original pact quantization for activation"""

    @staticmethod
    def forward(
        ctx,
        input_tensor,
        clip_val,
        clip_valn,
        num_bits,
        dequantize,
        inplace,
        align_zero,
        use_PT_native_Qfunc,
    ):
        clip_val, clip_valn = (
            clip_val.to(input_tensor.dtype),
            clip_valn.to(input_tensor.dtype),
        )
        ctx.save_for_backward(input_tensor, clip_val, clip_valn)
        if inplace:
            ctx.mark_dirty(input_tensor)
        scale, zero_point = asymmetric_linear_quantization_params(
            num_bits,
            clip_valn.data,
            clip_val.data,
            integral_zero_point=align_zero,
            signed=False,
        )

        if zero_point != 0:
            quant_min, quant_max = 0, 2**num_bits - 1  # eg (0, 255) or (0,15)
            int_dtype = torch.uint8
        else:
            quant_min, quant_max = (
                -(2 ** (num_bits - 1)),
                2 ** (num_bits - 1) - 1,
            )  # eg (-128,127) or (-8,7)
            int_dtype = torch.int8

        if use_PT_native_Qfunc:
            qint_dtype_dict = {
                (32, True): torch.int32,
                (8, True): torch.qint8,
                (8, False): torch.quint8,
                (4, True): torch.qint8,
                (4, False): torch.quint8,
            }
            # Note: quantized_tensor.int_repr() returns uint8
            int_dtype_dict = {
                torch.qint32: torch.int32,
                torch.qint8: torch.int8,
                torch.quint8: torch.uint8,
            }

            n_levels = 2**num_bits - 1
            scale = (clip_val - clip_valn) / n_levels  # overwrite scale
            zp = torch.round(-clip_valn / scale).to(torch.int)
            signed = (zp == 0).item()
            qint_dtype = qint_dtype_dict.get((num_bits, signed))
            int_dtype = int_dtype_dict.get(qint_dtype)
            input_tensor_dtype = (
                input_tensor.dtype
            )  # fake_quantize_per_xxx doesn't support fp16 out of the box

            if zp != 0:
                quant_min, quant_max = 0, 2**num_bits - 1  # eg (0, 255) or (0,15)
            else:
                quant_min, quant_max = (
                    -(2 ** (num_bits - 1)),
                    2 ** (num_bits - 1) - 1,
                )  # eg (-128,127) or (-8,7)

            if dequantize:
                out = torch.fake_quantize_per_tensor_affine(
                    input_tensor.float(), scale.float(), zp, quant_min, quant_max
                )
                out = out.to(input_tensor_dtype)
            else:
                # Clamp to [quant_min, quant_max] in case we are storing int4 into a uint8 tensor
                out = (
                    torch.quantize_per_tensor(
                        input_tensor, scale.float(), zp, qint_dtype
                    )
                    .int_repr()
                    .clamp(quant_min, quant_max)
                )
            return out
            # NOTE remember scale and zp from asym_lin_q_params is different from
            # normal scale and zp def

        if isinstance(clip_val, torch.Tensor):
            output = torch.where(
                input_tensor > clip_val,
                torch.ones_like(input_tensor) * clip_val,
                input_tensor,
            )
            output = torch.where(
                output < clip_valn, torch.ones_like(input_tensor) * clip_valn, output
            )
        else:
            output = clamp(input_tensor, clip_valn.data, clip_val.data, inplace)

        output = linear_quantize(output, scale, zero_point, inplace)
        if dequantize:
            output = linear_dequantize(output, scale, zero_point, inplace)
        else:
            output = output.clamp(quant_min, quant_max).to(int_dtype)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        input_tensor, clip_val, clip_valn = ctx.saved_tensors
        grad_input_tensor = grad_output.clone()
        grad_input_tensor = torch.where(
            input_tensor <= clip_valn,
            torch.zeros_like(grad_input_tensor),
            grad_input_tensor,
        )
        grad_input_tensor = torch.where(
            input_tensor >= clip_val,
            torch.zeros_like(grad_input_tensor),
            grad_input_tensor,
        )

        grad_alpha = grad_output.clone()
        grad_alpha = torch.where(
            input_tensor < clip_val, torch.zeros_like(grad_alpha), grad_alpha
        )
        grad_alpha = grad_alpha.sum().expand_as(clip_val)

        grad_alphan = grad_output.clone()
        grad_alphan = torch.where(
            input_tensor > clip_valn, torch.zeros_like(grad_alphan), grad_alphan
        )
        grad_alphan = grad_alphan.sum().expand_as(clip_valn)
        # Straight-through estimator for the scale factor calculation
        return (
            grad_input_tensor,
            grad_alpha,
            grad_alphan,
            None,
            None,
            None,
            None,
            None,
            None,
        )


class CGPACT2_STE(torch.autograd.Function):
    """2-sided CGPACT"""

    @staticmethod
    def forward(ctx, input_tensor, clip_val, clip_valn, num_bits, dequantize, inplace):
        ctx.save_for_backward(input_tensor, clip_val, clip_valn)
        if inplace:
            ctx.mark_dirty(input_tensor)
        scale, zero_point = asymmetric_linear_quantization_params(
            num_bits,
            clip_valn.data,
            clip_val.data,
            integral_zero_point=False,
            signed=False,
        )
        if isinstance(clip_val, torch.Tensor):
            output = torch.where(
                input_tensor > clip_val,
                torch.ones_like(input_tensor) * clip_val,
                input_tensor,
            )
            output = torch.where(
                output < clip_valn, torch.ones_like(input_tensor) * clip_valn, output
            )
        else:
            output = clamp(input_tensor, clip_valn.data, clip_val.data, inplace)
        output, ctx.residual = linear_quantize_residual(
            output, scale, zero_point, inplace
        )
        with torch.no_grad():
            n = (2**num_bits - 1) / 2.0
            ctx.residual.div_(n).sub_(1.0)

        if dequantize:
            output = linear_dequantize(output, scale, zero_point, inplace)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input_tensor, clip_val, clip_valn = ctx.saved_tensors
        grad_input_tensor = grad_output.clone()
        grad_input_tensor = torch.where(
            input_tensor <= clip_valn,
            torch.zeros_like(grad_input_tensor),
            grad_input_tensor,
        )
        grad_input_tensor = torch.where(
            input_tensor >= clip_val,
            torch.zeros_like(grad_input_tensor),
            grad_input_tensor,
        )

        grad_alpha = grad_output.clone()
        grad_alpha = torch.where(
            ((input_tensor < clip_val) & (input_tensor > clip_valn)),
            grad_alpha * ctx.residual,
            grad_alpha,
        )
        grad_alpha = torch.where(input_tensor <= clip_valn, -grad_alpha, grad_alpha)
        grad_alpha = grad_alpha.sum().expand_as(clip_val)

        grad_alphan = grad_output.clone()
        grad_alphan = torch.where(input_tensor >= clip_val, -grad_alphan, grad_alphan)
        grad_alphan = torch.where(
            ((input_tensor < clip_val) & (input_tensor > clip_valn)),
            -grad_alphan * ctx.residual,
            grad_alphan,
        )
        grad_alphan = grad_alphan.sum().expand_as(clip_valn)
        # Straight-through estimator for the scale factor calculation
        return grad_input_tensor, grad_alpha, grad_alphan, None, None, None


class CGPACT2_gScale_STE(torch.autograd.Function):
    """2-sided CGPACT + scale alpha gradients"""

    @staticmethod
    def forward(ctx, input_tensor, clip_val, clip_valn, num_bits, dequantize, inplace):
        ctx.save_for_backward(input_tensor, clip_val, clip_valn)
        if inplace:
            ctx.mark_dirty(input_tensor)
        scale, zero_point = asymmetric_linear_quantization_params(
            num_bits,
            clip_valn.data,
            clip_val.data,
            integral_zero_point=False,
            signed=False,
        )
        if isinstance(clip_val, torch.Tensor):
            output = torch.where(
                input_tensor > clip_val,
                torch.ones_like(input_tensor) * clip_val,
                input_tensor,
            )
            output = torch.where(
                output < clip_valn, torch.ones_like(input_tensor) * clip_valn, output
            )
        else:
            output = clamp(input_tensor, clip_valn.data, clip_val.data, inplace)
        output, ctx.residual = linear_quantize_residual(
            output, scale, zero_point, inplace
        )
        with torch.no_grad():
            n = (2**num_bits - 1) / 2.0
            ctx.residual.div_(n).sub_(1.0)

        if dequantize:
            output = linear_dequantize(output, scale, zero_point, inplace)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input_tensor, clip_val, clip_valn = ctx.saved_tensors
        grad_input_tensor = grad_output.clone()
        grad_input_tensor = torch.where(
            input_tensor <= clip_valn,
            torch.zeros_like(grad_input_tensor),
            grad_input_tensor,
        )
        grad_input_tensor = torch.where(
            input_tensor >= clip_val,
            torch.zeros_like(grad_input_tensor),
            grad_input_tensor,
        )

        grad_alpha = grad_output.clone()
        grad_alpha = torch.where(
            ((input_tensor < clip_val) & (input_tensor > clip_valn)),
            grad_alpha * ctx.residual,
            grad_alpha,
        )
        grad_alpha = torch.where(input_tensor <= clip_valn, -grad_alpha, grad_alpha)
        grad_alpha = grad_alpha.sum().expand_as(clip_val)

        grad_alphan = grad_output.clone()
        grad_alphan = torch.where(input_tensor >= clip_val, -grad_alphan, grad_alphan)
        grad_alphan = torch.where(
            ((input_tensor < clip_val) & (input_tensor > clip_valn)),
            -grad_alphan * ctx.residual,
            grad_alphan,
        )
        grad_alphan = grad_alphan.sum().expand_as(clip_valn)
        ndim = float(sum(list(grad_output.shape)))
        grad_scale = math.sqrt(1 / ndim / 3.0)
        return (
            grad_input_tensor,
            grad_alpha * grad_scale,
            grad_alphan * grad_scale,
            None,
            None,
            None,
        )


class PACT2_Plus_STE(torch.autograd.Function):
    """
    2-sided CGPACT+direct gradient derivative:
    i.e. assuming total independent of clip_val and clip_valn
    one option is to use running mean for gradient
    """

    @staticmethod
    def forward(ctx, input_tensor, clip_val, clip_valn, num_bits, dequantize, inplace):
        ctx.save_for_backward(input_tensor, clip_val, clip_valn)
        if inplace:
            ctx.mark_dirty(input_tensor)
        scale, zero_point = asymmetric_linear_quantization_params(
            num_bits,
            clip_valn.data,
            clip_val.data,
            integral_zero_point=False,
            signed=False,
        )
        if isinstance(clip_val, torch.Tensor):
            output = torch.where(
                input_tensor > clip_val,
                torch.ones_like(input_tensor) * clip_val,
                input_tensor,
            )
            output = torch.where(
                output < clip_valn, torch.ones_like(input_tensor) * clip_valn, output
            )
        else:
            output = clamp(input_tensor, clip_valn.data, clip_val.data, inplace)
        output, ctx.residual = linear_quantize_residual(
            output, scale, zero_point, inplace
        )
        # save quantization residue for clip_val grad computation in backprop
        with torch.no_grad():
            n = 2**num_bits - 1
            ctx.residual.div_(n)

        if dequantize:
            output = linear_dequantize(output, scale, zero_point, inplace)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input_tensor, clip_val, clip_valn = ctx.saved_tensors
        grad_input_tensor = grad_output.clone()
        grad_input_tensor = torch.where(
            input_tensor <= clip_valn,
            torch.zeros_like(grad_input_tensor),
            grad_input_tensor,
        )
        grad_input_tensor = torch.where(
            input_tensor >= clip_val,
            torch.zeros_like(grad_input_tensor),
            grad_input_tensor,
        )

        grad_alpha = grad_output.clone()
        grad_alpha = torch.where(
            ((input_tensor < clip_val) & (input_tensor > clip_valn)),
            grad_alpha * ctx.residual,
            grad_alpha,
        )
        grad_alpha = torch.where(
            input_tensor <= clip_valn, torch.zeros_like(grad_alpha), grad_alpha
        )
        grad_alpha = grad_alpha.sum().expand_as(clip_val)

        grad_alphan = grad_output.clone()
        grad_alphan = torch.where(
            input_tensor >= clip_val, torch.zeros_like(grad_alphan), grad_alphan
        )
        grad_alphan = torch.where(
            ((input_tensor < clip_val) & (input_tensor > clip_valn)),
            (grad_alphan - grad_alphan * ctx.residual),
            grad_alphan,
        )
        grad_alphan = grad_alphan.sum().expand_as(clip_valn)

        return grad_input_tensor, grad_alpha, grad_alphan, None, None, None


class CGPACT2_perChannel_STE(torch.autograd.Function):
    """2-side CGPACT+per weight channel quantization"""

    @staticmethod
    def forward(ctx, input_tensor, clip_val, clip_valn, num_bits, dequantize, inplace):
        ctx.save_for_backward(input_tensor, clip_val, clip_valn)
        if inplace:
            ctx.mark_dirty(input_tensor)
        scale, zero_point = asymmetric_linear_quantization_params(
            num_bits,
            clip_valn.data,
            clip_val.data,
            integral_zero_point=False,
            signed=False,
        )
        if isinstance(clip_val, torch.Tensor):
            output = torch.where(
                input_tensor > clip_val,
                torch.ones_like(input_tensor) * clip_val,
                input_tensor,
            )
            output = torch.where(
                output < clip_valn, torch.ones_like(input_tensor) * clip_valn, output
            )
        else:
            output = clamp(input_tensor, clip_valn.data, clip_val.data, inplace)
        output, ctx.residual = linear_quantize_residual(
            output, scale, zero_point, inplace
        )
        # save quantization residue for clip_val grad computation in backprop
        with torch.no_grad():
            n = (2**num_bits - 1) / 2.0
            ctx.residual.div_(n).sub_(1.0)

        if dequantize:
            output = linear_dequantize(output, scale, zero_point, inplace)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input_tensor, clip_val, clip_valn = ctx.saved_tensors
        grad_input_tensor = grad_output.clone()
        grad_input_tensor = torch.where(
            input_tensor <= clip_valn,
            torch.zeros_like(grad_input_tensor),
            grad_input_tensor,
        )
        grad_input_tensor = torch.where(
            input_tensor >= clip_val,
            torch.zeros_like(grad_input_tensor),
            grad_input_tensor,
        )

        grad_alpha = grad_output.clone()
        grad_alpha = torch.where(
            ((input_tensor < clip_val) & (input_tensor > clip_valn)),
            grad_alpha * ctx.residual,
            grad_alpha,
        )
        grad_alpha = torch.where(input_tensor <= clip_valn, -grad_alpha, grad_alpha)
        grad_alpha = grad_alpha.sum((0, 2, 3), keepdim=True)

        grad_alphan = grad_output.clone()
        grad_alphan = torch.where(input_tensor >= clip_val, -grad_alphan, grad_alphan)
        grad_alphan = torch.where(
            ((input_tensor < clip_val) & (input_tensor > clip_valn)),
            -grad_alphan * ctx.residual,
            grad_alphan,
        )
        grad_alphan = grad_alphan.sum((0, 2, 3), keepdim=True)
        # Straight-through estimator for the scale factor calculation
        return grad_input_tensor, grad_alpha, grad_alphan, None, None, None


# Double-sided PACT+, simplified version, no LUT support
class PACTplus2STE(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        input_tensor,
        clip_val,
        clip_valn,
        num_bits,
        dequantize,
        inplace,
        intg_zp,
        use_PT_native_Qfunc,
    ):
        clip_val, clip_valn = (
            clip_val.to(input_tensor.dtype),
            clip_valn.to(input_tensor.dtype),
        )
        n_levels = 2**num_bits - 1
        scale = n_levels / (clip_val - clip_valn)
        stepsize = 1.0 / scale
        zp = clip_valn * scale
        # --- TODO, move this to outer loop and use load_state_dict method
        # (as used by clip_val_asst) to avoid slow-down ---
        if intg_zp >= 1:  # adjust to preserve 0 in the quantized y
            # align zero only, no change in stepsize
            rzp = torch.round(zp)
            # deltastep = (torch.round(zp - rzp) + rzp) * stepsize - clip_valn
            zp = rzp
            if intg_zp == 2:  # align upper bound
                stepsize = clip_val / (n_levels + zp)
                clip_valn = (stepsize * zp).to(input_tensor.device)
            elif intg_zp == 3:  # align lower bound
                stepsize = clip_valn / zp
                clip_val = (stepsize * (n_levels + zp)).to(input_tensor.device)

        ctx.save_for_backward(input_tensor, clip_val, clip_valn)
        if inplace:
            ctx.mark_dirty(input_tensor)
        ctx.n_levels = n_levels
        ctx.n_bits = num_bits
        ctx.stepsize = stepsize

        if torch.is_nonzero(zp):  # was torch.any(zp != 0):
            # given the zp formula we use above, zp should always be non-zeros...
            quant_min, quant_max = 0, 2**num_bits - 1  # eg (0, 255) or (0,15)
            int_dtype = torch.uint8
        else:
            quant_min, quant_max = (
                -(2 ** (num_bits - 1)),
                2 ** (num_bits - 1) - 1,
            )  # eg (-128,127) or (-8,7)
            int_dtype = torch.int8

        if use_PT_native_Qfunc:
            qint_dtype_dict = {
                (32, True): torch.qint32,
                (8, True): torch.qint8,
                (8, False): torch.quint8,
                (4, True): torch.qint8,
                (4, False): torch.quint8,
            }
            # Note: quantized_tensor.int_repr() returns uint8
            int_dtype_dict = {
                (torch.qint32): torch.int32,
                (torch.qint8): torch.uint8,
                (torch.quint8): torch.uint8,
            }

            n_levels = 2**num_bits - 1
            scale = (clip_val - clip_valn) / n_levels
            zp = torch.round(-clip_valn / scale).to(torch.int)
            signed = not torch.is_nonzero(zp)  # was (zp==0).item()
            qint_dtype = qint_dtype_dict.get((num_bits, signed))
            int_dtype = int_dtype_dict.get(qint_dtype)
            input_tensor_dtype = (
                input_tensor.dtype
            )  # fake_quantize_per_xxx doesn't support fp16 out of the box

            if dequantize:
                out = torch.fake_quantize_per_tensor_affine(
                    input_tensor.float(), scale.float(), zp, quant_min, quant_max
                )
                out = out.to(input_tensor_dtype)
            else:
                # Clamp to [quant_min, quant_max] in case we are storing quint4 into a uint8 tensor
                out = (
                    torch.quantize_per_tensor(
                        input_tensor, scale.float(), zp, qint_dtype
                    )
                    .int_repr()
                    .clamp(quant_min, quant_max)
                )
            return out  # do not cast back to input_tensor_dtype!

        if isinstance(clip_val, torch.Tensor):
            input_tensor = torch.minimum(input_tensor, clip_val)
            input_tensor = torch.maximum(input_tensor, clip_valn)
            output = input_tensor
        else:
            output = clamp(input_tensor, clip_valn.data, clip_val.data, inplace)

        output = torch.round(output / stepsize.to(output.device) - zp.to(output.device))
        if dequantize:
            output = (output + zp.to(output.device)) * stepsize.to(output.device)
        else:
            output = output.clamp(quant_min, quant_max).to(int_dtype)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input_tensor, clip_val, clip_valn = ctx.saved_tensors
        n_levels = ctx.n_levels
        stepsize = ctx.stepsize
        z = (input_tensor - clip_valn) / stepsize
        delz = (z - torch.round(z)) / n_levels

        grad_input_tensor = grad_output.clone()
        grad_input_tensor = torch.where(
            input_tensor <= clip_valn,
            torch.zeros_like(grad_input_tensor),
            grad_input_tensor,
        )
        grad_input_tensor = torch.where(
            input_tensor >= clip_val,
            torch.zeros_like(grad_input_tensor),
            grad_input_tensor,
        )

        grad_alpha = -grad_output.clone() * delz
        grad_alphan = -grad_alpha
        grad_alpha = torch.where(
            input_tensor <= clip_valn, torch.zeros_like(grad_alpha), grad_alpha
        )
        grad_alpha = torch.where(input_tensor >= clip_val, grad_output, grad_alpha)
        grad_alphan = torch.where(
            input_tensor >= clip_val, torch.zeros_like(grad_alpha), grad_alphan
        )
        grad_alphan = torch.where(input_tensor <= clip_valn, grad_output, grad_alphan)

        grad_alpha = grad_alpha.sum().expand_as(clip_val)
        grad_alphan = grad_alphan.sum().expand_as(clip_valn)

        # Straight-through estimator for the scale factor calculation
        return (
            grad_input_tensor,
            grad_alpha,
            grad_alphan,
            None,
            None,
            None,
            None,
            None,
            None,
        )


####################### Fixed clip quantizer ###################################


class QFixSymmetric(nn.Module):
    def __init__(self, num_bits, init_clip_val=None, align_zero=False):
        """Quantization class with 2-sided FIXED, SYMMETRIC clip values (no gradient on clip)"""
        super().__init__()
        self.num_bits = num_bits
        self.align_zero = align_zero
        if self.align_zero:
            self.quantizer = QFixSymmetricZeroSTE  # align zero by removing one level
        else:
            self.quantizer = QFixSymmetricSTE
        self.clip_val = nn.Parameter(torch.Tensor([init_clip_val]), requires_grad=False)

    def forward(self, input_tensor):
        input_tensor = self.quantizer.apply(input_tensor, self.num_bits, self.clip_val)
        return input_tensor

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(num_bits={self.num_bits},"
            f"clip_val={self.clip_val[0]:.2f}, quantizer={self.quantizer})"
        )


class QFixSymmetricSTE(torch.autograd.Function):
    """Linear quantization with Straight Through Estimator on input_tensor tensor
    Clips are fixed and symmetric, without gradient
    """

    @staticmethod
    def forward(ctx, input_tensor, num_bits, clip_val):
        n_gaps = 2**num_bits - 1
        scale = n_gaps / (2 * clip_val)
        zero_point = -clip_val * scale

        output = torch.where(
            input_tensor > clip_val,
            torch.ones_like(input_tensor) * clip_val,
            input_tensor,
        )
        output_clipped = torch.where(
            output < -clip_val, -torch.ones_like(input_tensor) * clip_val, output
        )

        # quantize and dequantize
        output_q = (
            torch.round(scale * output_clipped - zero_point) + zero_point
        ) / scale

        ctx.mark_non_differentiable(clip_val)
        return output_q

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None, None


class QFixSymmetricZeroSTE(torch.autograd.Function):
    """Linear quantization with Straight Through Estimator on input_tensor tensor
    Clips are fixed and symmetric, without gradient
    Zero is aligned by using one less level
    """

    @staticmethod
    def forward(ctx, input_tensor, num_bits, clip_val):
        n_gaps = 2**num_bits - 2  # drop one of the available levels
        scale = n_gaps / (2 * clip_val)
        zero_point = -clip_val * scale

        output = torch.where(
            input_tensor > clip_val,
            torch.ones_like(input_tensor) * clip_val,
            input_tensor,
        )
        output_clipped = torch.where(
            output < -clip_val, -torch.ones_like(input_tensor) * clip_val, output
        )

        # quantize and dequantize
        output_q = (
            torch.round(scale * output_clipped - zero_point) + zero_point
        ) / scale

        ctx.mark_non_differentiable(clip_val)
        return output_q

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None, None


########################1-side LSQ #############################################
class LSQQuantization(nn.Module):
    def __init__(self, num_bits, init_clip_val, dequantize=True, inplace=False):
        """1-sided LSQ quantization module"""
        super().__init__()
        self.num_bits = num_bits
        if isinstance(init_clip_val, torch.Tensor):
            self.clip_val = nn.Parameter(init_clip_val)
        else:
            self.clip_val = nn.Parameter(torch.Tensor([init_clip_val]))
        self.dequantize = dequantize
        self.inplace = inplace

    def forward(self, input_tensor):
        input_tensor = LSQQuantizationSTE.apply(
            input_tensor, self.clip_val, self.num_bits, self.dequantize, self.inplace
        )
        return input_tensor

    def __repr__(self):
        inplace_str = ", inplace" if self.inplace else ""
        return (
            f"{self.__class__.__name__}(num_bits={self.num_bits}, clip_val={self.clip_val[0]}, "
            f"cggrad={inplace_str})"
        )


class LSQQuantizationSTE(torch.autograd.Function):
    """1-sided LSQ quantization STE"""

    @staticmethod
    def forward(ctx, input_tensor, clip_val, num_bits, dequantize, inplace):
        if inplace:
            ctx.mark_dirty(input_tensor)
        scale, zero_point = asymmetric_linear_quantization_params(
            num_bits, 0, clip_val.data, signed=False
        )
        if isinstance(clip_val, torch.Tensor):
            if input_tensor.min() < 0:
                raise ValueError(
                    "FMS: input_tensor to ClippedLinearQuantization should be non-negative."
                )
            output = torch.where(
                input_tensor > clip_val,
                torch.ones_like(input_tensor) * clip_val,
                input_tensor,
            )
        else:
            output = clamp(input_tensor, 0, clip_val.data, inplace)
        output, residual = linear_quantize_LSQresidual(
            output, scale, zero_point, inplace
        )
        with torch.no_grad():
            n = 2**num_bits - 1
            residual /= n

        if dequantize:
            output = linear_dequantize(output, scale, zero_point, inplace)

        ctx.save_for_backward(input_tensor, clip_val, residual)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input_tensor, clip_val, residual = ctx.saved_tensors
        grad_input_tensor = grad_output.clone()
        grad_input_tensor = torch.where(
            input_tensor < 0, torch.zeros_like(grad_input_tensor), grad_input_tensor
        )
        grad_input_tensor = torch.where(
            input_tensor > clip_val,
            torch.zeros_like(grad_input_tensor),
            grad_input_tensor,
        )

        grad_alpha = grad_output.clone()
        grad_alpha = torch.where(
            input_tensor < clip_val, grad_alpha * residual, grad_alpha
        )
        grad_alpha = grad_alpha.sum().expand_as(clip_val)
        ndim = float(sum(list(grad_output.shape)))
        grad_scale = math.sqrt(1 / ndim / 3.0)  # 3 is for 2bit, should be n
        return grad_input_tensor, grad_alpha * grad_scale, None, None, None


########################LSQ+ #############################################
class LSQPlus(nn.Module):
    def __init__(
        self, num_bits, init_clip_vals, init_clip_valb, dequantize=True, inplace=False
    ):
        super().__init__()
        self.init_clip_vals = init_clip_vals
        self.init_clip_valb = init_clip_valb
        self.clip_vals = nn.Parameter(torch.tensor(self.init_clip_vals))
        self.clip_valb = nn.Parameter(torch.tensor(self.init_clip_valb))
        self.counter = nn.Parameter(torch.zeros_like(self.clip_vals.data))
        self.dequantize = dequantize
        self.inplace = inplace
        self.num_bits = num_bits

    # NOTE: no dequantize here
    def forward(self, input_tensor):
        with torch.no_grad():
            if self.counter == 0:
                xmin = input_tensor.min().item()
                self.clip_vals.data = (
                    torch.ones_like(self.clip_vals.data)
                    * (input_tensor.max().item() - xmin)
                    / (2.0**self.num_bits - 1)
                )
                self.clip_valb.data = torch.ones_like(self.clip_valb.data) * (
                    xmin + 2.0 ** (self.num_bits - 1) * self.clip_vals.data.item()
                )
                self.counter += 1
        input_tensor = LSQPlus_func.apply(
            input_tensor,
            self.clip_vals,
            self.clip_valb,
            self.num_bits,
            self.dequantize,
            self.inplace,
        )
        return input_tensor


class LSQPlus_func(torch.autograd.Function):
    """2-side LSQ+ from CVPR workshop paper"""

    @staticmethod
    def forward(
        ctx, input_tensor, clip_vals, clip_valb, num_bits, _dequantize, _inplace
    ):
        ctx.save_for_backward(input_tensor, clip_vals.clone(), clip_valb.clone())
        clip_val = 2.0 ** (num_bits - 1) - 1
        clip_valn = -(2.0 ** (num_bits - 1))

        output = (input_tensor - clip_valb) / clip_vals
        # output = output.clamp(clip_valn, clip_val)
        output = torch.where(
            output > clip_val, torch.ones_like(output) * clip_val, output
        )
        output = torch.where(
            output < clip_valn, torch.ones_like(output) * clip_valn, output
        )
        rounded = output.round()
        ctx.residual = rounded - output

        with torch.no_grad():
            ctx.clip_val = clip_val * clip_vals + clip_valb
            ctx.clip_valn = clip_valn * clip_vals + clip_valb
            ctx.num_bits = num_bits

        output = rounded * clip_vals + clip_valb
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input_tensor, clip_vals, clip_valb = ctx.saved_tensors
        p = 2.0 ** (ctx.num_bits - 1) - 1
        n = -(2.0 ** (ctx.num_bits - 1))
        grad_input_tensor = grad_output.clone()
        grad_input_tensor = torch.where(
            input_tensor <= ctx.clip_valn,
            torch.zeros_like(grad_input_tensor),
            grad_input_tensor,
        )
        grad_input_tensor = torch.where(
            input_tensor >= ctx.clip_val,
            torch.zeros_like(grad_input_tensor),
            grad_input_tensor,
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

        return grad_input_tensor, grad_s, grad_beta, None, None, None


###################silu quantization############


class QSILU(nn.Module):
    def __init__(
        self,
        num_bits,
        init_clip_valn,
        init_clip_val,
        dequantize=True,
        inplace=False,
        cggrad=False,
        grad_scale=False,
    ):
        """For silu activations
        saturate the negative values to -0.28746
        """
        super().__init__()
        self.num_bits = num_bits

        if isinstance(init_clip_val, torch.Tensor) and isinstance(
            init_clip_valn, torch.Tensor
        ):
            self.clip_val = nn.Parameter(init_clip_val)
            self.clip_valn_cont = init_clip_valn
        elif not isinstance(init_clip_val, torch.Tensor) and not isinstance(
            init_clip_valn, torch.Tensor
        ):
            self.clip_val = nn.Parameter(torch.Tensor([init_clip_val]))
            self.clip_valn_const = torch.Tensor([init_clip_valn])
        else:
            raise ValueError(
                "FMS: init_clip_val and init_clip_valn in LearnedTwosidedClippedLinearQuantization "
                "should be the same instance type."
            )
        self.dequantize = dequantize
        self.inplace = inplace
        self.cggrad = cggrad
        self.grad_scale = grad_scale
        if self.cggrad:
            self.quantizer = CGSiluSTE
        elif self.grad_scale:
            self.quantizer = CGsiluGradScaleSTE
        else:
            self.quantizer = SiluSTE

    def forward(self, input_tensor):
        input_tensor = self.quantizer.apply(
            input_tensor,
            self.clip_val,
            self.clip_valn_const,
            self.num_bits,
            self.dequantize,
            self.inplace,
        )
        return input_tensor

    def __repr__(self):
        clip_str = (
            f", pos-clip={self.clip_val[0]}, neg-clip={self.clip_valn_const[0]}, "
            f"cggrad={self.cggrad}"
        )
        inplace_str = ", inplace" if self.inplace else ""
        return f"{self.__class__.__name__}(num_bits={self.num_bits}{clip_str}{inplace_str})"


##############Dorefa############
# TODO: specify inplace = True or False
def dorefa_quantize_activation(out, num_bits):
    clip_val = 1
    scale, zero_point = asymmetric_linear_quantization_params(
        num_bits, 0, clip_val, signed=False
    )
    out = clamp(out, 0, clip_val)
    out = LinearQuantizeSTE.apply(
        out, scale, zero_point, dequantize=True, inplace=False
    )
    return out


def dorefa_quantize_param(out, num_bits):
    scale, zero_point = asymmetric_linear_quantization_params(
        num_bits, 0, 1, signed=False
    )
    out = out.tanh()
    out = out / (2 * out.abs().max()) + 0.5
    out = LinearQuantizeSTE.apply(
        out, scale, zero_point, dequantize=True, inplace=False
    )
    out = 2 * out - 1
    return out


#############linear quantizer/dequantizer############
def _prep_saturation_val_tensor(sat_val):
    is_scalar = not isinstance(sat_val, torch.Tensor)
    if is_scalar:
        out = torch.tensor(sat_val)
    else:
        out = sat_val
    if not out.is_floating_point():
        out = out.to(torch.float32)
    if out.dim() == 0:
        out = out.unsqueeze(0)
    return is_scalar, out


def symmetric_linear_quantization_params(num_bits, saturation_val):
    is_scalar, sat_val = _prep_saturation_val_tensor(saturation_val)
    if any(sat_val < 0):
        raise ValueError("Saturation value must be >= 0")
    # Leave one bit for sign
    n = 2 ** (num_bits - 1) - 1
    # If float values are all 0, we just want the quantized values to be 0 as well.
    # So overriding the saturation value to 'n', so the scale becomes 1
    sat_val[sat_val == 0] = n
    scale = n / sat_val
    zero_point = torch.zeros_like(scale)
    if is_scalar:
        # If input_tensor was scalar, return scalars
        return scale.item(), zero_point.item()
    return scale, zero_point


def asymmetric_linear_quantization_params(
    num_bits, saturation_min, saturation_max, integral_zero_point=True, signed=False
):
    with torch.no_grad():
        scalar_min, sat_min = _prep_saturation_val_tensor(saturation_min)
        scalar_max, sat_max = _prep_saturation_val_tensor(saturation_max)
        is_scalar = scalar_min and scalar_max

        if scalar_max and not scalar_min:
            sat_max = sat_max.to(sat_min.device)
        elif scalar_min and not scalar_max:
            sat_min = sat_min.to(sat_max.device)

        n = 2**num_bits - 1
        # Make sure 0 is in the range
        sat_min = torch.min(sat_min, torch.zeros_like(sat_min))
        sat_max = torch.max(sat_max, torch.zeros_like(sat_max))
        if sat_min.device != sat_max.device:
            sat_min = sat_min.to(sat_max.device)
        diff = sat_max - sat_min
        # If float values are all 0, we just want the quantized values to be 0 as well.
        # So overriding the saturation value to 'n', so the scale becomes 1
        diff[diff == 0] = n

        scale = n / diff
        zero_point = scale * sat_min
        if integral_zero_point:
            zero_point = zero_point.round()
        if signed:
            zero_point += 2 ** (num_bits - 1)
        if is_scalar:
            return scale.item(), zero_point.item()
        return scale, zero_point


def clamp(input_tensor: torch.FloatTensor, clamp_min, clamp_max, inplace=False):
    """
    Returns:
        Clamped Torch Tensor.
    """
    if inplace:
        input_tensor.clamp_(clamp_min, clamp_max)
        return input_tensor
    return torch.clamp(input_tensor, clamp_min, clamp_max)


def linear_quantize(input_tensor, scale, zero_point, inplace=False):
    """
    Linearly discretize input_tensor (Tensor or scalar)

    NOTE for PyTorch-native equivalency: because of rounding, the following
    computations do NOT produce identical outputs in all scenarios
    OPTION 1:
        inv_scale = 1 / scale.to(input_tensor.device)
        return torch.round(input_tensor / inv_scale) - zero_point.to(input_tensor.device)
    OPTION 2:
        return torch.round(input_tensor * scale.to(input_tensor.device)) -
            zero_point.to(input_tensor.device)
    OPTION 3 [in use]:
        return torch.round(input_tensor * scale.to(input_tensor.device) -
            zero_point.to(input_tensor.device))
    """
    if inplace:
        input_tensor.mul_(scale).sub_(zero_point).round_()
        return input_tensor
    if isinstance(scale, torch.Tensor):
        return torch.round(
            scale.to(input_tensor.device) * input_tensor
            - zero_point.to(input_tensor.device)
        )
    return torch.round(scale * input_tensor - zero_point)


def linear_quantize_residual(input_tensor, scale, zero_point, _inplace=False):
    if isinstance(scale, torch.Tensor):
        unrounded = scale.to(input_tensor.device) * input_tensor - zero_point.to(
            input_tensor.device
        )
    else:
        unrounded = scale * input_tensor - zero_point
    rounded = torch.round(unrounded)
    with torch.no_grad():
        residual = torch.round(unrounded)
    return rounded, residual


def linear_quantize_LSQresidual(input_tensor, scale, zero_point, _inplace=False):
    if isinstance(scale, torch.Tensor):
        unrounded = scale.to(input_tensor.device) * input_tensor - zero_point.to(
            input_tensor.device
        )
    else:
        unrounded = scale * input_tensor - zero_point
    rounded = torch.round(unrounded)
    with torch.no_grad():
        residual = rounded - unrounded
    return rounded, residual


# TODO: This function is not used anywhere. To be removed.
def linear_quantize_clamp(
    input_tensor, scale, zero_point, clamp_min, clamp_max, inplace=False
):
    output = linear_quantize(input_tensor, scale, zero_point, inplace)
    return clamp(output, clamp_min, clamp_max, inplace)


def linear_dequantize(input_tensor, scale, zero_point, inplace=False):
    """
    TODO: Summarize
    """
    if inplace:
        input_tensor.add_(zero_point).div_(scale)
        return input_tensor
    if isinstance(scale, torch.Tensor):
        return (input_tensor + zero_point.to(input_tensor.device)) / scale.to(
            input_tensor.device
        )  # HACK for PACT
    return (input_tensor + zero_point) / scale


# this _zp used to do [-8, +8] then saturate +8 to +7 so that the final is [-8,7]
def linear_quantize_zp(input_tensor, scale, zero_point, num_bits, inplace=False):
    """
    TODO: Summarize
    """
    Qp = 2 ** (num_bits - 1) - 1
    if inplace:
        input_tensor.mul_(scale).sub_(zero_point).round_()
        return input_tensor
    if isinstance(scale, torch.Tensor):
        out = torch.round(
            scale.to(input_tensor.device) * input_tensor
            - zero_point.to(input_tensor.device)
        )  # HACK for PACT
        out = torch.where(out > Qp, torch.ones_like(out) * Qp, out)
        return out

    out = torch.round(scale * input_tensor - zero_point)
    out = torch.where(out > Qp, torch.ones_like(out) * Qp, out)
    return out


def linear_dequantize_zp(input_tensor, scale, zero_point, inplace=False):
    """
    TODO: Summarize
    """
    if inplace:
        input_tensor.add_(zero_point).div_(scale)
        return input_tensor
    if isinstance(scale, torch.Tensor):
        return (input_tensor + zero_point.to(input_tensor.device)) / scale.to(
            input_tensor.device
        )  # HACK for PACT
    return (input_tensor + zero_point) / scale


class LinearQuantizeSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_tensor, scale, zero_point, dequantize, inplace):
        if inplace:
            ctx.mark_dirty(input_tensor)
        output = linear_quantize(input_tensor, scale, zero_point, inplace)
        if dequantize:
            output = linear_dequantize(output, scale, zero_point, inplace)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        # Straight-through estimator
        return grad_output, None, None, None, None


class ZPLinearQuantizeSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_tensor, scale, zero_point, num_bits, dequantize, inplace):
        if inplace:
            ctx.mark_dirty(input_tensor)
        output = linear_quantize_zp(input_tensor, scale, zero_point, num_bits, inplace)
        if dequantize:
            output = linear_dequantize_zp(output, scale, zero_point, inplace)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None, None, None


class SiluSTE(torch.autograd.Function):
    """two-sided original pact quantization for activation"""

    @staticmethod
    def forward(ctx, input_tensor, clip_val, clip_valn, num_bits, dequantize, inplace):
        if clip_val.device != clip_valn.device:
            clip_valn = clip_valn.to(clip_val.device)
        ctx.save_for_backward(input_tensor, clip_val, clip_valn)
        if inplace:
            ctx.mark_dirty(input_tensor)
        scale, zero_point = asymmetric_linear_quantization_params(
            num_bits,
            clip_valn.data,
            clip_val.data,
            integral_zero_point=False,
            signed=False,
        )
        if isinstance(clip_val, torch.Tensor):
            output = torch.where(
                input_tensor > clip_val,
                torch.ones_like(input_tensor) * clip_val,
                input_tensor,
            )
            output = torch.where(
                output < clip_valn, torch.ones_like(input_tensor) * clip_valn, output
            )
        else:
            output = clamp(input_tensor, clip_valn.data, clip_val.data, inplace)
        output = linear_quantize(output, scale, zero_point, inplace)
        if dequantize:
            output = linear_dequantize(output, scale, zero_point, inplace)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input_tensor, clip_val, clip_valn = ctx.saved_tensors
        grad_input_tensor = grad_output.clone()
        grad_input_tensor = torch.where(
            input_tensor <= clip_valn,
            torch.zeros_like(grad_input_tensor),
            grad_input_tensor,
        )
        grad_input_tensor = torch.where(
            input_tensor >= clip_val,
            torch.zeros_like(grad_input_tensor),
            grad_input_tensor,
        )

        grad_alpha = grad_output.clone()
        grad_alpha = torch.where(
            input_tensor < clip_val, torch.zeros_like(grad_alpha), grad_alpha
        )
        grad_alpha = grad_alpha.sum().expand_as(clip_val)
        return grad_input_tensor, grad_alpha, None, None, None, None


class CGSiluSTE(torch.autograd.Function):
    """2-sided CGsilu"""

    @staticmethod
    def forward(ctx, input_tensor, clip_val, clip_valn, num_bits, dequantize, inplace):
        ctx.save_for_backward(input_tensor, clip_val, clip_valn)
        if inplace:
            ctx.mark_dirty(input_tensor)
        scale, zero_point = asymmetric_linear_quantization_params(
            num_bits,
            clip_valn.data,
            clip_val.data,
            integral_zero_point=False,
            signed=False,
        )
        if isinstance(clip_val, torch.Tensor):
            output = torch.where(
                input_tensor > clip_val,
                torch.ones_like(input_tensor) * clip_val,
                input_tensor,
            )
            output = torch.where(
                output < clip_valn, torch.ones_like(input_tensor) * clip_valn, output
            )
        else:
            output = clamp(input_tensor, clip_valn.data, clip_val.data, inplace)
        output, ctx.residual = linear_quantize_residual(
            output, scale, zero_point, inplace
        )
        with torch.no_grad():
            n = (2**num_bits - 1) / 2.0
            ctx.residual.div_(n).sub_(1.0)

        if dequantize:
            output = linear_dequantize(output, scale, zero_point, inplace)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input_tensor, clip_val, clip_valn = ctx.saved_tensors
        grad_input_tensor = grad_output.clone()
        grad_input_tensor = torch.where(
            input_tensor <= clip_valn,
            torch.zeros_like(grad_input_tensor),
            grad_input_tensor,
        )
        grad_input_tensor = torch.where(
            input_tensor >= clip_val,
            torch.zeros_like(grad_input_tensor),
            grad_input_tensor,
        )

        grad_alpha = grad_output.clone()
        grad_alpha = torch.where(
            ((input_tensor < clip_val) & (input_tensor > clip_valn)),
            grad_alpha * ctx.residual,
            grad_alpha,
        )
        grad_alpha = torch.where(input_tensor <= clip_valn, -grad_alpha, grad_alpha)
        grad_alpha = grad_alpha.sum().expand_as(clip_val)
        return grad_input_tensor, grad_alpha, None, None, None, None


class CGsiluGradScaleSTE(torch.autograd.Function):
    """2-sided silu+ scale alpha gradients"""

    @staticmethod
    def forward(ctx, input_tensor, clip_val, clip_valn, num_bits, dequantize, inplace):
        ctx.save_for_backward(input_tensor, clip_val, clip_valn)
        if inplace:
            ctx.mark_dirty(input_tensor)
        scale, zero_point = asymmetric_linear_quantization_params(
            num_bits,
            clip_valn.data,
            clip_val.data,
            integral_zero_point=False,
            signed=False,
        )
        if isinstance(clip_val, torch.Tensor):
            output = torch.where(
                input_tensor > clip_val,
                torch.ones_like(input_tensor) * clip_val,
                input_tensor,
            )
            output = torch.where(
                output < clip_valn, torch.ones_like(input_tensor) * clip_valn, output
            )
        else:
            output = clamp(input_tensor, clip_valn.data, clip_val.data, inplace)
        output, ctx.residual = linear_quantize_residual(
            output, scale, zero_point, inplace
        )
        with torch.no_grad():
            n = (2**num_bits - 1) / 2.0
            ctx.residual.div_(n).sub_(1.0)

        if dequantize:
            output = linear_dequantize(output, scale, zero_point, inplace)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input_tensor, clip_val, clip_valn = ctx.saved_tensors
        grad_input_tensor = grad_output.clone()
        grad_input_tensor = torch.where(
            input_tensor <= clip_valn,
            torch.zeros_like(grad_input_tensor),
            grad_input_tensor,
        )
        grad_input_tensor = torch.where(
            input_tensor >= clip_val,
            torch.zeros_like(grad_input_tensor),
            grad_input_tensor,
        )
        grad_alpha = grad_output.clone()
        grad_alpha = torch.where(
            ((input_tensor < clip_val) & (input_tensor > clip_valn)),
            grad_alpha * ctx.residual,
            grad_alpha,
        )
        grad_alpha = torch.where(input_tensor <= clip_valn, -grad_alpha, grad_alpha)
        grad_alpha = grad_alpha.sum().expand_as(clip_val)

        ndim = float(sum(list(grad_output.shape)))
        grad_scale = math.sqrt(1 / ndim / 3.0)
        return grad_input_tensor, grad_alpha * grad_scale, None, None, None, None


class SiluQuantization(nn.Module):
    def __init__(
        self,
        num_bits,
        init_clip_valn,
        init_clip_val,
        dequantize=True,
        inplace=False,
        cggrad=False,
        grad_scale=False,
    ):
        """two-sided original PACT"""
        super().__init__()
        self.num_bits = num_bits

        if isinstance(init_clip_val, torch.Tensor) and isinstance(
            init_clip_valn, torch.Tensor
        ):
            self.clip_val = nn.Parameter(init_clip_val)
            self.clip_valn_cont = init_clip_valn
        elif not isinstance(init_clip_val, torch.Tensor) and not isinstance(
            init_clip_valn, torch.Tensor
        ):
            self.clip_val = nn.Parameter(torch.Tensor([init_clip_val]))
            self.clip_valn_const = torch.Tensor([init_clip_valn])
        else:
            raise ValueError(
                "FMS: init_clip_val and init_clip_valn in LearnedTwosidedClippedLinearQuantization "
                "should be the same instance type."
            )
        self.dequantize = dequantize
        self.inplace = inplace
        self.cggrad = cggrad
        self.grad_scale = grad_scale

    def forward(self, input_tensor):
        if self.cggrad:
            if self.grad_scale:
                input_tensor = CGsiluGradScaleSTE.apply(
                    input_tensor,
                    self.clip_val,
                    self.clip_valn_const,
                    self.num_bits,
                    self.dequantize,
                    self.inplace,
                )
            else:
                input_tensor = CGSiluSTE.apply(
                    input_tensor,
                    self.clip_val,
                    self.clip_valn_const,
                    self.num_bits,
                    self.dequantize,
                    self.inplace,
                )
        else:
            input_tensor = SiluSTE.apply(
                input_tensor,
                self.clip_val,
                self.clip_valn_const,
                self.num_bits,
                self.dequantize,
                self.inplace,
            )
        return input_tensor

    def __repr__(self):
        clip_str = (
            f", pos-clip={self.clip_val[0]}, neg-clip={self.clip_valn_const[0]}, "
            f"cggrad={self.cggrad}"
        )
        inplace_str = ", inplace" if self.inplace else ""
        return f"{self.__class__.__name__}(num_bits={self.num_bits}{clip_str}{inplace_str})"


## Pytorch LSTM used for debugging
def lstm_cell_q(qinput_tensor, qhidden, qw_ih, qw_hh, b_ih=None, b_hh=None):
    """Regular LSTM Cell running operations on quantized activations and weights"""

    qhx, cx = qhidden

    gates_i = F.linear(qinput_tensor, qw_ih, b_ih)  # pylint: disable=not-callable
    gates_h = F.linear(qhx, qw_hh, b_hh)  # pylint: disable=not-callable
    ii, fi, ci, oi = torch.split(
        gates_i, qhx.size(1), dim=1
    )  # ii = to input_tensor gate, from input_tensor; fi = to forget gate, from input_tensor; etc.
    ih, fh, ch, oh = torch.split(
        gates_h, qhx.size(1), dim=1
    )  # ih = to input_tensor gate, from hidden; fh = to forget gate, from hidden; etc.

    ingate, forgetgate, cellgate, outgate = ii + ih, fi + fh, ch + ci, oh + oi

    ingate = torch.sigmoid(ingate)
    forgetgate = torch.sigmoid(forgetgate)
    cellgate = torch.tanh(cellgate)
    outgate = torch.sigmoid(outgate)

    cy = (forgetgate * cx) + (ingate * cellgate)  # element-wise multiplication
    hy = outgate * torch.tanh(cy)

    return hy, cy


class Qmax(nn.Module):
    """Max with custom backward (gradient pass through for clip function)
    Qmax can quantize both weights and activations
    """

    def __init__(
        self,
        num_bits,
        dequantize=True,
        inplace=False,
        align_zero=False,
        clipSTE=True,
        minmax=False,
        perCh=None,
        perGp=None,
        extend_act_range=False,
    ):
        super().__init__()
        self.num_bits = num_bits
        self.dequantize = dequantize
        self.inplace = inplace
        self.align_zero = align_zero
        self.clipSTE = clipSTE
        self.minmax = minmax
        self.perCh = perCh
        self.extend_act_range = extend_act_range
        if minmax:
            self.quantizer_name = "Qminmax"
            if perGp:
                self.quantizer = QminmaxPerGpSTE
            else:
                self.quantizer = QminmaxPerChSTE if perCh else QminmaxSTE
        elif extend_act_range:
            self.quantizer = QmaxExtendRangeSTE
            self.quantizer_name = "QmaxExtend"
            self.clip_ratio = -(2 ** (self.num_bits - 1)) / (
                2 ** (self.num_bits - 1) - 1
            )
        else:
            self.quantizer_name = "Qmax"
            if perGp:
                self.quantizer = QmaxPerGpSTE
            else:
                self.quantizer = QmaxPerChSTE if perCh else QmaxSTE
        self.perGp = perGp
        self.movAvgFac = 0.1
        self.Niter = 0
        if self.perGp:
            self.register_buffer("clip_val", torch.zeros(perGp[0]))
            self.register_buffer("clip_valn", torch.zeros(perGp[0]))
        else:
            self.register_buffer(
                "clip_val", torch.zeros(perCh) if perCh else torch.Tensor([0.0])
            )
            self.register_buffer(
                "clip_valn", torch.zeros(perCh) if perCh else torch.Tensor([0.0])
            )

    def forward(self, input_tensor):
        if self.perCh:
            if self.minmax:
                clipval_new = torch.max(
                    input_tensor.reshape([self.perCh, -1]), dim=1
                ).values
                clipvaln_new = torch.min(
                    input_tensor.reshape([self.perCh, -1]), dim=1
                ).values
            else:
                clipval_new = torch.max(
                    input_tensor.abs().reshape([self.perCh, -1]), dim=1
                ).values
                clipvaln_new = -clipval_new
            assert (
                len(clipval_new) == input_tensor.shape[0]
            ), f"dimension error, input_tensor{input_tensor.shape}, clipval{clipval_new.shape}"
        elif self.perGp:
            if self.minmax:
                clipval_new = torch.max(input_tensor.reshape(self.perGp), dim=1).values
                clipvaln_new = torch.min(input_tensor.reshape(self.perGp), dim=1).values
            else:
                clipval_new = torch.max(
                    input_tensor.abs().reshape(self.perGp), dim=1
                ).values
                clipvaln_new = -clipval_new
            assert (
                len(clipval_new)
                == (input_tensor.shape[0] * input_tensor.shape[1] // self.perGp[1])
            ), f"dimension error, input_tensor{input_tensor.shape}, clipval{clipval_new.shape}"
        elif self.extend_act_range:
            if input_tensor.max() >= input_tensor.min().abs():
                clipval_new = input_tensor.max()
                clipvaln_new = clipval_new * self.clip_ratio
            else:
                clipvaln_new = input_tensor.min()
                clipval_new = clipvaln_new / self.clip_ratio
        else:
            if self.minmax:  # asymmetric
                clipval_new = input_tensor.max()
                clipvaln_new = input_tensor.min()
            else:  # symmetric
                clipval_new = input_tensor.abs().max()
                clipvaln_new = -clipval_new

        if len(clipval_new.shape) == 0:
            clipval_new = clipval_new.unsqueeze(dim=0)
        if len(clipvaln_new.shape) == 0:
            clipvaln_new = clipvaln_new.unsqueeze(dim=0)

        if self.Niter == 0 and self.training:
            # to avoid unintended bwd ops added to the graph, cause memory leak sometimes
            with torch.no_grad():
                self.clip_val.copy_(
                    clipval_new
                )  # similar to fill_(), will not change id(self.clip_val) but update the values
                self.clip_valn.copy_(clipvaln_new)

        if self.training:
            input_tensor = self.quantizer.apply(
                input_tensor,
                self.num_bits,
                self.dequantize,
                self.inplace,
                clipval_new,
                clipvaln_new,
                self.align_zero,
            )
            # to avoid unintended bwd ops added to the graph, cause memory leak sometimes
            with torch.no_grad():
                self.clip_val.copy_(
                    self.clip_val * (1.0 - self.movAvgFac)
                    + clipval_new * self.movAvgFac
                )
                if self.quantizer_name == "QmaxExtend":
                    self.clip_valn.copy_(self.clip_val * self.clip_ratio)
                else:
                    self.clip_valn.copy_(
                        self.clip_valn * (1.0 - self.movAvgFac)
                        + clipvaln_new * self.movAvgFac
                    )
        else:
            input_tensor = self.quantizer.apply(
                input_tensor,
                self.num_bits,
                self.dequantize,
                self.inplace,
                self.clip_val,
                self.clip_valn,
                self.align_zero,
            )

        self.Niter += 1
        return input_tensor

    def __repr__(self):
        inplace_str = ", inplace" if self.inplace else ""
        clip_valn_str = (
            f"{self.clip_valn[0]:.2f}, " if self.quantizer_name == "Qminmax" else ""
        )
        perCh_str = f"per channel of {self.perCh}" if self.perCh else ""
        perGp_str = f"per group of {self.perGp}" if self.perGp else ""
        return (
            f"{self.quantizer_name}(num_bits={self.num_bits}, clip_val={self.clip_val[0]:.2f}, "
            f"{clip_valn_str}"
            f"{perCh_str}"
            f"{perGp_str}"
            f"class={self.__class__.__name__}{inplace_str})"
        )


class QmaxSTE(torch.autograd.Function):
    """Max with zero alignment (symmetric)
    "dequantize=False" option is functional
    """

    @staticmethod
    def forward(ctx, input_tensor, num_bits, dequantize, inplace, cv, _cvn, align_zero):
        if inplace:
            ctx.mark_dirty(input_tensor)
        scale = (2**num_bits - 2) if align_zero else (2**num_bits - 1)
        zero_point = 0.0
        clip_val = cv.to(input_tensor.dtype)

        output = input_tensor.mul(1 / clip_val).clamp(-1, 1).mul(0.5).add(0.5)
        output = linear_quantize(output, scale, zero_point, inplace)
        if dequantize:
            output = linear_dequantize(output, scale, zero_point, inplace)
            output = (2 * output - 1) * clip_val
        else:
            if not align_zero:
                # NOTE: not *strictly* necessary but due to shifting output before linear_quantize
                # (dequantize without align zero needs different math)
                raise ValueError("Dequantize=False in QmaxSTE requires zero alignment")
            output -= scale / 2
        return output

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None, None, None, None, None


class QmaxPerChSTE(torch.autograd.Function):
    """Max with zero alignment (symmetric)
    "dequantize=False" option is functional
    """

    @staticmethod
    def forward(ctx, input_tensor, num_bits, dequantize, inplace, cv, _cvn, align_zero):
        if inplace:
            ctx.mark_dirty(input_tensor)
        scale = (2**num_bits - 2) if align_zero else (2**num_bits - 1)
        zero_point = 0.0
        # here use symmetric similar to sawbperCh code
        nspace = 2**num_bits - 2  # lose one level
        int_l = -(2 ** (num_bits - 1)) + 1
        int_u = -int_l  # symmetric
        # original SAWB assumes odd number of bins when calc clip_val
        scale = cv * 2 / nspace
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

        if not dequantize:
            return (output.t() / scale).t()
        return output

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None, None, None, None, None


class QmaxPerGpSTE(torch.autograd.Function):
    """Max with zero alignment (symmetric)
    per group quantization
    """

    @staticmethod
    def forward(
        ctx, input_tensor, num_bits, _dequantize, inplace, cv, _cvn, align_zero
    ):
        if inplace:
            ctx.mark_dirty(input_tensor)
        input_tensor_shape = input_tensor.shape
        clip_val_shape = cv.shape
        # use clip_val shape to reshape input_tensor
        input_tensor = input_tensor.reshape(clip_val_shape[0], -1)
        scale = (2**num_bits - 2) if align_zero else (2**num_bits - 1)
        zero_point = 0.0

        # here use symmetric similar to sawbperCh code
        nspace = 2**num_bits - 2  # lose one level
        int_l = -(2 ** (num_bits - 1)) + 1
        int_u = -int_l  # symmetric
        # original SAWB assumes odd number of bins when calc clip_val
        scale = cv * 2 / nspace
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
        output = output.reshape(input_tensor_shape)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None, None, None, None, None


class QminmaxSTE(torch.autograd.Function):
    """minmax with zero alignment (asymmetric)
    Dequantization always enabled (cannot be turned off, at this time)
    """

    @staticmethod
    def forward(ctx, input_tensor, num_bits, _dequantize, inplace, cv, cvn, align_zero):
        if inplace:
            ctx.mark_dirty(input_tensor)
        cv, cvn = cv.to(input_tensor.dtype), cvn.to(input_tensor.dtype)
        scale = (2**num_bits - 1) / (cv - cvn)
        zero_point = cvn * scale
        if align_zero:
            zero_point = torch.round(zero_point)

        output = (input_tensor.clamp(cvn.item(), cv.item()) - cvn) * scale
        output = (torch.round(output) + zero_point) / scale
        return output

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None, None, None, None, None


class QminmaxPerChSTE(torch.autograd.Function):
    """per channel minmax with zero alignment (asymmetric)"""

    @staticmethod
    def forward(ctx, input_tensor, num_bits, _dequantize, inplace, cv, cvn, align_zero):
        if inplace:
            ctx.mark_dirty(input_tensor)
        cv, cvn = cv.to(input_tensor.dtype), cvn.to(input_tensor.dtype)
        scale = (2**num_bits - 1) / (cv - cvn)
        zero_point = cvn * scale
        if align_zero:
            zero_point = torch.round(zero_point)
        output = (input_tensor.clamp(cvn[:, None], cv[:, None]) - cvn[:, None]) * scale[
            :, None
        ]
        output = (torch.round(output) + zero_point[:, None]) / scale[:, None]
        return output

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None, None, None, None, None


class QminmaxPerGpSTE(torch.autograd.Function):
    """per group minmax with zero alignment (asymmetric)"""

    @staticmethod
    def forward(ctx, input_tensor, num_bits, _dequantize, inplace, cv, cvn, align_zero):
        if inplace:
            ctx.mark_dirty(input_tensor)
        cv, cvn = cv.to(input_tensor.dtype), cvn.to(input_tensor.dtype)
        input_tensor_shape = input_tensor.shape
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
        output = output.reshape(input_tensor_shape)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None, None, None, None, None


class QmaxExtendRangeSTE(torch.autograd.Function):
    """2-sided Max quantizer using a single clip

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
        input_tensor,
        num_bits,
        dequantize,
        _inplace,
        clip_val,
        clip_valn,
        _align_zero,
    ):
        clip_val = clip_val.to(input_tensor.dtype)
        clip_valn = clip_valn.to(input_tensor.dtype)

        stepsize = clip_val / (2 ** (num_bits - 1) - 1)

        # clip input_tensor distribution on both sides
        output = torch.where(
            input_tensor > clip_val,
            torch.ones_like(input_tensor) * clip_val,
            input_tensor,
        )
        output = torch.where(
            output < clip_valn, torch.ones_like(input_tensor) * clip_valn, output
        )

        # quantize to range [-2^(b-1), 2^(b-1)-1]
        output = torch.round(output / stepsize)

        if dequantize:
            output = output * stepsize
        return output

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None, None, None, None, None


class QminmaxSTEnoclip(torch.autograd.Function):
    """
    minmax with zero alignment, i.e. asymmetric
    """

    @staticmethod
    def forward(ctx, input_tensor, nlevels, _dequantize, inplace, cv, cvn, align_zero):
        if inplace:
            ctx.mark_dirty(input_tensor)
        scale = nlevels / (cv - cvn)
        zero_point = cvn * scale
        if align_zero:
            zero_point = torch.round(zero_point)

        output = (input_tensor.clamp(cvn.item(), cv.item()) - cvn) * scale
        output = (torch.round(output) + zero_point) / scale
        return output

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None, None, None, None, None


# simply let PyTorch handle backward()
class QmaxSimple(nn.Module):
    def __init__(self, num_bits, align_zero=False, minmax=False):
        """
        Simple max quantizer, let PyTorch handles backward.
        minmax = True means save clipvaln and clipval (asymmetric case),
        otherwise use clipval only (symmetric)
        Details to be added..
        """
        super().__init__()
        self.num_bits = num_bits
        self.nlevels = 2**num_bits - 2 if not minmax and align_zero else 2**num_bits - 1
        self.align_zero = align_zero
        self.minmax = minmax  # False -> use  abs.max and symmetric
        self.movAvgFac = 0.1
        self.Niter = 0
        self.register_buffer("clip_val", torch.Tensor([0.0]))
        self.register_buffer("clip_valn", torch.Tensor([0.0]))

    def forward(self, input_tensor):
        if self.minmax:  # asymmetric
            clipval_new = input_tensor.max()
            clipvaln_new = input_tensor.min()
        else:  # symmetric
            clipval_new = input_tensor.abs().max()
            clipvaln_new = -clipval_new

        if self.Niter == 0:
            self.clip_val, self.clip_valn = clipval_new, clipvaln_new

        if self.training:
            self.clip_val = (
                self.clip_val * (1.0 - self.movAvgFac) + clipval_new * self.movAvgFac
            )
            self.clip_valn = (
                self.clip_valn * (1.0 - self.movAvgFac) + clipvaln_new * self.movAvgFac
            )
        else:  # use saved clipvals if not training
            clipval_new, clipvaln_new = self.clip_val, self.clip_valn

        scale = self.nlevels / (clipval_new - clipval_new)
        zero_point = clipvaln_new * scale if self.minmax else 0.0
        if self.align_zero:
            zero_point = torch.round(zero_point)

        input_tensor = (
            input_tensor.clamp(clipvaln_new.item(), clipval_new.item()) - clipvaln_new
        ) * scale
        input_tensor = (torch.round(input_tensor) + zero_point) / scale

        self.Niter += 1
        return input_tensor

    def __repr__(self):
        return f"{self.__class__.__name__}(num_bits={self.num_bits}, quantizer=)"


class PerTokenMax(nn.Module):
    def __init__(self, num_bits):
        """
        For per-token activation quantization using abs().max() as scale,
        Zero is aligned so that the levels are symmetric around zero (lossing one level)
        Since the token length is un-known before running, the quatnization is dynamic, meaning
        no trainable quantization scales and the scales are computed at run time.
        """
        super().__init__()
        self.num_bits = num_bits
        self.register_buffer("clip_val", torch.Tensor([0.0]))
        self.register_buffer("clip_valn", torch.Tensor([0.0]))

    def forward(self, input_tensor):
        self.clip_val = input_tensor.abs().max(dim=-1, keepdim=True)[0]
        self.clip_valn = -self.clip_val
        levels = 2 ** (self.num_bits - 1) - 1
        scales = self.clip_val.clamp(min=1e-5).div(levels)
        input_tensor.div_(scales).round_().mul_(scales)
        return input_tensor

    def __repr__(self):
        return f"{self.__class__.__name__}(num_bits={self.num_bits}, quantizer=)"


class Qdynamic(nn.Module):
    def __init__(
        self,
        num_bits,
        qcfg,
        _non_neg=False,
        align_zero=False,
        qmode="max",
        symmetric=False,
        quantizer2sync=None,
        **kwargs,
    ):
        """
        Dynamic quantizer, can be used for
        1. (slower but) possibly more accurate QAT,
        2. PTQ
        3. calibration
        ignore non_neg -> always observe both clipval on the positive side and clipvaln
        qmode = [max(minmax if non_neg==True) or percentile]
        will keep 3 clipvals:
        a) last clipval,
        b) a moving average (For calib, use larger factor so newer clips contribute more)
        c) a simple average of the clipvals
        """
        super().__init__()
        self.num_bits = num_bits
        self.symmetric = symmetric or qmode.endswith("sym")
        self.nlevels = 2**self.num_bits - 2 if self.symmetric else 2**self.num_bits - 1
        self.align_zero = align_zero
        self.qmode = qmode
        self.quantizer = QminmaxSTEnoclip
        self.per = qcfg.get("clip_val_asst_percentile", (0.1, 99.9))
        self.isPTQ = qcfg.get("ptq_calibration", 0) > 0
        self.cvs_in_Q2sync = []
        self.quantizer2sync = quantizer2sync
        if (
            quantizer2sync
        ):  # will automatically update this given quantizer after each fwd()
            if hasattr(quantizer2sync, "clip_val"):
                self.cvs_in_Q2sync.append("clip_val")
            if hasattr(quantizer2sync, "clip_valn"):
                self.cvs_in_Q2sync.append("clip_valn")
            if hasattr(quantizer2sync, "delta"):
                self.cvs_in_Q2sync.append("delta")
            if hasattr(quantizer2sync, "zero_point"):
                self.cvs_in_Q2sync.append("zero_point")

        self.movAvgFac = 0.1
        self.Niter = 0
        for cv_name, cvn_name in [
            ("clip_val", "clip_valn"),
            ("cv_last", "cvn_last"),
            ("cv_movAvg", "cvn_movAvg"),
        ]:
            self.register_buffer(
                cv_name, torch.Tensor([8.0])
            )  # conventional "clip_val" will be "simple avg", also keep track of last and movAvg
            self.register_buffer(cvn_name, torch.Tensor([-8.0]))
        self.register_buffer("delta", torch.Tensor([1.0]))
        self.register_buffer("zero_point", torch.Tensor([0.0]))

    def forward(self, input_tensor):
        with torch.no_grad():
            if self.qmode.startswith("percentile"):
                nelem = input_tensor.nelement()
                cv_new_candidate = (
                    input_tensor.reshape(1, -1)
                    .float()
                    .kthvalue(
                        round(self.per[1] * 0.01 * nelem)
                    )  # built-in 'round' returns int
                    .values.data[0]
                ).to(input_tensor.dtype)

                # conventionaly percentile is input_tensor as 99.9 (% is implied),
                # so we need *0.01 here
                lower_k = round(self.per[0] * 0.01 * nelem)
                cvn_new_candidate = (
                    input_tensor.reshape(1, -1).float().kthvalue(lower_k).values.data[0]
                    if lower_k > 0
                    else input_tensor.min()
                ).to(
                    input_tensor.dtype
                )  # for very small tensor, lower_k could be 0, kthvalue(0) will cause error

                if self.symmetric:
                    cv_new = max(cv_new_candidate, cvn_new_candidate.abs())
                    cvn_new = -cv_new
                else:
                    cv_new = cv_new_candidate
                    cvn_new = cvn_new_candidate
            elif (
                self.qmode == "sawb" and self.num_bits == 4
            ):  # only works for PACT+sym for weights
                cv_new, _ = sawb_params_code(self.num_bits, 403, input_tensor)
                cvn_new = -cv_new

            else:  # i.e., minmax
                cv_new = input_tensor.max()
                cvn_new = -cv_new if self.symmetric else input_tensor.min()

            if self.Niter == 0 and self.training:
                # to avoid unintended bwd ops added to the graph, cause memory leak sometimes
                with torch.no_grad():
                    # similar to fill_(), will not change id(self.clip_val) but update the values
                    self.clip_val = cv_new
                    self.clip_valn = cvn_new

            if self.training:  # or self.isPTQ
                self.cv_last = cv_new
                self.cvn_last = cvn_new
                self.cv_movAvg = (
                    self.cv_movAvg * (1.0 - self.movAvgFac) + cv_new * self.movAvgFac
                )
                self.cvn_movAvg = (
                    self.cvn_movAvg * (1.0 - self.movAvgFac) + cvn_new * self.movAvgFac
                )
                self.cv_simpleAvg = (self.clip_val * self.Niter + cv_new) / (
                    self.Niter + 1
                )
                self.cvn_simpleAvg = (self.clip_valn * self.Niter + cvn_new) / (
                    self.Niter + 1
                )
                self.clip_val = self.cv_simpleAvg
                self.clip_valn = self.cvn_simpleAvg
                self.delta = (cv_new - cvn_new) / self.nlevels
                self.zero_point = torch.round(-cvn_new / self.delta)
            else:
                # when inference, use last saved clipvals (default is simple avg,
                # but can choose others as well)
                cv_new = self.clip_val
                cvn_new = self.clip_val

            if self.symmetric:
                # if dist is symmetric around 0, levels = 2^n-2
                cv_new = max(
                    cv_new.abs(), cvn_new.abs()
                )  # in case sign of cv and cvn are accidentally swapped
                cvn_new = -cv_new

            if (
                self.quantizer2sync
            ):  # automatically fill the new cvs into the real quantizer
                for cvi in self.cvs_in_Q2sync:
                    getattr(self.quantizer2sync, cvi).fill_(getattr(self, cvi).item())

        # handle backward by STE
        output = self.quantizer.apply(
            input_tensor, self.nlevels, True, False, cv_new, cvn_new, self.align_zero
        )

        self.Niter += 1
        return output

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(num_bits={self.num_bits}, qmode={self.qmode})"
        )


class PTQLossFunc(nn.Module):
    def __init__(  # pylint: disable=dangerous-default-value
        self,
        method="mse",
        Ntotal_iters=20000,
        layers=None,
        p=2.0,
        isOptimConv=False,
        adaR_anneal={
            "warmup": 0.1,
            "warmup2": 0.0,
            "hold": 0.9,
            "beta": {"range": (20, 2), "style": "linear"},
            "lambda": {"range": (0.01, 0.01), "style": "constant"},
        },
    ):
        super().__init__()
        self.method = method
        self.p = p
        self.count = 0
        self.Ntotal = Ntotal_iters
        self.layers = layers
        self.isOptimConv = isOptimConv
        self.warmup = adaR_anneal["warmup"]
        self.warmup2 = adaR_anneal["warmup2"]
        self.hold = adaR_anneal["hold"]
        self.loss_start = int(Ntotal_iters * self.warmup)
        self.reset_ReSig = None
        if self.warmup2 >= self.warmup:
            self.reset_ReSig = int(Ntotal_iters * self.warmup2)
        # NOTE, round_loss starts from warmup but decay could start from warmup2, controlled by
        # LinearTempDecay() and CyclicTempDecay() when using delayed-decay (warmup2 !=0),
        # we may further switch the formula, e.g. from 1ReSig to 3ReSig at decay_start
        # NOTE method could be ['mse','normalized_change','ssim','ssimlog','ssimp0.2','ssimp0.5',
        #   'ssimp2','fisher_diag','fisher_full', 'adaround']

        self.beta = adaR_anneal["beta"]  # brecq's settings was (20, 2)
        self.lambda_eq21 = adaR_anneal["lambda"]

        if self.beta["style"] == "constant":
            self.beta_decay = lambda x: self.beta["range"][0]
        elif self.beta["style"] == "linear":
            self.beta_decay = LinearTempDecay(
                Ntotal_iters,
                warm_up=max(self.warmup, self.warmup2),
                hold=self.hold,
                b_range=self.beta["range"],
            )
        else:
            self.beta_decay = CyclicTempDecay(
                Ntotal_iters,
                warm_up=max(self.warmup, self.warmup2),
                hold=self.hold,
                b_maxmin=self.beta["range"],
                style=self.beta["style"],
            )
            # style can be 'cos','V','cos*0.5', 'cos*2','V*2'...

        if self.lambda_eq21["style"] == "constant":
            self.lambda_decay = lambda x: self.lambda_eq21["range"][0]
        elif self.lambda_eq21["style"] == "linear":
            self.lambda_decay = LinearTempDecay(
                Ntotal_iters,
                warm_up=max(self.warmup, self.warmup2),
                hold=self.hold,
                b_range=self.lambda_eq21["range"],
            )
        else:
            self.lambda_decay = CyclicTempDecay(
                Ntotal_iters,
                warm_up=max(self.warmup, self.warmup2),
                hold=self.hold,
                b_maxmin=self.lambda_eq21["range"],
                style=self.lambda_eq21["style"],
            )

    def __call__(
        self, im1, im2, grad=None, gt=None
    ):  # input_tensor im1, im2 are supposed to be q_out, fp_out
        self.count += 1

        if self.method == "mse":
            return F.mse_loss(im1, im2)

        if self.method in ["mae", "l1"]:
            return F.l1_loss(im1, im2)

        if self.method == "normalized_change":
            return torch.norm(im1 - im2) / torch.norm(im2)

        if self.method == "fisher_diag":
            return ((im1 - im2).pow(2) * grad.pow(2)).sum(1).mean()

        if self.method == "fisher_full":
            a = (im1 - im2).abs()
            grad = grad.abs()
            batch_dotprod = torch.sum(a * grad, (1, 2, 3)).view(-1, 1, 1, 1)
            return (batch_dotprod * a * grad).mean() / 100

        if "adaround" in self.method:
            # default is mse + rounding loss as in the original paper
            round_loss = torch.tensor(0.0, device=im1.device)
            # ssimloss = torch.tensor(0.0, device=im1.device)
            losses = {}
            if self.count > self.loss_start:
                # we can choose to anneal beta and lambda separately, or together
                b = self.beta_decay(self.count)
                lambda_eq21 = self.lambda_decay(
                    self.count
                )  # eq21 in adaround paper, use brecq's settings

                for l in self.layers:
                    if hasattr(l, "quantize_weight") and isinstance(
                        l.quantize_weight, AdaRoundQuantizer
                    ):
                        if self.count == self.reset_ReSig:
                            l.quantize_weight.reset_ReSig_param(3)

                        round_vals = (
                            l.quantize_weight.get_soft_targets()
                        )  # calc h from eq23, now support multimodal
                        if l.quantize_weight.multimodal:
                            round_vals = (
                                round_vals
                                + (round_vals < 0.0).to(torch.float32)
                                - (round_vals > 1.0).to(torch.float32)
                            )
                        round_loss += (
                            lambda_eq21
                            * (1 - ((round_vals - 0.5).abs() * 2).pow(b)).sum()
                        )  # eq24

            if self.method == "adaroundKL":
                rec_loss = F.kl_div(
                    torch.log(im1 + 1e-6), im2 + 1e-6, reduction="batchmean"
                )
            elif self.method == "adaroundCos":
                rec_loss = torch.mean(
                    F.cosine_similarity(im1, im2)  # pylint: disable=not-callable
                )
            elif self.method == "adaroundL1":
                rec_loss = F.l1_loss(im1, im2)
            elif self.method.startswith("adaroundMonAll"):
                # monitor all losses, still use mse+round for total
                with torch.no_grad():
                    losses["l1"] = F.l1_loss(im1, im2)
                    losses["mse"] = F.mse_loss(
                        im1, im2
                    )  # backward will use "lp loss" instead of mse

                with torch.set_grad_enabled(self.method.endswith("_norm")):
                    losses["norm"] = torch.norm(im1 - im2) / torch.norm(im2)
                with torch.set_grad_enabled(self.method.endswith("_cos")):
                    losses["cos"] = 1.0 - torch.mean(
                        F.cosine_similarity(im1, im2)  # pylint: disable=not-callable
                    )
                with torch.set_grad_enabled(self.method.endswith("_ce")):
                    # ce loss only works for last layer, "im1" should be q_out, im2 won't be used,
                    # unless we want to check fp_out's ce loss
                    losses["qce"] = (
                        0.0 * F.cross_entropy(im1, gt) if gt is not None else None
                    )
                    # may need to adjust the weighting factor

                if self.method.endswith("_norm"):
                    rec_loss = losses["norm"]
                elif self.method.endswith("_cos"):
                    rec_loss = losses["cos"]
                elif self.method.endswith("_ce") and gt is not None:
                    # only last layer will have gt in input_tensor, others will default to lp_loss
                    rec_loss = losses["qce"]
                else:
                    rec_loss = lp_loss(im1, im2, p=self.p)

            else:
                # use brecq and qdrop's implementation
                rec_loss = lp_loss(im1, im2, p=self.p)

            losses["total"] = rec_loss + round_loss
            losses["reconstruct"] = rec_loss.detach()  # for tensorboard plot only
            losses["rounding"] = round_loss.detach()  # for tensorboard plot only
            return losses

        # method not defined!
        logger.info(f'PTQ Loss method {self.method} not defined. Use "MSE" instead.')
        return F.mse_loss(im1, im2)


def round_ste(x: torch.Tensor):
    """
    Implement Straight-Through Estimator for rounding operation.
    """
    return (x.round() - x).detach() + x


def lp_loss(pred, tgt, p=2.0, reduction="none"):
    """
    loss function measured in L_p Norm
    """
    if reduction == "none":
        return (pred - tgt).abs().pow(p).sum(1).mean()
    return (pred - tgt).abs().pow(p).mean()


class UniformAffineQuantizer(nn.Module):
    """
    PyTorch Function that can be used for asymmetric quantization (also called uniform affine
    quantization). Quantizes its argument in the forward pass, passes the gradient 'straight
    through' on the backward pass, ignoring the quantization that occurred.
    Based on https://arxiv.org/abs/1806.08342.
    :param n_bits: number of bit for quantization
    :param symmetric: if True, the zero_point should always be 0
    :param channel_wise: if True, compute scale and zero_point in each channel
    :param scale_method: determines the quantization scale and zero point
    """

    def __init__(
        self,
        n_bits: int = 8,
        symmetric: bool = False,
        channel_wise: bool = False,
        scale_method: str = "max",
        leaf_param: bool = False,
        inited: bool = False,
    ):
        super().__init__()
        self.sym = symmetric
        assert 2 <= n_bits <= 8, "bitwidth not supported"
        self.n_bits = n_bits
        self.n_levels = 2**self.n_bits
        self.delta = None
        #        self.zero_point = None
        self.inited = inited
        self.leaf_param = leaf_param
        self.channel_wise = channel_wise
        self.scale_method = scale_method
        self.delta = torch.nn.Parameter(torch.Tensor([1.0]))
        self.register_buffer("zero_point", torch.Tensor([0.0]))

    def forward(self, x: torch.Tensor):
        if self.inited is False:
            if self.leaf_param:
                delta, self.zero_point = self.init_quantization_scale(
                    x, self.channel_wise
                )
                self.delta = torch.nn.Parameter(delta)
            else:
                delta, zero_point = self.init_quantization_scale(x, self.channel_wise)
                self.delta.fill_(delta)
                self.zero_point.fill_(zero_point)
            self.inited = True

        # start quantization
        x_int = round_ste(x / self.delta) + self.zero_point
        x_quant = torch.clamp(x_int, 0, self.n_levels - 1)
        x_dequant = (x_quant - self.zero_point) * self.delta
        return x_dequant

    def init_quantization_scale(self, x: torch.Tensor, channel_wise: bool = False):
        delta, zero_point = None, None
        if channel_wise:
            x_clone = x.clone().detach()
            n_channels = x_clone.shape[0]
            if len(x.shape) == 4:
                x_max = x_clone.abs().max(dim=-1)[0].max(dim=-1)[0].max(dim=-1)[0]
            else:
                x_max = x_clone.abs().max(dim=-1)[0]
            delta = x_max.clone()
            zero_point = x_max.clone()
            # determine the scale and zero point channel-by-channel
            for c in range(n_channels):
                delta[c], zero_point[c] = self.init_quantization_scale(
                    x_clone[c], channel_wise=False
                )
            if len(x.shape) == 4:
                delta = delta.view(-1, 1, 1, 1)
                zero_point = zero_point.view(-1, 1, 1, 1)
            else:
                delta = delta.view(-1, 1)
                zero_point = zero_point.view(-1, 1)
        else:
            if "max" in self.scale_method:
                x_min = min(x.min().item(), 0)
                x_max = max(x.max().item(), 0)
                if "scale" in self.scale_method:
                    x_min = x_min * (self.n_bits + 2) / 8
                    x_max = x_max * (self.n_bits + 2) / 8

                x_absmax = max(abs(x_min), x_max)
                if self.sym:
                    x_min, x_max = -x_absmax if x_min < 0 else 0, x_absmax

                delta = float(x_max - x_min) / (self.n_levels - 1)
                if delta < 1e-8:
                    logger.info(f"Quantization range close to zero: [{x_min}, {x_max}]")
                    delta = 1e-8

                zero_point = round(-x_min / delta)

            elif self.scale_method == "mse":
                x_max = x.max()
                x_min = x.min()
                best_score = 1e10
                for i in range(80):
                    new_max = x_max * (1.0 - (i * 0.01))
                    new_min = x_min * (1.0 - (i * 0.01))
                    x_q = self.quantize(x, new_max, new_min)
                    # L_p norm minimization as described in LAPQ
                    # https://arxiv.org/abs/1911.07190
                    score = lp_loss(x, x_q, p=2.4, reduction="all")
                    if score < best_score:
                        best_score = score
                        delta = (new_max - new_min) / (2**self.n_bits - 1)
                        zero_point = (-new_min / delta).round()
            else:
                raise NotImplementedError

        return delta, zero_point

    def quantize(self, x, clip_max, clip_min):
        delta = (clip_max - clip_min) / (2**self.n_bits - 1)
        zero_point = (-clip_min / delta).round()
        # we assume weight quantization is always signed
        x_int = torch.round(x / delta)
        x_quant = torch.clamp(x_int + zero_point, 0, self.n_levels - 1)
        x_float_q = (x_quant - zero_point) * delta
        return x_float_q

    def bitwidth_refactor(self, refactored_bit: int):
        assert 2 <= refactored_bit <= 8, "bitwidth not supported"
        self.n_bits = refactored_bit
        self.n_levels = 2**self.n_bits

    def extra_repr(self):
        s = (
            "bit={n_bits}, scale_method={scale_method}, symmetric={sym}, "
            "channel_wise={channel_wise}, leaf_param={leaf_param}"
        )
        return s.format(**self.__dict__)


class AdaRoundQuantizer(nn.Module):
    """
    Adaptive Rounding Quantizer, used to optimize the rounding policy
    by reconstructing the intermediate output.
    Based on
     Up or Down? Adaptive Rounding for Post-Training Quantization: https://arxiv.org/abs/2004.10568

    :param uaq: UniformAffineQuantizer, used to initialize quantization parameters in this quantizer
    :param round_mode: controls the forward pass in this quantizer
    :param weight_tensor: initialize alpha
    """

    def __init__(
        self,
        n_bits: int = 8,
        symmetric: bool = False,
        round_mode="learned_round_sigmoid",
        useSAWB=False,
        perCh=False,
        multimodal=False,
        scalebyoptim=False,
    ):
        super().__init__()
        self.n_bits = n_bits
        self.sym = symmetric
        self.n_levels = 2**n_bits
        self.nspace = self.n_levels - 1
        self.register_buffer(
            "delta", torch.tensor([1.0], requires_grad=False)
        )  # delta = scale, placeholder, will be replaced during init_delta or during fwd()
        self.register_buffer("zero_point", torch.tensor([0.0], requires_grad=False))
        self.useSAWB = (
            useSAWB  # use SAWB to calc delta instead of sweeping/minimizing MSE
        )
        self.perCh = perCh
        self.SAWBcode = 403 if n_bits == 4 else 803 if n_bits == 8 else self.n_bits

        self.round_mode = round_mode
        self.alpha = None
        self.soft_targets = False
        self.multimodal = multimodal
        self.scalebyoptim = scalebyoptim

        # set params for sigmoid function, could be [1 (or True), 3, others]
        self.reset_ReSig_param(multimodal)

        self.beta = 2 / 3
        self.Wshape = None
        self.reshape2 = None

    def forward(self, x):
        if self.useSAWB:
            clip_val, _ = sawb_params_code(self.n_bits, self.SAWBcode, x, self.perCh)
            # NOTE this may cause problem when using multiGPUs or trying to learn delta
            self.delta = clip_val * 2 / (2**self.n_bits - 2)
            self.zero_point = (clip_val / self.delta).round()
        if self.perCh and len(self.delta.shape) < 4:
            # follow Qdrop's approach to create local delta/zp every time ...
            delta = self.delta.detach().reshape(
                self.reshape2
            )  # [-1,1,1,1]) # broadcast to [Cout, 1,1,1] for Convs, [Cout, 1] for Linear
            zero_point = self.zero_point.detach().reshape(self.reshape2)  # [-1,1,1,1])
        else:
            delta = self.delta.detach()
            zero_point = self.zero_point.detach()

        if self.round_mode == "nearest":
            x_int = torch.round(x / delta)
        elif self.round_mode == "nearest_ste":
            x_int = round_ste(x / delta)
        elif self.round_mode == "stochastic":
            x_floor = torch.floor(x / delta)
            rest = (x / delta) - x_floor  # rest of rounding
            x_int = x_floor + torch.bernoulli(rest)
            logger.info("Draw stochastic sample")
        elif self.round_mode == "learned_hard_sigmoid":
            x_floor = torch.floor(x / delta)
            if self.soft_targets:
                x_int = x_floor + self.get_soft_targets()
            else:
                x_int = x_floor + self.get_hard_targets()

        elif self.round_mode == "weight_STE":
            if self.training:
                # under training mode
                return AdaRoundSTE.apply(
                    x, self.nspace, self.delta, self.zero_point, self.alpha
                )
            # during eval mode, will apply safety to ensure rounding part is either 0 or 1,
            # no intermediate (floating) numbers
            x_floor = torch.floor(x / delta)
            x_int = x_floor + (self.alpha >= 0).float()

        else:
            raise ValueError("Wrong rounding mode")

        x_quant = torch.clamp(x_int + zero_point, 0, self.n_levels - 1)
        x_float_q = (x_quant - zero_point) * delta

        return x_float_q

    def reset_ReSig_param(self, mode=0):
        if mode == 1:  # stretch h to [-1,2]
            # multimodal Case 1: use a further stretched sigmoid.
            # NOTE that eff grad nead h=0 and 1 will be large
            self.gamma, self.zeta = -1.1, 2.1
            self.lower, self.upper = -1.0, 2.0  # clip range
            self.xscale = (
                1.0  # could scale sigmoid curve x by 0.5 (ie, stretch 2x in x-dir)
            )
        elif mode == 3:
            # multimodal Case 3: use 3 overlapping, scaled and shifted sigmoids
            # (based on the original ReSig)
            self.gamma, self.zeta = -0.1, 1.1
            self.lower, self.upper = 0.0, 1.0
            self.xscale = 4.0
            self.sigwidth = 1.5
        else:
            # default, original settings as in the paper
            self.gamma, self.zeta = -0.1, 1.1
            self.lower, self.upper = 0.0, 1.0
            self.xscale = 1.0

    def re_sig(self, xoffset=0.0):  # , xscale=1.0
        return torch.clamp(
            torch.sigmoid((self.alpha - xoffset) * self.xscale)
            * (self.zeta - self.gamma)
            + self.gamma,
            self.lower,
            self.upper,
        )

    def get_soft_targets(self):
        if self.multimodal == 3:
            # Case 3: use 3 overlapping, shifted sigmoid,
            # self.gamma, self.zeta = -0.1, 1.1, clip to [0, 1]
            return (
                self.re_sig(-self.sigwidth)
                + self.re_sig()
                + self.re_sig(self.sigwidth)
                - 1.0
            )
        # either normal rectified sigmoid or further-stretched one, no xoffset
        # self.gamma, self.zeta, and cvs are defined in self
        return torch.clamp(
            torch.sigmoid(self.alpha) * (self.zeta - self.gamma) + self.gamma,
            self.lower,
            self.upper,
        )

    def get_hard_targets(self):
        assert (
            self.round_mode == "learned_hard_sigmoid"
        ), "Please check round_mode settings"
        if self.multimodal:
            h = self.get_soft_targets()
            # only implement [-1, 0, 1, 2] for now
            h_int = (
                torch.ones_like(h, dtype=torch.int8) * 2
                - (h < 1.5).to(torch.int8)
                - (h < 0.5).to(torch.int8)
                - (h < -0.5).to(torch.int8)
            )
        else:
            h_int = self.alpha >= 0

        return h_int.float()

    def init_alpha(self, x: torch.Tensor, _optim=None):
        # follow Qdrop's style, use local delta every time
        self.reshape2 = [1] * len(self.Wshape)
        self.reshape2[0] = self.Wshape[
            0
        ]  # ch_axis is always 0, e.g. should be [Cout,1,1,1] for Conv and [Cout, 1] for Linear
        delta = (
            self.delta.detach().reshape(self.reshape2)
            if self.perCh and len(self.delta.shape) < 4
            else self.delta.detach()
        )

        x_floor = torch.floor(x / delta)
        if self.round_mode in ["learned_hard_sigmoid", "weight_STE"]:
            rest = (x / delta) - x_floor  # rest of rounding [0, 1)
            alpha = (
                -torch.log((self.zeta - self.gamma) / (rest - self.gamma) - 1)
                / self.xscale
            )  # => sigmoid(alpha*xscale) = rest

            if self.alpha is not None and isinstance(self.alpha, nn.Parameter):
                # if we want to re-init alpha after it's been added to the optimizer,
                # remember to update the parameter in optimizer as it will not update
                # itself automatically... We could implement an automated optimizer update here,
                # but don't see a real need so far.
                # logger.info("!! AdaRound alpha is being re-assigned. Make sure it's not added to "
                #             "your optimizer yet, or please update the optimizer accordingly !!")
                with torch.no_grad():
                    self.alpha.copy_(alpha)
                logger.info("!! AdaRound alpha is updated !!")
            else:
                self.alpha = nn.Parameter(alpha)
        else:
            raise NotImplementedError

    def init_delta(self, W: torch.Tensor, qw_mode=None, _mod_name=None):
        Wfp = W.clone().detach()
        self.Wshape = W.shape

        # use Qdrop's implementation instead of BRECQ's
        observer = MSEObserver(
            bit=self.n_bits,
            # a) should use self.sym, b) default is False but it's almost always True for weights
            symmetric=False,
            ch_axis=0 if "perCh" in qw_mode else -1,
            useOptim4Search=self.scalebyoptim,
        )
        observer(Wfp)  # will update observe.min_val and max_val
        currDev = self.zero_point.device
        _scale, _zero_point = observer.calculate_qparams(
            observer.min_val, observer.max_val
        )
        _scale, _zero_point = _scale.to(currDev), _zero_point.to(currDev)
        if self.delta.shape != _scale.shape:  # delta is scale
            self.delta.resize_(_scale.shape)
            self.zero_point.resize_(_zero_point.shape)
        self.delta.copy_(_scale)
        self.zero_point.copy_(_zero_point)


class AdaRoundSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_tensor, nspace, delta, zp, alpha):
        gamma, zeta = -0.1, 1.1
        x_floor = torch.floor(input_tensor / delta)
        x_int = x_floor + torch.clamp(
            torch.sigmoid(alpha) * (zeta - gamma) + gamma, 0, 1
        )

        x_quant = torch.clamp(x_int + zp, 0, nspace)
        x_float_q = (x_quant - zp) * delta

        ctx.save_for_backward(
            input_tensor, alpha, delta, x_float_q.max(), x_float_q.min()
        )

        return x_float_q

    @staticmethod
    def backward(ctx, grad_output):
        input_tensor, alpha, delta, cv, cvn = ctx.saved_tensors  # delta is scale
        gamma, zeta = -0.1, 1.1
        sig_alpha = torch.sigmoid(alpha)
        h = sig_alpha * (zeta - gamma) + gamma

        grad_alpha = grad_output.clone()
        grad_alpha = torch.where(
            torch.logical_or(input_tensor <= cvn, input_tensor >= cv),
            torch.zeros_like(grad_alpha),
            grad_alpha,
        )
        grad_alpha = torch.where(
            torch.logical_or(h <= 0.0, h >= 1.0),
            torch.zeros_like(grad_alpha),
            grad_alpha,
        )
        grad_alpha *= delta * (zeta - gamma) * sig_alpha * (1 - sig_alpha)

        return grad_output, None, None, None, grad_alpha


class LinearTempDecay:
    def __init__(self, t_max=20000, warm_up=0.2, hold=1.0, b_range=(20, 2)):
        self.t_max = hold * t_max
        self.start_decay = warm_up * t_max
        # NOTE from warm_up to warm_up2, round_loss starts to work but no decay in beta and lambda
        # see PTQLossFunc for more details
        self.start_b = b_range[0]
        self.end_b = b_range[1]
        self.curr_beta = 0.0

    def __call__(self, t):
        # NOTE from warm_up to warm_up2, round_loss starts to work but no decay in beta and lambda
        if t < self.start_decay:
            self.curr_beta = self.start_b
        elif t > self.t_max:
            self.curr_beta = self.end_b
        else:
            rel_t = (t - self.start_decay) / (self.t_max - self.start_decay)
            self.curr_beta = self.end_b + (self.start_b - self.end_b) * max(
                0.0, (1 - rel_t)
            )
        return self.curr_beta


class CyclicTempDecay:
    def __init__(self, t_max=20000, warm_up=0.2, hold=1.0, b_maxmin=(20, 2), style="V"):
        self.t_max = hold * t_max
        self.start_decay = warm_up * t_max
        # Annealing only happens between [warmup, hold]*t_max, from max_b -> min_b -> max_b
        # e.g. usually no round_loss before 0.2*tmax (controlled by PTQloss_func),
        # here we still return b_max, and hold at b_max after 0.8*tmax
        self.max_b = b_maxmin[0]
        self.min_b = b_maxmin[1]
        assert self.max_b > self.min_b, "max_b is smaller than min_b, Please check!"
        self.curr_beta = 0.0
        # style can be 'V', 'V*2', 'V*3'... 'cos*0.5', 'cos*2', 'cos' ... shape*N where
        # N means number of cycles, N could be 0.5
        if "*" not in style:
            style = style + "*1"
        self.style_cycles = style.split("*")
        self.period = (self.t_max - self.start_decay) // float(self.style_cycles[1])

    def __call__(self, t):
        # NOTE from warm_up to warm_up2, round_loss starts to work but no decay in beta and lambda
        if t < self.start_decay:
            self.curr_beta = self.max_b
        elif self.start_decay <= t < self.t_max:
            rel_t = ((t - self.start_decay) % self.period) / self.period
            if self.style_cycles[0] == "cos":
                self.curr_beta = (
                    self.min_b
                    + (self.max_b - self.min_b)
                    * (math.cos(math.pi * 2 * rel_t) + 1.0)
                    * 0.5
                )
            else:  # V-shape
                self.curr_beta = self.min_b + (self.max_b - self.min_b) * abs(
                    1.0 - 2 * rel_t
                )
        return self.curr_beta


def _transform_to_ch_axis(x, ch_axis):
    if ch_axis == -1:
        return x
    x_dim = x.size()
    new_axis_list = list(range(len(x_dim)))  # [i for i in range(len(x_dim))]
    new_axis_list[ch_axis] = 0
    new_axis_list[0] = ch_axis
    x_channel = x.permute(new_axis_list)
    y = torch.flatten(x_channel, start_dim=1)
    return y


def fake_quantize_per_tensor_affine(x, scale, zero_point, quant_min, quant_max):
    x_int = round_ste(x / scale) + zero_point
    x_quant = torch.clamp(x_int, quant_min, quant_max)
    x_dequant = (x_quant - zero_point) * scale
    return x_dequant


def fake_quantize_per_channel_affine(
    x, scale, zero_point, ch_axis, quant_min, quant_max
):
    new_shape = [1] * len(x.shape)
    new_shape[ch_axis] = x.shape[ch_axis]
    scale = scale.reshape(new_shape)
    zero_point = zero_point.reshape(new_shape)
    x_int = round_ste(x / scale) + zero_point
    x_quant = torch.clamp(x_int, quant_min, quant_max)
    x_dequant = (x_quant - zero_point) * scale
    return x_dequant


class ObserverBase(nn.Module):
    def __init__(self, bit=8, symmetric=False, ch_axis=-1):
        super().__init__()
        self.bit = bit
        self.symmetric = symmetric
        self.ch_axis = ch_axis
        self.eps = torch.tensor(1e-8, dtype=torch.float32)
        if self.symmetric:
            self.quant_min = -(2 ** (self.bit - 1))
            self.quant_max = 2 ** (self.bit - 1) - 1
        else:
            self.quant_min = 0
            self.quant_max = 2**self.bit - 1
        self.register_buffer("min_val", torch.tensor(float("inf")))
        self.register_buffer("max_val", torch.tensor(float("-inf")))

    def set_bit(self, bit):
        self.bit = bit
        if self.symmetric:
            self.quant_min = -(2 ** (self.bit - 1))
            self.quant_max = 2 ** (self.bit - 1) - 1
        else:
            self.quant_min = 0
            self.quant_max = 2**self.bit - 1

    def set_name(self, name):
        self.name = name

    @torch.jit.export
    def calculate_qparams(self, min_val, max_val):
        # one_dim or one element
        quant_min, quant_max = self.quant_min, self.quant_max
        min_val_neg = torch.min(min_val, torch.zeros_like(min_val))
        max_val_pos = torch.max(max_val, torch.zeros_like(max_val))

        device = min_val_neg.device
        scale = torch.ones(min_val_neg.size(), dtype=torch.float32, device=device)
        zero_point = torch.zeros(min_val_neg.size(), dtype=torch.int, device=device)
        if self.symmetric:
            max_val_pos = torch.max(-min_val_neg, max_val_pos)
            scale = max_val_pos / (float(quant_max - quant_min) / 2)
            scale = torch.max(scale, self.eps)
        else:
            scale = (max_val_pos - min_val_neg) / float(quant_max - quant_min)
            scale = torch.max(scale, self.eps)
            zero_point = quant_min - torch.round(min_val_neg / scale)
            zero_point = torch.clamp(zero_point, quant_min, quant_max)
        return scale, zero_point


class MSEObserver(ObserverBase):
    # grid search: more accurate but slow
    def __init__(self, bit=8, symmetric=False, ch_axis=-1, useOptim4Search=False):
        super().__init__(bit=bit, symmetric=symmetric, ch_axis=ch_axis)
        self.p = 2.4
        self.num = 100  # candidate num
        self.one_side_dist = None  # 'pos', 'neg', 'no'
        self.useOptim4Search = useOptim4Search

    def lp_loss(self, pred, tgt, p=2.0):
        x = (pred - tgt).abs().pow(p)
        if self.ch_axis == -1:
            return x.mean()
        y = _transform_to_ch_axis(x, self.ch_axis)
        return y.mean(1)

    def loss_fx(self, x, new_min, new_max):
        # should also consider channel here
        scale, zero_point = self.calculate_qparams(new_min, new_max)
        if self.ch_axis != -1:
            x_q = fake_quantize_per_channel_affine(
                x,
                scale.data,
                zero_point.data.int(),
                self.ch_axis,
                self.quant_min,
                self.quant_max,
            )
        else:
            x_q = fake_quantize_per_tensor_affine(
                x, scale.item(), int(zero_point.item()), self.quant_min, self.quant_max
            )
        score = self.lp_loss(x_q, x, p=self.p)
        return score

    def loss_fx_grad(self, x, new_min, new_max):
        # should also consider channel here
        scale, zero_point = self.calculate_qparams(new_min, new_max)
        if self.ch_axis != -1:
            x_q = fake_quantize_per_channel_affine(
                x, scale, zero_point, self.ch_axis, self.quant_min, self.quant_max
            )
        else:
            x_q = fake_quantize_per_tensor_affine(
                x, scale, zero_point, self.quant_min, self.quant_max
            )
        score = self.lp_loss(x_q, x, p=self.p)
        return score

    def perform_2D_search(self, x):
        if self.ch_axis != -1:
            y = _transform_to_ch_axis(x, self.ch_axis)
            x_min, x_max = torch._aminmax(y, 1)
            # may also have the one side distribution in some channels
            x_max = torch.max(x_max, torch.zeros_like(x_max))
            x_min = torch.min(x_min, torch.zeros_like(x_min))
        else:
            x_min, x_max = torch._aminmax(x)
        xrange = x_max - x_min
        best_score = torch.zeros_like(x_min) + (1e10)
        best_min = x_min.clone()
        best_max = x_max.clone()
        # enumerate xrange
        for i in range(1, self.num + 1):
            tmp_min = torch.zeros_like(x_min)
            tmp_max = xrange / self.num * i
            tmp_delta = (tmp_max - tmp_min) / float(self.quant_max - self.quant_min)
            # enumerate zp
            for zp in range(self.quant_min, self.quant_max + 1):
                new_min = tmp_min - zp * tmp_delta
                new_max = tmp_max - zp * tmp_delta
                score = self.loss_fx(x, new_min, new_max)
                best_min = torch.where(score < best_score, new_min, best_min)
                best_max = torch.where(score < best_score, new_max, best_max)
                best_score = torch.min(best_score, score)
        return best_min, best_max

    # Instead of grid search, use optimizer to find cv and cvn, then scale and zp accordingly
    def search_by_optim(self, x, _ref_min=None, _ref_max=None):
        Wfp = x.clone().detach()
        Wshape = Wfp.shape
        Wfp = Wfp.reshape((Wfp.shape[0], -1) if self.ch_axis != -1 else (1, -1))
        with torch.no_grad():
            if self.bit == 8:
                cvn = Wfp.min(dim=1).values
                cv = Wfp.max(dim=1).values
            elif self.bit == 4:
                # use 99.9 percentile
                N99p9 = int(Wfp.shape[1] * 0.999)
                cvn = torch.kthvalue(Wfp, Wfp.shape[1] - N99p9).values
                cv = torch.kthvalue(Wfp, N99p9).values
            else:
                logger.info("!! nbits != 4 or 8, not implemented !!")
                cvn = torch.empty(Wfp)
                cv = torch.empty(Wfp)
            Wfp = Wfp.reshape(Wshape)

        cv.requires_grad = True
        cvn.requires_grad = True
        optim = torch.optim.Adam([cvn, cv], lr=1e-2)

        for _i in range(self.num):
            optim.zero_grad()
            loss = torch.sum(self.loss_fx_grad(Wfp, cvn, cv))
            loss.backward()
            optim.step()

        return cvn.detach(), cv.detach()

    def perform_1D_search(self, x):
        if self.ch_axis != -1:
            y = _transform_to_ch_axis(x, self.ch_axis)
            x_min, x_max = torch._aminmax(y, 1)
        else:
            x_min, x_max = torch._aminmax(x)
        xrange = torch.max(x_min.abs(), x_max)
        best_score = torch.zeros_like(x_min) + (1e10)
        best_min = x_min.clone()
        best_max = x_max.clone()
        # enumerate xrange
        for i in range(1, self.num + 1):
            thres = xrange / self.num * i
            new_min = torch.zeros_like(x_min) if self.one_side_dist == "pos" else -thres
            new_max = torch.zeros_like(x_max) if self.one_side_dist == "neg" else thres
            score = self.loss_fx(x, new_min, new_max)
            best_min = torch.where(score < best_score, new_min, best_min)
            best_max = torch.where(score < best_score, new_max, best_max)
            best_score = torch.min(score, best_score)
        return best_min, best_max

    def forward(self, x_orig):
        if x_orig.numel() == 0:
            return x_orig
        x = x_orig.clone().detach().to(self.min_val.dtype)  # pylint: disable=access-member-before-definition
        if self.one_side_dist is None:
            self.one_side_dist = (
                "pos" if x.min() >= 0.0 else "neg" if x.max() <= 0.0 else "no"
            )

        if (
            self.one_side_dist != "no" or self.symmetric
        ):  # one-side distribution or symmetric value for 1-d search
            best_min, best_max = self.perform_1D_search(x)
        else:  # 2-d search
            if self.useOptim4Search:
                # # default method may not find the optimal values and it's very slow
                # best_min0, best_max0 = self.perform_2D_search(x)
                best_min, best_max = self.search_by_optim(
                    x
                )  # , best_min0, best_max0) # Adam 100 iters is ~ 10x faster
            else:
                best_min, best_max = self.perform_2D_search(x)

        self.min_val = torch.min(self.min_val, best_min)
        self.max_val = torch.max(self.max_val, best_max)


class Qbypass(nn.Module):
    """
    no quantization at all, straight-thru
    in place of lambda function when using nbits=32 and 16.
    to avoid issue when pickle (ie torch.save) of lambda
    (seems to be a problem only for DDP)
    """

    def __init__(self):
        super().__init__()

    def forward(self, input_tensor):
        return input_tensor


def transformers_prepare_input(
    data: Union[torch.Tensor, Any], dev="cuda"
) -> Union[torch.Tensor, Any]:
    """
    Prepares one `data` before feeding it to the model, be it a tensor or a
    nested list/dictionary of tensors.
    """
    if isinstance(data, Mapping):
        return type(data)(
            {k: transformers_prepare_input(v, dev=dev) for k, v in data.items()}
        )
    if isinstance(data, (tuple, list)):
        return type(data)(transformers_prepare_input(v, dev=dev) for v in data)
    if isinstance(data, torch.Tensor):
        kwargs = {"device": dev}
        # if self.deepspeed and data.dtype != torch.int64:
        #     # NLP models input_tensors are int64 and those get adjusted to the right dtype of the
        #     # embedding. Other models such as wav2vec2's input_tensors are already float and thus
        #     # may need special handling to match the dtypes of the model
        #     kwargs.update(dict(dtype=self.args.hf_deepspeed_config.dtype()))
        return data.to(**kwargs)

    return data


# ================================================================================================
#
# Swiched Capacitor (swcap) Quantizers
#
# ================================================================================================

# ==================================================================================
# SAWB Quantizer
# ==================================================================================


class SAWB_sw(nn.Module):
    """
    SAWB with custom backward: gradient passed through clip function
        SAWBZeroSTE_sw: grad of clipped input_tensors is zero (as per SAWB paper)
        SAWBPlusZeroSTE_sw: grad passed through everywhere
    For swcap compatibility, zero alignment is enforced (one level is lost)

    dequantize == False: output in range [-scale_sym, scale_sym]
    dequantize == True:  output in range [-clip_val, clip_val]
    """

    def __init__(self, num_bits, dequantize=False, clipSTE=True, recompute=False):
        super().__init__()
        self.num_bits = num_bits
        self.dequantize = dequantize
        self.recompute = recompute
        self.clipSTE = clipSTE
        self.code_dict = {2: 103, 4: 403, 7: 703, 8: 803}
        self.scale_sym = 2 ** (num_bits - 1) - 1  # INT4: 7; INT7: 63; INT8: 127
        self.movAvgFac = 0.1
        self.register_buffer("first_call", torch.tensor(1, dtype=torch.int))
        self.register_buffer("clip_val", torch.Tensor([0.0]))  # dim 1 tensor

        if self.clipSTE:
            self.quantizer = SAWBPlusZeroSTE_sw  # zero is always aligned for swcap
        else:
            self.quantizer = SAWBZeroSTE_sw  # zero is always aligned for swcap

    def forward(self, input_tensor):
        if self.training or self.recompute:
            clipval_new = (
                sawb_params_code_sw(self.code_dict[self.num_bits], input_tensor)
                if self.num_bits in self.code_dict
                else input_tensor.abs().max()
            )  # dim 0 tensor
            clipval_new = clipval_new.unsqueeze(dim=0)

            output = self.quantizer.apply(
                input_tensor, clipval_new, self.scale_sym, self.dequantize
            )

            with torch.no_grad():
                self.clip_val.fill_(
                    self.clip_val.item() * (1.0 - self.movAvgFac)
                    + clipval_new.item() * self.movAvgFac
                    if not self.first_call
                    else clipval_new.item()
                )
                self.first_call.fill_(
                    0
                )  # in-place fill ensures correct update on multi-gpu
        else:
            output = self.quantizer.apply(
                input_tensor, self.clip_val, self.scale_sym, self.dequantize
            )

        return output

    def __repr__(self):
        sawbplut_str = "+" if self.clipSTE else ""
        return (
            f"{self.__class__.__name__}{sawbplut_str}(num_bits={self.num_bits}, "
            f"quantizer={self.quantizer})"
        )


class SAWBPlusZeroSTE_sw(torch.autograd.Function):
    """
    symmetric SAWB+ quantizer
    no zero point
    input_tensor zero is always aligned to integer zero and mapped to zero output
    gradients are passed through also for values beyond clipping range
    """

    @staticmethod
    def forward(ctx, input_tensor, clip_val, scale_sym, dequantize):
        output = torch.round(
            input_tensor.mul(1 / clip_val).clamp(-1, 1) * scale_sym
        )  # INT4: [-7, 7]; INT8: [-127, 127]
        if dequantize:
            output = output / scale_sym * clip_val  # range [-clip_val, clip_val]
        ctx.scale = 1 if dequantize else scale_sym / clip_val
        return output

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output * ctx.scale, None, None, None


class SAWBZeroSTE_sw(torch.autograd.Function):
    """
    symmetric SAWB quantizer
    no zero point
    input_tensor zero is always aligned to integer zero and mapped to zero output
    """

    @staticmethod
    def forward(ctx, input_tensor, clip_val, scale_sym, dequantize):
        output = torch.round(
            input_tensor.mul(1 / clip_val).clamp(-1, 1) * scale_sym
        )  # INT4: [-7, 7]; INT8: [-127, 127]
        if dequantize:
            output = output / scale_sym * clip_val  # range [-clip_val, clip_val]
        ctx.save_for_backward(input_tensor, clip_val)
        ctx.scale = 1 if dequantize else scale_sym / clip_val
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input_tensor, clip_val = ctx.saved_tensors
        grad_input_tensor = grad_output.clone()
        grad_input_tensor = torch.where(
            input_tensor < -clip_val,
            torch.zeros_like(grad_input_tensor),
            grad_input_tensor,
        )
        grad_input_tensor = torch.where(
            input_tensor > clip_val,
            torch.zeros_like(grad_input_tensor),
            grad_input_tensor,
        )
        return grad_input_tensor * ctx.scale, None, None, None


def sawb_params_code_sw(code, out):
    with torch.no_grad():
        x = out.flatten()
        mu = x.abs().mean()
        std = x.mul(x).mean().sqrt()

        coeff_dict = {
            102: (3.12, -2.064),  # [-a, -a/3, a/3, a] equivalent to 2 bits
            103: (2.6, -1.71),  # [-a, 0, a]
            403: (12.035, -12.03),  # [-a, -6/7a, ..., 0, ..., 6/7a, a]
            703: (28.24, -30.81),
            803: (31.76, -35.04),
        }

        if not coeff_dict.get(code) is None:
            coeff = coeff_dict[code]
        else:
            raise ValueError(f"SAWB not implemented for code={code}")

        return coeff[1] * mu + coeff[0] * std  # a*/mu = c1 + c0 * std/mu


# ==================================================================================
# Max Quantizer
# ==================================================================================


class Qmax_sw(nn.Module):
    """
    Symmetric MAX with custom backward (Straight Through Estimator)
    For swcap compatibility:
    - zero always aligned (due to symmetry)
    - minmax not supported

    dequantize == False: output in range [-scale_sym, scale_sym]
    dequantize == True:  output in range [-clip_val, clip_val]
    """

    def __init__(self, num_bits, dequantize=False, recompute=False):
        super().__init__()
        self.num_bits = num_bits
        self.dequantize = dequantize
        self.recompute = recompute
        self.quantizer = QmaxSTE_sw
        self.scale_sym = 2 ** (num_bits - 1) - 1  # INT4: 7; INT8: 127
        self.movAvgFac = 0.1
        self.register_buffer("first_call", torch.tensor(1, dtype=torch.int))
        self.register_buffer("clip_val", torch.Tensor([0.0]))  # dim 1 tensor

    def forward(self, input_tensor):
        if self.training or self.recompute:
            clipval_new = input_tensor.abs().max().unsqueeze(dim=0)  # symmetric

            output = self.quantizer.apply(
                input_tensor, clipval_new, self.scale_sym, self.dequantize
            )

            with torch.no_grad():
                self.clip_val.fill_(
                    self.clip_val.item() * (1.0 - self.movAvgFac)
                    + clipval_new.item() * self.movAvgFac
                    if not self.first_call
                    else clipval_new.item()
                )
                self.first_call.fill_(
                    0
                )  # in-place fill ensures correct update on multi-gpu
        else:
            output = self.quantizer.apply(
                input_tensor, self.num_bits, self.clip_val, self.dequantize
            )
        return output

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(num_bits={self.num_bits}, "
            f"quantizer={self.quantizer})"
        )


class QmaxSTE_sw(torch.autograd.Function):
    """
    MAX quantizer with symmetric clips
    no zero point
    input_tensor zero is always aligned to integer zero and mapped to zero output
    """

    @staticmethod
    def forward(ctx, input_tensor, cv, scale_sym, dequantize):
        output = torch.round(
            input_tensor.mul(1 / cv).clamp(-1, 1) * scale_sym
        )  # INT4: [-7, 7]; INT8: [-127, 127]
        if dequantize:
            output = output / scale_sym * cv  # range [-clip_val, clip_val]
        ctx.scale = 1 if dequantize else scale_sym / cv
        return output

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.clone() * ctx.scale, None, None, None


# ==================================================================================
# PACT Quantizers:
# - PACT2_sw: asymmetric, 2-sided (supports PACT2 and PACT2+)
# - PACT_sw: asymmetric, 1-sided (= non-negative)
# - PACT2sym_sw: symmetric, 2-sided (= single clip_val)
# ==================================================================================


class PACT2_sw(nn.Module):
    """
    2-sided asymmetric PACT quantizer, compatible with swcap
    Store scale and zero point to be accessed for dequantization
    """

    def __init__(
        self,
        num_bits,
        init_clip_valn=-8.0,
        init_clip_val=8.0,
        dequantize=False,
        align_zero=False,
        pact_plus=False,
    ):
        super().__init__()
        self.num_bits = num_bits
        if isinstance(init_clip_val, torch.Tensor) and isinstance(
            init_clip_valn, torch.Tensor
        ):
            self.clip_val = nn.Parameter(init_clip_val)
            self.clip_valn = nn.Parameter(init_clip_valn)
        elif not isinstance(init_clip_val, torch.Tensor) and not isinstance(
            init_clip_valn, torch.Tensor
        ):
            self.clip_val = nn.Parameter(torch.Tensor([init_clip_val]))
            self.clip_valn = nn.Parameter(torch.Tensor([init_clip_valn]))
        else:
            raise ValueError(
                "FMS: init_clip_val and init_clip_valn should be the same instance type."
            )

        # break clip symmetry to avoid additional Q level due to rounding
        if self.clip_val.item() == -self.clip_valn.item():  # pylint: disable=invalid-unary-operand-type
            self.clip_valn.data += 0.00001

        self.dequantize = dequantize
        self.align_zero = align_zero
        self.pact_plus = pact_plus
        self.register_buffer("scale", torch.Tensor([1.0]))
        self.register_buffer("zero_point", torch.Tensor([0.0]))
        self.quantizer = PACT2plus_STE_sw if pact_plus else PACT2_STE_sw

    def forward(self, input_tensor):
        output = self.quantizer.apply(
            input_tensor,
            self.clip_val,
            self.clip_valn,
            self.num_bits,
            self.dequantize,
            self.align_zero,
        )

        # recalculate scale and zero point (passing them from quantizer creates memory leak)
        with torch.no_grad():
            self.scale.fill_(
                (2**self.num_bits - 1) / (self.clip_val.item() - self.clip_valn.item())
            )
            self.zero_point.fill_(
                (self.scale * self.clip_valn).round().item()
                if self.align_zero
                else (self.scale * self.clip_valn).item()
            )
        return output

    def __repr__(self):
        return (
            f"PACT2{'plus' if self.pact_plus else ''}_sw(bits={self.num_bits}), "
            f"pos-clip={self.clip_val[0]:.4f}, "
            f"neg-clip={self.clip_valn[0]:.4f}, quantizer={self.quantizer})"
        )


class PACT2_STE_sw(torch.autograd.Function):
    """
    swcap-compatible, 2-sided asymmetric PACT quantize/dequantize with custom backward
    """

    @staticmethod
    def forward(
        ctx, input_tensor, clip_val, clip_valn, num_bits, dequantize, align_zero
    ):
        ctx.save_for_backward(input_tensor, clip_val, clip_valn)
        scale, zero_point = asymmetric_linear_quantization_params(
            num_bits,
            clip_valn.data,
            clip_val.data,
            integral_zero_point=align_zero,
            signed=False,
        )
        if isinstance(clip_val, torch.Tensor):
            output = torch.where(
                input_tensor > clip_val,
                torch.ones_like(input_tensor) * clip_val,
                input_tensor,
            )
            output = torch.where(
                output < clip_valn, torch.ones_like(input_tensor) * clip_valn, output
            )
        else:
            output = clamp(input_tensor, clip_valn, clip_val)
        output = linear_quantize(output, scale, zero_point)
        if dequantize:
            output = linear_dequantize(output, scale, zero_point)
            scale = 1
        ctx.scale = scale
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input_tensor, clip_val, clip_valn = ctx.saved_tensors
        grad_input_tensor = grad_output.clone()
        grad_input_tensor = torch.where(
            input_tensor <= clip_valn,
            torch.zeros_like(grad_input_tensor),
            grad_input_tensor,
        )
        grad_input_tensor = torch.where(
            input_tensor >= clip_val,
            torch.zeros_like(grad_input_tensor),
            grad_input_tensor,
        )

        grad_alpha = grad_output.clone()
        grad_alpha = torch.where(
            input_tensor < clip_val, torch.zeros_like(grad_alpha), grad_alpha
        )
        grad_alpha = grad_alpha.sum().expand_as(clip_val)

        grad_alphan = grad_output.clone()
        grad_alphan = torch.where(
            input_tensor > clip_valn, torch.zeros_like(grad_alphan), grad_alphan
        )
        grad_alphan = grad_alphan.sum().expand_as(clip_valn)

        scale = ctx.scale  # need to scale gradient if did not dequantize
        return (
            grad_input_tensor * scale,
            grad_alpha * scale,
            grad_alphan * scale,
            None,
            None,
            None,
        )


class PACT2plus_STE_sw(torch.autograd.Function):
    """
    swcap-compatible, 2-sided asymmetric PACT quantize/dequantize with custom backward
    PACT2plus (a.k.a PACT2+) improves gradient handling of PACT2
    """

    @staticmethod
    def forward(
        ctx, input_tensor, clip_val, clip_valn, num_bits, dequantize, align_zero
    ):
        ctx.save_for_backward(input_tensor, clip_val, clip_valn)

        num_gaps = 2**num_bits - 1  # quantization levels - 1
        scale = num_gaps / (clip_val - clip_valn)
        stepsize = 1.0 / scale
        zero_point = scale * clip_valn

        if align_zero:
            zero_point = zero_point.round()

        if isinstance(clip_val, torch.Tensor):
            output = torch.where(
                input_tensor > clip_val,
                torch.ones_like(input_tensor) * clip_val,
                input_tensor,
            )
            output = torch.where(
                output < clip_valn, torch.ones_like(input_tensor) * clip_valn, output
            )
        else:
            output = clamp(input_tensor, clip_valn, clip_val)
        output = linear_quantize(output, scale, zero_point)
        if dequantize:
            output = linear_dequantize(output, scale, zero_point)
            scale = 1

        ctx.num_gaps = num_gaps
        ctx.num_bits = num_bits
        ctx.stepsize = stepsize
        ctx.scale = (
            scale  # 1/stepsize if NO dequantization, otherwise 1 (used to scale grads)
        )
        return output, scale, zero_point

    @staticmethod
    def backward(ctx, grad_output, _grad_scale, _grad_zp):
        input_tensor, clip_val, clip_valn = ctx.saved_tensors
        num_gaps = ctx.num_gaps
        stepsize = ctx.stepsize
        z = (input_tensor - clip_valn) / stepsize
        delta_z = (z - torch.round(z)) / num_gaps

        grad_input_tensor = grad_output.clone()
        grad_input_tensor = torch.where(
            input_tensor <= clip_valn,
            torch.zeros_like(grad_input_tensor),
            grad_input_tensor,
        )
        grad_input_tensor = torch.where(
            input_tensor >= clip_val,
            torch.zeros_like(grad_input_tensor),
            grad_input_tensor,
        )

        grad_alpha = -grad_output.clone() * delta_z
        grad_alphan = -grad_alpha

        grad_alpha = torch.where(
            input_tensor <= clip_valn, torch.zeros_like(grad_alpha), grad_alpha
        )
        grad_alpha = torch.where(input_tensor >= clip_val, grad_output, grad_alpha)
        grad_alphan = torch.where(
            input_tensor >= clip_val, torch.zeros_like(grad_alphan), grad_alphan
        )
        grad_alphan = torch.where(input_tensor <= clip_valn, grad_output, grad_alphan)

        grad_alpha = grad_alpha.sum().expand_as(clip_val)
        grad_alphan = grad_alphan.sum().expand_as(clip_valn)

        scale = ctx.scale  # need to scale gradient if did not dequantize
        return (
            grad_input_tensor * scale,
            grad_alpha * scale,
            grad_alphan * scale,
            None,
            None,
            None,
        )


class PACT2sym_sw(nn.Module):
    """
    2-sided symmetric PACT quantizer (clip_val = -clip_valn), compatible with swcap
    For swcap compatibility, zero alignment is enforced (one level is lost)
    Multiplicative scaling for dequantization can be computed as scale = scale_sym / clip_val
    """

    def __init__(self, num_bits, init_clip_val=8.0, dequantize=False):
        super().__init__()
        self.num_bits = num_bits
        self.clip_val = (
            nn.Parameter(init_clip_val)
            if isinstance(init_clip_val, torch.Tensor)
            else nn.Parameter(torch.Tensor([init_clip_val]))
        )
        self.scale_sym = 2 ** (num_bits - 1) - 1  # INT4: 7; INT8: 127
        self.dequantize = dequantize
        self.quantizer = PACT2sym_STE_sw

    def forward(self, input_tensor):
        return self.quantizer.apply(
            input_tensor, self.clip_val, self.scale_sym, self.dequantize
        )

    def __repr__(self):
        return (
            f"PACT2sym_sw(bits={self.num_bits}), clip={self.clip_val[0]:.4f}, "
            f"quantizer={self.quantizer})"
        )


class PACT2sym_STE_sw(torch.autograd.Function):
    """
    2-sided symmetric PACT quantize/dequantize with custom backward
    Compatible with swcap: implements grad scaling for dequantize = False
    Does not use zero_point
    """

    @staticmethod
    def forward(ctx, input_tensor, clip_val, scale_sym, dequantize):
        ctx.save_for_backward(input_tensor, clip_val)

        if isinstance(clip_val, torch.Tensor):
            output = torch.where(
                input_tensor > clip_val,
                torch.ones_like(input_tensor) * clip_val,
                input_tensor,
            )
            output = torch.where(
                output < -clip_val, torch.ones_like(input_tensor) * (-clip_val), output
            )
        else:
            output = clamp(input_tensor, -clip_val, clip_val)

        output = torch.round(output / clip_val * scale_sym)
        if dequantize:
            output = output / scale_sym * clip_val

        ctx.scale = 1 if dequantize else scale_sym / clip_val
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input_tensor, clip_val = ctx.saved_tensors
        grad_input_tensor = grad_output.clone()
        grad_input_tensor = torch.where(
            input_tensor <= -clip_val,
            torch.zeros_like(grad_input_tensor),
            grad_input_tensor,
        )
        grad_input_tensor = torch.where(
            input_tensor >= clip_val,
            torch.zeros_like(grad_input_tensor),
            grad_input_tensor,
        )

        grad_alpha = grad_output.clone()
        grad_alpha = torch.where(
            torch.logical_and(input_tensor < clip_val, input_tensor > -clip_val),
            torch.zeros_like(grad_alpha),
            grad_alpha,
        )
        grad_alpha = grad_alpha.sum().expand_as(clip_val)

        scale = ctx.scale  # need to scale gradient if did not dequantize
        return grad_input_tensor * scale, grad_alpha * scale, None, None, None


class PACT_sw(nn.Module):
    """
    1-sided asymmetric PACT, compatible with swcap
    Store scaling to be accessed for dequantization (zero_point = 0)
    """

    def __init__(self, num_bits, init_clip_val=8.0, dequantize=False, align_zero=False):
        super().__init__()
        self.num_bits = num_bits
        self.clip_val = (
            nn.Parameter(init_clip_val)
            if isinstance(init_clip_val, torch.Tensor)
            else nn.Parameter(torch.Tensor([init_clip_val]))
        )
        self.dequantize = dequantize
        self.align_zero = align_zero
        self.register_buffer("scale", torch.Tensor([1.0]))
        self.quantizer = PACT_STE_sw

    def forward(self, input_tensor):
        output = self.quantizer.apply(
            input_tensor, self.clip_val, self.num_bits, self.dequantize, self.align_zero
        )

        with torch.no_grad():
            self.scale.fill_((2**self.num_bits - 1) / self.clip_val.item())
        return output

    def __repr__(self):
        return (
            f"PACT_sw(bits={self.num_bits}), clip={self.clip_val[0]:.4f}, "
            f"quantizer={self.quantizer})"
        )


class PACT_STE_sw(torch.autograd.Function):
    """
    1-sided original pact quantization for activation
    compatible with swcap (implemented grad scaling for dequantize = False)
    """

    @staticmethod
    def forward(ctx, input_tensor, clip_val, num_bits, dequantize, align_zero):
        assert input_tensor.min() >= 0, (
            "FMS: input_tensor to one-sided PACT" "should be non-negative."
        )
        ctx.save_for_backward(input_tensor, clip_val)
        scale, zero_point = asymmetric_linear_quantization_params(
            num_bits,
            saturation_min=0,
            saturation_max=clip_val.data,
            integral_zero_point=align_zero,
            signed=False,
        )
        if isinstance(clip_val, torch.Tensor):
            output = torch.where(
                input_tensor > clip_val,
                torch.ones_like(input_tensor) * clip_val,
                input_tensor,
            )
        else:
            output = clamp(input_tensor, 0, clip_val.data)
        output = linear_quantize(output, scale, zero_point)
        if dequantize:
            output = linear_dequantize(output, scale, zero_point)
            scale = 1
        ctx.scale = scale
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input_tensor, clip_val = ctx.saved_tensors
        grad_input_tensor = grad_output.clone()
        grad_input_tensor = torch.where(
            input_tensor >= clip_val,
            torch.zeros_like(grad_input_tensor),
            grad_input_tensor,
        )

        grad_alpha = grad_output.clone()
        grad_alpha = torch.where(
            input_tensor < clip_val, torch.zeros_like(grad_alpha), grad_alpha
        )
        grad_alpha = grad_alpha.sum().expand_as(clip_val)

        scale = ctx.scale  # need to scale gradient if did not dequantize
        return grad_input_tensor * scale, grad_alpha * scale, None, None, None


if Version(torch.__version__) >= Version("2.1"):
    # define the e4m3/e5m2 constants
    E4M3_MAX_POS = torch.finfo(torch.float8_e4m3fn).max
    E5M2_MAX_POS = torch.finfo(torch.float8_e5m2).max
    FP16_MAX_POS = torch.finfo(torch.float16).max
    EPS = 1e-12

    def to_fp8_saturated(
        x,
        float8_dtype=torch.float8_e4m3fn,
        orig_dtype=torch.bfloat16,
        emulate=True,
    ):
        """
        Directly clamping x by dtype.max
        FP8 dtype: torch.float8_e4m3fn or torch.float8_e5m2
        For emulation, i.e. the computation is done at orig_dtype
        """
        if float8_dtype == torch.float8_e4m3fn:
            x = x.clamp(min=-1 * E4M3_MAX_POS, max=E4M3_MAX_POS)
        else:
            x = x.clamp(min=-1 * E5M2_MAX_POS, max=E5M2_MAX_POS)
        if emulate:
            return x.to(float8_dtype).to(orig_dtype)
        return x.to(float8_dtype)

    def max_to_scale(
        x_max,
        float8_dtype,
        orig_dtype,
    ):
        scale = torch.empty_like(x_max, dtype=orig_dtype)
        if float8_dtype == torch.float8_e4m3fn:
            res = E4M3_MAX_POS / torch.clamp(x_max, min=EPS)
        else:
            res = E5M2_MAX_POS / torch.clamp(x_max, min=EPS)

        if orig_dtype is torch.float16:
            res = torch.clamp(res, max=torch.finfo(torch.float16).max)

        scale.copy_(res)
        return scale

    def to_fp8_scaled(
        x,
        float8_dtype=torch.float8_e4m3fn,
        orig_dtype=torch.float16,
        emulate=True,
    ):
        """
        x is scaled first to (dtype.max / x.max) then cast to FP8
        FP8 dtype: torch.float8_e4m3fn or torch.float8_e5m2
        For emulation, i.e. the computation is done at orig_dtype
        """
        x_max = torch.max(torch.abs(x))
        x_scale = max_to_scale(x_max, float8_dtype, orig_dtype)
        x_scaled = x * x_scale
        if emulate:
            return (
                to_fp8_saturated(x_scaled, float8_dtype, orig_dtype, emulate) / x_scale
            ).to(orig_dtype), x_scale
        return (
            to_fp8_saturated(x_scaled, float8_dtype, orig_dtype, emulate),
            x_scale,
        )

    def to_fp8_scaled_perCh(
        x,
        float8_dtype=torch.float8_e4m3fn,
        orig_dtype=torch.float16,
        emulate=True,
    ):
        """
        x is scaled first to (dtype.max / x.max) then cast to FP8
        Per-channel quantization for weights
        FP8 dtype: torch.float8_e4m3fn or torch.float8_e5m2
        For emulation, i.e. the computation is done at orig_dtype
        """
        max_perCh = torch.max(torch.abs(x), dim=1, keepdim=True)[0]
        x_scale = max_to_scale(max_perCh, float8_dtype, orig_dtype)
        x_scaled = x * x_scale
        if emulate:
            return (
                to_fp8_saturated(x_scaled, float8_dtype, orig_dtype, emulate) / x_scale
            ).to(orig_dtype), x_scale
        return (
            to_fp8_saturated(x_scaled, float8_dtype, orig_dtype, emulate),
            x_scale,
        )

    def to_fp8_scaled_perToken(
        x,
        float8_dtype=torch.float8_e4m3fn,
        orig_dtype=torch.float16,
        emulate=True,
    ):
        """
        x is scaled first to (dtype.max / x.max) then cast to FP8
        Per-Token quantization for activations
        FP8 dtype: torch.float8_e4m3fn or torch.float8_e5m2
        For emulation, i.e. the computation is done at orig_dtype
        """
        max_per_token = torch.max(torch.abs(x), dim=-1, keepdim=True)[0]
        x_scale = max_to_scale(max_per_token, float8_dtype, orig_dtype)
        x_scaled = x * x_scale
        if emulate:
            return (
                to_fp8_saturated(x_scaled, float8_dtype, orig_dtype, emulate) / x_scale
            ).to(orig_dtype), x_scale
        return (
            to_fp8_saturated(x_scaled, float8_dtype, orig_dtype, emulate),
            x_scale,
        )

    class to_fp8(nn.Module):
        def __init__(
            self,
            n_bits: int = 8,
            q_mode: str = "fp8_e4m3_sat",
            perCh: int = None,
            perToken: bool = False,
            emulate: bool = True,
            # scale_method: str = "max",
        ):
            """
            to cast tensor to torch float8 format
            """
            super().__init__()
            assert n_bits == 8, "only for 8-bit floating point conversion"
            self.q_mode = q_mode
            if "e4m3" in q_mode:
                self.float8_dtype = torch.float8_e4m3fn
            elif "e5m2" in q_mode:
                self.float8_dtype = torch.float8_e5m2G
            else:
                raise ValueError("FP8 only supports e4m3 and e5m2")
            self.emulate = emulate
            self.perCh = perCh  # for weight only
            self.perToken = perToken  # for activation only
            self.register_buffer(
                "scale", torch.zeros(perCh) if perCh else torch.Tensor([0.0])
            )  # TODO need to apply scale to self scale when needed.

        def forward(self, x: torch.Tensor):
            x_fp8 = None
            orig_dtype = x.dtype
            if self.perCh:
                x_fp8, _ = to_fp8_scaled_perCh(
                    x,
                    float8_dtype=self.float8_dtype,
                    orig_dtype=orig_dtype,
                    emulate=self.emulate,
                )
            elif self.perToken:
                x_fp8, _ = to_fp8_scaled_perToken(
                    x,
                    float8_dtype=self.float8_dtype,
                    orig_dtype=orig_dtype,
                    emulate=self.emulate,
                )
            else:
                if "_sat" in self.q_mode:
                    x_fp8 = to_fp8_saturated(
                        x,
                        float8_dtype=self.float8_dtype,
                        orig_dtype=orig_dtype,
                        emulate=self.emulate,
                    )
                elif "_scale" in self.q_mode:
                    x_fp8, _ = to_fp8_scaled(
                        x,
                        float8_dtype=self.float8_dtype,
                        orig_dtype=orig_dtype,
                        emulate=self.emulate,
                    )
                # NOTE when emulate=False, need to return scales for post gemm de-scale.
            return x_fp8

        def __repr__(self):
            return f"{self.__class__.__name__}(q_mode={self.q_mode}, emulate={self.emulate})"


# ====================================================================================
# FP8 casting customize to other fp behavior

MAX_FP8_E4M3 = 448
MAX_FP8_E5M2 = 98304


def custom_fp8_quantizer(
    x: torch.Tensor,
    bits: int = 8,
    mantissa_bits: int = 3,
    use_subnormal: bool = False,
    scale_to_max: bool = False,
) -> torch.Tensor:
    """Convert tensor tensor to FP8 format, remanining in decimal form (no binary conversion)
    and using some clever manipulation to round each tensor values to the closest representable
    FP8 value.

    Implement emulation only, always quantizing & dequantizing.

    Support use of subnormal or normal-only representation.
    Support tensor "saturation" (clamped to maximum representable value) or "scale_to_max"
    (the whole tensor is scaled up or down dynamically, such that its maximum matches the
    max representable value of selected format).

    Note: extending saturation is not functional if the max representable value is larger
    than the original dtype max value (as in custom e5m2), because the FP8-converted value is
    cast back to the original dtype."""

    assert bits == 8, "only 8-bit floating point supported"
    assert mantissa_bits in (3, 2)
    dtype_input_tensor = x.dtype
    x = x.to(
        torch.float32
    )  # HACK: clamp can't exceed format range (e.g. maxval > fp16_maxval)

    exp_bits = bits - 1 - mantissa_bits
    bias = 2 ** (exp_bits - 1) - 1

    x_max = MAX_FP8_E4M3 if mantissa_bits == 3 else MAX_FP8_E5M2
    if scale_to_max:
        stm = x_max / torch.clamp(
            x.max(), 1e-12
        )  # epsilon=1e-12 has no effect on FP16 but may limit scaling for BF16
        x *= stm
    else:
        x = torch.clamp(x, -x_max, x_max)

    # set lower bound for |x| (x_min)
    # no need to do it with subnormals, as x_min = scales_min = 2**(1 - b - m)
    if not use_subnormal:
        x_min = 2 ** (-bias) * (1 + 2 ** (-mantissa_bits))
        x = torch.where(
            torch.abs(x) < x_min,
            x_min * torch.round(x / x_min),
            x,
        )

    scales = 2 ** (
        torch.floor(torch.log2(torch.abs(x))) - mantissa_bits
    )  # x = 0 -> log2 = -inf -> scales = 0
    if use_subnormal:  # PYTORCH
        scales = torch.clamp(scales, min=2 ** (1 - bias - mantissa_bits))
    else:
        scales = torch.clamp(scales, min=2 ** (-bias - mantissa_bits))

    x_qdq = scales * torch.round(x / scales)

    if scale_to_max:
        x_qdq /= stm
    return x_qdq.to(
        dtype_input_tensor
    )  # NOTE: for e5m2, this will truncate back the outliers


class to_custom_fp8(nn.Module):
    def __init__(
        self,
        bits: int = 8,
        q_mode: str = "fp8_e4m3_custom",
        use_subnormal: bool = True,
        scale_to_max: bool = False,
    ):
        """Quantizer class to convert input_tensor to FP8 (simulated).
        Support customization in the use of subnormal (enabled/disabled).
        """
        super().__init__()
        assert bits == 8, "Only 8-bit floating point supported"
        self.bits = bits
        self.q_mode = q_mode
        if "e4m3" in q_mode:
            self.mantissa_bits = 3
        elif "e5m2" in q_mode:
            self.mantissa_bits = 2
        else:
            raise ValueError(f"Unsupported fp8 format: {q_mode} [options: e4m3, e5m2]")
        self.use_subnormal = use_subnormal
        self.scale_to_max = scale_to_max
        self.quantizer = custom_fp8_quantizer

    def forward(self, x: torch.Tensor):
        x_fp8 = self.quantizer(
            x,
            self.bits,
            self.mantissa_bits,
            self.use_subnormal,
            self.scale_to_max,
        )
        return x_fp8


# =====================================================================================


###for pruning
def mask_conv2d_kij(weight, group=4, prune_ratio=0.5):
    # NVIDIA style kij group pruning
    w_size = weight.shape
    weight_reshape = weight.clone().reshape(w_size[0] * w_size[1] // group, group, -1)
    weight_reshape = weight_reshape.permute(2, 0, 1).reshape(-1, group)
    mask = torch.ones(w_size[2] * w_size[3] * w_size[0] * w_size[1] // group, group).to(
        weight.device
    )
    mask = mask.scatter(
        1, weight_reshape.abs().argsort(1)[:, : int(group * prune_ratio)], 0
    )
    mask2d = mask.reshape(w_size[2], w_size[3], w_size[0], w_size[1])
    mask2d = mask2d.permute(2, 3, 0, 1)
    assert mask2d.shape == w_size
    return mask2d


def mask_fc_kij(weight, group, prune_ratio=0.50):
    w_size = weight.shape
    weight_reshape = weight.clone().reshape(w_size[0] * w_size[1] // group, group)
    mask = torch.ones(w_size[0] * w_size[1] // group, group).to(weight.device)
    mask = mask.scatter(
        1, weight_reshape.abs().argsort(1)[:, : int(group * prune_ratio)], 0
    )
    mask2d = mask.reshape(w_size[0], w_size[1])
    assert mask2d.shape == w_size
    return mask2d


class HardPrune(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_tensor, mask, inplace):
        ctx.save_for_backward(input_tensor, mask)
        if inplace:
            ctx.mark_dirty(input_tensor)
        with torch.no_grad():
            output = input_tensor.mul(mask)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        _input_tensor, mask = ctx.saved_tensors
        return grad_output.mul(mask), None, None
