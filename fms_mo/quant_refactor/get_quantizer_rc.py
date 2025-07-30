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
Functions to create quantizers for activation and weights.  Called from Qmodule level.
"""

# Third Party
import torch

# Local
from fms_mo.quant.quantizers import (
    AdaRoundQuantizer,
    PerTokenMax,
    QFixSymmetric,
    UniformAffineQuantizer,
    to_custom_fp8,
    to_fp8,
)
from fms_mo.quant_refactor.base_quant import Qscheme
from fms_mo.quant_refactor.lsq_rc import LSQPlus_rc, LSQQuantization_rc
from fms_mo.quant_refactor.pact2_rc import PACT2_rc
from fms_mo.quant_refactor.pact2sym_rc import PACT2Sym_rc
from fms_mo.quant_refactor.pact_rc import PACT_rc
from fms_mo.quant_refactor.pactplussym_rc import PACTplusSym_rc
from fms_mo.quant_refactor.qmax_rc import Qmax_rc
from fms_mo.quant_refactor.sawb_rc import SAWB_rc


def get_activation_quantizer_rc(
    qa_mode: str = "PACT",
    nbits: int = 32,
    clip_val: torch.Tensor = None,
    clip_valn: torch.Tensor = None,
    non_neg: bool = False,
    align_zero: bool = True,  # pylint: disable=unused-argument
    extend_act_range: bool = False,
    use_PT_native_Qfunc: bool = False,
    use_subnormal: bool = False,
):
    """Return a quantizer for activation quantization
    Regular quantizers:
    - pact, pact2 (non_neg, pact+)
    - pactsym/pactsym+
    - max, minmax, maxsym
    - lsq+, lsq (inactive), fix
    - brecq (PTQ)
    """

    QPACTLUT = {
        "pact_uni": PACT_rc,
        "pact_bi": PACT2_rc,
        "pact+_uni": PACT_rc,
        "pact+_bi": PACT2_rc,
    }
    if "pact" in qa_mode and "sym" not in qa_mode:
        keyQact = qa_mode + "_uni" if non_neg else qa_mode + "_bi"
        pact_plus = "pact+" in qa_mode
        act_quantizer = QPACTLUT[keyQact](
            num_bits=nbits,
            init_clip_valn=clip_valn,
            init_clip_val=clip_val,
            Qscheme=Qscheme(
                unit="perT",
                symmetric="sym" in qa_mode,
                Nch=None,
                Ngrp=None,
                single_sided="uni" in keyQact,
                qlevel_lowering=False,
            ),
            dequantize=True,
            pact_plus=pact_plus,
            use_PT_native_Qfunc=use_PT_native_Qfunc,
        )
    elif qa_mode == "pactsym":
        act_quantizer = PACT2Sym_rc(
            num_bits=nbits,
            init_clip_val=clip_val,
            Qscheme=Qscheme(
                unit="perT",
                symmetric=True,
                Nch=None,
                Ngrp=None,
                single_sided=False,
                qlevel_lowering=False,
            ),
            dequantize=True,
            use_PT_native_Qfunc=use_PT_native_Qfunc,
        )
    elif qa_mode == "pactsym+":
        act_quantizer = PACTplusSym_rc(
            nbits,
            init_clip_val=clip_val,
            Qscheme=Qscheme(
                unit="perT",
                symmetric=True,
                Nch=None,
                Ngrp=None,
                single_sided=False,
                qlevel_lowering=False,
            ),
            dequantize=True,
            extend_act_range=extend_act_range,
            use_PT_native_Qfunc=use_PT_native_Qfunc,
        )
    elif qa_mode == "lsq+":
        act_quantizer = LSQPlus_rc(
            nbits,
            init_clip_valn=clip_valn,
            init_clip_val=clip_val,
            Qscheme=Qscheme(
                unit="perT",
                symmetric=False,
                Nch=None,
                Ngrp=None,
                single_sided=False,
                qlevel_lowering=False,
            ),
            dequantize=True,
            # use_PT_native_Qfunc=use_PT_native_Qfunc, # nativePT not enabled for LSQ+
        )
    elif qa_mode == "lsq":
        act_quantizer = LSQQuantization_rc(
            nbits, init_clip_val=clip_val, dequantize=True, inplace=False
        )
    # NOTE: need to be careful using this for activation, particular to 1 sided.
    elif qa_mode == "max":
        act_quantizer = Qmax_rc(
            nbits,
            Qscheme=Qscheme(
                unit="perT",
                symmetric=False,
                Nch=None,
                Ngrp=None,
                single_sided=False,
                qlevel_lowering=False,
            ),
            dequantize=True,
            minmax=False,
            use_PT_native_Qfunc=use_PT_native_Qfunc,
        )
    elif qa_mode == "minmax":
        act_quantizer = Qmax_rc(
            nbits,
            Qscheme=Qscheme(
                unit="perT",
                symmetric=False,
                Nch=None,
                Ngrp=None,
                single_sided=False,
                qlevel_lowering=False,
            ),
            dequantize=False,
            minmax=True,
            use_PT_native_Qfunc=use_PT_native_Qfunc,
        )
    elif qa_mode == "maxsym":
        act_quantizer = Qmax_rc(
            nbits,
            Qscheme=Qscheme(
                unit="perT",
                symmetric=True,
                Nch=None,
                Ngrp=None,
                single_sided=False,
                qlevel_lowering=False,
            ),
            minmax=False,
            extend_act_range=extend_act_range,
            use_PT_native_Qfunc=use_PT_native_Qfunc,
        )

    # Following quantizers do not have unit tests
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

    return act_quantizer


def get_weight_quantizer_rc(
    qw_mode: str = "SAWB+",
    nbits: int = 32,
    clip_val: torch.Tensor = None,
    clip_valn: torch.Tensor = None,
    align_zero: bool = True,
    w_shape: torch.Size = None,
    recompute: bool = False,  # pylint: disable=unused-argument
    perGp: int = None,
    use_PT_native_Qfunc: bool = False,
    use_subnormal: bool = False,
):
    """Return a quantizer for weight quantization
    Regular quantizers:
    - sawb (16, perCh, +)
    - max, minmax
    - pact, pact+
    - lsq+, fix, dorefa
    - brecq, adaround
    """

    Nch = w_shape[0] if w_shape is not None and "perCh" in qw_mode else False
    Ngrp = (
        [w_shape[0] * w_shape[1] // perGp, perGp] if "perGp" in qw_mode else False
    )  # store clip_val size and group size
    unit = "perCh" if Nch is not False else "perGrp" if perGp is not None else "perT"
    if "sawb" in qw_mode:
        clipSTE = "+" in qw_mode
        weight_quantizer = SAWB_rc(
            nbits,
            Qscheme=Qscheme(
                unit=unit,
                symmetric=True,
                Nch=Nch if Nch is not False else None,
                Ngrp=Ngrp if Ngrp is not False else None,
                single_sided=False,
                qlevel_lowering=align_zero,
            ),
            dequantize=True,
            clipSTE=clipSTE,
            use_PT_native_Qfunc=use_PT_native_Qfunc,
        )
    elif "max" in qw_mode:
        weight_quantizer = Qmax_rc(
            nbits,
            Qscheme=Qscheme(
                unit=unit,
                symmetric=True,
                Nch=Nch if Nch is not False else None,
                Ngrp=Ngrp if Ngrp is not False else None,
                single_sided=False,
                qlevel_lowering=align_zero,
            ),
            minmax="min" in qw_mode,
            use_PT_native_Qfunc=use_PT_native_Qfunc,
        )
    elif qw_mode == "pact":
        weight_quantizer = PACT2_rc(
            nbits,
            init_clip_valn=clip_valn,
            init_clip_val=clip_val,
            Qscheme=Qscheme(
                unit="perT",
                symmetric=True,  # Assumed symmetric for weights
                Nch=Nch if Nch is not False else None,
                Ngrp=Ngrp if Ngrp is not False else None,
                single_sided=False,
                qlevel_lowering=align_zero,
            ),
            dequantize=True,
            use_PT_native_Qfunc=use_PT_native_Qfunc,
        )
    elif qw_mode == "pact+":
        weight_quantizer = PACTplusSym_rc(
            nbits,
            init_clip_val=clip_val,
            Qscheme=Qscheme(
                unit="perT",
                symmetric=True,  # Assumed symmetric for weights
                Nch=Nch if Nch is not False else None,
                Ngrp=Ngrp if Ngrp is not False else None,
                single_sided=False,
                qlevel_lowering=align_zero,
            ),
            dequantize=True,
            use_PT_native_Qfunc=use_PT_native_Qfunc,
        )
    elif qw_mode == "lsq+":
        weight_quantizer = LSQPlus_rc(
            nbits,
            init_clip_valb=clip_valn,
            init_clip_vals=clip_val,
            Qscheme=Qscheme(
                unit="perT",
                symmetric=True,  # Assumed symmetric for weights
                Nch=Nch if Nch is not False else None,
                Ngrp=Ngrp if Ngrp is not False else None,
                single_sided=False,
                qlevel_lowering=align_zero,
            ),
            dequantize=True,
            use_PT_native_Qfunc=use_PT_native_Qfunc,
        )

    # Following quantizers do not have unit testing
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
            # qw_mode should be one of
            # [fp8_e4m3_sat, fp8_e5m2_sat, fp8_e4m3_scale, fp8_e5m2_scale] + 'perCh'
            # by default, emulate = True, unless using a GPU that support FP8 computation
            # NOTE: emulate will be similar to dequantize.
            Nch = w_shape[0] if w_shape is not None and "perCh" in qw_mode else False
            weight_quantizer = to_fp8(
                nbits,
                q_mode=qw_mode,
                emulate=True,
                perCh=Nch,
            )
    else:
        raise ValueError(f"unrecognized weight quantized mode {qw_mode}")

    return weight_quantizer
