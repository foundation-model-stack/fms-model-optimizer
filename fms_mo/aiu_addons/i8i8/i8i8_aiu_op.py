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
"""Registration of INT8xINT8 node compatible with AIU compiler"""

# Standard
import logging

# Third Party
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# pylint: disable=unused-argument
# i8i8 op must be registered with specific I/O, even if not in use by the op function

# pylint: disable=not-callable
# torch.nn.functional.linear not recognized as callable
# open issue in PyLint: https://github.com/pytorch/pytorch/issues/119482


def register_aiu_i8i8_op():
    """Register AIU-specific op to enable torch compile without graph break.
    The op preserves I/O shapes of a `X @ W^T` matmul but performs no operation.

    It takes weights and activations, their quantization boundaries (clip val),
    as well as smoothquant parameters as arguments. All arguments are explicitly
    attached to the computational graph.
    """
    if hasattr(torch.ops, "fms_mo") and hasattr(torch.ops.fms_mo, "i8i8_aiu"):
        logger.warning("AIU op has already been registered")
        return

    op_namespace_id = "fms_mo::i8i8_aiu"
    torch.library.define(
        op_namespace_id,
        "(Tensor x, Tensor weight, Tensor bias, Tensor qdata, "
        "str weight_quant_type, str activ_quant_type, "
        "bool smoothquant) "
        "-> Tensor",
    )

    @torch.library.impl(op_namespace_id, "default")
    def i8i8_aiu(
        x,
        weight,
        bias,
        qdata,
        weight_quant_type,
        activ_quant_type,
        smoothquant,
    ):
        """Implement addmm of X and W.
        Support various quantization options for weights and activations.

        X -> quant -> dequant -> X_dq
        W_int8 -> dequant -> W_dq
        Y = X @ W^T + B
        """
        dtype = x.dtype
        out_feat, in_feat = weight.size()

        # unused returns are w_cvn and zero_shift
        w_cv, _, a_cv, a_cvn, _, sq = extract_qdata(
            qdata,
            weight_quant_type,
            activ_quant_type,
            w_in_feat=in_feat,
            w_out_feat=out_feat,
            smoothquant=smoothquant,
        )

        x_dq = quant_dequant_activ(x, a_cv, a_cvn, sq, activ_quant_type)
        w_dq = dequant_weights(weight, w_cv, sq, weight_quant_type)

        return F.linear(x_dq.to(dtype), w_dq.to(dtype), bias.to(dtype))

    @torch.library.impl_abstract(op_namespace_id)
    def i8i8_aiu_abstract(
        x,
        weight,
        bias,
        qdata,
        weight_quant_type,
        activ_quant_type,
        smoothquant,
    ):
        """OP template of I/O sizes"""

        outshape = x.size()[:-1] + (weight.size(0),)
        return torch.empty(
            outshape, dtype=x.dtype, device=x.device, requires_grad=False
        )

    logger.info("W8A8 op `i8i8_aiu` has been registered under the `fms_mo` namespace")
    return


def extract_qdata(
    qdata: torch.Tensor,
    weight_quant_type: str,
    activ_quant_type: str,
    w_in_feat: int,
    w_out_feat: int,
    smoothquant: bool,
) -> tuple[torch.Tensor, ...]:
    """6 tensors are to be de-concatenated from qdata:
    w_clip_val      [    : idx1]
    w_clip_valn     [idx1: idx2]
    a_clip_val      [idx2: idx3]
    a_clip_valn     [idx3: idx4]
    zero_shift      [idx4: idx5]
    smoothquant     [idx5:     ]
    """

    if weight_quant_type == "per_tensor":
        idx1 = 1
        idx2 = idx1 + 1
    elif weight_quant_type == "per_channel":
        idx1 = w_out_feat
        idx2 = idx1 + w_out_feat
    else:
        raise NotImplementedError(
            f"weight quantizantion type {weight_quant_type} is not supported"
        )

    # all activ_quant_type result in 1-element clip
    idx3 = idx2 + 1
    idx4 = idx3 + 1

    if activ_quant_type == "per_tensor_asymm":
        idx5 = idx4 + w_out_feat
    else:  # per_tensor_symm or per_token
        idx5 = idx4 + 1

    if smoothquant:
        idx6 = idx5 + w_in_feat
    else:
        idx6 = idx5 + 1

    if len(qdata) != idx6:
        raise ValueError(
            f"qdata length ({len(qdata)}) does not match series of indices for split:"
            f"[{idx1}, {idx2}, {idx3}, {idx4}, {idx5}, {idx6}]"
        )

    return torch.tensor_split(qdata, (idx1, idx2, idx3, idx4, idx5))


def dequant_weights(
    weight: torch.Tensor,
    w_cv: torch.Tensor,
    sq: torch.Tensor,
    weight_quant_type: str,
) -> torch.Tensor:
    """Dequantize integer weights based on quantizer type"""

    if weight_quant_type == "per_tensor":  # assume 8-bit symmetric W quantization
        # w size: (out_feat, in_feat)
        # sq size: (in_feat) or (1), no need to unsqueeze
        return (weight * w_cv / 127) / sq
    if weight_quant_type == "per_channel":
        # w_cv is (out_feat), need to unsqueeze to broadcast mul to weight
        return (weight * w_cv.unsqueeze(dim=1) / 127) / sq
    raise NotImplementedError(
        f"weight quantizantion type {weight_quant_type} is not supported"
    )


def quant_dequant_activ(
    x: torch.Tensor,
    a_cv: torch.Tensor,
    a_cvn: torch.Tensor,
    sq: torch.Tensor,
    activ_quant_type: str,
) -> torch.Tensor:
    """
    Quantize and dequantize activations based on quantizer type

    x size    (*, hid_dim)
    sq size   (hid_dim) or (1)
    => no need to unsqueeze to perform x / sq
    """
    if activ_quant_type == "per_tensor_symm":
        scale_x = 127 / a_cv
        x_int = torch.round(x / sq * scale_x).clamp(-127, 127).to(torch.int8)
        return x_int.div(scale_x).mul(sq)
    if activ_quant_type == "per_tensor_asymm":
        scale_x = 255 / (a_cv - a_cvn)
        zp_x = a_cvn * scale_x
        x_int = torch.round(x / sq * scale_x - zp_x).clamp(0, 255)
        return x_int.add(zp_x).div(scale_x).mul(sq)
    if activ_quant_type == "per_token":
        x_sq = x / sq
        a_cv_per_token = x_sq.abs().max(dim=-1, keepdim=True)[0]
        scale_x = 127 / a_cv_per_token
        x_int = torch.round(x_sq * scale_x).clamp(-127, 127)
        return x_int.div(scale_x).mul(sq)
    raise NotImplementedError(
        f"activation quantizantion type {activ_quant_type} is not supported"
    )
