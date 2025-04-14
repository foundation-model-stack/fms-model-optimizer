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

# Standard
from pathlib import Path
import logging

# Third Party
from transformers.modeling_utils import PreTrainedModel
import torch

# logging is only enabled for verbose output (performance is less critical during debug),
# and f-string style logging is preferred for code readability
# pylint: disable=logging-not-lazy


logger = logging.getLogger()


def get_quantized_linear_names(model_type: str) -> list[str]:
    """Return a list of unique identifiers for the linear layers in a given model."""

    if model_type in ["granite", "llama"]:
        return [
            "self_attn.q_proj",
            "self_attn.k_proj",
            "self_attn.v_proj",
            "self_attn.o_proj",
            "mlp.gate_proj",
            "mlp.up_proj",
            "mlp.down_proj",
        ]
    if model_type == "gpt_bigcode":
        return [
            "attn.c_attn",
            "attn.c_proj",
            "mlp.c_fc",
            "mlp.c_proj",
        ]
    if model_type in ["bert", "roberta"]:
        return [
            "attention.self.query",
            "attention.self.key",
            "attention.self.value",
            "attention.output.dense",
            "intermediate.dense",
            "output.dense",
        ]
    raise NotImplementedError(
        f"Model type {model_type} is not supported for quantized checkpoint saving"
    )


def convert_sd_for_aiu(
    model: PreTrainedModel,
    verbose: bool,
) -> dict[str, torch.Tensor]:
    """Convert the state dictionary (sd) of an FMS-MO-quantized model into a format
    compatible with the AIU.

    Expected tensors in input state dictionary:
    weights:
        [out_feat, in_feat]
    w_cv:
        perT   [1]
        perCh  [out_feat]
        w_cvn = - w_cv   <--- always symmetric!
    a_cv:
        per-token-max   n/a
        perT            [1]
        a_cvn: symmetric or asymmetric

    Smoothquant combined scale is computed as:
        s_sq = a_sq_scale ^ alpha / w_sq_scale ^ (1- alpha)

    All parameters except quantized weights are cast to FP16, per AIU requirement.
    """

    if verbose:
        logger.info("Before conversion:")
        logger.info("* ALL MODEL PARAMETERS (name, size, dtype)")
        logger.info(
            "\n"
            + "\n".join(
                f"{k:80} {str(list(v.size())):15} {v.dtype}"
                for k, v in model.named_parameters()
            )
        )
        logger.info("* ALL BUFFERS (name, size, dtype)")
        logger.info(
            "\n"
            + "\n".join(
                f"{k:80} {str(list(v.size())):15} {v.dtype}"
                for k, v in model.named_buffers()
            )
        )
        logger.info("=" * 60)

    model_type = getattr(model.config, "model_type", None)
    if model_type:
        quantized_layers = get_quantized_linear_names(model_type)
    else:
        raise ValueError(
            "Could not determine model type to save quantized state dictionary."
        )
    excluded_keys_from_new_sd = [
        "calib_counter",
        "num_module_called",
        "smoothq_act_scale",
        "smoothq_alpha",
        "obsrv_w_clipval",
        "obsrv_clipval",
        "obsrv_clipvaln",
    ]

    new_sd = {}
    for k, v in model.state_dict().items():
        if k.endswith(".weight") and any(qlayer in k for qlayer in quantized_layers):
            layername = k[:-7]

            # smoothquant processing:
            # - if smoothquant wasn't used, smoothq_alpha doesn't exist or is zero
            # - compute combined weight/activation smoothquant scaling factor (sq_scale)
            # - rescale weights before quantization
            # - store scaling factor into state dict
            v_scaled = None
            if layername + ".smoothq_alpha" in model.state_dict():
                sq_a_scale = model.state_dict()[layername + ".smoothq_act_scale"]
                if sum(sq_a_scale) != 0:
                    sq_alpha = model.state_dict()[layername + ".smoothq_alpha"]
                    sq_w_scale = v.abs().max(dim=0, keepdim=True).values.clamp(min=1e-5)
                    sq_scale = sq_a_scale.pow(sq_alpha) / sq_w_scale.pow(1 - sq_alpha)
                    v_scaled = v * sq_scale  # weights sq-scaled before quantization
                    # guarding FP16 casting
                    if sq_scale.abs().max() > torch.finfo(torch.float16).max:
                        raise ValueError(
                            "Quantization parameters (qscale) exceeds float16 range. "
                            "Aborted state dict saving."
                        )
                    new_sd[layername + ".smoothq_scale"] = (
                        sq_scale.squeeze().to(torch.float16).to("cpu")
                    )

            # quantize weights and store them into state dict
            if layername + ".quantize_weight.clip_val" in model.state_dict():
                w_cv = model.state_dict()[layername + ".quantize_weight.clip_val"]
                if w_cv.numel() > 1:
                    w_cv = w_cv.unsqueeze(dim=1)
                weight_pre_quant = v_scaled if v_scaled is not None else v
                weight_int = torch.clamp(
                    127 / w_cv * weight_pre_quant, -127, 127
                ).round()
                new_sd[k] = weight_int.to(torch.int8).to("cpu")  # signed int8

            a_cv_name = layername + ".quantize_feature.clip_val"
            a_cvn_name = a_cv_name + "n"
            a_cv = None
            a_cvn = None
            if a_cv_name in model.state_dict():
                a_cv = model.state_dict()[a_cv_name]
                if a_cvn_name in model.state_dict():
                    a_cvn = model.state_dict()[a_cvn_name]

                # compute "zero_shift" correction factor only for asymmetric activations
                if a_cv and a_cvn and a_cv != -a_cvn:
                    if v.dim() == 2:
                        # weight_int: [out_feat, in_feat]
                        # sum (squash) along in_feat dimension: dim=1
                        new_sd[layername + ".zero_shift"] = (
                            torch.sum(
                                weight_int,
                                dim=1,
                            )
                            .to(torch.float16)
                            .to("cpu")
                        )
                    else:
                        raise NotImplementedError(
                            "Zero shift computation for tensor "
                            "with more than 2 dims is not supported yet."
                        )
        elif all(excluded_key not in k for excluded_key in excluded_keys_from_new_sd):
            # guarding FP16 cast
            if v.abs().max() > torch.finfo(torch.float16).max:
                raise ValueError(
                    f"Quantization parameters ({k}) exceeds float16 range. "
                    "Aborted state dict saving."
                )
            new_sd[k] = v.to("cpu").to(torch.float16)

    logger.info("New state dict processed.")
    if verbose:
        logger.info(
            "\n"
            + "\n".join(
                f"{k:80} {str(list(v.size())):15} "
                f"{str(v.dtype):18} {str(v.device):10} "
                f"{v.reshape(-1)[0].item():12.4f} "
                f"{v.min().item():12.4f} {v.max().item():12.4f}"
                for k, v in new_sd.items()
            )
        )

    return new_sd


def save_sd_for_aiu(
    model: PreTrainedModel,
    output_dir: str,
    savename: str = "qmodel_state_dict.pt",
    verbose: bool = False,
) -> None:
    """Save model state dictionary after conversion for AIU compatibility."""

    converted_sd = convert_sd_for_aiu(model, verbose)
    torch.save(converted_sd, Path(output_dir) / savename)
    logger.info("Model saved.")
