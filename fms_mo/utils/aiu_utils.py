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
from transformers.modeling_utils import PreTrainedModel  # type: ignore[import-untyped]
import torch

# Local
from fms_mo.quant.quantizers import SAWB
from fms_mo.utils.qconfig_utils import qconfig_save

# logging is only enabled for verbose output (performance is less critical during debug),
# and f-string style logging is preferred for code readability
# pylint: disable=logging-not-lazy


# empirical threshold to standard deviation of INT weights to trigger their recomputation
STD_THRESHOLD = 20

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


def print_params(sd: dict) -> None:
    """Print to logger some info and stats of all items in provided state dictionary."""

    logger.info(
        "\n"
        + "\n".join(
            f"{k:80} {str(list(v.size())):15} "
            f"{str(v.dtype):18} {str(v.device):10} "
            f"{v.reshape(-1)[0].item():12.4f} "
            f"{v.min().item():12.4f} {v.max().item():12.4f}"
            for k, v in sd.items()
        )
    )


def process_smoothquant(
    model: PreTrainedModel,
    layer_name: str,
    new_sd: dict,
    verbose: bool = False,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Check if smoothquant was in use and, if so:
    1. compute combined weight/activation scaling factor
    2. store it in new_sd dictionary
    3. return scaled weights and smoothquant activation scale (for future use)

    If smoothquant was not in use, activation scale does not exist or sums to zero.
    In this case, a (None, None) tuple is returned.
    """

    weight_scaled = None
    sq_a_scale = None
    w = model.state_dict()[layer_name + ".weight"]
    if layer_name + ".smoothq_alpha" in model.state_dict():
        sq_a_scale = model.state_dict()[layer_name + ".smoothq_act_scale"]
        if sum(sq_a_scale) != 0:
            sq_alpha = model.state_dict()[layer_name + ".smoothq_alpha"]
            sq_w_scale = w.abs().max(dim=0, keepdim=True).values.clamp(min=1e-5)
            sq_scale = sq_a_scale.pow(sq_alpha) / sq_w_scale.pow(1 - sq_alpha)
            weight_scaled = w * sq_scale  # weights sq-scaled before quantization
            # guarding FP16 casting
            if sq_scale.abs().max() > torch.finfo(torch.float16).max:
                raise ValueError(
                    "Quantization parameters (qscale) exceeds float16 range. "
                    "Aborted state dict saving."
                )
            k = layer_name + ".smoothq_scale"
            if verbose:
                logger.info(f"  Save key: {k}")
            new_sd[k] = sq_scale.squeeze().to(torch.float16).to("cpu")
    return weight_scaled, sq_a_scale


def recompute_weight_with_sawb(
    weight_pre_quant: torch.Tensor,
    weight_int_as_fp: torch.Tensor,
    weight_per_channel: bool,
    sq_a_scale: torch.Tensor | None,
    layer_name: str,
    new_sd: dict,
    verbose: bool = False,
) -> tuple[torch.Tensor | None, bool]:
    """Use SAWB quantizer to recompute weights showing narrow distributions in the
    integer domain.
    """

    weight_pre_quant = weight_pre_quant.to("cpu")
    weight_int_as_fp = weight_int_as_fp.to("cpu")

    is_w_recomputed = False
    weight_int_sawb: torch.Tensor | None = None
    weight_int_std: torch.Tensor | float | None = None
    if weight_per_channel:
        # recompute if any channel shows narrow int weights
        weight_int_std = weight_int_as_fp.std(dim=-1)
        weight_int_std_min = weight_int_std.min()
        recompute = any(w < STD_THRESHOLD for w in weight_int_std)
    else:
        # recompute if full tensor shows narrow int weights
        weight_int_std = weight_int_as_fp.std().item()
        recompute = weight_int_std < STD_THRESHOLD

    if recompute:
        is_w_recomputed = True
        if sq_a_scale is not None and sum(sq_a_scale) != 0:
            # TODO: add support for smoothquant
            raise ValueError(
                "Weight recomputation while smoothquant is in use is "
                "not yet supported."
            )

        # 1. Select an SAWB quantizer for weight recomputation
        quantizer = SAWB(
            num_bits=8,
            dequantize=False,
            align_zero=True,
            perCh=weight_int_as_fp.size(0) if weight_per_channel else False,
        )
        quantizer.training = True  # set SAWB to recompute clips
        # some SAWB quantizers only process FP32 inputs, so weights are
        # temporarily upscaled
        weight_int_sawb = quantizer(weight_pre_quant.to(torch.float32))
        assert weight_int_sawb is not None

        # 2. Recompute clip values using new SAWB quantizer
        w_cv_key = layer_name + ".quantize_weight.clip_val"
        w_cvn_key = layer_name + ".quantize_weight.clip_valn"
        if verbose:
            logger.info(
                f"  {'Overwrite' if w_cv_key in new_sd else 'Add'} key: {w_cv_key}"
            )
            logger.info(
                f"  {'Overwrite' if w_cvn_key in new_sd else 'Add'} key: {w_cvn_key}"
            )

        cv_sawb = quantizer.clip_val.to("cpu").to(torch.float16)
        if weight_per_channel:
            # Select SAWB rows only where clip value does not exceed row max
            cv_max = weight_pre_quant.abs().max(dim=-1)[0]
            weight_int_guarded = torch.where(
                (cv_sawb < cv_max)[:, None],
                weight_int_sawb,
                weight_int_as_fp,
            )
            cv_guarded = torch.where(cv_sawb < cv_max, cv_sawb, cv_max)
            weight_int_sawb = weight_int_guarded
        else:
            cv_max = weight_pre_quant.abs().max()
            weight_int_guarded = (
                weight_int_sawb if cv_sawb < cv_max else weight_int_as_fp
            )
            cv_guarded = torch.min(cv_sawb, cv_max)

        new_sd[w_cv_key] = cv_guarded
        new_sd[w_cvn_key] = -cv_guarded

        # 3. [optional] Recompute standard deviation of integer weights
        if verbose:
            weight_int_sawb_as_fp = weight_int_guarded.to(torch.float32)
            if weight_per_channel:
                weight_int_sawb_std_min = weight_int_sawb_as_fp.std(dim=-1).min()
                if verbose:
                    logger.info(
                        "  Reprocessed weights "
                        f"(std_min={weight_int_std_min:.1f} "
                        f"-> {weight_int_sawb_std_min:.1f}) "
                        f"and clips of {layer_name + '.weight'}"
                    )
            else:
                weight_int_sawb_as_fp_std = weight_int_sawb_as_fp.std()
                if verbose:
                    logger.info(
                        "  Reprocessed weights "
                        f"(std={weight_int_std:.1f} "
                        f"-> {weight_int_sawb_as_fp_std:.1f}) "
                        f"and clips of {layer_name + '.weight'}"
                    )
    elif verbose:
        log_min_std = "min_" if weight_per_channel else ""
        log_w_std = weight_int_std_min if weight_per_channel else weight_int_std
        logger.info(f"  Weights preserved ({log_min_std}std={log_w_std:.1f})")

    return weight_int_sawb, is_w_recomputed


def process_weight(
    model: PreTrainedModel,
    layer_name: str,
    weight_pre_quant: torch.Tensor,
    recompute_narrow_weights: bool,
    weight_per_channel: bool,
    sq_a_scale: torch.Tensor | None,
    new_sd: dict,
    verbose: bool = False,
) -> tuple[torch.Tensor | None, bool | None]:
    """Compute integer weights and store them into new state dictionary.
    If recomputation is enabled, int weights are updated using SAWB quantizer.
    """

    # in most scenarios, weights are quantized, so clip_val exists
    weight_int = None
    is_w_recomputed = False
    if layer_name + ".quantize_weight.clip_val" in model.state_dict():
        w_cv = model.state_dict()[layer_name + ".quantize_weight.clip_val"]

        # Check that clip values are initialized
        if torch.any(w_cv.isclose(torch.tensor(0.0))):
            raise ValueError(
                f"Quantization clip values for {layer_name=} have near-zero values and "
                "are likely uninitialized."
            )

        if w_cv.numel() > 1:
            w_cv = w_cv.unsqueeze(dim=1)
        weight_int_as_fp = torch.clamp(127 / w_cv * weight_pre_quant, -127, 127).round()

        weight_int_sawb = None
        if recompute_narrow_weights:
            weight_int_sawb, is_w_recomputed = recompute_weight_with_sawb(
                weight_pre_quant,
                weight_int_as_fp,
                weight_per_channel,
                sq_a_scale,
                layer_name,
                new_sd,
                verbose,
            )

        weight_int = (
            weight_int_sawb if weight_int_sawb is not None else weight_int_as_fp
        )
        new_sd[layer_name + ".weight"] = weight_int.to(torch.int8).to("cpu")

    return weight_int, is_w_recomputed


def process_zero_shift(
    model: PreTrainedModel,
    layer_name: str,
    weight_int: torch.Tensor | None,
    new_sd: dict,
    verbose: bool = False,
) -> None:
    """Compute and store the zero shift, a correction factor that compensates the
    output of (W integer, X integer) matmuls to match the corresponding FP operation.

    Only needed if activations are asymmetrically quantized.
    """

    k = layer_name + ".zero_shift"
    a_cv_name = layer_name + ".quantize_feature.clip_val"
    a_cvn_name = a_cv_name + "n"
    a_cv = None
    a_cvn = None
    if a_cv_name in model.state_dict():
        a_cv = model.state_dict()[a_cv_name]
        if a_cvn_name in model.state_dict():
            a_cvn = model.state_dict()[a_cvn_name]

        # compute "zero_shift" correction factor only for asymmetric activations
        if not (a_cv is None or a_cvn is None or torch.equal(a_cv, -a_cvn)):
            if weight_int is None:
                logger.info(
                    f"As weights appear to be not quantized, zero shift for {k} "
                    "will not be generated."
                )
            elif weight_int.dim() == 2:
                # weight_int: [out_feat, in_feat]
                # sum (squash) along in_feat dimension: dim=1
                zero_shift = torch.sum(weight_int, dim=1)

                if verbose:
                    logger.info(f"  Save key: {k}")

                # zero shift can exceed FP16 max value, especially if INT weights have
                # been recomputed, so it is saved as FP32
                new_sd[k] = zero_shift.to(torch.float32).to("cpu")
            else:
                raise NotImplementedError(
                    "Zero shift computation for tensor "
                    "with more than 2 dims is not supported yet."
                )


def convert_sd_for_aiu(
    model: PreTrainedModel,
    recompute_narrow_weights: bool = False,
    weight_per_channel: bool = False,
    verbose: bool = False,
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
        logger.info("Parameters before conversion")
        print_params(model.state_dict())
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

    new_sd: dict = {}
    num_w_recomputed = 0
    num_w_preserved = 0
    for k, v in model.state_dict().items():
        if verbose:
            logger.info(f"Processing key: {k}")
        if k.endswith(".weight") and any(qlayer in k for qlayer in quantized_layers):
            layer_name = k[:-7]

            v_scaled, sq_a_scale = process_smoothquant(
                model=model,
                layer_name=layer_name,
                new_sd=new_sd,
                verbose=verbose,
            )

            weight_int, is_w_recomputed = process_weight(
                model=model,
                layer_name=layer_name,
                weight_pre_quant=v_scaled if v_scaled is not None else v,
                recompute_narrow_weights=recompute_narrow_weights,
                weight_per_channel=weight_per_channel,
                sq_a_scale=sq_a_scale,
                new_sd=new_sd,
                verbose=verbose,
            )
            if is_w_recomputed:
                num_w_recomputed += 1
            else:
                num_w_preserved += 1

            process_zero_shift(
                model=model,
                layer_name=layer_name,
                weight_int=weight_int,
                new_sd=new_sd,
                verbose=verbose,
            )

        elif all(excluded_key not in k for excluded_key in excluded_keys_from_new_sd):
            if k not in new_sd:
                # guarding FP16 cast
                if v.abs().max() > torch.finfo(torch.float16).max:
                    raise ValueError(
                        f"Quantization parameters ({k}) exceeds float16 range. "
                        "Aborted state dict saving."
                    )
                logger.info(f"  Save key: {k}")
                new_sd[k] = v.to("cpu").to(torch.float16)
            else:
                logger.info(f"  Skip parameter already processed: {k}")

    logger.info("New state dict processed.")
    if verbose:
        logger.info("Parameters after conversion")
        print_params(new_sd)
        logger.info("=" * 60)

    if recompute_narrow_weights:
        logger.info(
            f"Recomputed {num_w_recomputed} weight matrices with SAWB, "
            f"{num_w_preserved} preserved."
        )

    return new_sd


def save_sd_for_aiu(
    model: PreTrainedModel,
    qcfg: dict | None = None,
    output_dir: str | Path = "./",
    file_name: str | Path = "qmodel_for_aiu.pt",
    verbose: bool = False,
) -> None:
    """Save model state dictionary after conversion for AIU compatibility."""

    if qcfg is None:
        logger.info(
            "Attention: saving state dictionary without specifying a quantization "
            "configuration (qcfg) performs no recomputation for narrow weight "
            "distributions and assumes the weight quantizer used was 8-bit per-tensor."
        )
    else:
        nbits_w = qcfg.get("nbits_w", None)
        if nbits_w is None:
            logger.info(
                "Number of bits for weight quantization is not set in qcfg. "
                "Assuming default (nbits_w=8)."
            )
        elif nbits_w != 8:
            raise ValueError(
                "Saving checkpoint in AIU-compliant format only supports INT8 "
                f"quantization for now, but found {nbits_w=} in qcfg."
            )

    converted_sd = convert_sd_for_aiu(
        model=model,
        recompute_narrow_weights=(
            qcfg.get("recompute_narrow_weights", False) if qcfg is not None else False
        ),
        weight_per_channel=(
            "perch" in qcfg.get("qw_mode", False).lower() if qcfg is not None else False
        ),
        verbose=verbose,
    )
    torch.save(converted_sd, Path(output_dir) / file_name)
    logger.info(f"Quantized model checkpoint saved to {Path(output_dir) / file_name}")


def save_for_aiu(
    model: PreTrainedModel,
    qcfg: dict | None = None,
    output_dir: str | Path = "./",
    file_name: str | Path = "qmodel_for_aiu.pt",
    cfg_name: str | Path = "qcfg.json",
    recipe: str | None = None,
    verbose: bool = False,
) -> None:
    """Main entry point to save quantized model state dictionary and configuration
    in the format requested by the AIU.

    Checkpoint saving is customized for AIU compatibility, with the option to recompute
    weights presenting narrow distributions in the integer domain.
    The general qconfig_save function is used to save the quantization configuration.

    Required arguments: model (quantized)
    """

    save_sd_for_aiu(model, qcfg, output_dir, file_name, verbose)

    if qcfg is None:
        logger.info(
            "Quantization configuration was not provided. Only converted checkpoint is "
            "saved."
        )
        return

    # enforce specific keys needed when reloading model for AIU
    qcfg["keys_to_save"] = [
        "qa_mode",
        "qw_mode",
        "smoothq",
        "smoothq_scale_layers",
        "qskip_layer_name",
        "qskip_large_mag_layers",
        "recompute_narrow_weights",
    ]
    qconfig_save(
        qcfg=qcfg,
        recipe=recipe,
        minimal=True,
        fname=str(Path(output_dir) / cfg_name),  # only str is fname accepted type
    )
