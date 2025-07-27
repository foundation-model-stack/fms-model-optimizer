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
"""Implement FMS adapter for INT8xINT8 checkpoints"""

# Standard
from typing import Mapping

# Third Party
import torch

# Local
from fms_mo.utils.import_utils import available_packages

if not available_packages["fms"]:
    raise ImportError(
        "AIU functionality requires ibm-fms to be installed."
        "See https://github.com/foundation-model-stack/foundation-model-stack for details."
    )

# Third Party
# pylint: disable=import-error,wrong-import-position
from fms.utils import serialization


def _int8_qparams_aiu(
    input_sd: Mapping[str, torch.Tensor],
    **kwargs,  # pylint: disable=unused-argument
) -> Mapping[str, torch.Tensor]:
    new_sd = {}
    modules_seen = set()
    for name, param in input_sd.items():
        new_name = name
        if "clip_val" in name:
            name_split = name.split(".")
            is_weight = "weight" in name_split[-2]
            module_name = ".".join(name_split[:-2])
            modules_seen.add(module_name)

            param_type = "w" if is_weight else "a"
            new_name = f"{module_name}.{param_type}_{name_split[-1]}"
        elif "smoothq" in name and "smoothquant" not in name:
            new_name = name.replace("smoothq", "smoothquant")

        new_sd[new_name] = param

    _add_defaults_and_concat(new_sd, modules_seen)
    return new_sd


def _add_defaults_and_concat(
    new_sd: dict[str, torch.Tensor],
    modules_seen: set[str],
) -> None:
    """
    Add default activation clip values, zero_shift, and smoothquant_scale (if not
    already present) to every linear module processed in the partial state dict.
    It is assumed that weight clip values are always present and don't need default.

    For every module, also create float32 `qdata` tensor, as concatenation of
    quantization metadata tensors, as per AIU requirement.
    """

    for module_name in modules_seen:
        # add default activation clip values (both), if not present
        if module_name + ".a_clip_val" not in new_sd:
            a_clip_val = torch.zeros(1, dtype=torch.float16)
            a_clip_valn = torch.zeros(1, dtype=torch.float16)
            new_sd[module_name + ".a_clip_val"] = a_clip_val
            new_sd[module_name + ".a_clip_valn"] = a_clip_valn
        else:
            a_clip_val = new_sd[module_name + ".a_clip_val"]
            a_clip_valn = new_sd[module_name + ".a_clip_valn"]

        # add default zero shift, if not present
        if module_name + ".zero_shift" not in new_sd:
            zero_shift = torch.zeros(1, dtype=torch.float32)
            new_sd[module_name + ".zero_shift"] = zero_shift
        else:
            zero_shift = new_sd[module_name + ".zero_shift"]

        # add default smoothquant scale, if not present
        if module_name + ".smoothquant_scale" not in new_sd:
            sq_scale = torch.ones(1, dtype=torch.float16)
            new_sd[module_name + ".smoothquant_scale"] = sq_scale
        else:
            sq_scale = new_sd[module_name + ".smoothquant_scale"]

        # add concatenated quantization metadata to state dict
        new_sd[module_name + ".qdata"] = torch.cat(
            (
                new_sd[module_name + ".w_clip_val"].to(torch.float32),
                new_sd[module_name + ".w_clip_valn"].to(torch.float32),
                a_clip_val.to(torch.float32),
                a_clip_valn.to(torch.float32),
                zero_shift.to(torch.float32),  # should be already fp32
                sq_scale.to(torch.float32),
            )
        )


# registration of new adapter step and adapter for each architecture
for arch in [
    "llama",
    "gpt_bigcode",
    "granite",
    "roberta",
    "roberta_question_answering",
]:
    serialization.register_adapter_step(arch, "int8_qparams_aiu", _int8_qparams_aiu)
    if arch in ["llama", "granite"]:
        steps_to_register = [
            "hf_to_fms_names",
            "hf_to_fms_rope",
            "weight_fusion",
            "int8_qparams_aiu",
        ]
    else:
        steps_to_register = ["hf_to_fms_names", "weight_fusion", "int8_qparams_aiu"]
    serialization.register_adapter(arch, "fms_mo", steps_to_register)
