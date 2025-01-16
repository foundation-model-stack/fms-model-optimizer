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
"""Implement FMS adapter for GPTQ W4A16 checkpoints"""

# Standard
from typing import Mapping

# Third Party
from fms.utils import serialization
import torch


def _gptq_qweights_transpose_aiu(
    input_sd: Mapping[str, torch.Tensor],
) -> Mapping[str, torch.Tensor]:
    new_sd = {}
    for name, param in input_sd.items():
        new_sd[name] = param
        # for AIU, qweights are needed as [out_feat, in_feat]
        if "qweight" in name:
            new_sd[name] = new_sd[name].t()
        elif "g_idx" in name:
            new_sd[name] = torch.zeros(1, dtype=torch.int32, device=param.device)
    return new_sd


serialization.register_adapter_step(
    "llama", "gptq_qweights_transpose_aiu", _gptq_qweights_transpose_aiu
)
serialization.register_adapter_step(
    "gpt_bigcode", "gptq_qweights_transpose_aiu", _gptq_qweights_transpose_aiu
)
serialization.register_adapter(
    "llama",
    "hf_gptq_aiu",
    [
        "hf_to_fms_names",
        "hf_to_fms_rope",
        "hf_gptq_fusion_check",
        "weight_fusion",
        "gptq_qweights_transpose_aiu",
    ],
)
serialization.register_adapter(
    "gpt_bigcode",
    "hf_gptq_aiu",
    ["hf_to_fms_names", "weight_fusion", "gptq_qweights_transpose_aiu"],
)
