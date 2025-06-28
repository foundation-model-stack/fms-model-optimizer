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
"""Implement and register FMS adapters for FP8 checkpoint loading."""

# Standard
from typing import Any, Mapping

# Third Party
from fms.modules.linear import get_linear_type
from fms.utils import serialization
from fms.utils.config import ModelConfig

# pylint: disable=unused-argument
# Retaining kwargs input arguments for consistency with other adapter steps.


# NOTE: this adapter step must be registered before the adapter that uses it (such as
# the llama adapter in fms.models.llama)
# TODO: may be shared with gptq llama
# TODO: generalize across architectures if possible
def _hf_fp8_llama_check(
    input_sd: Mapping[str, Any], model_config: ModelConfig | None = None, **kwargs
) -> Mapping[str, Any]:
    """Implementation of adapter step for FMS Llama: ensure that when FP8 quantization
    is in use, weights are unfused.
    """

    has_fused_weights = True
    linear_type = "torch_linear"
    if model_config:
        if not model_config.fused_weights:
            has_fused_weights = False
        if model_config.linear_config:
            linear_type = model_config.linear_config["linear_type"]
            if callable(linear_type):
                # Calling this with "any" guarantees "fp8" to be returned
                # when loading an HF fp8 checkpoint, and never in any other condition
                linear_type = get_linear_type(model_config.linear_config, "any")

    if "fp8" in linear_type and has_fused_weights:
        raise ValueError(
            "FP8 HF llama checkpoints cannot be loaded into a model with fused weights"
        )

    return input_sd


serialization.register_adapter_step("llama", "hf_fp8_llama_check", _hf_fp8_llama_check)
