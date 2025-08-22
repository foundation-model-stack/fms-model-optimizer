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
"""Utility functions for Direct Quantization" (DQ)."""

# Standard
import logging

logger = logging.getLogger(__name__)


def config_quantize_smooth_layers(qcfg: dict):
    """Update qcfg with model-dependent config parameters:
    - qlayer_name_pattern: identifier of transformer layers containing linear layers
        to quantize (if any, tracing is bypassed)
    - qskip_layer_name: full name of linear layers that will not be quantized
    - smoothq_scale_layers: identifier of linear layers to apply smoothquant on
    - smoothq_act_scale_path: path to save/load smoothquant activation scales, should be kept
        Path(f"./act_scales/{qcfg['model'].replace('/', '-')}.pt"), no need to specify here.

    Selected model is determined by comparing all architecture identifiers against
    `model` and `model_type` fields in qcfg.

    NOTE: layer quantization skip is determined by bool `qskip_large_mag_layers`
    NOTE: different versions of granite models are based on different architectures
    (chronologically: bigcode -> llama -> granite)
    """

    llama_architecture = [
        "llama",
        "Nemotron",
        "granite-3b-code",
        "granite-8b-code",
    ]
    bigcode_architecture = [
        "granite-3b-base",
        "granite-13b-base",
        "granite-20b-code",
        "granite-20b-code",
    ]
    granite_architecture = [
        "granite-3.0-8b-base",
        "granite-3.0-8b-instruct",
        "granite-3.1-8b-base",
        "granite-3.1-8b-instruct",
        "granite-3.2-8b-instruct",
        "granite-3.3-8b-base",
        "granite-3.3-8b-instruct",
    ]

    if any(model in qcfg["model"] for model in llama_architecture) or any(
        model in qcfg["model_type"] for model in llama_architecture
    ):
        qcfg["qlayer_name_pattern"] = ["model.layers."]
        qcfg["smoothq_scale_layers"] = ["k_proj", "v_proj", "gate_proj", "up_proj"]
        if qcfg["qskip_large_mag_layers"]:
            large_mag_layers = {
                "2-7b": [1, 30],
                "2-70b": [2, 8, 79],
                "3-8B": [1, 31],
                "3-70B": [3, 78, 79],
                "405B-Instruct": [5, 124, 125],
            }
            for llama_family, layers in large_mag_layers.items():
                if llama_family in qcfg["model"]:
                    qcfg["qskip_layer_name"] += [
                        f"model.layers.{i}.mlp.down_proj" for i in layers
                    ]
                    break
    elif any(model in qcfg["model"] for model in granite_architecture) or any(
        model in qcfg["model_type"] for model in granite_architecture
    ):
        qcfg["qlayer_name_pattern"] = ["model.layers."]
        qcfg["smoothq_scale_layers"] = ["k_proj", "v_proj", "gate_proj", "up_proj"]
        # NOTE: supported granite-v3 models do not need layer skip for large magnitude
    elif "mixtral" in qcfg["model"]:
        qcfg["qlayer_name_pattern"] = (
            ["model.layers"] if qcfg["nbits_bmm1"] == 32 else []
        )
        qcfg["smoothq_scale_layers"] = ["q_proj", "k_proj", "v_proj", "w1", "w3"]
        qcfg["qskip_layer_name"] += [
            f"model.layers.{i}.block_sparse_moe.gate" for i in range(32)
        ]
        if qcfg["qskip_large_mag_layers"]:
            qcfg["qskip_layer_name"] += [
                f"model.layers.{i}.block_sparse_moe.experts.{j}.w2"
                for [i, j] in [
                    [1, 3],
                    [30, 5],
                    [30, 7],
                    [31, 1],
                    [31, 2],
                    [31, 5],
                    [31, 7],
                ]
            ]
    elif any(model in qcfg["model"] for model in bigcode_architecture):
        qcfg["qlayer_name_pattern"] = ["transformer.h"]
        qcfg["smoothq_scale_layers"] = ["c_attn", "c_fc"]
        # NOTE: supported bigcode models do not need layer skip for large magnitude
    elif "roberta" in qcfg["model"]:
        qcfg["act_scale_path"] = "./act_scales"
        qcfg["smoothq_scale_layers"] = [
            "attention.self.query",
            "attention.self.key",
            "attention.self.value",
            "intermediate.dense",
        ]
        qcfg["qskip_layer_name"] = []
        qcfg["qlayer_name_pattern"] = ["roberta.encoder"]
    else:
        logger.info(
            "The model architecture is not supported for DQ. No architecture-specific settings is"
            "applied. All Linear layers will be quantized, which may not yield the optimal results."
        )
