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
Utils for DQ

"""


def config_quantize_smooth_layers(qcfg):
    """
    To set the config for each model, for example
    layers to quantize
    layers to skip
    layers to apply smooth-scale
    block_size
    smooth_alpha
    """
    llama_architecture = [
        "llama",
        "Nemotron",
        "granite-3b-code",
        "granite-8b-code",
    ]
    granite_BigCode_architecture = [
        "granite-3b-base",
        "granite-13b-base",
        "granite-20b-code",
        "granite-20b-code",
    ]
    if (
        any(model in qcfg["model"] for model in llama_architecture)
        or any(model in qcfg["model_type"] for model in llama_architecture)
        and qcfg["qskip_large_mag_layers"]
    ):
        qcfg["qlayer_name_pattern"] = ["model.layers."]
        qcfg["scale_layers"] = ["k_proj", "v_proj", "gate_proj", "up_proj"]
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

    elif "mixtral" in qcfg["model"]:
        qcfg["qlayer_name_pattern"] = (
            ["model.layers"] if qcfg["nbits_bmm1"] == 32 else []
        )
        qcfg["scale_layers"] = ["q_proj", "k_proj", "v_proj", "w1", "w3"]
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
        qcfg["act_scale_path"] = "./act_scales/Mixtral-8x7B-v0.1.pt"
    elif any(model in qcfg["model"] for model in granite_BigCode_architecture):
        qcfg["qlayer_name_pattern"] = ["transformer.h"]
        qcfg["scale_layers"] = ["c_attn", "c_fc"]
        qcfg["qskip_layer_name"] = []
        if "granite-3b-base-v2" in qcfg["model"]:
            qcfg["act_scale_path"] = "./act_scales/granite_3b_base_v2_500_nw.pt"
        if "granite-13b-base-v2" in qcfg["model"]:
            qcfg["act_scale_path"] = "./act_scales/granite_13b_base_v2.pt"
        if "granite-20b-code-base" in qcfg["model"]:
            qcfg["act_scale_path"] = "./act_scales/graniteCodeHF_20b_base12.pt"
        if "granite-20b-code-instruct" in qcfg["model"]:
            qcfg["act_scale_path"] = "./act_scales/graniteCodeHF_20b_base12.pt"
        if "granite-34b-code-base" in qcfg["model"]:
            qcfg["act_scale_path"] = "./act_scales/graniteCodeHF_34b_base12.pt"
        if "granite-34b-code-instruct" in qcfg["model"]:
            qcfg["act_scale_path"] = "./act_scales/graniteCodeHF_34b_base12.pt"
    else:
        raise ValueError("The model architecture is not supported for DQ.")
