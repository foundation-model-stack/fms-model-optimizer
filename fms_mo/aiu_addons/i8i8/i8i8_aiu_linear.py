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
"""Implement INT8xINT8 linear module compatible with AIU compiler"""

# Standard
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Optional, Union
import copy

# Third Party
from fms.modules.linear import (
    LinearModuleShardingInfo,
    LinearParameterShardingInfo,
    register_linear_type_to_module_map,
    register_linear_type_to_sharding_map,
    shard_base_linear,
)
from fms.modules.tp import ShardType, TPModule
from fms.utils.config import ModelConfig
import torch

# Local
from fms_mo.aiu_addons.i8i8.i8i8_aiu_op import register_aiu_i8i8_op

register_aiu_i8i8_op()


@dataclass
class W8A8LinearConfig(ModelConfig):
    """Configuration for W8A8 Linear module"""

    linear_type: str = "int8"
    bits: int = 8
    weight_per_channel: bool = False
    activ_quant_type: Optional[str] = "per_token"
    smoothquant: bool = False
    smoothquant_layers: Optional[list] = None


class W8A8LinearAIU(torch.nn.Module):
    """Simplified QLinear that wraps quantize/dequantize operation.
    fms_mo.i8i8_aiu must have been pre-registered to use this class.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool,
        config: W8A8LinearConfig,
        use_smoothquant: bool,
    ):
        super().__init__()

        if config.bits != 8:
            raise ValueError(
                "Only INT8 quantization is supported by W8A8LinearAIU module"
            )
        if config.activ_quant_type not in [
            "per_token",
            "per_tensor_symm",
            "per_tensor_asymm",
        ]:
            raise ValueError(
                f"Unrecognized activation quantization type {config.activ_quant_type}. "
                "Choose between per_token, per_tensor_symm, per_tensor_asymm"
            )
        self.in_features = in_features
        self.out_features = out_features

        self.register_buffer(
            "weight",
            torch.zeros(out_features, in_features, dtype=torch.int8),
        )

        self.has_bias = bias
        bias_size = out_features if self.has_bias else 1
        self.register_buffer("bias", torch.zeros((bias_size), dtype=torch.float16))

        if config.weight_per_channel:
            w_clip_size = out_features
            self.weight_quant_type = "per_channel"
        else:
            w_clip_size = 1
            self.weight_quant_type = "per_tensor"
        self.register_buffer(
            "w_clip_val",
            torch.zeros(w_clip_size, dtype=torch.float16),
        )
        self.register_buffer(
            "w_clip_valn",
            torch.zeros(w_clip_size, dtype=torch.float16),
        )

        # a_clip_val, a_clip_valn buffers are always created but remain dummy if
        # quantization type is "per_token" (they are computed on-the-fly)
        self.register_buffer("a_clip_val", torch.zeros(1, dtype=torch.float16))
        self.register_buffer("a_clip_valn", torch.zeros(1, dtype=torch.float16))

        # define zero shift buffer
        if config.activ_quant_type in ["per_token", "per_tensor_symm"]:
            zero_shift_size = 1
        else:  # "per_tensor_asymm"
            zero_shift_size = out_features
        self.register_buffer(
            "zero_shift",
            torch.zeros(zero_shift_size, dtype=torch.float32),
        )
        self.activ_quant_type = config.activ_quant_type

        # define smoothquant buffer
        self.smoothquant = config.smoothquant and use_smoothquant
        smoothquant_size = in_features if self.smoothquant else 1
        self.register_buffer(
            "smoothquant_scale",
            torch.ones(smoothquant_size, dtype=torch.float16),
        )

        # check op is registered before loading
        if not hasattr(torch.ops, "fms_mo") or not hasattr(
            torch.ops.fms_mo, "i8i8_aiu"
        ):
            raise ValueError("Custom AIU op `fms_mo.i8i8_aiu` has not been registered.")
        self.aiu_op = torch.ops.fms_mo.i8i8_aiu

        self.register_buffer(
            "qdata",
            torch.cat(
                (
                    self.w_clip_val,
                    self.w_clip_valn,
                    self.a_clip_val,
                    self.a_clip_valn,
                    self.zero_shift,
                    self.smoothquant_scale,
                )
            ),
        )

    def forward(self, x):
        """
        qdata: `quantization metadata` obtained from the concatenation
                of 6 tensors: w_clip_val, w_clip_valn, a_clip_val, a_clip_valn,
                zero_shift, smoothquant_scale

        Tensors characteristics depend on choice of quantization, as follows:

        name        quant type      #dims   #elements   values
        ------------------------------------------------------
        w_clip_val  per tensor      1       1           >0
                    per channel     1       out         >0
        a_clip_val  per tensor      1       1           >0
                    per token       1       1           0

        zero_shift  asymm activ     1       out         !=0
        zero_shift  symm activ      1       1           0

        smoothquant_scale ON        1       in          >0
        smoothquant_scale OFF       1       1           1
        ------------------------------------------------------

        smoothquant_scale is pre-computed scaling factor:
        ```
        S_sq = max(|X_j|)^alpha / max(|W_j|)^(1 - alpha)
        ```
        It is used in hardware to scale activations, prior quantization
        """

        return self.aiu_op(
            x,
            self.weight,
            self.bias,
            self.qdata,
            self.weight_quant_type,
            self.activ_quant_type,
            self.smoothquant,
        )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}"
            f"(in={self.in_features}, out={self.out_features}, "
            f"bias={self.has_bias}, wq={self.weight_quant_type}, "
            f"aq={self.activ_quant_type}, smoothq={self.smoothquant}, "
            f"op={self.aiu_op})"
        )


def update_from_partial(
    linear_config: dict[Union[str, Callable], Any],
) -> dict[Union[str, Callable], Any]:
    """Update linear config parameters using those of partial callable"""

    linear_config_updated = copy.deepcopy(linear_config)
    for k, v in linear_config["linear_type"].keywords.items():
        linear_config_updated[k] = v
    return linear_config_updated


def get_int8_aiu_linear(
    in_features: int,
    out_features: int,
    bias: bool,
    linear_config: dict[Union[str, Callable], Any],
    linear_type: Optional[str] = None,
    use_smoothquant: bool = False,
) -> torch.nn.Module:
    """Retrieve a W8A8 Linear module"""

    # Preprocess linear_config if its linear_type field is a callable
    # (which would not initialize correctly the dataclass parameters).
    # We don't want to alter the original linear_config though.
    linear_config_for_dataclass: Optional[dict[Union[str, Callable], Any]] = None
    if callable(linear_config["linear_type"]):
        linear_config_for_dataclass = update_from_partial(linear_config)
        linear_config_for_dataclass["linear_type"] = linear_type
    if not linear_config_for_dataclass:
        linear_config_for_dataclass = linear_config

    int8_config = W8A8LinearConfig(**linear_config_for_dataclass)
    linear = W8A8LinearAIU(
        in_features=in_features,
        out_features=out_features,
        bias=bias,
        config=int8_config,
        use_smoothquant=use_smoothquant,
    )
    return linear


def shard_int8_aiu_linear(
    tensor_values: dict[str, torch.Tensor],
    tp_module: TPModule,
    module_sharding_info: dict[str, LinearModuleShardingInfo],
) -> Optional[set]:
    """Set up INT8 (W8A8) quantization parameters to be sharded onto
    AIU-compliant linear modules

                         |     GPU     |
    sharding  | qparam   | shard | dim |
    ----------+----------+-------+-----|
    colwise   | weight   |   Y   |  0  |
              | bias     |   Y   |  0  |
              | others*  |   N   |  -  |
    ----------+----------+-------+-----|
    rowwise   | weight   |   Y   |  1  |
              | bias     |   0   |  -  |
              | others*  |   N   |  -  |

    Other quantization parameters: w_clip_val, w_clip_valn,
    a_clip_val, a_clip_valn, zero_shift, smoothquant_scale
    No sharding on all these parameters, except w_clip_val and w_clip_valn when
    per-channel quantization is used
    """
    param_sharding_info: dict[str, dict[str, LinearParameterShardingInfo]] = {}
    for module_name, module_info in module_sharding_info.items():
        int8_aiu_mod = module_info.linear_module
        params: dict[str, LinearParameterShardingInfo] = {
            "weight": LinearParameterShardingInfo(
                module_info.sharding_dim, ShardType.SHARD
            ),
            # FIXME: with per-channel W, clips need to be sharded
            # but if per-tensor w, there should be no sharding
            # HOW CAN WE DISCRIMINATE THE TWO CASES?
            "w_clip_val": LinearParameterShardingInfo(0, ShardType.CLONE),
            "w_clip_valn": LinearParameterShardingInfo(0, ShardType.CLONE),
            # "w_clip_val": LinearParameterShardingInfo(
            #     module_info.sharding_dim,
            #     ShardType.SHARD if module_info.sharding_dim == 0 else ShardType.RANK0,
            # ),
            # "w_clip_valn": LinearParameterShardingInfo(
            #     module_info.sharding_dim,
            #     ShardType.SHARD if module_info.sharding_dim == 0 else ShardType.RANK0,
            # ),
            "a_clip_val": LinearParameterShardingInfo(0, ShardType.CLONE),
            "a_clip_valn": LinearParameterShardingInfo(0, ShardType.CLONE),
            "zero_shift": LinearParameterShardingInfo(0, ShardType.CLONE),
            "smooqthquant_scale": LinearParameterShardingInfo(0, ShardType.CLONE),
        }
        if int8_aiu_mod.bias is not None:
            params["bias"] = LinearParameterShardingInfo(
                module_info.sharding_dim,
                ShardType.SHARD if module_info.sharding_dim == 0 else ShardType.RANK0,
            )
        param_sharding_info[module_name] = params

    unused_keys = shard_base_linear(
        tensor_values, tp_module, module_sharding_info, param_sharding_info
    )

    raise NotImplementedError("TP not yet supported for INT8. Work in progress")
    # return unused_keys


register_linear_type_to_module_map(
    "int8_aiu",
    partial(
        get_int8_aiu_linear,
        linear_type="int8_aiu",
        use_smoothquant=False,
    ),
)
register_linear_type_to_module_map(
    "int8_smoothquant_aiu",
    partial(
        get_int8_aiu_linear,
        linear_type="int8_smoothquant_aiu",
        use_smoothquant=True,
    ),
)
register_linear_type_to_sharding_map("int8_aiu", shard_int8_aiu_linear)
