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
"""Implement GPTQ W4A16 linear module compatible with AIU compiler"""

# Standard
from typing import Any, Mapping, Optional
import math

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
# pylint: disable=import-error,wrong-import-position,ungrouped-imports
from fms.modules.linear import (
    LinearModuleShardingInfo,
    LinearParameterShardingInfo,
    register_linear_type_to_module_map,
    register_linear_type_to_sharding_map,
    shard_base_linear,
)
from fms.modules.tp import ShardType, TPModule
from fms.utils.gptq import GPTQLinearConfig

# Local
from fms_mo.aiu_addons.gptq.gptq_aiu_op import register_aiu_gptq_op

register_aiu_gptq_op()


class GPTQLinearAIU(torch.nn.Module):
    """Simplified QLinear that wraps GPTQ W4A16 custom operation.
    gptq_gemm.i4f16_fxinputs_aiu must have been pre-registered to use this class.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool,
        config: GPTQLinearConfig,
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.bits = config.bits
        self.group_size = config.group_size if config.group_size != -1 else in_features
        self.desc_act = config.desc_act
        # self.weight_transposed = True

        if self.bits not in [4]:
            raise NotImplementedError(
                "AIU GPTQLinear only supports 4 bits quantization."
            )
        if in_features % self.group_size != 0:
            raise ValueError("`in_features` must be divisible by `group_size`.")
        if in_features % 32 or out_features % 32:
            raise ValueError("`in_features` and `out_features` must be divisible by 32")
        if self.desc_act:
            raise NotImplementedError(
                "AIU GPTQLinear does not support activation reordering (`desc_act`)"
            )

        # Register quantization parameters
        self.register_buffer(
            "qweight",
            torch.zeros(
                # transposed w.r.t. GPTQ ckpt (AIU requirement)
                (out_features, in_features // 32 * self.bits),
                dtype=torch.int32,
            ),
        )
        self.register_buffer(
            "qzeros",
            torch.zeros(
                (
                    math.ceil(in_features / self.group_size),
                    out_features // 32 * self.bits,
                ),
                dtype=torch.int32,
            ),
        )
        self.register_buffer(
            "scales",
            torch.zeros(
                (math.ceil(in_features / self.group_size), out_features),
                dtype=torch.float16,
            ),
        )
        # AIU requirement
        self.register_buffer("g_idx", torch.tensor([0], dtype=torch.int32))
        if bias:
            self.register_buffer(
                "bias",
                torch.zeros((out_features), dtype=torch.float16),
            )
        else:
            self.bias = None

        # Register op
        if not hasattr(torch.ops, "gptq_gemm") or not hasattr(
            torch.ops.gptq_gemm, "i4f16_fxinputs_aiu"
        ):
            raise ValueError(
                "Custom AIU op `gptq_gemm.i4f16_fxinputs_aiu` has not been registered."
            )
        self.aiu_op = torch.ops.gptq_gemm.i4f16_fxinputs_aiu

    def forward(self, x):
        """Call pre-registered custom GPTQ operation"""

        x = self.aiu_op(
            x.half(),
            self.qweight,
            self.qzeros,
            self.scales,
            self.g_idx,
        )
        if self.bias is not None:
            x.add_(self.bias)
        return x

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}"
            f"(in={self.in_features}, out={self.out_features}, "
            f"bias={self.bias is not None}, group={self.group_size}, "
            f"op={self.aiu_op})"
        )


def get_gptq_aiu_linear(
    in_features: int,
    out_features: int,
    bias: bool,
    linear_config: Mapping[str, Any],
) -> torch.nn.Module:
    """Retrieve a GPTQ W4A16 Linear module"""

    gptq_config = GPTQLinearConfig(**linear_config)
    if gptq_config.desc_act:
        raise NotImplementedError(
            "Activation reordering (desc_act=True) not supported on AIU"
        )
    linear = GPTQLinearAIU(
        in_features=in_features,
        out_features=out_features,
        bias=bias,
        config=gptq_config,
    )
    setattr(linear, "desc_act", gptq_config.desc_act)
    return linear


def shard_gptq_aiu_linear(
    tensor_values: dict[str, torch.Tensor],
    tp_module: TPModule,
    module_sharding_info: dict[str, LinearModuleShardingInfo],
) -> Optional[set]:
    """
    Set up GPTQ quantization parameters to be sharded onto
    AIU-compliant linear modules

                         |     GPU     |
    sharding  | qparam   | shard | dim |
    ----------+----------+-------+-----|
    colwise   | qweight  |   Y   |  0  |
              | bias     |   Y   |  0  |
              | scales   |   Y   |  1  |
              | qzeros   |   Y   |  1  |
              | g_idx    |   N   |  -  |
    ----------+----------+-------+-----|
    rowwise   | qweight  |   Y   |  1  |
              | bias     |   0   |  -  |
              | scales   |   Y   |  0  |
              | qzeros   |   Y   |  0  |
              | g_idx    |   N   |  -  |
    """
    param_sharding_info: dict[str, dict[str, LinearParameterShardingInfo]] = {}
    for module_name, module_info in module_sharding_info.items():
        gptq_aiu_mod = module_info.linear_module
        params: dict[str, LinearParameterShardingInfo] = {
            "qweight": LinearParameterShardingInfo(
                module_info.sharding_dim, ShardType.SHARD
            ),
            "scales": LinearParameterShardingInfo(
                1 - module_info.sharding_dim, ShardType.SHARD
            ),
            "qzeros": LinearParameterShardingInfo(
                1 - module_info.sharding_dim, ShardType.SHARD
            ),
            # g_idx on aiu is 1-dim zero tensor, always cloned on each shard
            "g_idx": LinearParameterShardingInfo(0, ShardType.CLONE),
        }
        if gptq_aiu_mod.bias is not None:
            params["bias"] = LinearParameterShardingInfo(
                module_info.sharding_dim,
                ShardType.SHARD if module_info.sharding_dim == 0 else ShardType.RANK0,
            )
        param_sharding_info[module_name] = params

    unused_keys = shard_base_linear(
        tensor_values, tp_module, module_sharding_info, param_sharding_info
    )
    return unused_keys


register_linear_type_to_module_map("gptq_aiu", get_gptq_aiu_linear)
register_linear_type_to_sharding_map("gptq_aiu", shard_gptq_aiu_linear)
