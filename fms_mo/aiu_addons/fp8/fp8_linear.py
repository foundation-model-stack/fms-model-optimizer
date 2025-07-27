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
"""Implement FP8 linear module to be loaded via FMS."""

# Standard
from typing import Any, Mapping

# Third Party
import torch

# Local
from fms_mo.aiu_addons.fp8 import fp8_spyre_op  # pylint: disable=unused-import
from fms_mo.prep import available_packages

# pylint: disable=not-callable
# torch.nn.functional.linear not recognized as callable
# open issue in PyLint: https://github.com/pytorch/pytorch/issues/119482

# Gated torchao imports for FP8 implementation
if available_packages["fms"] and available_packages["torchao"]:
    # Third Party
    from fms.modules.linear import (
        LinearModuleShardingInfo,
        LinearParameterShardingInfo,
        register_linear_type_to_module_map,
        register_linear_type_to_sharding_map,
        shard_base_linear,
    )
    from fms.modules.tp import ShardType, TPModule
    from torchao.dtypes.affine_quantized_tensor import (
        AffineQuantizedTensor,
        to_affine_quantized_floatx,
        to_affine_quantized_floatx_static,
    )
    from torchao.dtypes.floatx.float8_layout import (
        Float8AQTTensorImpl,
        Float8Layout,
        Float8MMConfig,
        preprocess_data,
        preprocess_scale,
    )
    from torchao.dtypes.utils import get_out_shape
    from torchao.float8.inference import (
        _is_rowwise_scaled,
        addmm_float8_unwrapped_inference,
    )
    from torchao.quantization.granularity import PerRow, PerTensor
    from torchao.quantization.observer import get_block_size
    from torchao.quantization.quant_primitives import ZeroPointDomain

    class FP8Linear(torch.nn.Module):
        """Class handles FP8 weights loading and uses torchao for the matmuls."""

        def __init__(
            self,
            in_features: int,
            out_features: int,
            bias: bool,
            linear_config: Mapping[str, Any],
        ):
            super().__init__()

            self.in_features = in_features
            self.out_features = out_features
            self.has_bias = bias
            self.linear_config = linear_config

            assert (
                self.linear_config["weights"] is not None
            ), "Weights must always be quantized for FP8Linear"
            assert self.linear_config["weights"][
                "symmetric"
            ], "We only support symmetric weights for now"
            assert not self.linear_config["weights"][
                "dynamic"
            ], "We only support pre-quantized weights for now"

            self.weight = torch.nn.Parameter(
                torch.zeros(out_features, in_features, dtype=torch.float8_e4m3fn),
                requires_grad=False,
            )

            weight_scale_shape = (
                (1,)
                if self.linear_config["weights"]["strategy"] == "tensor"
                else (out_features, 1)
            )
            self.weight_scale = torch.nn.Parameter(
                torch.ones(weight_scale_shape), requires_grad=False
            )

            self.has_bias = bias
            if self.has_bias:
                self.bias = torch.nn.Parameter(torch.zeros((out_features,)))

            if (
                self.linear_config["input_activations"] is not None
                and not self.linear_config["input_activations"]["dynamic"]
            ):
                input_scale_shape = (
                    (1,)
                    if self.linear_config["input_activations"]["strategy"] == "tensor"
                    else (out_features, 1)
                )
                self.input_scale = torch.nn.Parameter(
                    torch.ones(input_scale_shape), requires_grad=False
                )

        def _input_activation_quant_func_fp8(
            self,
            x: torch.Tensor,
            activation_granularity,
            activation_dtype: torch.dtype,
            scale: torch.Tensor | None = None,
        ):
            """Quantize the input activation tensor for an aqt_float variant.
            If scale is not provided, it will be dynamically calculated, otherwise the
            provided scale will be used.
            """
            block_size = get_block_size(x.shape, activation_granularity)
            if scale is None:
                activation = to_affine_quantized_floatx(
                    input_float=x,
                    block_size=block_size,
                    target_dtype=activation_dtype,
                    scale_dtype=torch.float32,
                    _layout=Float8Layout(mm_config=None),  # Config is stored on weight
                )
            else:
                assert isinstance(
                    activation_granularity, PerTensor
                ), "Static quantization only supports PerTensor granularity"
                activation = to_affine_quantized_floatx_static(
                    input_float=x,
                    block_size=block_size,
                    scale=scale,
                    target_dtype=activation_dtype,
                    _layout=Float8Layout(mm_config=None),  # Config is stored on weight
                )
            return activation

        def _construct_qweight_structure(self) -> "AffineQuantizedTensor":
            """Construct the torchao machinery for the fp8 matmul"""
            weight_granularity = (
                PerTensor()
                if self.linear_config["weights"]["strategy"] == "tensor"
                else PerRow()
            )
            fp8_layout = Float8Layout(Float8MMConfig(use_fast_accum=True))
            return AffineQuantizedTensor(
                Float8AQTTensorImpl.from_plain(
                    self.weight,
                    self.weight_scale.squeeze().to(torch.float32),
                    None,
                    fp8_layout,
                ),
                get_block_size(self.weight.shape, weight_granularity),
                self.weight.shape,
                zero_point_domain=ZeroPointDomain.NONE,
                dtype=self.weight_scale.dtype,
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """If input quantization is active, compute FP8xFP8 addmm leveraging torchao
            functionalities. Otherwise compute non-quantized addmm."""
            # fp8 weight tensor for torchao
            qweight: AffineQuantizedTensor = self._construct_qweight_structure()

            if self.linear_config["input_activations"] is not None:
                # activations are also fp8, quantize as required by model
                act_granularity = (
                    PerTensor()
                    if self.linear_config["input_activations"]["strategy"] == "tensor"
                    else PerRow()
                )
                input_quant_kwargs = {
                    "activation_granularity": act_granularity,
                    "activation_dtype": torch.float8_e4m3fn,
                }
                if not self.linear_config["input_activations"]["dynamic"]:
                    input_quant_kwargs["scale"] = self.input_scale.squeeze().to(
                        torch.float32
                    )
                qx = self._input_activation_quant_func_fp8(x, **input_quant_kwargs)

                # Copied from torchao _linear_fp8_act_fp8_weight_impl
                # (with changes to support fp8 out)
                scaled_mm_config = Float8MMConfig(use_fast_accum=True)
                out_shape = get_out_shape(qx.shape, qweight.shape)

                # Weight tensor preprocessing
                w_tensor_impl = qweight.tensor_impl
                assert not w_tensor_impl.transposed, "Weight tensor must be contiguous"
                w_data = w_tensor_impl.float8_data
                w_scale = w_tensor_impl.scale

                # Input tensor preprocessing
                inpt_data = qx.tensor_impl.float8_data
                input_scale = qx.tensor_impl.scale
                # Handle case where input tensor is more than 2D
                inpt_data = inpt_data.reshape(-1, inpt_data.shape[-1])

                # Handle rowwise case
                if _is_rowwise_scaled(qweight):
                    assert _is_rowwise_scaled(
                        qx
                    ), "Input tensor must be rowwise block size"
                    w_scale = w_scale.unsqueeze(-1).T
                    input_scale = preprocess_scale(input_scale, qx.shape)

                # Preprocess data
                inpt_data, w_data = preprocess_data(
                    inpt_data, w_data.T, scaled_mm_config
                )

                # Perform the computation
                return addmm_float8_unwrapped_inference(
                    inpt_data,
                    input_scale,
                    w_data,
                    w_scale,
                    output_dtype=qx.dtype,
                    bias=getattr(self, "bias", None),
                    use_fast_accum=scaled_mm_config.use_fast_accum,
                ).reshape(out_shape)

            # activations not quantized, dequant fp8 weight and do regular matmul
            out = torch.nn.functional.linear(
                x, qweight.dequantize(), self.bias if self.has_bias else None
            )
            return out

        def __repr__(self) -> str:
            return (
                f"{self.__class__.__name__}"
                f"(in={self.in_features}, out={self.out_features}, "
                f"bias={self.has_bias}, fp8_config={self._repr_fp8_config()})"
            )

        def _repr_fp8_config(self) -> str:
            return (
                "("
                "acts: ("
                f"dynamic: {self.linear_config['input_activations']['dynamic']}, "
                f"strategy: {self.linear_config['input_activations']['strategy']}"
                "), "
                "weights: ("
                f"dynamic: {self.linear_config['weights']['dynamic']}, "
                f"strategy: {self.linear_config['weights']['strategy']}"
                ")"
                ")"
            )

    def get_fp8_linear(
        in_features: int,
        out_features: int,
        bias: bool,
        linear_config: Mapping[str, Any],
    ) -> FP8Linear:
        """Retrieve an FP8 Linear module"""
        return FP8Linear(in_features, out_features, bias, linear_config)

    def shard_fp8_linear(
        tensor_values: dict[str, torch.Tensor],
        tp_module: TPModule,
        module_sharding_info: dict[str, LinearModuleShardingInfo],
    ) -> set | None:
        """
                                |     GPU     |
        sharding  | param          | shard | dim |
        ----------+----------------+-------+-----|
        colwise   | weight         |   Y   |  0  |
                  | weight_scale   |   N   |  -  |
                  | input_scale    |   N   |  -  |
                  | bias           |   Y   |  0  |
        ----------+----------------+-------+-----|
        rowwise   | weight         |   Y   |  1  |
                  | weight_scale   |  Y/N  | 0/- |
                  | input_scale    |  Y/N  | 0/- |
                  | bias           |   0   |  -  |
        """

        param_sharding_info: dict[str, dict[str, LinearParameterShardingInfo]] = {}
        for module_name, module_info in module_sharding_info.items():
            linear_mod: torch.nn.Module = module_info.linear_module
            weight_strategy = getattr(linear_mod, "linear_config")["input_activations"][
                "strategy"
            ]
            # Scales are per-row or per-tensor
            # Only sharding needed when row parallel and per-row
            shard_scales = weight_strategy != "tensor" and module_info.sharding_dim == 1
            params: dict[str, LinearParameterShardingInfo] = {
                "weight": LinearParameterShardingInfo(
                    module_info.sharding_dim, ShardType.SHARD
                ),
                "weight_scale": LinearParameterShardingInfo(
                    module_info.sharding_dim,
                    ShardType.SHARD if shard_scales else ShardType.CLONE,
                ),
            }
            if hasattr(linear_mod, "input_scale"):
                params["input_scale"] = LinearParameterShardingInfo(
                    module_info.sharding_dim,
                    ShardType.SHARD if shard_scales else ShardType.CLONE,
                )
            if hasattr(linear_mod, "bias") and linear_mod.bias is not None:
                params["bias"] = LinearParameterShardingInfo(
                    module_info.sharding_dim,
                    ShardType.SHARD
                    if module_info.sharding_dim == 0
                    else ShardType.RANK0,
                )
            param_sharding_info[module_name] = params

        unused_keys = shard_base_linear(
            tensor_values,
            tp_module,
            module_sharding_info,
            param_sharding_info,
        )
        return unused_keys

    register_linear_type_to_module_map("fp8", get_fp8_linear)
    register_linear_type_to_sharding_map("fp8", shard_fp8_linear)
