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
"""Utility functions and components for FP8 addon implementation."""

# Standard
import functools

# Third Party
import torch

# pylint: disable=unused-argument
# unusued arguments are needed for templates


_HANDLED_FUNCTIONS = {}


def _implements(torch_function):
    """Register a torch function override"""

    def decorator(func):
        @functools.wraps(torch_function)
        def wrapper(f, types, args, kwargs):
            return func(f, types, args, kwargs)

        _HANDLED_FUNCTIONS[torch_function] = wrapper
        return func

    return decorator


class ScaledTensor(torch.Tensor):
    """Representation of a quantized tensor and its scale."""

    def __new__(
        cls,
        data: torch.Tensor,
        scale: torch.Tensor,
    ):
        return torch.Tensor._make_wrapper_subclass(
            cls,
            data.size(),
            strides=data.stride(),
            storage_offset=data.storage_offset(),
            dtype=data.dtype,
            layout=data.layout,
            requires_grad=data.requires_grad,
            device=data.device,
        )

    def __init__(  # pylint: disable=super-init-not-called
        self,
        data: torch.Tensor,
        scale: torch.Tensor,
    ):
        self._data = data
        self._scale = scale

    def __tensor_flatten__(self):
        ctx = {}
        return ["_data", "_scale"], ctx

    @staticmethod
    def __tensor_unflatten__(inner_tensors, metadata, outer_size, outer_stride):
        assert len(inner_tensors) == 2
        return ScaledTensor(
            inner_tensors["_data"],
            inner_tensors["_scale"],
        )

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs=None):
        if func in _HANDLED_FUNCTIONS:
            return _HANDLED_FUNCTIONS[func](func, types, args, kwargs)

        arg_types = tuple(type(arg) for arg in args)
        kwarg_types = {k: type(arg) for k, arg in kwargs.items()}
        raise NotImplementedError(
            f"{cls.__name__} dispatch: attempting to run unimplemented "
            f"operator/function: {func=}, {types=}, {arg_types=}, {kwarg_types=}"
        )

    def __repr__(self):
        return f"{self._data.__repr__()}\n{self._scale.__repr__()}"


def _infer_quantization_config(quant_config: dict) -> dict | None:
    """Construct linear_config dictionary carrying FP8 configuration for FMS.

    There's many quantization packages compatible with HF
    We initially focus on llm-compressor as it is the one used in FMS-MO

    llm-compressor saves its checkpoints with quant_method = compressed-tensors
    quantization_status tells us whether the model has already been quantized
    We only support loading already quantized models (compressed status)
    """

    if (
        quant_config["quant_method"] == "compressed-tensors"
        and quant_config["quantization_status"] == "compressed"
    ):
        # FP8 quantization will have FP8 weights
        # We assume a single quantization group (group_0), to follow fms-mo checkpoints
        # num_bits and type tells us "float" with "8" bits, aka FP8
        if (
            quant_config["config_groups"]["group_0"]["weights"]["type"] == "float"
            and quant_config["config_groups"]["group_0"]["weights"]["num_bits"] == 8
        ):
            # This is used by get_linear to decide whether a linear layer
            # will be quantized or not inside the model
            def fp8_linear_type(name: str) -> str:
                # We need to translate HF names to FMS names
                translations = {
                    "lm_head": "head",
                }
                for ignored_layer in quant_config["ignore"]:
                    assert isinstance(ignored_layer, str)
                    fms_ign_layer = translations.get(ignored_layer, ignored_layer)
                    if name in fms_ign_layer:
                        return "torch_linear"
                for pattern in quant_config["config_groups"]["group_0"]["targets"]:
                    # Special case from llm-compressor that covers all linear layers
                    # not in the ignore pattern
                    assert isinstance(pattern, str)
                    if pattern == "Linear":
                        return "fp8"
                    if name in translations.get(pattern, pattern):
                        return "fp8"
                return "torch_linear"

            return {
                "linear_type": fp8_linear_type,
                "input_activations": quant_config["config_groups"]["group_0"][
                    "input_activations"
                ],
                "output_activations": quant_config["config_groups"]["group_0"][
                    "output_activations"
                ],
                "weights": quant_config["config_groups"]["group_0"]["weights"],
            }
    return None
