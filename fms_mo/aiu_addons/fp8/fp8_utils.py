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
        scaled: bool = True,
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

    def __init__(
        self,
        data: torch.Tensor,
        scale: torch.Tensor,
        scaled: bool = True,
    ):
        super().__init__()
        self._data = data
        self._scale = scale
        self._scaled = scaled

    def __tensor_flatten__(self):
        ctx = {"scaled": self._scaled}
        return ["_data", "_scale"], ctx

    @staticmethod
    def __tensor_unflatten__(inner_tensors, metadata, outer_size, outer_stride):
        assert len(inner_tensors) == 2
        return ScaledTensor(
            inner_tensors["_data"],
            inner_tensors["_scale"],
            metadata["scaled"],
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
