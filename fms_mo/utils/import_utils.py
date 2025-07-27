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
Utils for storing what optional dependencies are available
"""

# Standard
from importlib.util import find_spec
import pkgutil
import sys

# Third Party
import torch

all_available_modules = []
for finder, name, ispkg in pkgutil.iter_modules(sys.path):
    all_available_modules.append(name)

optional_packages = [
    "gptqmodel",
    "gptqmodel_exllama_kernels",
    "gptqmodel_exllamav2_kernels",
    "llmcompressor",
    "mx",
    "matplotlib",
    "graphviz",
    "pygraphviz",
    "fms",
    "triton",
    "torchvision",
    "huggingface_hub",
    "torchao",
]

available_packages = {}
for package in optional_packages:
    available_packages[package] = (
        find_spec(package) is not None or package in all_available_modules
    )

# cutlass is detected through torch.ops.cutlass_gemm
available_packages["cutlass"] = hasattr(torch.ops, "cutlass_gemm") and hasattr(
    torch.ops.cutlass_gemm, "i8i32nt"
)
