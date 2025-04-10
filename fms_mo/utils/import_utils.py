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

# Third Party
from transformers.utils.import_utils import _is_package_available
import torch

optional_packages = [
    "auto_gptq",
    "exllama_kernels",
    "exllamav2_kernels",
    "llmcompressor",
    "matplotlib",
    "graphviz",
    "pygraphviz",
    "fms",
    "triton",
]

available_packages = {}
for package in optional_packages:
    available_packages[package] = _is_package_available(package)

# cutlass is detected through torch.ops.cutlass_gemm
available_packages["cutlass"] = hasattr(torch.ops, "cutlass_gemm") and hasattr(
    torch.ops.cutlass_gemm, "i8i32nt"
)
