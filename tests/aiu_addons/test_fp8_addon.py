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
"""Test suite for FMS addon introducing FP8 functionalities"""

# Third Party
import pytest
import torch

# Local
from fms_mo.prep import available_packages
import fms_mo.aiu_addons.fp8.fp8_spyre_op  # pylint: disable=unused-import


def test_fp8_registration() -> None:
    """
    Ensure fp8 ops are registered properly.
    """

    assert hasattr(torch.ops, "spyre")
    assert hasattr(torch.ops.spyre, "scaled_bmm")
    assert hasattr(torch.ops.spyre, "scaled_paged_attn_store")
    assert hasattr(torch.ops.spyre, "scaled_paged_attn_compute")


# This test requires an H100 or higher GPU to run
@pytest.mark.skipif(
    not available_packages["torchao"] or not available_packages["fms"],
    reason="FMS and torchao required to run this test",
)
@pytest.mark.skipif(
    not torch.cuda.is_available()
    or (torch.cuda.is_available() and torch.cuda.get_device_capability() < (8, 9)),
    reason="FP8 is only available on GPUs with device level 8.9 or higher",
)
def test_fp8_op() -> None:
    """Validate output shapes of GPTQ W4A16 tensors.
    Note: this AIU-compatible operation only returns a zero tensor of the
    expected shape, it does not perform a real W4A16 matmul operation.
    """
    # Local
    from fms_mo.aiu_addons.fp8.fp8_attn import _math_fp8_compute_op

    query = torch.randn((1, 32, 64, 128), dtype=torch.bfloat16, device="cuda")
    key = torch.randn((1, 32, 64, 128), dtype=torch.bfloat16, device="cuda")
    value = torch.randn((1, 32, 64, 128), dtype=torch.bfloat16, device="cuda")

    out = _math_fp8_compute_op(query, key, value, 32, 32, 0.0, None)
    assert out.size() == query.size()
