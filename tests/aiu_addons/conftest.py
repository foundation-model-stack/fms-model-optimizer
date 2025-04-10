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
"""Pytest configuration file with fixtures for add-ons functionality testing"""

# Standard
from pathlib import Path

# Third Party
import pytest
import torch

# ================================================
# GPTQ W4A16 fixtures
# ================================================

gptq_input_sizes = [
    {
        "bs": 4,
        "seq_len": 5,
        "hid_dim": 256,
        "out_feat": 512,
        "n_grp": 4,
    },
]


@pytest.fixture(scope="session", params=gptq_input_sizes)
def get_gptq_gemm_inputs(request) -> tuple[torch.Tensor, ...]:
    """pytest fixture returning test inputs for GPTQ op"""

    sizes = request.param
    compression_factor = 8  # assume 4-bits compression

    x = torch.randn(
        (sizes["bs"], sizes["seq_len"], sizes["hid_dim"]), dtype=torch.float16
    )
    qweight = torch.randint(
        low=0,
        high=torch.iinfo(torch.int32).max,
        size=(sizes["out_feat"], sizes["hid_dim"] // compression_factor),
        dtype=torch.int32,
    )
    qzeros = 8 * torch.ones(
        (sizes["n_grp"], sizes["out_feat"] // compression_factor),
        dtype=torch.int32,
    )
    scales = torch.randn(
        (sizes["n_grp"], sizes["out_feat"]),
        dtype=torch.float16,
    )
    g_idx = torch.zeros(sizes["hid_dim"], dtype=torch.int32)

    return (x, qweight, qzeros, scales, g_idx)


# ================================================
# INT8xINT8 fixtures
# ================================================

i8i8_metadata = [
    {
        "wtype": "per_tensor",  # per_channel
        "atype": "per_tensor_symm",  # per_tensor_asymm, per_token
        "smoothquant": False,
    },
    {
        "wtype": "per_tensor",  # per_channel
        "atype": "per_tensor_asymm",  # per_tensor_asymm, per_token
        "smoothquant": False,
    },
    {
        "wtype": "per_channel",  # per_channel
        "atype": "per_tensor_symm",  # per_tensor_asymm, per_token
        "smoothquant": False,
    },
    {
        "wtype": "per_tensor",  # per_channel
        "atype": "per_token",  # per_tensor_asymm, per_token
        "smoothquant": False,
    },
    {
        "wtype": "per_channel",  # per_channel
        "atype": "per_tensor_asymm",  # per_tensor_asymm, per_token
        "smoothquant": False,
    },
    {
        "wtype": "per_channel",  # per_channel
        "atype": "per_token",  # per_tensor_asymm, per_token
        "smoothquant": False,
    },
]


@pytest.fixture(scope="session", params=i8i8_metadata)
def get_i8i8_gemm_inputs(
    request,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    str,
    str,
    bool,
    torch.Tensor,
]:
    """pytest fixture returning test inputs for INT8xINT8 op"""

    data = request.param

    filename = (
        f"ref_w-{data['wtype']}_"
        f"a-{data['atype']}_"
        f"sq-{'Y' if data['smoothquant'] else 'N'}.pt"
    )
    addon_references = Path("tests/artifacts/aiu_addons")
    i8i8_data = torch.load(addon_references / filename, weights_only=True)

    assert isinstance(i8i8_data, dict)
    assert data["wtype"] == i8i8_data["weight_quant_type"]
    assert data["atype"] == i8i8_data["activ_quant_type"]
    assert data["smoothquant"] == i8i8_data["smoothquant"]
    assert all(
        item in i8i8_data for item in ["x", "w_int", "bias", "qdata", "reference_out"]
    )

    return (
        i8i8_data["x"],
        i8i8_data["w_int"],
        i8i8_data["bias"],
        i8i8_data["qdata"],
        i8i8_data["weight_quant_type"],
        i8i8_data["activ_quant_type"],
        i8i8_data["smoothquant"],
        i8i8_data["reference_out"],
    )
