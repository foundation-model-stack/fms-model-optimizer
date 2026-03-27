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

# Standard
import warnings

# Third Party
import pytest
import torch

# Local
from fms_mo.prep import available_packages

# Suppress the UserWarning about overriding kernel registration in PyTorch 2.8+
# This warning is expected when we override the native CPU kernel for _scaled_mm
warnings.simplefilter("ignore", UserWarning)
# Local
import fms_mo.aiu_addons.fp8.fp8_spyre_op  # noqa: E402  # pylint: disable=unused-import,wrong-import-position

warnings.simplefilter("default", UserWarning)  # Reset to default after import

# ============================================================================
# Constants
# ============================================================================

# FP8 E4M3 maximum value
FP8_E4M3_MAX = torch.finfo(torch.float8_e4m3fn).max

# ============================================================================
# Helper Functions
# ============================================================================


def initialize_fp8_weights(
    fp8_linear,
    weight_strategy: str,
    in_features: int,
    out_features: int,
) -> None:
    """Initialize FP8Linear weights with proper absmax scaling.

    Args:
        fp8_linear: FP8Linear module to initialize
        weight_strategy: "tensor" or "channel" for weight quantization
        in_features: Input feature dimension
        out_features: Output feature dimension
    """
    with torch.no_grad():
        # Create random float weights
        float_weights = torch.randn(out_features, in_features)

        # Set appropriate scales based on strategy using absmax
        if weight_strategy == "tensor":
            # Per-tensor: single scale for entire weight matrix
            absmax = float_weights.abs().max()
            scale = absmax / FP8_E4M3_MAX
            # Ensure scale is not zero
            scale = torch.clamp(scale, min=1e-12)
            fp8_linear.weight_scale.fill_(scale.item())
        else:  # channel (per-row for weight matrix)
            # Per-channel: one scale per output channel (row)
            absmax = float_weights.abs().amax(dim=1)
            scale = absmax / FP8_E4M3_MAX
            # Ensure scales are not zero
            scale = torch.clamp(scale, min=1e-12)
            # Reshape to match weight_scale parameter shape (out_features, 1)
            fp8_linear.weight_scale.copy_(scale.reshape(-1, 1))

        # Quantize weights to FP8
        quantized_weights = (float_weights / fp8_linear.weight_scale).clamp(
            -FP8_E4M3_MAX, FP8_E4M3_MAX
        )
        fp8_linear.weight.copy_(quantized_weights.to(torch.float8_e4m3fn))

        # Initialize bias if present
        if fp8_linear.has_bias:
            fp8_linear.bias.copy_(torch.randn(out_features))


# ============================================================================
# Pytest Fixtures
# ============================================================================


@pytest.fixture
def fp8_test_dimensions():
    """Common test dimensions for FP8Linear tests."""
    return {
        "batch_size": 2,
        "seq_len": 4,
        "in_features": 8,
        "out_features": 16,
    }


# ============================================================================
# Tests
# ============================================================================


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
    """Validate output shapes of FP8 attention operation.

    Tests the FP8 attention compute operation to ensure it produces
    outputs with the expected shape.
    """
    # Local
    from fms_mo.aiu_addons.fp8.fp8_attn import _math_fp8_compute_op

    query = torch.randn((1, 64, 32, 128), dtype=torch.bfloat16, device="cuda")
    key = torch.randn((1, 64, 32, 128), dtype=torch.bfloat16, device="cuda")
    value = torch.randn((1, 64, 32, 128), dtype=torch.bfloat16, device="cuda")

    out = _math_fp8_compute_op(query, key, value, 32, 32, 0.0, None)
    assert out.size() == query.size()


@pytest.mark.skipif(
    not available_packages["torchao"] or not available_packages["fms"],
    reason="FMS and torchao required to run this test",
)
@pytest.mark.parametrize(
    "weight_strategy,activation_strategy",
    [
        ("tensor", "tensor"),  # Per-tensor W + per-tensor dynamic A
        ("channel", "token"),  # Per-channel W + per-token dynamic A
    ],
)
def test_fp8_linear_cpu_support(  # pylint: disable=redefined-outer-name
    weight_strategy: str,
    activation_strategy: str,
    fp8_test_dimensions: dict,
) -> None:
    """Test FP8Linear on CPU with supported quantization strategies.

    This test ensures that FP8Linear works correctly on CPU with:
    - Per-tensor quantization (weights and activations both per-tensor)
    - Per-channel quantization (weights and activations both per-channel/per-token)

    Note: Mixed granularity (e.g., per-tensor weights with per-token activations)
    is not supported on the target custom hardware.

    Args:
        weight_strategy: "tensor" or "channel" weight quantization
        activation_strategy: "tensor" or "token" dynamic activation quantization
        fp8_test_dimensions: Test dimensions fixture
    """
    # Local
    from fms_mo.aiu_addons.fp8.fp8_linear import FP8Linear

    # Get test dimensions
    batch_size = fp8_test_dimensions["batch_size"]
    seq_len = fp8_test_dimensions["seq_len"]
    in_features = fp8_test_dimensions["in_features"]
    out_features = fp8_test_dimensions["out_features"]

    # Create FP8Linear configuration
    linear_config = {
        "weights": {
            "strategy": weight_strategy,
            "symmetric": True,
            "dynamic": False,
        },
        "input_activations": {
            "strategy": activation_strategy,
            "symmetric": True,
            "dynamic": True,
        },
    }

    # Create FP8Linear module
    fp8_linear = FP8Linear(
        in_features=in_features,
        out_features=out_features,
        bias=True,
        linear_config=linear_config,
    )

    # Initialize weights using helper function
    initialize_fp8_weights(fp8_linear, weight_strategy, in_features, out_features)

    # Create input tensor on CPU
    x = torch.randn(batch_size, seq_len, in_features, dtype=torch.bfloat16)

    # Run forward pass - should not raise an error
    output = fp8_linear(x)

    # Validate output shape
    assert output.shape == (batch_size, seq_len, out_features)

    # Validate output is not NaN or Inf
    assert not torch.isnan(output).any()
    assert not torch.isinf(output).any()

    # Validate output dtype matches input dtype
    assert output.dtype == x.dtype


@pytest.mark.skipif(
    not available_packages["torchao"] or not available_packages["fms"],
    reason="FMS and torchao required to run this test",
)
def test_fp8_linear_cpu_no_activation_quantization(fp8_test_dimensions: dict) -> None:  # pylint: disable=redefined-outer-name
    """Test FP8Linear on CPU with only weight quantization (no activation quantization).

    This tests the code path where activations are not quantized but weights are FP8.

    Args:
        fp8_test_dimensions: Test dimensions fixture
    """
    # Local
    from fms_mo.aiu_addons.fp8.fp8_linear import FP8Linear

    # Get test dimensions
    batch_size = fp8_test_dimensions["batch_size"]
    seq_len = fp8_test_dimensions["seq_len"]
    in_features = fp8_test_dimensions["in_features"]
    out_features = fp8_test_dimensions["out_features"]

    # Create FP8Linear configuration with no activation quantization
    linear_config = {
        "weights": {
            "strategy": "channel",
            "symmetric": True,
            "dynamic": False,
        },
        "input_activations": None,  # No activation quantization
    }

    # Create FP8Linear module
    fp8_linear = FP8Linear(
        in_features=in_features,
        out_features=out_features,
        bias=True,
        linear_config=linear_config,
    )

    # Initialize weights using helper function
    initialize_fp8_weights(fp8_linear, "channel", in_features, out_features)

    # Create input tensor on CPU
    x = torch.randn(batch_size, seq_len, in_features, dtype=torch.bfloat16)

    # Run forward pass
    output = fp8_linear(x)

    # Validate output
    assert output.shape == (batch_size, seq_len, out_features)
    assert not torch.isnan(output).any()
    assert not torch.isinf(output).any()
