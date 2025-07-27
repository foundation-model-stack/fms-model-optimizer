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
Test functionality of quantization workflow on toy models
"""

# Standard
from copy import deepcopy

# Third Party
import pytest
import torch

# Local
from fms_mo import qmodel_prep
from fms_mo.modules.conv import DetQConv2d, QConv2d, QConv2dPTQ, QConv2dPTQv2
from fms_mo.modules.linear import QLinear
from tests.models.test_model_utils import (
    delete_file,
    is_qconv2d,
    is_qlinear,
    is_quantized_layer,
)


@pytest.fixture(autouse=True)
def delete_files():
    """
    Delete any known files lingering before starting test
    """
    delete_file("qcfg.json")


def test_subclass():
    """
    Verify proper subclass for quantization layer.
    Add to this test everytime we create a new quantization class.
    """
    assert issubclass(
        DetQConv2d, torch.nn.Conv2d
    ), "DetQConv2d is not subclass of nn.Conv2d"
    assert issubclass(QConv2d, torch.nn.Conv2d), "QConv2d is not subclass of nn.Conv2d"
    assert issubclass(
        QConv2dPTQ, torch.nn.Conv2d
    ), "QConv2dPTQ is not subclass of nn.Conv2d"
    assert issubclass(
        QConv2dPTQv2, torch.nn.Conv2d
    ), "QConv2dPTQv2 is not subclass of nn.Conv2d"
    assert issubclass(QLinear, torch.nn.Linear), "QLinear is not subclass of nn.Linear"


def test_first_layer_unchanged_fp32(
    model_fp32: torch.nn.Module,
    sample_input_fp32: torch.FloatTensor,
    config_fp32: dict,
):
    """
    This test should verify that under default conditions first and last layers are not quantized.

    Args:
        model_fp32 (torch.nn.Module): Toy Model w/ fp32 data
        sample_input_fp32 (torch.FloatTensor): Sample fp32 input for calibration.
        config_fp32 (dict): Config w/ fp32 settings
    """
    # Grab copy of first layer before quantization
    first_layer = deepcopy(model_fp32.first_layer)

    # Quanatize model
    qmodel_prep(model_fp32, sample_input_fp32, config_fp32)

    # Check that model's first layer is unchanged
    assert torch.equal(
        first_layer.weight, model_fp32.first_layer.weight
    ), "first_layer weight objects are not equal"
    assert torch.equal(
        first_layer.weight.data, model_fp32.first_layer.weight.data
    ), "first_layer weight data are not equal"

    # If model.first_layer has a bias, check equality
    if model_fp32.first_layer.bias is not None:
        assert torch.equal(
            first_layer.bias, model_fp32.first_layer.bias
        ), "first_layer bias objects are not equal"
        assert torch.equal(
            first_layer.bias.data, model_fp32.first_layer.bias.data
        ), "first_layer bias data are not equal"

    # Free first_layer copy
    del first_layer


def test_first_layer_unchanged_fp16(
    model_fp16: torch.nn.Module,
    sample_input_fp16: torch.FloatTensor,
    config_fp16: dict,
):
    """
    This test should verify that under default conditions first and last layers are not quantized.

    Args:
        model_fp16 (torch.nn.Module): Toy Model w/ fp16 data
        sample_input_fp16 (torch.FloatTensor): Sample fp16 input for calibration.
        config_fp16 (dict): Config w/ fp16 settings
    """
    # Grab copy of first layer before quantization
    first_layer = deepcopy(model_fp16.first_layer)

    # Quanatize model
    qmodel_prep(model_fp16, sample_input_fp16, config_fp16)

    # Check that model's first layer is unchanged
    assert torch.equal(
        first_layer.weight, model_fp16.first_layer.weight
    ), "first_layer weight objects are not equal"
    assert torch.equal(
        first_layer.weight.data, model_fp16.first_layer.weight.data
    ), "first_layer weight data are not equal"

    if model_fp16.first_layer.bias is not None:
        assert torch.equal(
            first_layer.bias, model_fp16.first_layer.bias
        ), "first_layer bias objects are not equal"
        assert torch.equal(
            first_layer.bias.data, model_fp16.first_layer.bias.data
        ), "first_layer bias data are not equal"

    del first_layer


def test_second_layer_quantized_fp32(
    model_fp32: torch.nn.Module,
    sample_input_fp32: torch.FloatTensor,
    config_fp32: dict,
):
    """
    This test should verify that the second layer is always quantized.
    It should also verify that the weight and bias data remain unmodified.

    first_layer, second_layer, third_layer count only include Linear and Conv2d.
    In other words, layers such as nn.Sequential, nn.ReLU should not be named in such
    a way for the ToyModel.

    Args:
        model_fp32 (torch.nn.Module): Toy Model w/ fp32 data
        sample_input_fp32 (torch.FloatTensor): Sample fp32 input for calibration.
        config_fp32 (dict): Config w/ fp32 settings
    """
    if hasattr(model_fp32, "second_layer"):
        # Copy 2nd layer before quantization
        second_layer = deepcopy(model_fp32.second_layer)

        # Quanatize model
        qmodel_prep(model_fp32, sample_input_fp32, config_fp32)

        # Check 2nd layer is quantized
        if isinstance(second_layer, torch.nn.Conv2d):
            assert is_qconv2d(
                model_fp32.second_layer
            ), "quantized second_layer is not Qconv2d type"
        elif isinstance(second_layer, torch.nn.Linear):
            assert is_qlinear(
                model_fp32.second_layer
            ), "quantized second_layer is not QLinear type"

        # Check that data is unmodified in the quantized second layer.
        assert id(second_layer) != id(
            model_fp32.second_layer
        ), "second_layer id's are not equal"
        assert torch.equal(
            second_layer.weight, model_fp32.second_layer.weight
        ), "second_layer weight objects are not equal"
        assert torch.equal(
            second_layer.weight.data, model_fp32.second_layer.weight.data
        ), "second_layer weight data are not equal"
        assert torch.equal(
            second_layer.bias, model_fp32.second_layer.bias
        ), "second_layer bias objects are not equal"
        assert torch.equal(
            second_layer.bias.data, model_fp32.second_layer.bias.data
        ), "second_layer bias data are not equal"
        assert (
            model_fp32.second_layer.num_bits_weight == config_fp32["nbits_w"]
        ), "model.second_layer nbit_w not equal to config"
        assert (
            model_fp32.second_layer.num_bits_feature == config_fp32["nbits_a"]
        ), "model.second_layer nbit_a not equal to config"

        del second_layer


def test_second_layer_quantized_fp16(
    model_fp16: torch.nn.Module,
    sample_input_fp16: torch.FloatTensor,
    config_fp16: dict,
):
    """
    This test should verify that the second layer is always quantized.
    It should also verify that the weight and bias data remain unmodified.

    first_layer, second_layer, third_layer count only include Linear and Conv2d.
    In other words, layers such as nn.Sequential, nn.ReLU should not be named in such
    a way for the ToyModel.

    Args:
        model_fp16 (torch.nn.Module): Toy Model w/ fp32 data
        sample_input_fp16 (torch.FloatTensor): Sample fp16 input for calibration.
        config_fp16 (dict): Config w/ fp16 settings
    """
    if hasattr(model_fp16, "second_layer"):
        # Copy 2nd layer before quantization
        second_layer = deepcopy(model_fp16.second_layer)

        # Quanatize model
        qmodel_prep(model_fp16, sample_input_fp16, config_fp16)

        # Check 2nd layer is quantized
        if isinstance(model_fp16.second_layer, torch.nn.Conv2d):
            assert is_qconv2d(
                model_fp16.second_layer
            ), "second_layer is not Qconv2d type"
        elif isinstance(model_fp16.second_layer, torch.nn.Linear):
            assert is_qlinear(
                model_fp16.second_layer
            ), "second_layer is not QLinear type"

        # Check that data is unmodified in the quantized second layer.
        assert id(second_layer) != id(
            model_fp16.second_layer
        ), "second_layer id's are not equal"
        assert torch.equal(
            second_layer.weight, model_fp16.second_layer.weight
        ), "second_layer weight objects are not equal"
        assert torch.equal(
            second_layer.weight.data, model_fp16.second_layer.weight.data
        ), "second_layer weight data are not equal"
        assert torch.equal(
            second_layer.bias, model_fp16.second_layer.bias
        ), "second_layer bias objects are not equal"
        assert torch.equal(
            second_layer.bias.data, model_fp16.second_layer.bias.data
        ), "second_layer bias data are not equal"
        assert (
            model_fp16.second_layer.num_bits_weight == config_fp16["nbits_w"]
        ), "model.second_layer nbit_w not equal to config"
        assert (
            model_fp16.second_layer.num_bits_feature == config_fp16["nbits_a"]
        ), "model.second_layer nbit_a not equal to config"

        del second_layer


def test_qmodel_prep_output_fp32(
    model_fp32: torch.nn.Module,
    sample_input_fp32: torch.FloatTensor,
    config_fp32: dict,
):
    """
    Testing that 32-bit qmodel_prep doesn't change fp32 model results

    Args:
        model_fp32 (torch.nn.Module): Toy Model w/ fp32 data
        sample_input_fp32 (torch.FloatTensor): Sample fp32 input for calibration.
        config_fp32 (dict): Config w/ fp32 settings
    """
    # need to detach to make deepcopy
    original_output = deepcopy(model_fp32(sample_input_fp32).detach())

    # Default quantization uses fp32
    qmodel_prep(model_fp32, sample_input_fp32, config_fp32)

    # default model_prep should use single-precision so output should be the same.
    assert torch.equal(
        model_fp32(sample_input_fp32), original_output
    ), "fp32 model output after default quantization has changed"


def test_qmodel_prep_output_fp16(
    model_fp16: torch.nn.Module,
    sample_input_fp16: torch.FloatTensor,
    config_fp16: dict,
):
    """
    Testing that fp16 qmodel_prep doesn't change fp16 model results

    Args:
        model_fp16 (torch.nn.Module): Toy Model w/ fp16 data
        sample_input_fp16 (torch.FloatTensor): Sample fp16 input for calibration.
        config_fp16 (dict): Config w/ fp16 settings
    """
    # need to detach to make deepcopy
    original_output = deepcopy(model_fp16(sample_input_fp16).detach())
    qmodel_prep(model_fp16, sample_input_fp16, config_fp16)

    # qmodel_prep should use fp16 so output should be the same.
    assert torch.equal(
        model_fp16(sample_input_fp16), original_output
    ), "fp16 model output after 16-bit quantization has changed"


def test_qmodel_prep_num_bits_output_fp32(
    model_fp32: torch.nn.Module,
    sample_input_fp32: torch.FloatTensor,
    num_bits_activation_fp32: int,
    num_bits_weight_fp32: int,
    config_fp32: dict,
):
    """
    Testing that model_prep changes output for num_bits of each.

    Args:
        model_fp32 (torch.nn.Module): Toy Model w/ fp32 data
        sample_input_fp32 (torch.FloatTensor): Sample fp32 input for calibration.
        num_bits_activation_fp32 (int): nbits_a valid for fp32
        num_bits_weight_fp32 (int): nbits_w valid for fp32
        config_fp32 (dict): Config w/ fp32 settings
    """
    # Get model output before quantization
    original_output = deepcopy(model_fp32(sample_input_fp32).detach())

    # Get model dtype before quantizing
    _, param = next(model_fp32.named_parameters())
    model_dtype = param.dtype

    # Quantize model
    config_fp32["nbits_a"] = num_bits_activation_fp32
    config_fp32["nbits_w"] = num_bits_weight_fp32
    qmodel_prep(model_fp32, sample_input_fp32, config_fp32)

    # Check to see if quantized model has Qmodules
    has_quantization_layer = False
    for _, module in model_fp32.named_modules():
        if is_quantized_layer(module):
            has_quantization_layer = True
            break

    # Check if quantized output is equal to original output
    isEqual = torch.equal(model_fp32(sample_input_fp32), original_output)

    # If there is a quantization layer, then the quantization most likely will change the output
    if has_quantization_layer:
        # Qbypass allowed for both fp16,fp32
        if (
            num_bits_activation_fp32 in [16, 32]
            and num_bits_weight_fp32 in [16, 32]
            and model_dtype == torch.float32
        ):
            assert isEqual, "32-bit quantization has changed output of fp32 model"
        elif (
            num_bits_activation_fp32 == 16
            and num_bits_weight_fp32 == 16
            and model_dtype in (torch.float16, torch.bfloat16)
        ):
            assert isEqual, "16-bit quantization has changed output of fp16 model"
        else:
            assert (
                not isEqual
            ), "Quantized int model has same output as pre-quantized model"
    else:
        # Non-quantized models should not change output
        assert isEqual, "Non-quantized model has changed output"

    # Free original output
    del original_output


def test_qmodel_prep_num_bits_output_fp16(
    model_fp16: torch.nn.Module,
    sample_input_fp16: torch.FloatTensor,
    num_bits_activation_fp16: int,
    num_bits_weight_fp16: int,
    config_fp16: dict,
):
    """
    Testing that model_prep changes output for num_bits of each.

    Args:
        model_fp16 (torch.nn.Module): Toy Model w/ fp16 data
        sample_input_fp16 (torch.FloatTensor): Sample fp16 input for calibration.
        num_bits_activation_fp16 (int): nbits_a valid for fp16
        num_bits_weight_fp16 (int): nbits_w valid for fp16
        config_fp16 (dict): Config w/ fp16 settings
    """
    # Get model output before quantization
    original_output = deepcopy(model_fp16(sample_input_fp16).detach())

    # Get model dtype before quantizing
    _, param = next(model_fp16.named_parameters())
    model_dtype = param.dtype

    # Quantize model
    config_fp16["nbits_a"] = num_bits_activation_fp16
    config_fp16["nbits_w"] = num_bits_weight_fp16
    qmodel_prep(model_fp16, sample_input_fp16, config_fp16)

    # Check to see if quantized model has Qmodules
    has_quantization_layer = False
    for _, module in model_fp16.named_modules():
        if is_quantized_layer(module):
            has_quantization_layer = True
            break

    # Check if quantized output is equal to original output
    isEqual = torch.equal(model_fp16(sample_input_fp16), original_output)

    # If there is a quantization layer, then the quantization most likely will change the output
    if has_quantization_layer:
        if (
            num_bits_activation_fp16 == 16
            and num_bits_weight_fp16 == 16
            and model_dtype in (torch.float16, torch.bfloat16)
        ):
            assert isEqual, "16-bit quantization has changed output of fp16 model"
        else:
            assert (
                not isEqual
            ), "Quantized int model has same output as pre-quantized model"
    else:
        # Non-quantized models should not change output
        assert isEqual, "Non-quantized model has changed output"

    del original_output
