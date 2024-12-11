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
Test qmodel_prep functionality on Toy Models, Resnet50, Vision Transformer, and BERT
"""

# Third Party
import pytest
import torch
import torchvision
import transformers

# First Party
# fms_mo imports
from fms_mo import qmodel_prep
from fms_mo.prep import has_quantized_module
from tests.models.test_model_utils import delete_config, qmodule_error

################
# Qmodel tests #
################

# Requires Nvidia GPU to run
if torch.cuda.is_available():

    def test_model_quantized(
        model_quantized: torch.nn.Module,
        sample_input_fp32: torch.FloatTensor,
        config_fp32: dict,
    ):
        """
        qmodel_prep should always throw RuntimeError if a model is already quantized

        Args:
            model_quantized (torch.nn.Module): Quantized Toy Model
            sample_input_fp32 (torch.FloatTensor): Sample fp32 input for calibration.
            config_fp32 (dict): Config w/ fp32 settings
        """
        delete_config()
        with pytest.raises(RuntimeError):
            qmodel_prep(model_quantized, sample_input_fp32, config_fp32)


def test_double_qmodel_prep_assert(
    model_fp32: torch.nn.Module,
    sample_input_fp32: torch.FloatTensor,
    config_fp32: dict,
):
    """
    qmodel_prep twice shouldn't be allowed.  If the model becomes quantized after
    1st qmodel_prep, it should throw a RuntimeError

    Args:
        model_fp32 (torch.nn.Module): Toy Model w/ fp32 data
        sample_input_fp32 (torch.FloatTensor): Sample fp32 input for calibration
        config_fp32 (dict): Config w/ fp32 settings
    """
    delete_config()

    # Run qmodel_prep once
    qmodel_prep(model_fp32, sample_input_fp32, config_fp32)
    delete_config()

    # If model now has a quantized node, ensure it has raises a RuntimeError
    if has_quantized_module(model_fp32):
        with pytest.raises(RuntimeError):
            qmodel_prep(model_fp32, sample_input_fp32, config_fp32)


# Test recipe configs are working on Toy Models
def test_config_recipes_fp32(
    model_fp32: torch.nn.Module,
    sample_input_fp32: torch.FloatTensor,
    config_int8: dict,
):
    """
    Testing qmodel_prep recipes allow for successful quantization on fp32 model+data

    Args:
        model_fp32 (torch.nn.Module): Toy Model w/ fp32 data
        sample_input_fp32 (torch.FloatTensor): Sample fp32 input for calibration
        config (dict): Recipe Config w/ int8 settings
    """
    delete_config()
    qmodel_prep(model_fp32, sample_input_fp32, config_int8)


def test_config_recipes_fp16(
    model_fp16: torch.nn.Module,
    sample_input_fp16: torch.FloatTensor,
    config_int8: dict,
):
    """
    Testing qmodel_prep recipes allow for successful quantization on fp16 model+data

    Args:
        model_fp16 (torch.nn.Module): Toy Model w/ fp16 data
        sample_input_fp16 (torch.FloatTensor): Sample fp16 input for calibration
        config (dict): Recipe Config w/ int8 settings
    """
    delete_config()
    qmodel_prep(model_fp16, sample_input_fp16, config_int8)


def test_config_fp32_qmodes(
    model_config_fp32: torch.nn.Module,
    sample_input_fp32: torch.FloatTensor,
    config_int8: dict,
    qa_mode: str,
    qw_mode: str,
):
    """
    Test that all supported qa/qw modes are callable

    Args:
        model_config_fp32 (torch.nn.Module): Toy Model w/ fp32 data
        sample_input_fp32 (torch.FloatTensor): Sample fp32 input for calibration
        config (dict): Recipe Config w/ int8 settings
        qa_mode (str): Activation quantizer
        qw_mode (str): Weight quantizer
    """
    delete_config()
    config_int8["qa_mode"] = qa_mode
    config_int8["qw_mode"] = qw_mode
    qmodel_prep(model_config_fp32, sample_input_fp32, config_int8)


###########################
# Vision/BERT Model Tests #
###########################


def test_resnet50_torchscript(
    model_resnet: torchvision.models.resnet.ResNet,
    batch_resnet: torch.FloatTensor,
    config_int8: dict,
):
    """
    Perform int8 quantization on Resnet50 w/ TorchScript tracer

    Args:
        model_resnet (torchvision.models.resnet.ResNet): Resnet50 model + weights
        batch_resnet (torch.FloatTensor): Batch image data for Resnet50
        config (dict): Recipe Config w/ int8 settings
    """
    # Run qmodel_prep w/ default torchscript tracer
    delete_config()
    qmodel_prep(model_resnet, batch_resnet, config_int8, use_dynamo=False)
    qmodule_error(model_resnet, 6, 48)


def test_resnet50_dynamo(
    model_resnet: torchvision.models.resnet.ResNet,
    batch_resnet: torch.FloatTensor,
    config_int8: dict,
):
    """
    Perform int8 quantization on Resnet50 w/ Dynamo tracer

    Args:
        model_resnet (torchvision.models.resnet.ResNet): Resnet50 model + weights
        batch_resnet (torch.FloatTensor): Batch image data for Resnet50
        config (dict): Recipe Config w/ int8 settings
    """
    # Run qmodel_prep w/ Dynamo tracer
    delete_config()
    qmodel_prep(model_resnet, batch_resnet, config_int8, use_dynamo=True)
    qmodule_error(model_resnet, 6, 48)


def test_resnet50_dynamo_layers(
    model_resnet: torchvision.models.resnet.ResNet,
    batch_resnet: torch.FloatTensor,
    config_int8: dict,
):
    """
    Perform int8 quantization on Resnet50 w/ Dynamo tracer.
    Use manual Qlayer patterns to identify Qmodule targets.

    Args:
        model_resnet (torchvision.models.resnet.ResNet): Resnet50 model + weights
        batch_resnet (torch.FloatTensor): Batch image data for Resnet50
        config (dict): Recipe Config w/ int8 settings
    """
    # Run qmodel_prep w/ qlayer_name_pattern + Dynamo tracer
    delete_config()
    config_int8["qlayer_name_pattern"] = ["layer[1,2,4]"]  # allow regex
    qmodel_prep(model_resnet, batch_resnet, config_int8, use_dynamo=True)
    qmodule_error(model_resnet, 21, 33)


# Vision Transformer tests
def test_vit_torchscript(
    model_vit: torchvision.models.vision_transformer.VisionTransformer,
    batch_vit: torch.FloatTensor,
    config_int8: dict,
):
    """
    Perform int8 quantization on ViT w/ TorchScript tracer

    Args:
        model_vit (torchvision.models.vision_transformer.VisionTransformer): ViT model + weights
        batch_vit (torch.FloatTensor): Batch image data for ViT
        config (dict): Recipe Config w/ int8 settings
    """
    # Run qmodel_prep w/ default torchscript tracer
    delete_config()
    qmodel_prep(model_vit, batch_vit, config_int8, use_dynamo=False)
    qmodule_error(model_vit, 2, 36)


def test_vit_dynamo(
    model_vit: torchvision.models.vision_transformer.VisionTransformer,
    batch_vit: torch.FloatTensor,
    config_int8: dict,
):
    """
    Perform int8 quantization on ViT w/ Dynamo tracer

    Args:
        model_vit (torchvision.models.vision_transformer.VisionTransformer): ViT model + weights
        batch_vit (torch.FloatTensor): Batch image data for ViT
        config (dict): Recipe Config w/ int8 settings
    """
    # Run qmodel_prep w/ Dynamo tracer
    delete_config()
    qmodel_prep(model_vit, batch_vit, config_int8, use_dynamo=True)
    qmodule_error(model_vit, 2, 36)


def test_bert_dynamo(
    model_bert: transformers.models.bert.modeling_bert.BertModel,
    input_bert: torch.FloatTensor,
    config_int8: dict,
):
    """
    Perform int8 quantization on BERT w/ Dynamo tracer

    Args:
        model_bert (transformers.models.bert.modeling_bert.BertModel): BERT model + weights
        input_bert (torch.FloatTensor): Tokenized input for BERT
        config (dict): Recipe Config w/ int8 settings
    """
    # Run qmodel_prep w/ Dynamo tracer
    delete_config()
    qmodel_prep(model_bert, input_bert, config_int8, use_dynamo=True)
    qmodule_error(model_bert, 1, 72)
