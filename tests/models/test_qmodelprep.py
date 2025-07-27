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
import transformers

# Local
# fms_mo imports
from fms_mo import qconfig_init, qmodel_prep
from fms_mo.prep import has_quantized_module
from fms_mo.utils.import_utils import available_packages
from fms_mo.utils.utils import patch_torch_bmm
from tests.models.test_model_utils import count_qmodules, delete_file, qmodule_error


@pytest.fixture(autouse=True)
def delete_files():
    """
    Delete any known files lingering before starting test
    """
    delete_file("qcfg.json")


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
        with pytest.raises(RuntimeError):
            qmodel_prep(model_quantized, sample_input_fp32, config_fp32)


def test_recipe_not_present(
    wrong_recipe_name: str,
):
    """
    Test if giving a bad recipe .json file name results in a ValueError.

    Args:
        wrong_recipe_name (str): Wrong .json file name
    """
    with pytest.raises(ValueError):
        qconfig_init(recipe=wrong_recipe_name)


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
    # Run qmodel_prep once
    qmodel_prep(model_fp32, sample_input_fp32, config_fp32)
    delete_file()

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
    config_int8["qa_mode"] = qa_mode
    config_int8["qw_mode"] = qw_mode
    qmodel_prep(model_config_fp32, sample_input_fp32, config_int8)


###########################
# Vision/BERT Model Tests #
###########################


@pytest.mark.skipif(
    not available_packages["torchvision"],
    reason="Requires torchvision",
)
def test_resnet50_torchscript(
    model_resnet,
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
    qmodel_prep(model_resnet, batch_resnet, config_int8, use_dynamo=False)
    qmodule_error(model_resnet, 6, 48)


@pytest.mark.skipif(
    not available_packages["torchvision"],
    reason="Requires torchvision",
)
def test_resnet50_dynamo(
    model_resnet,
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
    qmodel_prep(model_resnet, batch_resnet, config_int8, use_dynamo=True)
    qmodule_error(model_resnet, 6, 48)


@pytest.mark.skipif(
    not available_packages["torchvision"],
    reason="Requires torchvision",
)
def test_resnet50_dynamo_layers(
    model_resnet,
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
    config_int8["qlayer_name_pattern"] = ["layer[1,2,4]"]  # allow regex
    qmodel_prep(model_resnet, batch_resnet, config_int8, use_dynamo=True)
    qmodule_error(model_resnet, 21, 33)


# Vision Transformer tests
@pytest.mark.skipif(
    not available_packages["torchvision"],
    reason="Requires torchvision",
)
def test_vit_torchscript(
    model_vit,
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
    qmodel_prep(model_vit, batch_vit, config_int8, use_dynamo=False)
    qmodule_error(model_vit, 2, 36)


@pytest.mark.skipif(
    not available_packages["torchvision"],
    reason="Requires torchvision",
)
def test_vit_dynamo(
    model_vit,
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
    qmodel_prep(model_vit, batch_vit, config_int8, use_dynamo=True)
    qmodule_error(model_vit, 2, 36)


def test_resnet18(
    model_resnet18,
    batch_resnet18,
    config_int8: dict,
):
    """
    Perform int8 quantization on ResNet-18 w/ Dynamo tracer

    Args:
        model_resnet18 (AutoModelForImageClassification): Resnet18 model + weights
        batch_resnet18 (torch.FloatTensor): Batch image data for Resnet18
        config (dict): Recipe Config w/ int8 settings
    """
    # Run qmodel_prep w/ Dynamo tracer
    qmodel_prep(model_resnet18, batch_resnet18, config_int8, use_dynamo=True)
    qmodule_error(model_resnet18, 4, 17)


def test_vit_base(
    model_vit_base,
    batch_vit_base,
    config_int8: dict,
):
    """
    Perform int8 quantization on ViT-base w/ Dynamo tracer

    Args:
        model_vit_base (AutoModelForImageClassification): Resnet18 model + weights
        batch_vit_base (torch.FloatTensor): Batch image data for Resnet18
        config (dict): Recipe Config w/ int8 settings
    """
    # Run qmodel_prep w/ Dynamo tracer
    qmodel_prep(model_vit_base, batch_vit_base, config_int8, use_dynamo=True)
    qmodule_error(model_vit_base, 1, 73)


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
    qmodel_prep(model_bert, input_bert, config_int8, use_dynamo=True)
    qmodule_error(model_bert, 1, 72)


def test_bert_dynamo_wi_qbmm(
    model_bert_eager: transformers.models.bert.modeling_bert.BertModel,
    input_bert: torch.FloatTensor,
    config_int8: dict,
):
    """
    Perform int8 quantization on BERT w/ Dynamo tracer and QBmm modules. QBmms will be run in place
    of torch.matmul/torch.bmm automatically, if everything is set up correctly. See the 3 checks
    below for more details.
    NOTE:
        1. QBmm modules will be added after qmodel_prep(), see check 1.
        2. The self-attention forward() will still call torch.matmul as written in the original
            python code, i.e. if we check QLinear.num_called and QBmm.num_called, they will be 1 and
            0, respectively, meaning QBmms were attached but not called.
        3. By using patch_torch_bmm() context manager, QBmm modules will be triggered and those
            torch.matmul (usually 2 per attn module) calls will be redirect to QBmm's forward.

    Args:
        model_bert (transformers.models.bert.modeling_bert.BertModel): BERT model + weights
        input_bert (torch.FloatTensor): Tokenized input for BERT
        config (dict): Recipe Config w/ int8 settings
    """
    config_int8["nbits_bmm1"] = 8
    config_int8["nbits_bmm2"] = 8
    qmodel_prep(model_bert_eager, input_bert, config_int8, use_dynamo=True)

    # check 1: make sure QBmm are added, i.e. 72 QLinear + 24 QBmm
    qmodule_error(model_bert_eager, 1, 96)

    _, fms_qmodules = count_qmodules(model_bert_eager)
    qbmms = []
    other_qmodules = []
    for n, m in fms_qmodules:
        if "QBmm" in n:
            qbmms.append(m)
        else:
            other_qmodules.append(m)

    # check 2: model call without our "patch" context manager, will not reach QBmm
    #           we have an auto check in place, but it will only log warning, unless this flag
    #           qcfg["force_stop_if_qbmm_auto_check_failed"] = True
    with torch.no_grad():
        model_bert_eager(**input_bert)
    assert all(
        m.num_module_called == 0 for m in qbmms
    ), "Some QBmm was called when they shouldn't be."

    # check 3: model call with context manager, will reach QBmm
    with torch.no_grad(), patch_torch_bmm(config_int8):
        model_bert_eager(**input_bert)
    assert all(
        m.num_module_called == 1 for m in qbmms
    ), "Some QBmm was not called properly."

    assert all(
        m.num_module_called == 2 for m in other_qmodules
    ), "Modules other than QBmm were not called properly."
