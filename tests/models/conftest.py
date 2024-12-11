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
Fixtures for testing models
"""

# Standard
import os
from collections import OrderedDict
from copy import deepcopy

# Third Party
import pytest
import torch
from torchvision.io import read_image
from torchvision.models import (ResNet50_Weights, ViT_B_16_Weights, resnet50,
                                vit_b_16)
from transformers import BertModel, BertTokenizer

# First Party
# fms_mo imports
from fms_mo import qconfig_init
from fms_mo.modules import QLSTM, QConv2d, QConvTranspose2d, QLinear

########################
# check_config Fixtures #
########################

# Create fixtures for the config data
nbits_str = [
    "nbits_a",
    "nbits_w",
    "nbits_a_qkv",
    "nbits_w_qkv",
    "nbits_bmm1",
    "nbits_bmm2",
    "nbits_kvcache",
    "nbits_a_alt",
    "nbits_w_alt",
]


@pytest.fixture(scope="session", params=nbits_str)
def nbit_str(request):
    """
    Get a nbit name in a qconfig

    Args:
        request (str): nbit name

    Returns:
        str: nbit name
    """
    return request.param


q_modes_str = [
    "qa_mode",
    "qa_qkv_mode",
    "qw_mode",
    "qw_qkv_mode",
    "bmm1_qm1_mode",
    "bmm1_qm2_mode",
    "bmm2_qm1_mode",
    "bmm2_qm2_mode",
    # calib + init modes use same strings
    "qa_mode_calib",
    "qw_mode_calib",
    "a_init_method",
    "w_init_method",
]


@pytest.fixture(scope="session", params=q_modes_str)
def q_mode_str(request):
    """
    Get a q_mode name in a qconfig

    Args:
        request (str): q_mode name

    Returns:
        str: q_mode name
    """
    return request.param


boolean_vars_str = [
    "extend_act_range",
    "qshortcutconv",
    "q1stlastconv",
    "qdw",
    "qskipfpn",
    "qkvsync",
    "plotsvg",
    "ptq_freezecvs",
    "ptq_qdrop",
    #'use_PT_native_Qfunc', # is not defined in qconfig_init
]


@pytest.fixture(scope="session", params=boolean_vars_str)
def bool_str(request):
    """
    Get a bool name in a qconfig

    Args:
        request (str): bool name

    Returns:
        str: bool name
    """
    return request.param


integer_vars_str = [
    "qmodel_calibration",
    "qmodel_calibration_new",
    "ptq_nbatch",
    "ptq_batchsize",
    "ptq_nouterloop",
    "ptq_ninnerloop",
]


@pytest.fixture(scope="session", params=integer_vars_str)
def int_str(request):
    """
    Get a int name in a qconfig

    Args:
        request (str): int name

    Returns:
        str: int name
    """
    return request.param


fp_vars_str = [
    # 'pact_a_lr', # is not in qconfig_init
    # 'pact_a_decay', # is not in qconfig_init
    "ptq_lrw",
    "ptq_lrcv_w",
    "ptq_lrcv_a",
]


@pytest.fixture(scope="session", params=fp_vars_str)
def fp_str(request):
    """
    Get a fp name in a qconfig

    Args:
        request (str): fp name

    Returns:
        str: fp name
    """
    return request.param


iterable_vars_str = [
    "qskip_layer_name",
    "qspecial_layers",
    "qsinglesided_name",
    "ptqmod_to_be_optimized",
    "firstptqmodule",
    "params2optim",
    "clip_val_asst_percentile",
]


@pytest.fixture(scope="session", params=iterable_vars_str)
def iterable_str(request):
    """
    Get a iterable name in a qconfig

    Args:
        request (str): iterable name

    Returns:
        str: iterable name
    """
    return request.param


@pytest.fixture(scope="session")
def wrong_nbits():
    """
    Get list of invalid bit values

    Returns:
        list: invalid bits list
    """
    return [1, 3, 5, 7, 15, 31]


@pytest.fixture(scope="session")
def wrong_qmodes():
    """
    Get list of invalid qmodes values

    Returns:
        list: invalid qmodes list
    """
    return ["pact-", "saub", "wmax", "minmux", "pactsyn", "perventile"]


@pytest.fixture(scope="session")
def not_booleans():
    """
    Get list of invalid booleans values

    Returns:
        list: invalid booleans list
    """
    return [0.1, 5, torch.tensor(True)]


@pytest.fixture(scope="session")
def not_ints():
    """
    Get list of invalid ints values.
    Note: booleans also considered ints ; also check_config will cast float(k.0) to int(k)

    Returns:
        list: invalid ints list
    """
    return [0.1, torch.tensor(1)]


@pytest.fixture(scope="session")
def not_fps():
    """
    Get list of invalid fp values.
    Note: check_config will cast int(k) to float(k.0)

    Returns:
        list: invalid fp list
    """
    return [False, torch.tensor(0.1)]


@pytest.fixture(scope="session")
def not_iterables():
    """
    Get list of invalid iterable values.
    Note: tensors are iterable, even if single-valued.

    Returns:
        list: invalid iterable list
    """
    return [0.1, 5, True]


@pytest.fixture(scope="session")
def not_clip_val_asst_percentile_settings():
    """
    Get list of invalid clip_val_asst_percentile values.
    Note: needs to be an iterable list + 2 fp values

    Returns:
        list: invalid clip_val_asst_percentile list
    """
    return [1, False, [0.1], [0.1, 0.2, 0.3], [True, 0.3], [0.1, torch.tensor(1.0)]]


@pytest.fixture(scope="session")
def not_ptq_loss_func_settings():
    """
    Get list of invalid ptq_loss_func values

    Returns:
        list: invalid ptq_loss_func list
    """
    return ["mde", "ssin"]


@pytest.fixture(scope="session")
def not_ptq_coslr_settings():
    """
    Get list of invalid ptq_coslr values

    Returns:
        list: invalid ptq_coslr list
    """
    return [" ", "B", "Q", "WS"]


@pytest.fixture(scope="session")
def not_which2patch_contextmanager_settings():
    """
    Get list of invalid which2patch_contextmanager values

    Returns:
        list: invalid which2patch_contextmanager list
    """
    return ["torch.vmm", "torch.natnul", "None"]


################################
# Toy Model Classes + Fixtures #
################################
class ToyModel1(torch.nn.Module):
    """
    Single layer Conv2d model

    Extends:
        torch.nn.Module
    """

    def __init__(self):
        super().__init__()
        self.first_layer = torch.nn.Conv2d(3, 3, 1, 1, bias=False)

    def forward(self, input_tensor):
        """
        Forward func for Toy Model

        Args:
            input_tensor (torch.FloatTensor): Tensor to operate on

        Returns:
            torch.FloatTensor:
        """
        return self.first_layer(input_tensor)


class ToyModel2(torch.nn.Module):
    """
    Single layer Linear model

    Extends:
        torch.nn.Module
    """

    def __init__(self):
        super().__init__()
        self.first_layer = torch.nn.Linear(3, 3, bias=True)

    def forward(self, input_tensor: torch.FloatTensor):
        """
        Forward func for Toy Model

        Args:
            input_tensor (torch.FloatTensor): Tensor to operate on

        Returns:
            torch.FloatTensor:
        """
        return self.first_layer(input_tensor)


class ToyModel3(torch.nn.Module):
    """
    Three layer Linear model

    Extends:
        torch.nn.Module
    """

    def __init__(self):
        super().__init__()
        self.first_layer = torch.nn.Linear(3, 3, bias=True)
        self.second_layer = torch.nn.Linear(3, 3, bias=True)
        self.third_layer = torch.nn.Linear(3, 3, bias=True)

    def forward(self, input_tensor):
        """
        Forward func for Toy Model

        Args:
            input_tensor (torch.FloatTensor): Tensor to operate on

        Returns:
            torch.FloatTensor:
        """
        out = self.first_layer(input_tensor)
        out = self.second_layer(out)
        out = self.third_layer(out)
        return out


class ToyModel4(torch.nn.Module):
    """
    Four layer Linear + Conv2d model

    Extends:
        torch.nn.Module
    """

    def __init__(self):
        super().__init__()
        self.first_layer = torch.nn.Linear(3, 3, bias=True)
        self.relu = torch.nn.ReLU()
        self.second_layer = torch.nn.Linear(3, 3, bias=True)
        self.third_layer = torch.nn.Conv2d(3, 3, 1, 1)
        self.fourth_layer = torch.nn.Linear(3, 3, bias=True)

    def forward(self, input_tensor):
        """
        Forward func for Toy Model

        Args:
            input_tensor (torch.FloatTensor): Tensor to operate on

        Returns:
            torch.FloatTensor:
        """
        out = self.first_layer(input_tensor)
        out = self.relu(out)
        out = self.second_layer(out)
        out = self.third_layer(out)
        out = self.fourth_layer(out)
        return out


model_fp32_params = [
    ToyModel1(),
    ToyModel2(),
    ToyModel3(),
    ToyModel4(),
    torch.nn.Sequential(OrderedDict([("first_layer", torch.nn.Conv2d(3, 3, 1, 1))])),
]
model_fp16_params = [
    ToyModel1().half(),
    ToyModel2().half(),
    ToyModel3().half(),
    ToyModel4().half(),
    torch.nn.Sequential(
        OrderedDict([("first_layer", torch.nn.Conv2d(3, 3, 1, 1))])
    ).half(),
]

# Note: All models require deepcopy, since qmodel_prep will change the object direclty


@pytest.fixture(scope="function", params=model_fp32_params)
def model_fp32(request):
    """
    Toy Model w/ fp32 data.
    Use deepcopy of each model, so any edits do not persist

    Args:
        request (torch.nn.Module): Toy Model

    Returns:
        torch.nn.Module: Toy Model
    """
    return deepcopy(request.param)


@pytest.fixture(scope="function", params=model_fp16_params)
def model_fp16(request):
    """
    Toy Model w/ fp16 data.

    Args:
        request (torch.nn.Module): Toy Model

    Returns:
        torch.nn.Module: Toy Model
    """
    return deepcopy(request.param)


# Get single model of each datatype to test check_config
@pytest.fixture(scope="function")
def model_config_fp32():
    """
    Four layer Linear+Conv2d Toy Model w/ fp32 data

    Args:
        request (torch.nn.Module): Toy Model

    Returns:
        torch.nn.Module: Toy Model
    """
    return deepcopy(ToyModel4())


@pytest.fixture(scope="function")
def model_config_fp16():
    """
    Four layer Linear+Conv2d Toy Model w/ fp16 data

    Args:
        request (torch.nn.Module): Toy Model

    Returns:
        torch.nn.Module: Toy Model
    """
    return deepcopy(ToyModel4().half())


# QLinear class requires Nvidia GPU and cuda
if torch.cuda.is_available():

    class ToyModelQuantized(torch.nn.Module):
        """
        Three layer Linear model that has a quantized layer

        Extends:
            torch.nn.Module
        """

        def __init__(self):
            super().__init__()
            kwargs = {"qcfg": qconfig_init()}  # QLinear requires qconfig to work
            self.first_layer = torch.nn.Linear(3, 3, bias=True)
            self.second_layer = QLinear(3, 3, bias=True, **kwargs)
            self.third_layer = torch.nn.Linear(3, 3, bias=True)

        def forward(self, input_tensor):
            """
            Forward func for Toy Model

            Args:
                input_tensor (torch.FloatTensor): Tensor to operate on

            Returns:
                torch.FloatTensor:
            """
            out = self.first_layer(input_tensor)
            out = self.second_layer(out)
            out = self.third_layer(out)
            return out

    model_quantized_params = [ToyModelQuantized()]

    @pytest.fixture(scope="function", params=model_quantized_params)
    def model_quantized(request):
        """
        Toy Model that has quantized layer

        Args:
            request (torch.nn.Module): Toy Model

        Returns:
            torch.nn.Module: Toy Model
        """
        return deepcopy(request.param)


# Get a model to test layer uniqueness
class ToyModelNoUniqueLayers(torch.nn.Module):
    """
    Three layer Linear model that hold non-unique layers

    Extends:
        torch.nn.Module
    """

    def __init__(self):
        super().__init__()
        layer = torch.nn.Linear(3, 3, bias=True)
        self.first_layer = layer
        self.second_layer = layer
        self.third_layer = layer

    def forward(self, input_tensor):
        """
        Forward func for Toy Model

        Args:
            input_tensor (torch.FloatTensor): Tensor to operate on

        Returns:
            torch.FloatTensor:
        """
        out = self.first_layer(input_tensor)
        out = self.second_layer(out)
        out = self.third_layer(out)
        return out


model_no_unique_layers_params = [ToyModelNoUniqueLayers()]


@pytest.fixture(scope="function", params=model_no_unique_layers_params)
def model_no_unique_layers(request):
    """
    Toy Model w/ no unique layers

    Args:
        request (torch.nn.Module): Toy Model

    Returns:
        torch.nn.Module: Toy Model
    """
    return deepcopy(request.param)


class ToyModelHalf(torch.nn.Module):
    """
    Three layer Linear model that holds fp16 data explicitly

    Extends:
        torch.nn.Module
    """

    def __init__(self):
        super().__init__()
        self.first_layer = torch.nn.Linear(3, 3, bias=True, dtype=torch.float16)
        self.second_layer = torch.nn.Linear(3, 3, bias=True, dtype=torch.float16)
        self.third_layer = torch.nn.Linear(3, 3, bias=True, dtype=torch.float16)

    def forward(self, input_tensor):
        """
        Forward func for Toy Model

        Args:
            input_tensor (torch.FloatTensor): Tensor to operate on

        Returns:
            torch.FloatTensor:
        """
        out = self.first_layer(input_tensor)
        out = self.second_layer(out)
        out = self.third_layer(out)
        return out


model_half_params = [ToyModelHalf()]


@pytest.fixture(scope="function", params=model_half_params)
def model_half(request):
    """
    Toy Model w/ explicit fp16 data.

    Args:
        request (torch.nn.Module): Toy Model

    Returns:
        torch.nn.Module: Toy Model
    """
    return deepcopy(request.param)


###################################
# Model Inputs Tensors + Fixtures #
###################################
sample_input_fp32_params = [
    torch.randn(1, 3, 3, 3),
    torch.zeros(1, 3, 3, 3),
    torch.ones(1, 3, 3, 3),
]
sample_input_fp16_params = [
    torch.randn(1, 3, 3, 3).half(),
    torch.zeros(1, 3, 3, 3).half(),
    torch.ones(1, 3, 3, 3).half(),
]


@pytest.fixture(scope="session", params=sample_input_fp32_params)
def sample_input_fp32(request):
    """
    Sample input fp32 data

    Args:
        request (torch.FloatTensor): fp32 data

    Returns:
        torch.FloatTensor: fp32 data
    """
    return request.param


@pytest.fixture(scope="session", params=sample_input_fp16_params)
def sample_input_fp16(request):
    """
    Sample input fp16 data

    Args:
        request (torch.FloatTensor): fp16 data

    Returns:
        torch.FloatTensor: fp16 data
    """
    return request.param


# fp32 models do not use fp16
num_bits_activation_fp32_params = [2, 4, 8, 32]
num_bits_weight_fp32_params = [2, 4, 8, 32]

# fp16 models can't have fp32
num_bits_activation_fp16_params = [2, 4, 8, 16]
num_bits_weight_fp16_params = [2, 4, 8, 16]


@pytest.fixture(scope="session", params=num_bits_activation_fp32_params)
def num_bits_activation_fp32(request):
    """
    Valid nbits_a for fp32

    Args:
        request (int): nbits_a

    Returns:
        int: nbits_a
    """
    return request.param


@pytest.fixture(scope="session", params=num_bits_weight_fp32_params)
def num_bits_weight_fp32(request):
    """
    Valid nbits_w for fp32

    Args:
        request (int): nbits_w

    Returns:
        int: nbits_w
    """
    return request.param


@pytest.fixture(scope="session", params=num_bits_activation_fp16_params)
def num_bits_activation_fp16(request):
    """
    Valid nbits_a for fp16

    Args:
        request (int): nbits_a

    Returns:
        int: nbits_a
    """
    return request.param


@pytest.fixture(scope="session", params=num_bits_weight_fp16_params)
def num_bits_weight_fp16(request):
    """
    Valid nbits_w for fp16

    Args:
        request (int): nbits_w

    Returns:
        int: nbits_w
    """
    return request.param


###################
# Config Fixtures #
###################

# Note: All configs require deepcopy, as we will be modifying them for various tests

default_config_params = [qconfig_init()]


@pytest.fixture(scope="function", params=default_config_params)
def config_fp32(request):
    """
    Create fp32 qconfig

    Args:
        request (dict): qconfig_init

    Returns:
        dict: qconfig_init
    """
    qconfig = request.param
    return deepcopy(qconfig)


@pytest.fixture(scope="function", params=default_config_params)
def config_fp16(request):
    """
    Create fp16 qconfig

    Args:
        request (dict): qconfig_init w/ fp16 settings

    Returns:
        dict: qconfig_init w/ fp16 settings
    """
    qconfig = deepcopy(request.param)
    qconfig["nbits_a"] = 16
    qconfig["nbits_w"] = 16
    return qconfig


# Create QAT/PTQ int8 config fixture.
config_params = ["qat_int8", "ptq_int8"]


@pytest.fixture(scope="function", params=config_params)
def config_int8(request):
    """
    Create QAT/PTQ int8 recipe qconfig

    Args:
        request (str): qconfig recipe

    Returns:
        dict: qconfig_init
    """
    recipe = request.param
    cfg = qconfig_init(recipe=recipe)
    cfg["qmodel_calibration"] = 0
    return deepcopy(cfg)


qa_mode_params = [
    "pact",
    "pact+",
    "pactsym",
    "pactsym+",
    "max",
    "minmax",
    "maxsym",
    "lsq+",
]

qw_mode_params = [
    "sawb",
    "sawb16",
    # "sawbperCh",
    "sawb+",
    "sawb+16",
    # "sawb+perCh",
    "max",
    # "maxperCh",
    # "maxperGp",
    "minmax",
    # "minmaxperCh",
    # "minmaxperGp",
    "pact",
    "pact+",
    "lsq+",
]


@pytest.fixture(scope="session", params=qa_mode_params)
def qa_mode(request):
    """
    Fixture for qa_modes

    Args:
        request (str): qa_mode

    Returns:
        str: qa_mode
    """
    return request.param


@pytest.fixture(scope="session", params=qw_mode_params)
def qw_mode(request):
    """
    Fixture for qw_modes

    Args:
        request (str): qw_mode

    Returns:
        str: qw_mode
    """
    return request.param


########################
# Save Config Fixtures #
########################


# Dummy class
class BadClass:
    """
    Dummy class for object detection in qconfig_save
    """

    def __init__(self):
        self.name = "bad"

    def __str__(self):
        return f"BadClass(name={self.name})"


# Create embedded objects to test recursive detection
bad_pair_params = [
    ("bad1", torch.tensor(1.0)),
    ("bad2", {1, 2, 3, 4}),  # sets are not serializable
    ("bad3", [[BadClass()], True]),
    ("bad4", {"W": [torch.tensor(42)], "X": "y", "Z": None}),
    ("bad5", ([{"boom", BadClass()}])),
    (torch.tensor(True), "yikes"),
]


@pytest.fixture(scope="function", params=bad_pair_params)
def bad_pair(request):
    """
    Create bad key,value pair for qconfig_save

    Args:
        request (tuple): bad key,value pair

    Returns:
        tuple: bad key,value pair
    """
    return request.param


wanted_pair_params = [
    ("nbits_a", 32),
    ("qw_mode", "sawb"),
    ("extend_act_range", False),
    ("qspecial_layers", {}),
]


@pytest.fixture(scope="function", params=wanted_pair_params)
def wanted_pair(request):
    """
    Create wanted key,value pair for qconfig_save

    Args:
        request (tuple): wanted key,value pair

    Returns:
        tuple: wanted key,value pair
    """
    return request.param


# Checks for loading raw config
required_pair_params = [
    ("sweep_cv_percentile", False),
    ("tb_writer", None),
    (
        "mapping",
        {
            torch.nn.Conv2d: {
                "from": torch.nn.Conv2d,
                "to": QConv2d,
                "otherwise": QConv2d,
            },
            torch.nn.ConvTranspose2d: {
                "from": torch.nn.ConvTranspose2d,
                "to": QConvTranspose2d,
                "otherwise": QConvTranspose2d,
            },
            torch.nn.Linear: {
                "from": torch.nn.Linear,
                "to": QLinear,
                "otherwise": QLinear,
            },
            torch.nn.LSTM: {"from": torch.nn.LSTM, "to": QLSTM, "otherwise": QLSTM},
        },
    ),
    ("checkQerr_frequency", False),
    ("newlySwappedModules", []),
    ("force_calib_once", False),
    # if we keep the follwing LUTs, it will save the entire model
    ("LUTmodule_name", {}),
]


@pytest.fixture(scope="function", params=required_pair_params)
def required_pair(request):
    """
    Create required key,value pair for qconfig_save

    Args:
        request (tuple): required key,value pair

    Returns:
        tuple: required key,value pair
    """
    return request.param


#########################
# Vision Model Fixtures #
#########################

# Create img
# downloaded from torchvision github (vision/test/assets/encoder_jpeg/ directory)
img = read_image(
    os.path.realpath(
        os.path.join(os.path.dirname(__file__), "grace_hopper_517x606.jpg")
    )
)


# Create resnet/vit batch fixtures from weights
def prepocess_img(image, weights):
    """
    Preprocess an image w/ a weights.transform()

    Args:
        img (torch.FloatTensor): Image data
        weights (torchvision.models): Weight object

    Returns:
        torch.FloatTensor: Preprocessed image
    """
    preprocess = weights.transforms()
    batch = preprocess(image).unsqueeze(0)
    return batch


@pytest.fixture(scope="session")
def batch_resnet():
    """
    Preprocess an image w/ Resnet weights.transform()

    Returns:
        torch.FloatTensor: Preprocessed image
    """
    return prepocess_img(img, ResNet50_Weights.IMAGENET1K_V2)


@pytest.fixture(scope="session")
def batch_vit():
    """
    Preprocess an image w/ ViT weights.transform()

    Returns:
        torch.FloatTensor: Preprocessed image
    """
    return prepocess_img(img, ViT_B_16_Weights.IMAGENET1K_V1)


# Create resnet/vit model fixtures from weights
@pytest.fixture(scope="function")
def model_resnet():
    """
    Create Resnet50 model + weights

    Returns:
        torchvision.models.resnet.ResNet: Resnet50 model
    """
    return resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)


@pytest.fixture(scope="function")
def model_vit():
    """
    Create ViT model + weights

    Returns:
        torchvision.models.vision_transformer.VisionTransformer: ViT model
    """
    return vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)


#######################
# BERT Model Fixtures #
#######################


@pytest.fixture(scope="session")
def input_bert():
    """
    Create a BERT input

    Returns:
        torch.FloatTensor: BERT sample input
    """
    text = "Replace me by any text you'd like."
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    return tokenizer(text, return_tensors="pt")


@pytest.fixture(scope="function")
def model_bert():
    """
    Get a BERT model

    Returns:
        transformers.models.bert.modeling_bert.BertModel: BERT model
    """
    return BertModel.from_pretrained("bert-base-uncased")
