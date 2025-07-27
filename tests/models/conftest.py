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
from collections import OrderedDict
from copy import deepcopy
import os

# Third Party
from PIL import Image  # pylint: disable=import-error
from torch.utils.data import DataLoader, TensorDataset
from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
    BertConfig,
    BertModel,
    BertTokenizer,
    GraniteConfig,
    GraniteModel,
    LlamaConfig,
    LlamaModel,
)
import numpy as np
import pytest
import torch
import torch.nn.functional as F

# Local
# fms_mo imports
from fms_mo import qconfig_init
from fms_mo.modules import QLSTM, QBmm, QConv2d, QConvTranspose2d, QLinear
from fms_mo.utils.import_utils import available_packages
from fms_mo.utils.qconfig_utils import get_mx_specs_defaults, set_mx_specs

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


@pytest.fixture(scope="session")
def bad_mx_specs_settings():
    """
    Get list of invalid mx_spec key,value pairs

    Returns:
        list: invalid mx_spec list
    """
    return [
        ("w_elem_format", "fp8_e5m3"),
        ("a_elem_format", "fp8_m4e3"),
        ("scale_bits", False),
        ("block_size", "32"),
        ("bfloat", [16]),
        ("round", "bankers"),
        ("custom_cuda", "yes"),
    ]


@pytest.fixture(scope="session")
def bad_mx_config_settings():
    """
    Get list of invalid mx config key,value pairs for config and mx_specs

    Returns:
        list: invalid mx_spec list
    """
    return [
        ("qw_mode", "w_elem_format", "mx_fp8_e5m3", "fp8_e5m3"),
        ("qa_mode", "a_elem_format", "mx_fp8_m4e3", "fp8_m4e3"),
        ("mx_scale_bits", "scale_bits", False, False),
        ("mx_block_size", "block_size", "32", "32"),
        ("mx_bfloat", "bfloat", {16}, {16}),
        ("mx_round", "round", "bankers", "bankers"),
        ("mx_custom_cuda", "custom_cuda", "yes", "yes"),
    ]


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
    torch.randn(1, 3, 3, 3) * 0.001,
    torch.ones(1, 3, 3, 3),
]
sample_input_fp16_params = [
    torch.randn(1, 3, 3, 3).half(),
    (torch.randn(1, 3, 3, 3) * 0.001).half(),
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
mx_config_params = [qconfig_init(use_mx=True)]


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
def config_fp32_mx(request):
    """
    Create fp32 qconfig w/ mx_specs vars set in qconfig.

    Args:
        request (dict): qconfig_init

    Returns:
        dict: qconfig_init
    """
    qconfig = deepcopy(request.param)
    mx_specs = get_mx_specs_defaults()

    # Set config vars prefixed w/ "mx_"
    for key, val in mx_specs.items():
        qconfig["mx_" + key] = val

    # Only 1 variable that has "mx_" prefix from MX lib
    qconfig["mx_flush_fp32_subnorms"] = qconfig["mx_mx_flush_fp32_subnorms"]
    del qconfig["mx_mx_flush_fp32_subnorms"]

    # Move x_elem_format to q_modes and delete mx_x_elem_format
    # Needs prefix settings to avoid collision w/ fms_mo modes
    qconfig["qa_mode"] = "mx_" + qconfig["mx_a_elem_format"]
    qconfig["qw_mode"] = "mx_" + qconfig["mx_w_elem_format"]
    del qconfig["mx_a_elem_format"]
    del qconfig["mx_w_elem_format"]

    return qconfig


@pytest.fixture(scope="function", params=mx_config_params)
def config_fp32_mx_specs(request):
    """
    Create fp32 qconfig w/ mx_specs.


    Args:
        request (dict): qconfig_init

    Returns:
        dict: qconfig_init
    """
    qconfig = deepcopy(request.param)
    qconfig["mx_specs"] = get_mx_specs_defaults()

    # Set mx_specs as if we ran qconfig_init
    set_mx_specs(qconfig)

    return qconfig


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


keys_to_save_params = [
    ["qa_mode", "qw_mode", "nbits_a", "nbits_w", "qskip_layer_name"],
]


@pytest.fixture(scope="session", params=keys_to_save_params)
def save_list(request):
    """
    Generate a save list for testing user-requested save config.

    Args:
        request (list): List of variables to save in a quantized config.

    Returns:
        list: List of variables to save in a quantized config.
    """
    return request.param


wrong_recipe_name_params = ["qat_int7", "pzq_int8"]


@pytest.fixture(scope="session", params=wrong_recipe_name_params)
def wrong_recipe_name(request):
    """
    Get a bad recipe json file name in fms_mo/recipes

    Args:
        request (str): Bad recipe name in fms_mo/recipes

    Returns:
        str: Bad recipe name
    """
    return request.param


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
    ("qw_mode", "sawb+"),
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
            torch.nn.Conv2d: QConv2d,
            torch.nn.ConvTranspose2d: QConvTranspose2d,
            torch.nn.Linear: QLinear,
            torch.nn.LSTM: QLSTM,
            "matmul_or_bmm": QBmm,
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


if available_packages["torchvision"]:
    # Third Party
    # pylint: disable = import-error
    from torchvision.io import read_image
    from torchvision.models import (
        ResNet50_Weights,
        ViT_B_16_Weights,
        resnet50,
        vit_b_16,
    )

    # Create img
    # downloaded from torchvision github (vision/test/assets/encoder_jpeg/ directory)
    img_tv = read_image(
        os.path.realpath(
            os.path.join(os.path.dirname(__file__), "grace_hopper_517x606.jpg")
        )
    )

    # Create resnet/vitbatch fixtures from weights
    def prepocess_img(image, weights):
        """
        Preprocess an image w/ a weights.transform()

        Args:
            img_tv (torch.FloatTensor): Image data
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
        return prepocess_img(img_tv, ResNet50_Weights.IMAGENET1K_V2)

    @pytest.fixture(scope="session")
    def batch_vit():
        """
        Preprocess an image w/ ViT weights.transform()

        Returns:
            torch.FloatTensor: Preprocessed image
        """
        return prepocess_img(img_tv, ViT_B_16_Weights.IMAGENET1K_V1)

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


img = Image.open(
    os.path.realpath(
        os.path.join(os.path.dirname(__file__), "grace_hopper_517x606.jpg")
    )
).convert("RGB")


def process_img(
    pretrained_model: str,
    input_img: Image.Image,
):
    """
    Process an image w/ AutoImageProcessor

    Args:
        processor (AutoImageProcessor): Processor weights for pretrained model
        pretrained_model (str): Weight object
        input_img (Image.Image): Image data

    Returns:
        torch.FloatTensor: Processed image
    """
    img_processor = AutoImageProcessor.from_pretrained(pretrained_model, use_fast=True)
    batch_dict = img_processor(images=input_img, return_tensors="pt")
    return batch_dict["pixel_values"]


@pytest.fixture(scope="function")
def batch_resnet18():
    """
    Preprocess an image w/ ms resnet18 processor

    Returns:
        torch.FloatTensor: Preprocessed image
    """
    return process_img("microsoft/resnet-18", img)


@pytest.fixture(scope="function")
def model_resnet18():
    """
    Create MS ResNet18 model + weights

    Returns:
        AutoModelForImageClassification: Resnet18 model
    """
    return AutoModelForImageClassification.from_pretrained("microsoft/resnet-18")


@pytest.fixture(scope="function")
def batch_vit_base():
    """
    Preprocess an image w/ Google ViT-base processor

    Returns:
        torch.FloatTensor: Preprocessed image
    """
    return process_img("google/vit-base-patch16-224", img)


@pytest.fixture(scope="function")
def model_vit_base():
    """
    Create Google ViT-base model + weights

    Returns:
        AutoModelForImageClassification: Google ViT-base model
    """
    return AutoModelForImageClassification.from_pretrained(
        "google/vit-base-patch16-224"
    )


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
    tokenizer = BertTokenizer.from_pretrained("google-bert/bert-base-uncased")
    return tokenizer(text, return_tensors="pt")


@pytest.fixture(scope="function")
def model_bert():
    """
    Get a BERT model

    Returns:
        transformers.models.bert.modeling_bert.BertModel: BERT model
    """
    return BertModel.from_pretrained("google-bert/bert-base-uncased", torchscript=True)


@pytest.fixture(scope="function")
def model_bert_eager():
    """
    Get a BERT model

    Returns:
        transformers.models.bert.modeling_bert.BertModel: BERT model
    """
    return BertModel.from_pretrained(
        "google-bert/bert-base-uncased", torchscript=True, attn_implementation="eager"
    )


# MX reference class for quantization
if torch.cuda.is_available():

    class ResidualMLP(torch.nn.Module):
        """
        Test Linear model for MX library
        """

        def __init__(self, hidden_size, device="cuda"):
            super().__init__()

            self.layernorm = torch.nn.LayerNorm(hidden_size, device=device)
            self.dense_4h = torch.nn.Linear(hidden_size, 4 * hidden_size, device=device)
            self.dense_h = torch.nn.Linear(4 * hidden_size, hidden_size, device=device)
            self.dummy = torch.nn.Linear(hidden_size, hidden_size, device=device)
            # add a dummy layer because by default we skip 1st/last,
            # if there are only 2 layers, all will be skipped

        def forward(self, inputs):
            """
            Forward function for Residual MLP

            Args:
                inputs (torch.tensor): Input tensor

            Returns:
                torch.tensor: Output tensor
            """
            norm_outputs = self.layernorm(inputs)

            # MLP
            proj_outputs = self.dense_4h(norm_outputs)
            # pylint: disable=not-callable
            proj_outputs = F.gelu(proj_outputs)
            mlp_outputs = self.dense_h(proj_outputs)
            mlp_outputs = self.dummy(mlp_outputs)

            # Residual Connection
            outputs = inputs + mlp_outputs

            return outputs


mx_format_params = ["int8", "int4", "fp8_e4m3", "fp8_e5m2", "fp4_e2m1"]


@pytest.fixture(scope="session", params=mx_format_params)
def mx_format(request):
    """
    Get a MX element format to test

    Returns:
        str: MX element format name
    """
    return request.param


@pytest.fixture(scope="function")
def input_residualMLP():
    """
    Get a random input for a residual MLP model

    Returns:
        torch.FloatTensor: Random 16x128 tensor
    """
    x = np.random.randn(16, 128)
    return torch.tensor(x, dtype=torch.float32, device="cuda")


@pytest.fixture(scope="function")
def model_residualMLP():
    """
    Get a ResidualMLP model

    Returns:
        torch.nn.Module: _description_
    """
    return ResidualMLP(128)


#########################
# Tiny Model Fake Input #
#########################

# Changing vocab_size and max_position_embeddings impacts all tiny models as well!
vocab_size = 512
max_position_embeddings = 512
batch_size = 2
size = (batch_size, max_position_embeddings)


@pytest.fixture(scope="function")
def input_tiny() -> DataLoader:
    """
    Create a fake input for tiny models w/ fixed vocab_size and max_position_embeddings

    Returns:
        DataLoader: Fake Encoding for a Tokenizer
    """
    # Random tokens and attention mask == 1
    random_tokens = torch.randint(low=0, high=vocab_size, size=size)
    attention_mask = torch.ones(size)

    dataset = TensorDataset(random_tokens, attention_mask)
    # qmodel_prep expects dataloader batch=tuple(tensor, tensor)
    # Without collate_fn, it returns batch=list(tensor,tensor)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda batch: tuple(torch.stack(samples) for samples in zip(*batch)),
    )


#############################
# Tiny BERT Model Fixtures #
#############################


tiny_bert_config_params = [
    BertConfig(
        vocab_size=vocab_size,  # 30522
        hidden_size=128,  # 768
        num_hidden_layers=2,  # 12
        num_attention_heads=2,  # 12
        intermediate_size=512,  # 3072
        max_position_embeddings=max_position_embeddings,  # 512
        type_vocab_size=1,  # 2
    ),
]


@pytest.fixture(scope="function", params=tiny_bert_config_params)
def model_tiny_bert(request) -> BertModel:
    """
    Get a tiny Llama Model based on the config

    Args:
        config_tiny_bert (BertConfig): Trimmed Tiny Bert config

    Returns:
        BertConfig: Tiny Bert model
    """
    model = BertModel(config=request.param)
    return model


qcfg_tiny_bert_update_params = [
    {
        "nbits_a": 8,
        "nbits_w": 8,
        "qa_mode": "pertokenmax",
        "qw_mode": "max",
        "qmodel_calibration": 1,
        "smoothq": False,
        "smoothq_scale_layers": [],
        "qskip_layer_name": [
            "embeddings.position_embeddings",
            "embeddings.word_embeddings",
            "embeddings.token_type_embeddings",
            "pooler.dense",
        ],
        "qskip_large_mag_layers": False,
        "recompute_narrow_weights": True,
    },
    {
        "nbits_a": 8,
        "nbits_w": 8,
        "qa_mode": "maxsym",
        "qw_mode": "maxperCh",
        "qmodel_calibration": 1,
        "smoothq": False,
        "smoothq_scale_layers": [],
        "qskip_layer_name": [
            "embeddings.position_embeddings",
            "embeddings.word_embeddings",
            "embeddings.token_type_embeddings",
            "pooler.dense",
        ],
        "qskip_large_mag_layers": False,
        "recompute_narrow_weights": False,
    },
]


@pytest.fixture(scope="function", params=qcfg_tiny_bert_update_params)
def qcfg_bert(request) -> dict:
    """
    Quantization config for Tiny Bert

    Args:
        request (dict): Quantization config

    Returns:
        dict: Quantization config
    """
    qcfg = qconfig_init()

    qcfg.update(request.param)

    return qcfg


@pytest.fixture(scope="function")
def bert_linear_names() -> list:
    """
    Get Bert linear layers names in state dict

    Returns:
        list: Bert linear layer names
    """
    return [
        "attention.self.query",
        "attention.self.key",
        "attention.self.value",
        "attention.output.dense",
        "intermediate.dense",
        "output.dense",
    ]


#############################
# Tiny Llama Model Fixtures #
#############################

tiny_llama_config_params = [
    LlamaConfig(
        vocab_size=vocab_size,  # 32000
        hidden_size=128,  # 4096
        intermediate_size=256,  # 11008
        num_hidden_layers=2,  # 32
        num_attention_heads=2,  # 32
        max_position_embeddings=max_position_embeddings,  # 2048
    ),
]


@pytest.fixture(scope="function", params=tiny_llama_config_params)
def model_tiny_llama(request) -> LlamaModel:
    """
    Get a tiny Llama Model based on the config

    Args:
        config_tiny_llama (LlamaConfig): Trimmed Tiny Llama config

    Returns:
        LlamaModel: Tiny Llama model
    """
    model = LlamaModel(config=request.param)
    return model


qcfg_tiny_llama_update_params = [
    {
        "nbits_a": 8,
        "nbits_w": 8,
        "qa_mode": "pertokenmax",
        "qw_mode": "max",
        "qmodel_calibration": 1,
        "smoothq": False,
        "smoothq_scale_layers": [],
        "qskip_layer_name": [
            "embeddings.position_embeddings",
            "embeddings.word_embeddings",
            "embeddings.token_type_embeddings",
            "pooler.dense",
        ],
        "qskip_large_mag_layers": False,
        "recompute_narrow_weights": True,
    },
]


@pytest.fixture(scope="function", params=qcfg_tiny_llama_update_params)
def qcfg_llama(request) -> dict:
    """
    Quantization config for Tiny Llama

    Args:
        request (dict): Quantization config

    Returns:
        dict: Quantization config
    """
    qcfg = qconfig_init()

    qcfg.update(request.param)

    return qcfg


@pytest.fixture(scope="function")
def llama_linear_names() -> list:
    """
    Get Llama linear layers names in state dict

    Returns:
        list: Llama linear layer names
    """
    return [
        "self_attn.q_proj",
        "self_attn.k_proj",
        "self_attn.v_proj",
        "self_attn.o_proj",
        "mlp.gate_proj",
        "mlp.up_proj",
        "mlp.down_proj",
    ]


###############################
# Tiny Granite Model Fixtures #
###############################

tiny_granite_config_params = [
    GraniteConfig(
        vocab_size=vocab_size,  # 32000
        hidden_size=128,  # 4096
        intermediate_size=256,  # 11008
        num_hidden_layers=2,  # 32
        num_attention_heads=2,  # 32
        max_position_embeddings=max_position_embeddings,  # 2048
    ),
]


@pytest.fixture(scope="function", params=tiny_granite_config_params)
def model_tiny_granite(request) -> GraniteModel:
    """
    Get a tiny Granite Model based on the config

    Args:
        config_tiny_granite (GraniteConfig): Trimmed Tiny Granite config

    Returns:
        GraniteModel: Tiny Granite model
    """
    model = GraniteModel(config=request.param)
    return model


qcfg_tiny_granite_update_params = [
    {
        "nbits_a": 8,
        "nbits_w": 8,
        "qa_mode": "pertokenmax",
        "qw_mode": "maxperCh",
        "qmodel_calibration": 1,
        "smoothq": False,
        "smoothq_scale_layers": ["k_proj", "v_proj", "gate_proj", "up_proj"],
        "qskip_layer_name": ["lm_head"],
        "qskip_large_mag_layers": False,
        "recompute_narrow_weights": False,
    },
]


@pytest.fixture(scope="function", params=qcfg_tiny_granite_update_params)
def qcfg_granite(request) -> dict:
    """
    Quantization config for Tiny Granite

    Args:
        request (dict): Quantization config

    Returns:
        dict: Quantization config
    """
    qcfg = qconfig_init()

    qcfg.update(request.param)

    return qcfg


@pytest.fixture(scope="function")
def granite_linear_names() -> list:
    """
    Get Granite linear layers names in state dict

    Returns:
        list: Granite linear layer names
    """
    return [
        "self_attn.q_proj",
        "self_attn.k_proj",
        "self_attn.v_proj",
        "self_attn.o_proj",
        "mlp.gate_proj",
        "mlp.up_proj",
        "mlp.down_proj",
    ]
