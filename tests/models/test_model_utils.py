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
Utility functions for testing Models
"""

# Standard
import json
import logging
import os

# Third Party
from torch.nn import Conv2d, Linear
import torch

# Local
from fms_mo.modules.bmm import QBmm
from fms_mo.modules.conv import DetQConv2d, QConv2d, QConv2dPTQ, QConv2dPTQv2
from fms_mo.modules.linear import QLinear
from fms_mo.utils.qconfig_utils import serialize_config

logger = logging.getLogger(__name__)

####################
# Helper Functions #
####################

qconv2d_nodes = (QConv2d, QConv2dPTQ, QConv2dPTQv2, DetQConv2d)
qlinear_nodes = QLinear

quantized_nodes = (QConv2d, QConv2dPTQ, QConv2dPTQv2, QLinear, DetQConv2d)


def is_qconv2d(node: torch.nn.Module):
    """
    Tests if a node is a known Qconv2d node

    Args:
        node (torch.nn.Module): Node to test

    Returns:
        bool: If node is in qconv2d_nodes
    """
    return isinstance(node, qconv2d_nodes)


def is_qlinear(node: torch.nn.Module):
    """
    Tests if a node is a known QLinear node

    Args:
        node (torch.nn.Module): Node to test

    Returns:
        bool: If node is in qlinear_nodes
    """
    return isinstance(node, qlinear_nodes)


def is_quantized_layer(node: torch.nn.Module):
    """
    Tests if a node is a Quantized node

    Args:
        node (torch.nn.Module): Node to test

    Returns:
        bool: If node is in quantized_nodes
    """
    return isinstance(node, quantized_nodes)


#########################
# qmodel_prep functions #
#########################


def count_qmodules(model: torch.nn.Module):
    """
    Function that counts quantized/non-quantized modules in model
    Used to assert the qmodules are being traced and classified properly

    Args:
        model (torch.nn.Module): Model to search

    Returns:
        [list, list]: List of Torch and FMS qmodules found in model
    """
    torch_modules, fms_qmodules = [], []
    for n, m in model.named_modules():
        if isinstance(m, (QConv2d, QLinear, QBmm)):
            fms_qmodules.append((n, m))
        elif isinstance(m, (Conv2d, Linear)):
            torch_modules.append((n, m))
    return torch_modules, fms_qmodules


def qmodule_error(
    model: torch.nn.Module,
    num_torch_modules: int,
    num_fms_qmodules: int,
):
    """
    Checks number of torch.nn modules, fms qmodules, and any unknown modules are present
    after qmodel_prep.

    Args:
        model (torch.nn.Module): Model to search
        num_torch_modules (int): Number of expected torch modules
        num_fms_qmodules (int): Number of expected FMS qmodules

    Raises:
        e_model: Unexpected number of modules in model
    """
    torch_modules, fms_qmodules = count_qmodules(model)

    try:
        # Check lengths of all categories against expected values
        assert len(torch_modules) == num_torch_modules
        assert len(fms_qmodules) == num_fms_qmodules

    except AssertionError as e_model:
        logger.error(f"model = \n{model}")
        logger.error(f"torch_modules = \n{torch_modules}")
        logger.error(f"Expected num_torch_modules = {num_torch_modules}")
        logger.error(f"fms_qmodules = \n{fms_qmodules}")
        logger.error(f"Expected num_fms_qmodules = {num_fms_qmodules}")
        raise e_model


###############################
# General save/load functions #
###############################


def delete_config(file_path: str = "qcfg.json"):
    """
    Delete a qconfig at the file path provided

    Args:
        file_path (str, optional): Qconfig file to delete. Defaults to "qcfg.json".
    """
    if os.path.exists(file_path):
        os.remove(file_path)


def load_json(file_path: str = "qcfg.json"):
    """
    Load a qconfig at the file path provided

    Args:
        file_path (str, optional): Qconfig file to load. Defaults to "qcfg.json".

    Returns:
        dict: Qconfig dict
    """
    json_file = None
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as openfile:
            json_file = json.load(openfile)
    assert json_file is not None, f"JSON at {file_path} was not found"
    return json_file


def save_serialized_json(config: dict, file_path: str = "qcfg.json"):
    """
    Save qconfig by serializing it first

    Args:
        config (dict): Qconfig dict to save
        file_path (str, optional): File path to save to. Defaults to "qcfg.json".
    """
    # Remove warnings from serialize config
    keys = ["mapping", "logger"]
    for key in keys:
        if key in config:
            del config[key]

    serialize_config(config)  # Only remove stuff necessary to dump
    with open(file_path, "w", encoding="utf-8") as outfile:
        json.dump(config, outfile, indent=4)
