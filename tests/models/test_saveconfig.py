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
Test qconfig_save functionality
"""

# Third Party
import pytest

# First Party
from fms_mo.utils.qconfig_utils import qconfig_load, qconfig_save
from tests.models.test_model_utils import (delete_config, load_json,
                                           save_serialized_json)

#########
# Tests #
#########


def test_save_config_warn_bad_pair(
    config_fp32: dict,
    bad_pair: tuple,
):
    """
    When given a bad key,val pair, it should generate a UserWarning and recover

    Args:
        config_fp32 (dict): Config for fp32 quantization
        bad_pair (tuple): A pair that can't be serialized for qconfig_save
    """
    key, val = bad_pair
    delete_config()

    # Add bad key,val pair and save ; should generate UserWarning(s) for removing bad pair
    config_fp32[key] = val
    with pytest.warns(UserWarning):
        qconfig_save(config_fp32)

    # Load saved config and assert the key is not saved
    loaded_config = load_json("qcfg.json")  # load json as is - do not modify
    assert key not in loaded_config

    delete_config()


def test_save_config_wanted_pairs(
    config_fp32: dict,
    wanted_pair: tuple,
):
    """
    When saving a config without wanted pairs, they should be re-initialized to default vals

    Args:
        config_fp32 (dict): Config for fp32 quantization
        wanted_pair (tuple): A pair that needs to be re-init if not present in qconfig_save
    """
    key, default_val = wanted_pair
    delete_config()

    # Delete wanted pair from config and save ; should be reset to default
    if key in config_fp32:
        del config_fp32[key]
    qconfig_save(config_fp32)

    # Load saved config and check the wanted pair was reset to default
    loaded_config = load_json()
    assert loaded_config.get(key) == default_val

    delete_config()


def test_load_config_restored_pair(
    config_fp32: dict,
    wanted_pair: tuple,
):
    """
    Loading from unsaved config should restore to wanted pairs default value

    Args:
        config_fp32 (dict): Config for fp32 quantization
        wanted_pair (tuple): A pair that needs to be re-init if not present in qconfig_load
    """
    key, default_val = wanted_pair
    delete_config()

    if key in config_fp32:
        del config_fp32[key]
    save_serialized_json(
        config_fp32
    )  # Save config as is, no other edits other than to serialize

    loaded_config = qconfig_load("qcfg.json")
    assert loaded_config.get(key) == default_val


def test_load_config_required_pair(
    config_fp32: dict,
    required_pair: tuple,
):
    """
    Loading from unsaved config should restore to required pairs default value

    Args:
        config_fp32 (dict): Config for fp32 quantization
        required_pair (tuple): A pair that needs to be re-init if not present in qconfig_load
    """
    key, default_val = required_pair
    delete_config()

    if key in config_fp32:
        del config_fp32[key]

    # Save config with minimal removals to dump as json
    save_serialized_json(config_fp32)

    loaded_config = qconfig_load("qcfg.json")
    assert loaded_config.get(key) == default_val
