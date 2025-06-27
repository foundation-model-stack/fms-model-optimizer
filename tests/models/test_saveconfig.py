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

# Local
from fms_mo import qconfig_init
from fms_mo.utils.qconfig_utils import qconfig_load, qconfig_save
from tests.models.test_model_utils import (
    delete_file,
    load_json,
    save_json,
    save_serialized_json,
)


@pytest.fixture(autouse=True)
def delete_files():
    """
    Delete any known files lingering before starting test
    """
    delete_file("qcfg.json")
    delete_file("keys_to_save.json")


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

    # Add bad key,val pair and save ; should generate UserWarning(s) for removing bad pair
    config_fp32[key] = val
    with pytest.warns(UserWarning):
        qconfig_save(config_fp32, minimal=False)

    # Load saved config and assert the key is not saved
    loaded_config = load_json("qcfg.json")  # load json as is - do not modify
    assert key not in loaded_config


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

    # Delete wanted pair from config and save ; should be reset to default
    if key in config_fp32:
        del config_fp32[key]
    qconfig_save(config_fp32, minimal=False)

    # Load saved config and check the wanted pair was reset to default
    loaded_config = load_json()
    assert loaded_config.get(key) == default_val


def test_save_config_with_qcfg_save(
    config_fp32: dict,
    save_list: list,
):
    """
    Test for checking that the "save_list" functionality works from within a quantized config

    Args:
        config_fp32 (dict): Config for fp32 quantization
        save_list (list): List of variables to save in a quantized config.
    """
    config_fp32["keys_to_save"] = save_list

    qconfig_save(config_fp32, minimal=False)

    loaded_config = load_json()

    # Remove pkg_versions and date before processing
    del loaded_config["pkg_versions"]
    del loaded_config["date"]

    assert len(loaded_config) == len(save_list)

    # Now ensure that every value in save_list was properly saved
    for key in save_list:
        assert key in loaded_config
        assert loaded_config.get(key) == config_fp32.get(key)

    del config_fp32["keys_to_save"]


def test_save_config_with_recipe_save(
    config_fp32: dict,
    save_list: list,
):
    """
    Test for checking that the "save_list" functionality works from a saved json file

    Args:
        config_fp32 (dict): Config for fp32 quantization
        save_list (list): List of variables to save in a quantized config.
    """
    # Save new "save.json"
    save_path = "keys_to_save.json"
    save_json(save_list, file_path=save_path)

    qconfig_save(config_fp32, recipe="keys_to_save")

    # Check that saved qcfg matches
    loaded_config = load_json()

    # Remove pkg_versions and date before processing
    del loaded_config["pkg_versions"]
    del loaded_config["date"]

    assert len(loaded_config) == len(save_list)

    # Now ensure that every value in save_list was properly saved
    for key in save_list:
        assert key in loaded_config
        assert loaded_config.get(key) == config_fp32.get(key)


def test_save_config_minimal(
    config_fp32: dict,
):
    """
    Test for checking that the minimal functionality works for saving a quantized config.

    Args:
        config_fp32 (dict): Config for fp32 quantization
    """
    qconfig_save(config_fp32, minimal=True)

    # Check that saved qcfg matches
    loaded_config = load_json()

    # Remove pkg_versions and date before processing
    del loaded_config["pkg_versions"]
    del loaded_config["date"]

    # No items should exist - default config should be completely removed
    assert len(loaded_config) == 0


def test_double_qconfig_save(
    config_fp32: dict,
):
    """
    Ensure that using qconfig_save multiple times doesn't fail.

    Args:
        config_fp32 (dict): Config for fp32 quantization
    """
    qconfig_save(config_fp32, minimal=False)
    qconfig_save(config_fp32, minimal=False)


def test_qconfig_save_list_as_dict(
    config_fp32: dict,
):
    """
    Test that save recipes can't be used as dictionary

    Args:
        config_fp32 (dict): Config for fp32 quantization
    """
    delete_file()

    # Fill in keys_to_save as dict with nonsense val
    config_fp32["keys_to_save"] = {
        "qa_mode": None,
        "qw_mode": None,
        "smoothq": None,
        "scale_layers": None,
        "qskip_layer_name": None,
        "qskip_large_mag_layers": None,
    }

    with pytest.raises(ValueError):
        qconfig_save(config_fp32, minimal=True)


def test_qconfig_save_recipe_as_dict(
    config_fp32: dict,
):
    """
    Test that save recipes can't be used as dictionary

    Args:
        config_fp32 (dict): Config for fp32 quantization
    """
    # Fill in keys_to_save as dict with nonsense val
    save_dict = {
        "qa_mode": None,
        "qw_mode": None,
        "smoothq": None,
        "scale_layers": None,
        "qskip_layer_name": None,
        "qskip_large_mag_layers": None,
    }
    save_json(save_dict, file_path="keys_to_save.json")

    with pytest.raises(ValueError):
        qconfig_save(config_fp32, recipe="keys_to_save.json", minimal=True)


def test_qconfig_load_with_recipe_as_list(
    config_fp32: dict,
):
    """
    Test if using qconfig_load errors when loading a json list

    Args:
        config_fp32 (dict): Config for fp32 quantization
    """
    config_list = list(config_fp32.keys())

    save_json(config_list, file_path="qcfg.json")

    with pytest.raises(ValueError):
        _ = qconfig_load(fname="qcfg.json")


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

    if key in config_fp32:
        del config_fp32[key]

    # Save config with minimal removals to dump as json
    save_serialized_json(config_fp32)

    loaded_config = qconfig_load("qcfg.json")
    assert loaded_config.get(key) == default_val


def test_save_init_recipe(
    config_int8: dict,
):
    """
    Change a config, save it,

    Args:
        config_fp32 (dict): Config for fp32 quantization
    """
    # Change some elements of config to ensure its being saved/loaded properly
    config_int8["qa_mode"] = "minmax"
    config_int8["qa_mode"] = "pertokenmax"
    config_int8["qmodel_calibration"] = 17
    config_int8["qskip_layer_name"] = ["lm_head"]

    qconfig_save(config_int8)
    recipe_config = qconfig_init(recipe="qcfg.json")

    # Remove date field from recipe_config - only added at save
    del recipe_config["date"]

    assert len(recipe_config) == len(config_int8)

    for key, val in config_int8.items():
        assert key in recipe_config
        assert recipe_config.get(key) == val
