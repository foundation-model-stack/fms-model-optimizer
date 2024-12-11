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
Test check_config functionality

"""

# Third Party
import pytest
import torch

# First Party
from fms_mo.utils.qconfig_utils import check_config


def test_fp32model_with_fp16config(
    model_config_fp32: torch.nn.Module,
    config_fp16: dict,
):
    """
    check_config should raise RuntimeError when model is fp32, but requesting a fp16 nbits config

    Args:
        model_config_fp32 (torch.nn.Module): Single fp32 model
        config_fp16 (dict): Config for fp16 quantization
    """

    with pytest.raises(RuntimeError):
        model_dtype = next(model_config_fp32.parameters()).dtype
        check_config(config_fp16, model_dtype)


def test_fp16model_with_fp32config(
    model_config_fp16: torch.nn.Module,
    config_fp32: dict,
):
    """
    check_config should set fp32 config to fp16 if model is fp16

    Args:
        model_config_fp16 (torch.nn.Module): Single fp16 model
        config_fp32 (dict): Config for fp32 quantization
    """
    assert config_fp32.get("nbits_a", 32) == 32, "config nbits_a is not fp32"
    assert config_fp32.get("nbits_w", 32) == 32, "config nbits_w is not fp32"

    model_dtype = next(model_config_fp16.parameters()).dtype
    check_config(config_fp32, model_dtype)

    assert config_fp32["nbits_a"] == 16, "config nbits_a is not fp16 after check_config"
    assert config_fp32["nbits_w"] == 16, "config nbits_w is not fp16 after check_config"


# Config wrong value tests
def test_config_nbit_error(
    model_config_fp32: torch.nn.Module,
    config_fp32: dict,
    nbit_str: str,
    wrong_nbits: list,
):
    """
    Check that all nbit variables throw ValueError when presented with bad value

    Args:
        model_config_fp32 (torch.nn.Module): Single fp32 model
        config_fp32 (dict): Config for fp32 quantization
        nbit_str (str): Config dict key for nbit pairs
        wrong_nbits (list): List of bad nbit values
    """

    # Save value of current config
    nbit_temp = config_fp32.get(nbit_str, 32)
    # Write bad value to config
    for wrong_nbit in wrong_nbits:
        config_fp32[nbit_str] = wrong_nbit
        with pytest.raises(ValueError):
            model_dtype = next(model_config_fp32.parameters()).dtype
            check_config(config_fp32, model_dtype)
        # Reset to saved value
        config_fp32[nbit_str] = nbit_temp


def test_config_qmode_error(
    model_config_fp32: torch.nn.Module,
    config_fp32: dict,
    q_mode_str: str,
    wrong_qmodes: list,
):
    """
    Check that all qmode variables throw ValueError when presented with bad value

    Args:
        model_config_fp32 (torch.nn.Module): Single fp32 model
        config_fp32 (dict): Config for fp32 quantization
        q_mode_str (str): Config dict key for qmode pairs
        wrong_qmodes (list): List of bad qmode values
    """

    # Save value of current config
    q_mode_temp = config_fp32.get(q_mode_str, "pact")
    # Write bad value to config
    for wrong_qmode in wrong_qmodes:
        config_fp32[q_mode_str] = wrong_qmode
        with pytest.raises(ValueError):
            model_dtype = next(model_config_fp32.parameters()).dtype
            check_config(config_fp32, model_dtype)
        # Reset to saved value
        config_fp32[q_mode_str] = q_mode_temp


def test_config_boolean_error(
    model_config_fp32: torch.nn.Module,
    config_fp32: dict,
    bool_str: str,
    not_booleans: list,
):
    """
    Check that all boolean variables throw ValueError when presented with bad value

    Args:
        model_config_fp32 (torch.nn.Module): Single fp32 model
        config_fp32 (dict): Config for fp32 quantization
        bool_str (str): Config dict key for bool value pairs
        not_booleans (list): List of non-bool values
    """

    # Save value of current config
    bool_temp = config_fp32.get(bool_str, False)
    # Write bad value to config
    for not_boolean in not_booleans:
        config_fp32[bool_str] = not_boolean
        with pytest.raises(ValueError):
            model_dtype = next(model_config_fp32.parameters()).dtype
            check_config(config_fp32, model_dtype)
        # Reset to saved value
        config_fp32[bool_str] = bool_temp


def test_config_int_error(
    model_config_fp32: torch.nn.Module,
    config_fp32: dict,
    int_str: str,
    not_ints: list,
):
    """
    Check that all int variables throw ValueError when presented with bad value

    Args:
        model_config_fp32 (torch.nn.Module): Single fp32 model
        config_fp32 (dict): Config for fp32 quantization
        int_str (str): Config dict key for int value pairs
        not_ints (list): List of non-int values
    """

    # Save value of current config
    int_temp = config_fp32.get(int_str, 1)
    # Write bad value to config
    for not_int in not_ints:
        config_fp32[int_str] = not_int
        with pytest.raises(ValueError):
            model_dtype = next(model_config_fp32.parameters()).dtype
            check_config(config_fp32, model_dtype)
        # Reset to saved value
        config_fp32[int_str] = int_temp


def test_config_float_error(
    model_config_fp32: torch.nn.Module,
    config_fp32: dict,
    fp_str: str,
    not_fps: list,
):
    """
    Check that all float variables throw ValueError when presented with bad value

    Args:
        model_config_fp32 (torch.nn.Module): Single fp32 model
        config_fp32 (dict): Config for fp32 quantization
        fp_str (str): Config dict key for fp value pairs
        not_fps (list): List of non-fp values
    """

    # Save value of current config
    fp_temp = config_fp32.get(fp_str, 0.01)
    # Write bad value to config
    for not_fp in not_fps:
        config_fp32[fp_str] = not_fp
        with pytest.raises(ValueError):
            model_dtype = next(model_config_fp32.parameters()).dtype
            check_config(config_fp32, model_dtype)
        # Reset to saved value
        config_fp32[fp_str] = fp_temp


def test_config_iterable_error(
    model_config_fp32: torch.nn.Module,
    config_fp32: dict,
    iterable_str: str,
    not_iterables: list,
):
    """
    Check that all iterable variables throw ValueError when presented with bad value

    Args:
        model_config_fp32 (torch.nn.Module): Single fp32 model
        config_fp32 (dict): Config for fp32 quantization
        iterable_str (str): Config dict key for iterable value pairs
        not_iterables (list): List of non-iterable values
    """

    # Save value of current config
    iterable_temp = config_fp32.get(iterable_str, [])
    # Write bad value to config
    for not_iterable in not_iterables:
        config_fp32[iterable_str] = not_iterable
        with pytest.raises(ValueError):
            model_dtype = next(model_config_fp32.parameters()).dtype
            check_config(config_fp32, model_dtype)
        # Reset to saved value
        config_fp32[iterable_str] = iterable_temp


def test_config_clip_val_asst_percentile_error(
    model_config_fp32: torch.nn.Module,
    config_fp32: dict,
    not_clip_val_asst_percentile_settings: list,
):
    """
    Check that clip_val_asst_percentile throw ValueError when presented with bad value
    It is both iterable and has to contain floats

    Args:
        model_config_fp32 (torch.nn.Module): Single fp32 model
        config_fp32 (dict): Config for fp32 quantization
        not_clip_val_asst_percentile_settings (list):
            List of invalid values for clip_val_asst_percentile
    """
    other_str = "clip_val_asst_percentile"

    # Save value of current config
    clip_temp = config_fp32.get(other_str, [0.1, 99.9])
    # Write bad value to config
    for not_clip_val_asst_percentile in not_clip_val_asst_percentile_settings:
        config_fp32[other_str] = not_clip_val_asst_percentile
        with pytest.raises(ValueError):
            model_dtype = next(model_config_fp32.parameters()).dtype
            check_config(config_fp32, model_dtype)
        # Reset to saved value
        config_fp32[other_str] = clip_temp


def test_config_ptq_loss_func_error(
    model_config_fp32: torch.nn.Module,
    config_fp32: dict,
    not_ptq_loss_func_settings: list,
):
    """
    Check that ptq_loss_func throw ValueError when presented with bad value

    Args:
        model_config_fp32 (torch.nn.Module): Single fp32 model
        config_fp32 (dict): Config for fp32 quantization
        not_ptq_loss_func_settings (list): List of invalid values for ptq_loss_func
    """
    other_str = "ptq_loss_func"

    # Save value of current config
    other_temp = config_fp32.get(other_str, "mse")
    # Write bad value to config
    for not_ptq_loss_func in not_ptq_loss_func_settings:
        config_fp32[other_str] = not_ptq_loss_func
        with pytest.raises(ValueError):
            model_dtype = next(model_config_fp32.parameters()).dtype
            check_config(config_fp32, model_dtype)
        # Reset to saved value
        config_fp32[other_str] = other_temp


def test_config_ptq_coslr_error(
    model_config_fp32: torch.nn.Module, config_fp32: dict, not_ptq_coslr_settings: list
):
    """
    Check that ptq_coslr throw ValueError when presented with bad value

    Args:
        model_config_fp32 (torch.nn.Module): Single fp32 model
        config_fp32 (dict): Config for fp32 quantization
        not_ptq_coslr_settings (list): List of invalid values for ptq_coslr
    """
    other_str = "ptq_coslr"

    # Save value of current config
    other_temp = config_fp32.get(other_str, "")
    # Write bad value to config
    for not_ptq_coslr in not_ptq_coslr_settings:
        config_fp32[other_str] = not_ptq_coslr
        with pytest.raises(ValueError):
            model_dtype = next(model_config_fp32.parameters()).dtype
            check_config(config_fp32, model_dtype)
        # Reset to saved value
        config_fp32[other_str] = other_temp


def test_config_which2patch_contextmanager_error(
    model_config_fp32: torch.nn.Module,
    config_fp32: dict,
    not_which2patch_contextmanager_settings: list,
):
    """
    Check that which2patch_contextmanager throw ValueError when presented with bad value

    Args:
        model_config_fp32 (torch.nn.Module): Single fp32 model
        config_fp32 (dict): Config for fp32 quantization
        not_which2patch_contextmanager_settings (list):
            List of invalid values for which2patch_contextmanager
    """
    other_str = "which2patch_contextmanager"

    # Save value of current config
    other_temp = config_fp32.get(other_str, None)
    # Write bad value to config
    for not_which2patch_contextmanager in not_which2patch_contextmanager_settings:
        config_fp32[other_str] = not_which2patch_contextmanager
        with pytest.raises(ValueError):
            model_dtype = next(model_config_fp32.parameters()).dtype
            check_config(config_fp32, model_dtype)
        # Reset to saved value
        config_fp32[other_str] = other_temp
