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
"""Util functions for qconfig."""

# Standard
from copy import deepcopy
from datetime import date
from importlib.metadata import version
from pathlib import Path
from typing import Any, Union
import argparse
import json
import logging
import os
import sys
import warnings

# Third Party
from torch import nn
import torch

# Local
from fms_mo.modules import QLSTM, QBmm, QConv2d, QConvTranspose2d, QLinear
from fms_mo.utils.import_utils import available_packages

# import numpy as np # only used in experimental func


logger = logging.getLogger(__name__)


def get_pkg_info(other_pkgs: list = None) -> dict:
    """
    Get the package name and version of important packages currently in use.

    Args:
        other_pkgs (list): List of other packages to include

    Returns:
        dict: pkg,version pairs
    """
    # Get shortened python version - only obtainable from sys.version
    pkgs = {"python": ".".join(map(str, sys.version_info[:3]))}
    if other_pkgs is None:
        other_pkgs = []

    # Get installed packages name:version
    pkgs.update(
        {
            pkg: version(pkg)
            for pkg in [
                "fms-model-optimizer",
                "torch",
                "transformers",
                "triton",
            ]
            + other_pkgs
        }
    )

    return pkgs


def config_defaults() -> dict:
    """Create defaults for qconfig"""
    cfg_defaults = {
        # nbits vars
        "nbits_a": 32,
        "nbits_w": 32,
        "nbits_a_alt": None,
        "nbits_w_alt": None,
        "nbits_a_qkv": None,
        "nbits_w_qkv": None,
        "nbits_bmm1": None,
        "nbits_bmm2": None,
        "nbits_kvcache": None,
        "nbits_w_lstm": None,
        "nbits_i_lstm": None,
        "nbits_h_lstm": None,
        # qmodes vars
        "qa_mode": "pact+",
        "qw_mode": "sawb+",
        "qa_qkv_mode": "pact",
        "qw_qkv_mode": "sawb",
        "bmm1_qm1_mode": "pact",
        "bmm1_qm2_mode": "pact",
        "bmm2_qm1_mode": "pact",
        "bmm2_qm2_mode": "pact",
        "qa_mode_lstm": "pact+",
        # mode_calib vars
        "qa_mode_calib": "percentile",
        "qw_mode_calib": "percentile",
        # init_method vars
        "a_init_method": "percentile",
        "w_init_method": "sawb",
        # qmodel_calibration
        "qmodel_calibration": 0,
        "qmodel_calibration_new": 0,
        # Boolean vars
        "qshortcutconv": False,
        "q1stlastconv": False,
        "qdw": False,
        "qskipfpn": False,
        "qkvsync": False,
        "extend_act_range": False,
        "plotsvg": False,
        "qskip_large_mag_layers": False,
        "recompute_narrow_weights": False,
        # Iterable vars
        "qlayer_name_pattern": [],
        "qskip_layer_name": [],
        "qspecial_layers": {},
        "qsinglesided_name": [],
        "clip_val_asst_percentile": (0.1, 99.9),
        "params2optim": {
            "W": [[] for _ in range(torch.cuda.device_count())],
            "cvs": [[] for _ in range(torch.cuda.device_count())],
        },
        # PTQ vars
        "ptq_nbatch": 100,
        "ptq_batchsize": 12,
        "ptq_nouterloop": 20000,
        "ptq_ninnerloop": 1,
        "ptq_coslr": "",
        "ptq_lrw": 1e-05,
        "ptq_lrcv_a": 0.001,
        "ptq_lrcv_w": 0.001,
        "ptq_freezecvs": False,
        "ptq_qdrop": False,
        "ptq_loss_func": "mse",
        "firstptqmodule": [],
        "temp_disable_quantizers": False,
        "temp_disable_PTQ": False,
        "temp_disable_calib": False,
        "org_batch_size": {},
        "ptqmod_to_be_optimized": [],
        # SmoothQuant vars
        "smoothq": False,
        "smoothq_scale_layers": [],
        "smoothq_act_scale_path": None,
        # Other vars
        "which2patch_contextmanager": None,
        "force_stop_if_qbmm_auto_check_failed": False,
        "world_size": max(1, torch.cuda.device_count()),
        "global_rank": 0,
        "batch_size": 2,
        "keys_to_save": [],
        # items could be obsoleted
        "output_attentions": False,
        "bias_corr": False,
        "qwav2vec": False,
        "qvit": False,
        "numparamsfromloadertomodel": 1,
        "gradclip": 0.0,
    }

    return cfg_defaults


def find_recipe_json(recipe: str, subdir: str = None) -> Path:
    """
    Search for recipe .json file in fms_mo and return the path

    Args:
        recipe (str): Recipe file name (can be the "name" or "prefix.json").
        subdir (str, optional): Alternative subdir path from pkg_root. Defaults to None.

    Returns:
        Path: File Path to .json recipe if found, else None
    """
    if recipe is None:
        return None

    cwd = Path().resolve()
    pkg_root = Path(__file__).parent.parent.resolve()
    file_in_cwd = cwd / recipe
    file_in_cwd2 = cwd / f"{recipe}.json"
    if subdir:
        file_in_recipes = pkg_root / subdir / "recipes" / recipe
        file_in_recipes2 = pkg_root / subdir / "recipes" / f"{recipe}.json"
    else:
        file_in_recipes = pkg_root / "recipes" / recipe
        file_in_recipes2 = pkg_root / "recipes" / f"{recipe}.json"

    if not recipe.endswith(".json") and file_in_recipes2.exists():
        json_file = file_in_recipes2
    elif not recipe.endswith(".json") and file_in_cwd2.exists():
        json_file = file_in_cwd2
    elif file_in_cwd.exists():
        json_file = file_in_cwd
    elif file_in_recipes.exists():
        json_file = file_in_recipes
    else:
        json_file = None

    return json_file


def get_recipe(recipe: str, subdir: str = None) -> Union[list, dict]:
    """
    Get a json recipe.

    Args:
        recipe (str): Name of the recipe file.
        subdir (str, optional): A subdirectory to search from root. Defaults to None.

    Returns:
        Any: Data from a saved .json.
    """
    json_file = find_recipe_json(recipe, subdir)
    temp_data = None
    if json_file:
        with open(json_file, "r", encoding="utf-8") as openfile:
            temp_data = json.load(openfile)
        logger.info(f"Loaded settings from {json_file}.")

        # Any recipe should be a dict (qcfg) or list (keys_to_save)
        if not isinstance(temp_data, (dict, list)):
            raise ValueError(f"Loaded recipe {json_file} was not a dict or list")

    return temp_data


def qconfig_init(recipe: str = None, args: Any = None, use_mx: bool = False) -> dict:
    """Three possible ways to create qcfg:
        1. create a default qcfg
        2. load from a json
        3. parse the args
        NOTE: Content from higher number, e.g. arg parser, will override thier counterpart from
            lower numbers, e.g. json.

    Args:
        recipe: str. Recipe filename (json) that contains settings, if specified and exists.
            Will search cwd and fms_mo/recipes folder. ok to omit '.json' extension.
        args: argparser object that may contain relevant parameters.

        Important items in the config dictionary:
        nbits_[w|a]_alt: "_alt" stands for "alternative" -> the default prec for those "skipped"
            layers e.g. usually the 1st/last layers are "skipped" and will NOT be swapped to
            QLinear. But, if "nbits_x_alt = 8", they will.
        qmodel_calibration[_new]: set to non-zero will trigger calibration. "_new" means
            calibration will happen during the first N calls of fwd path, better for long
            training or fine-tuning that you don't mind losing the first N iters
        qlayer_name_pattern: allows partial or regex name matching, the layers satisfy the
            criteria will be skipped. NOTE: tracing will be bypassed entirely if this arg is used
        qskip_layer_name: user can specify exact name to skip
        qspecial_layers: special case handling. user can specify any quant params for any
            given layer, e.g. {'1st.conv':{'nbits_w':8,'qw_mode':'pact+sym'}, '2nd.layers':{...} }
        extend_act_range: symmetric act quantizers (maxsym, pactsym+, ...) to use full range, e.g.,
            [-128, 127] instead [-127,127], TODO: should default to True?
        ptq_nbatch: total number of batches of data that will be fetched from loader for PTQ tuning
        ptq_batchsize: data used in PTQ tuning usually is fetched from loader directly,
            i.e. batchsize is the unchanged from dataloader.batch_size. although it could be
            different if needed, e.g. PTQ may allow larger bs due to only partial model tuning.
            But fine-grain shuffling will be needed in that case.
        ptq_nouterloop: number of optimization "steps" in the PTQ outer loop. 1 outer loop uses
            1 cached data batch. when Nouter >= Nbatch, data will be re-used
        ptq_ninnerloop: number of "inner loop" for PTQ optimization. When 1 batch of data is
            fetched, run (loss->loss.back->optim.step) this many times before fetching the next
            batch. NOTE: usually doesn't make big differences, hence, default to 1
        ptq_coslr: can be "", "W" or "A" or "WA", indicating which (or both) optimizer will use
            cosLR, otherwise use constantLR as default
    """

    qcfg = {}
    # 1. create a dict with default values
    qcfg["mapping"] = {
        nn.Linear: QLinear,
        nn.Conv2d: QConv2d,
        nn.ConvTranspose2d: QConvTranspose2d,
        nn.LSTM: QLSTM,
        "matmul_or_bmm": QBmm,
    }

    qcfg["pkg_versions"] = get_pkg_info()

    qcfg["nbits_w"] = 32
    qcfg["nbits_a"] = 32
    qcfg["qa_mode"] = "pact+"
    qcfg["qw_mode"] = "sawb+"
    qcfg["nbits_w_alt"] = None
    qcfg["nbits_a_alt"] = None
    qcfg["qmodel_calibration"] = 0
    qcfg["qmodel_calibration_new"] = 0
    qcfg["qa_mode_calib"] = "percentile"
    qcfg["qw_mode_calib"] = "percentile"
    # TODO: qx_mode_calib is used by new calib, w_init_method is used by old calib. Need to unify
    qcfg["w_init_method"] = "sawb"
    qcfg["a_init_method"] = "percentile"
    qcfg["clip_val_asst_percentile"] = (0.1, 99.9)

    # ways to control which layers to be quantized/skipped
    qcfg["qlayer_name_pattern"] = []
    qcfg["qskip_layer_name"] = []
    qcfg["qskip_large_mag_layers"] = False
    qcfg["recompute_narrow_weights"] = False
    qcfg["qspecial_layers"] = {}

    # settings about quantizing bmm/matmul
    qcfg["nbits_bmm1"] = None
    qcfg["nbits_bmm2"] = None
    qcfg["nbits_kvcache"] = None
    qcfg["qa_qkv_mode"] = "pact"
    qcfg["qw_qkv_mode"] = "sawb"
    qcfg["bmm1_qm1_mode"] = "pact"
    qcfg["bmm2_qm1_mode"] = "pact"
    qcfg["bmm1_qm2_mode"] = "pact"
    qcfg["bmm2_qm2_mode"] = "pact"
    qcfg["qkvsync"] = False
    qcfg["which2patch_contextmanager"] = (
        None  # an internal var that should not be set by user
    )
    qcfg["force_stop_if_qbmm_auto_check_failed"] = False

    # LSTM related, if any of these is not None, then last layer (FC) will not be skipped.
    qcfg["nbits_w_lstm"] = None
    qcfg["nbits_i_lstm"] = None
    qcfg["nbits_h_lstm"] = None
    qcfg["nbits_w_qkv"] = None
    qcfg["nbits_a_qkv"] = None
    qcfg["qa_mode_lstm"] = "pact+"

    qcfg["extend_act_range"] = False

    # PTQ related settings
    qcfg["temp_disable_quantizers"] = False
    qcfg["temp_disable_PTQ"] = False
    qcfg["temp_disable_calib"] = False
    qcfg["force_calib_once"] = False
    qcfg["ptq_nbatch"] = 100
    qcfg["ptq_batchsize"] = 12
    qcfg["ptq_nouterloop"] = 20000
    qcfg["ptq_ninnerloop"] = 1
    qcfg["ptq_coslr"] = ""
    qcfg["ptq_lrw"] = 1e-5  # 1e-3 or 1e-5 for AdaQuant
    qcfg["ptq_lrcv_w"] = 1e-3
    qcfg["ptq_lrcv_a"] = 1e-3  # lr was 1e-1 or 1e-3 in AdaQuant, 4e-5 for BRECQ
    qcfg["org_batch_size"] = {}
    qcfg["ptqmod_to_be_optimized"] = []
    qcfg["ptq_freezecvs"] = False
    qcfg["ptq_qdrop"] = False
    qcfg["ptq_loss_func"] = "mse"
    qcfg["firstptqmodule"] = []
    qcfg["params2optim"] = {
        "W": [[] for _ in range(torch.cuda.device_count())],
        "cvs": [[] for _ in range(torch.cuda.device_count())],
    }
    # collect parameters based on device index, in case DP is used

    qcfg["tb_writer"] = None
    qcfg["world_size"] = max(1, torch.cuda.device_count())  # in case no GPU is found
    qcfg["global_rank"] = 0
    qcfg["batch_size"] = 2

    # items could be obsoleted
    qcfg["output_attentions"] = False
    qcfg["bias_corr"] = False
    qcfg["qwav2vec"] = False
    qcfg["qvit"] = False
    qcfg["qsinglesided_name"] = []
    qcfg["qshortcutconv"] = False
    qcfg["q1stlastconv"] = False
    qcfg["qdw"] = False
    qcfg["qskipfpn"] = False
    qcfg["plotsvg"] = False
    qcfg["numparamsfromloadertomodel"] = 1  # TODO: to be obsoleted
    # Sometimes, dataloader unpack into 2 elements or more, e.g. (img, labels) = next(dataloader)
    # but only one will be passed to model during forward, e.g. pred = model(img)
    # => set numparamsfromloadertomodel = 1, use "prefwdfunc" may be a better option
    qcfg["gradclip"] = 0.0

    # 2. load values from json, if specified and exists
    #    this can be used to load a previously saved ckpt as well
    if recipe:
        # qcfg recipes should reside in fms_mo/recipes
        temp_cfg = qconfig_load(recipe)

        if temp_cfg:
            if not isinstance(temp_cfg, dict):
                raise ValueError(
                    f"Quantized config recipe={recipe} is not a dictionary"
                )

            qcfg.update(temp_cfg)
            logger.info("Updated config with recipe values")
        else:
            raise ValueError(f"Config recipe {recipe} was not found.")

    # 3. parse args, if provided
    if hasattr(args, "__dict__"):
        vars_dict = vars(
            args
        )  # vars() returns "args" properties as a dict, easier than dir()?
        if "__flags" in vars_dict:
            for k, v in vars_dict[
                "__flags"
            ].items():  # NOTE: k is str but v is object, hence v.value
                qcfg[k] = v.value
        else:
            qcfg.update(vars_dict)
        logger.info(
            "Some args are parsed into qcfg."
            " Default or values from json of the same key will be overwritten."
        )

    # 4. Check if mapping must change for MX library
    # For now, simply use qa_mode or qw_mode to trigger, e.g. "mx_fp4_e2m1" -> "fp4_e2m1"
    # user may create qcfg without "mx_fpxxx" then manually changes qw_mode/qa_mode to "mx_fpxxx"
    # => need to check again at the beginning of qmodel_prep(), i.e. in check_config()
    set_mx_specs(qcfg, args, use_mx)

    return qcfg


def set_mx_specs(
    config: dict,
    args: argparse.ArgumentParser = None,
    use_mx: bool = False,
):
    """
    Set mx_specs dict in quantized config to be used for MX quantization.
    Will use fms_mo default values for variables when none are given.

    Options available:
    1. Pass job args to create mx_specs.  Must have --a_elem_format and --w_elem_format set.
    2. Consume a premade mx_specs dict from quantized config if present.
    3. Consume quantized config variables prefixed with "mx_".

    Options 2 and 3 are mutually exclusive with preference for Option 2 if both are given.

    Args:
        config (dict): Quantization config dict
        args (argparse.ArgumentParser, optional): Job arg parser. Defaults to None.
        use_mx (bool): Create default mx_specs when qcfg or args aren't present.
            Defaults to False.
    """
    mx_prefix = "mx_"

    # MX lib defaults for these values are None, 0, nearest, max, or bool
    fms_defaults = get_mx_specs_defaults()

    # Already have a mx_specs saved in config
    use_mx_specs_config = "mx_specs" in config

    # Check for any "mx_" vars set in config
    use_mx_config = any(key.startswith(mx_prefix) for key in config.keys())

    # Check args for any mx_specs vars
    use_mx_args = args is not None and any(
        hasattr(args, key)
        for key, _ in fms_defaults.items()
        if key != "block_size"
        # some items are not unique to mx, add names here if needed
    )

    # Lastly, check for BMM consistency to enable QBmmMX
    fms_bmm_modes = [
        config["bmm1_qm1_mode"].startswith(mx_prefix),
        config["bmm1_qm2_mode"].startswith(mx_prefix),
        config["bmm2_qm1_mode"].startswith(mx_prefix),
        config["bmm2_qm2_mode"].startswith(mx_prefix),
    ]
    # If any mx bmm set, they all must be set for QBmmMX ; will be checked in check_config
    use_fms_bmm_modes = all(fms_bmm_modes)

    use_mx = (
        use_mx
        or use_mx_specs_config
        or use_mx_config
        or use_mx_args
        or use_fms_bmm_modes
    )

    if use_mx:
        # If "mapping" has been removed from qcfg -> chk_cfg is being called by save_config() at
        #     the end of qmodel_prep() => don't need to update anything.
        # NOTE: If "mx_" qa_/qw_mode was used through args, the "mx_" prefix would have been removed
        #     already in chk_cfg() => "use_mx" flag will be False. Keep in mind that THE ONLY WAY TO
        #     TRIGGER REFRESH of mx_specs AFTER qconfig_init() is to manually set qa_/qw_mode to
        #     "mx_something"!

        if available_packages["mx"]:
            # Standard
            from functools import partial

            # Third Party
            # pylint: disable = import-error
            import mx

            # Local
            from fms_mo.modules.bmm import QBmmMX
            from fms_mo.modules.linear import QLinearMX

            # Create a MxSpecs object based on input args and overwrite w/ qcfg if provided
            mx_specs = mx.get_mx_specs(args) if use_mx_args else mx.MxSpecs()

            # Ensure fms defaults are set assuming job args haven't already changed them
            for key, val in fms_defaults.items():
                if mx_specs[key] in [None, 0, False, True, "nearest", "max"]:
                    mx_specs[key] = val

            # Use config["mx_specs"] settings
            if use_mx_specs_config:
                mx_specs.update(config["mx_specs"])

            # Use qcfg mx equivalents
            else:
                # k_elem_format special case - in q_modes
                if config["qw_mode"].startswith(mx_prefix):
                    mx_specs["w_elem_format"] = config["qw_mode"].replace(mx_prefix, "")
                if config["qa_mode"].startswith(mx_prefix):
                    mx_specs["a_elem_format"] = config["qa_mode"].replace(mx_prefix, "")

                for mx_var, _ in fms_defaults.items():
                    fms_var = "mx_" + mx_var
                    # Only update if its in config; default values already set
                    if fms_var in config:
                        mx_specs[mx_var] = config.get(fms_var)

                # Only 1 variable that has "mx_" prefix from MX lib
                mx_var = "mx_flush_fp32_subnorms"
                if mx_var in config:
                    mx_specs[mx_var] = config.get(mx_var)

            # Many mx_spec vars are synched with other vars -- may have changed now
            mx_specs = mx.finalize_mx_specs(mx_specs)

            # Save finalized mx_spec to config
            config["mx_specs"] = mx_specs.data

            # Update mapping for torch.nn and matmul_or_bmm to MX variants
            # QLinearMX will be used, but QBmmMX requires bmm specifically
            config["mapping"][nn.Linear] = partial(
                QLinearMX, mx_specs=config["mx_specs"]
            )
            # config["mapping"][nn.Conv2d] = partial(
            #     QConv2dMX, mx_specs=config["mx_specs"]
            # )
            # config["mapping"][nn.ConvTranspose2d] = partial(
            #     QConvTranspose2dMX, mx_specs=config["mx_specs"]
            # )
            if use_fms_bmm_modes:  # all bmm_modes are "mx_" prefixed
                config["mapping"]["matmul_or_bmm"] = partial(
                    QBmmMX, mx_specs=config["mx_specs"]
                )

        else:
            logger.info("MX variables provided, but MX package is not installed")


def is_nvcc_installed():
    """
    Check whether we can call on the NVIDIA CUDA Compiler from the OS level

    Returns:
        bool: If nvcc is found and callable at the OS level
    """
    # Standard
    import subprocess

    try:
        # Run the nvcc command to check if it's installed
        subprocess.check_output("nvcc --version", shell=True, stderr=subprocess.STDOUT)
        logger.info("nvcc is installed and callable")
        return True
    except subprocess.CalledProcessError:
        logger.info("nvcc is installed, but there was an issue running nvcc.")
        return False
    except FileNotFoundError:
        logger.info("nvcc is not installed on the system.")
        return False


def get_mx_specs_defaults():
    """
    Get key,value pairs for mx_specs defaults for fms_mo

    Returns:
        dict: fms_mo defaults of mx_specs
    """
    return {
        "w_elem_format": "fp8_e4m3",
        "a_elem_format": "fp8_e4m3",
        "w_elem_format_bp": "fp8_e4m3",
        "a_elem_format_bp": "fp8_e4m3",
        "a_elem_format_bp_ex": "fp8_e4m3",
        "a_elem_format_bp_os": "fp8_e4m3",
        "shared_exp_method": "max",
        "scale_bits": 8,
        "block_size": 32,  # this item is not unique to mx
        "bfloat": 16,  # bfloat and fp cannot be set at the same time
        "fp": 0,
        "round": "nearest",
        "round_m": "nearest",
        "round_weight": "nearest",
        "round_output": "nearest",
        "round_grad_weight": "nearest",
        "round_grad_input": "nearest",
        "round_mx_output": "nearest",
        "round_mx_input_grad_input": "nearest",
        "round_mx_weight_grad_input": "nearest",
        "round_mx_grad_output_grad_input": "nearest",
        "round_mx_input_grad_weight": "nearest",
        "round_mx_grad_output_grad_weight": "nearest",
        "quantize_backprop": True,
        "bfloat_subnorms": True,
        "mx_flush_fp32_subnorms": False,
        "softmax_exp2": False,
        "vec_use_exp2": False,
        "vec_use_recip": False,
        "custom_cuda": torch.cuda.is_available() and is_nvcc_installed(),
    }


def has_non_serializable_object(anything: Any) -> bool:
    """
    Generalized recursive function looking for any non-serializable Python object
    Only types that are JSON serializable are None, primitives, tuples, lists, and dicts.
    Any other types must be converted into one of the types above.
    """
    if isinstance(anything, (list, tuple)):
        is_not_serializable = any(has_non_serializable_object(i) for i in anything)
        if is_not_serializable:
            message = f"{anything} contains non-serializable object(s)!"
            warnings.warn(message, UserWarning)

    elif isinstance(anything, dict):
        is_not_serializable = any(
            (has_non_serializable_object(k) or has_non_serializable_object(v))
            for k, v in anything.items()
        )
        if is_not_serializable:
            message = f"{anything} contains non-serializable object(s)!"
            warnings.warn(message, UserWarning)

    else:
        is_not_primitive = not isinstance(anything, (int, float, bool, str))
        is_not_none = anything is not None
        is_not_serializable = is_not_primitive and is_not_none
        if is_not_serializable:
            message = f"{anything} w/ type {type(anything)} not a serializable!"
            warnings.warn(message, UserWarning)

    return is_not_serializable


def serialize_config(config: dict) -> tuple[dict, dict]:
    """
    Util function to clean config of any non-serializable key,val pairs
    """
    items_to_delete = []
    for key, val in config.items():
        if has_non_serializable_object(key) or has_non_serializable_object(val):
            items_to_delete.append(key)
            message = (
                f"Deleting non-serializable pair {key},{val} from config. "
                "If you want this pair in your config, use json.dump() directly"
            )
            warnings.warn(message, UserWarning)

    len_before = len(config)
    dump = {k: config.pop(k) for k in items_to_delete}
    assert (
        len(config) + len(dump) == len_before
    ), "Inconsistency in config. Please check."

    return config, dump


def remove_unwanted_from_config(
    config: dict, minimal: bool = True
) -> tuple[dict, dict]:
    """Remove deprecated items or things cannot be saved as text (json)"""
    unwanted_items = [
        "sweep_cv_percentile",
        "Qlist",
        "tb_writer",
        "mapping",
        "checkQerr_frequency",
        "newlySwappedModules",
        "force_calib_once",
        # if we keep the following LUTs, it will save the entire model
        "LUTmodule_name",
        "qkvsync_my_1st_sibling",
        "graph_in_out",
        "hook_qbmm_auto_check",
    ]

    # If minimal qcfg to be saved, remove any variable that is equal to a default
    if minimal:
        default_config = config_defaults()
        for key, val in config.items():
            # If config has a default setting, add to unwanted items
            if default_config.get(key) == val:
                unwanted_items.append(key)

    len_before = len(config)
    dump = {k: config.pop(k) for k in unwanted_items if k in config}
    assert (
        len(config) + len(dump) == len_before
    ), "Inconsistency in config. Please check."
    return config, dump


def get_unserializable_defaults() -> dict:
    """Add back those unserializable items if needed"""
    unserializable_items = {
        "sweep_cv_percentile": False,
        "tb_writer": None,
        "mapping": {
            nn.Conv2d: QConv2d,
            nn.ConvTranspose2d: QConvTranspose2d,
            nn.Linear: QLinear,
            nn.LSTM: QLSTM,
            "matmul_or_bmm": QBmm,
        },
        "checkQerr_frequency": False,
        "newlySwappedModules": [],
        "force_calib_once": False,
        # if we keep the follwing LUTs, it will save the entire model
        "LUTmodule_name": {},
    }
    return unserializable_items


def add_if_not_present(config: dict, items_to_add: dict) -> None:
    """
    Add items to config dict only if they aren't present

    Args:
        config (dict): Quantized config
        items_to_add (dict): Items that will be added if not present in config
    """
    for key, val in items_to_add.items():
        if key not in config:
            config[key] = val


def add_required_defaults_to_config(config: dict) -> None:
    """Recover "unserializable" items that are previously removed from config"""
    add_if_not_present(config, get_unserializable_defaults())


def add_wanted_defaults_to_config(config: dict, minimal: bool = True) -> None:
    """Util function to add basic config defaults that are missing into a config
    if a wanted item is not in the config, add it w/ default value
    """
    if not minimal:
        add_if_not_present(config, config_defaults())


def qconfig_save(
    qcfg: dict,
    recipe: str | None = None,
    minimal: bool = True,
    fname: str = "qcfg.json",
) -> None:
    """
    Try to save qcfg into a JSON file (or use .pt format if something really can't be text-only).
    For example, qcfg['mapping'] has some classes as keys and values, json won't work. We will try
    to remove unserializable items first.

    Args:
        qcfg (dict): Quantized config.
        recipe (str, optional): String name for a save recipe. Defaults to None.
        minimal (bool, optional): Save a minimal quantized config. Defaults to True.
        fname (str, optional): File name to save quantized config. Defaults to "qcfg.json".
    """

    # First check in qcfg for added save list
    # This value is hardcoded to avoid probing qcfg with real keys like "qa_mode"
    keys_to_save = qcfg.get("keys_to_save", [])

    # Next, check in fms_mo/recipes and merge them into a unique set (in case they differ)
    keys_to_save_json = get_recipe(recipe)

    if keys_to_save_json:
        if not isinstance(keys_to_save_json, list):
            raise ValueError(f"Save recipe={recipe} is not a list!")

        # Merge keys_to_save lists
        keys_to_save = list(set(keys_to_save + keys_to_save_json))

    # If we found keys to save, fetch them from qcfg
    if keys_to_save:
        temp_qcfg = {}
        for key in keys_to_save:
            if key in qcfg:
                temp_qcfg[key] = qcfg[key]
            else:
                raise ValueError(f"Desired save {key=} not in qcfg!")

    else:
        # We assume a full qcfg is being saved - trim it!
        temp_qcfg = deepcopy(qcfg)

        # Remove deprecated/unwanted key,vals in config
        temp_qcfg, _ = remove_unwanted_from_config(temp_qcfg, minimal)

        # Add back wanted defaults for any missing vars
        add_wanted_defaults_to_config(temp_qcfg, minimal)

        # Clean config of any unwanted key,vals not found in unwanted list
        temp_qcfg, _ = serialize_config(temp_qcfg)

    # Add in date and system information for archival
    temp_qcfg["date"] = date.today().strftime("%Y-%B-%d")
    temp_qcfg["pkg_versions"] = get_pkg_info()

    # Finally, check to ensure all values are valid before saving
    check_config(temp_qcfg)

    # Save config as json
    if os.path.isfile(fname):
        logger.info(f"{fname} already exist, will overwrite.")
    with open(fname, "w", encoding="utf-8") as outfile:
        json.dump(temp_qcfg, outfile, indent=4)


def qconfig_load(fname: str = "qcfg.json") -> dict:
    """Read config in json format, work together with qconfig_save"""
    config = get_recipe(fname)

    if config:
        # Check that loaded file is a dict
        if not isinstance(config, dict):
            raise ValueError(f"Quantized config={fname} is not a dictionary")

        # Add back wanted defaults for any missing vars
        add_wanted_defaults_to_config(config, minimal=False)
        add_required_defaults_to_config(config)

        # Ensure config has correct values before continuing
        check_config(config)

        return config

    logger.info(f"{fname} doesn't exist. cannot load the qcfg")


def check_config(config: dict, model_dtype: torch.dtype = None) -> None:
    """
    Check config values are valid before consuming them in qmodel_prep
    The following errors are detected:
        Any non-valid variables will throw a ValueError
        A RuntimeError will be thrown if a model is fp32 and is requested to be fp16

    If a recoverable option is available, we can overwrite it:
        If a model is fp16 and we request quantization at a higher precision -> set nbits to fp16
        supposed to be an int but provided a float (float(k.0) vs int(k)) -> cast to int(k)
        supposed to be a float but provided an int (int(k) vs float(k.0)) -> cast to float(k.0)
    """
    num_bits_settings = [2, 4, 8, 16, 32]
    nbits_a = config.get("nbits_a", 32)
    # Check if integer was given as float (1.0 when it should be 1)
    if isinstance(nbits_a, float) and nbits_a.is_integer():
        config["nbits_a"] = int(nbits_a)
        nbits_a = int(nbits_a)
    if nbits_a not in num_bits_settings:
        raise ValueError(
            f"nbits_a = {nbits_a} is not a supported quantization setting.  "
            f"Should be set one of the following: {num_bits_settings}"
        )

    nbits_w = config.get("nbits_w", 32)
    # Check if integer was given as float (1.0 when it should be 1)
    if isinstance(nbits_w, float) and nbits_w.is_integer():
        config["nbits_w"] = int(nbits_w)
        nbits_w = int(nbits_w)
    if nbits_w not in num_bits_settings:
        raise ValueError(
            f"nbits_w = {nbits_w} is not a supported quantization setting.  "
            f"Should be set one of the following: {num_bits_settings}"
        )

    # If no model_dtype given, compute based on min nbits
    if model_dtype is None:
        min_nbits = min(nbits_a, nbits_w)
        if min_nbits == 32:
            model_dtype = torch.float32
        elif min_nbits == 16:
            model_dtype = torch.float16
        else:
            model_dtype = torch.int8

    # Check if model is fp32 and nbits == 16, throw RuntimeError
    if model_dtype == torch.float32 and (nbits_a, nbits_w) == (16, 16):
        raise RuntimeError(f"Model has dtype {model_dtype}, but nbits_a,nbits_w = 16.")

    # If model is fp16 and higher precision is requested, change any nbits to fp16
    if model_dtype in [torch.float16, torch.bfloat16]:
        if nbits_a > 16:
            config["nbits_a"] = 16
            logger.warning(
                f"Model has dtype {model_dtype}, but nbits_a = {nbits_a} is requesting higher "
                "precision.  Setting nbits_a to 16",
            )

        if nbits_w > 16:
            config["nbits_w"] = 16
            logger.warning(
                f"Model has dtype {model_dtype}, but nbits_w = {nbits_w} is requesting higher "
                "precision.  Setting nbits_w to 16",
            )

    # Check other nbit settings
    other_nbits_str = [
        "nbits_a_qkv",
        "nbits_w_qkv",
        "nbits_bmm1",
        "nbits_bmm2",
        "nbits_kvcache",
        "nbits_a_alt",
        "nbits_w_alt",
    ]
    other_nbits_settings = [2, 4, 8, 16, 32, None]
    # None = null in JSON - these do not need to be set

    for other_nbit_str in other_nbits_str:
        other_nbit = config.get(other_nbit_str, None)
        # Check if integer was given as float (1.0 when it should be 1)
        if isinstance(other_nbit, float) and other_nbit.is_integer():
            config[other_nbit] = int(other_nbit)
            other_nbit = int(other_nbit)
        if other_nbit not in other_nbits_settings:
            raise ValueError(
                f"{other_nbit_str} = {other_nbit} is not set to one of the following: "
                f"{other_nbits_settings}"
            )

    # Set allowed qa_modes, qw_modes, bmm_modes
    qa_mode_settings = [
        "pact",
        "pact+",
        "pactsym",
        "pactsym+",
        "max",
        "minmax",
        "maxsym",
        "pertokenmax",
        "lsq+",
        "fix",
        "brecq",
        # fp8_e4m3
        "fp8_e4m3_sat",
        "fp8_e4m3_scale",
        "fp8_e4m3_sat_perCh",
        "fp8_e4m3_scale_perCh",
        "fp8_e4m3_sat_perToken",
        "fp8_e4m3_scale_perToken",
        # fp8_e5m2
        "fp8_e5m2_sat",
        "fp8_e5m2_scale",
        "fp8_e5m2_sat_perCh",
        "fp8_e5m2_scale_perCh",
        "fp8_e5m2_sat_perToken",
        "fp8_e5m2_scale_perToken",
    ]
    qw_mode_settings = [
        "sawb",
        "sawb16",
        "sawbperCh",
        "sawb+",
        "sawb+16",
        "sawb+perCh",
        "max",
        "maxperCh",
        "maxperGp",
        "minmax",
        "minmaxperCh",
        "minmaxperGp",
        "pact",
        "pact+",
        "lsq+",
        "fix",
        "dorefa",
        "brecq",
        "adaround",
        "pertokenmax",
        # fp8_e4m3
        "fp8_e4m3_sat",
        "fp8_e4m3_scale",
        "fp8_e4m3_sat_perCh",
        "fp8_e4m3_scale_perCh",
        "fp8_e4m3_sat_perToken",
        "fp8_e4m3_scale_perToken",
        # fp8_e5m2
        "fp8_e5m2_sat",
        "fp8_e5m2_scale",
        "fp8_e5m2_sat_perCh",
        "fp8_e5m2_scale_perCh",
        "fp8_e5m2_sat_perToken",
        "fp8_e5m2_scale_perToken",
    ]
    bmm_mode_settings = [
        "pact",
        "pactsym",
        "pactsym+",
        "maxsym",
        "max",
        "minmax",
        "pertokenmax",
        "fp8_e4m3_sat",
        "fp8_e4m3_scale_perToken",
        "fp8_e5m2_sat",
        "fp8_e5m2_scale_perToken",
    ]

    # Get strings in config for qa_modes, qw_modes, bmm_modes
    qa_modes_str = [
        "qa_mode",
        "qa_qkv_mode",
    ]
    qw_modes_str = [
        "qw_mode",
        "qw_qkv_mode",
    ]
    bmm_modes_str = [
        "bmm1_qm1_mode",
        "bmm1_qm2_mode",
        "bmm2_qm1_mode",
        "bmm2_qm2_mode",
    ]

    # mx related modes for config:
    mx_spec_config_modes = [
        "mx_fp8_e5m2",
        "mx_fp8_e4m3",
        "mx_fp4_e2m1",
        "mx_fp4",
        "mx_int8",
        "mx_int4",
        "mx_fp16",
        "mx_float16",
        "mx_bf16",
        "mx_bfloat16",
    ]

    # Check each for correct ranges
    for qa_mode_str in qa_modes_str:
        qa_mode = config.get(qa_mode_str, "pact+")
        if not qa_mode in (qa_mode_settings + mx_spec_config_modes):
            raise ValueError(
                f"{qa_mode_str} = {qa_mode} is not set to one of the following: "
                f"{qa_mode_settings + mx_spec_config_modes}"
            )

    for qw_mode_str in qw_modes_str:
        qw_mode = config.get(qw_mode_str, "sawb+")
        if not qw_mode in (qw_mode_settings + mx_spec_config_modes):
            raise ValueError(
                f"{qw_mode_str} = {qw_mode} is not set to one of the following: "
                f"{qw_mode_settings + mx_spec_config_modes}"
            )

    bmm_mode_consistency = 0  # all or none when using mx
    for bmm_mode_str in bmm_modes_str:
        bmm_mode = config.get(bmm_mode_str, "pactsym+")
        bmm_mode_consistency += bmm_mode.startswith("mx_")
        # mx_specs doesn't have 4 individual bmmX_qmY_modes, it re-uses w and a fmt instead.
        # We will keep them in qcfg (with "mx_" prefix NOT removed).
        if not bmm_mode in (bmm_mode_settings + mx_spec_config_modes):
            raise ValueError(
                f"{bmm_mode_str} = {bmm_mode} is not set to one of the following: "
                f"{bmm_mode_settings + mx_spec_config_modes}"
            )
    if bmm_mode_consistency != 0 and bmm_mode_consistency != len(bmm_modes_str):
        raise ValueError("bmmX_qmY_modes inconsistent! Should be all mx or no mx.")

    # Check mode calibration and initialization values
    calib_init_settings = ["percentile", "pact", "sawb", "max"]
    calib_inits_str = [
        "qa_mode_calib",
        "qw_mode_calib",
        "a_init_method",
        "w_init_method",
    ]
    for calib_init_str in calib_inits_str:
        calib_init = config.get(calib_init_str, "max")
        if not calib_init in calib_init_settings:
            raise ValueError(
                f"{calib_init_str} = {calib_init} is not set to one of the following: "
                f"{calib_init_settings}"
            )

    # Check boolean values
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
        "qskip_large_mag_layers",
        "recompute_narrow_weights",
        "smoothq",
    ]
    for boolean_var_str in boolean_vars_str:
        boolean_var = config.get(
            boolean_var_str, False
        )  # assume default = False is not specified
        # Note: bool is a subclass of int, so we can't rely on isinstance
        # pylint: disable = unidiomatic-typecheck
        if type(boolean_var) is not bool:
            raise ValueError(f"{boolean_var_str} = {boolean_var} is not a boolean")

    default_config = config_defaults()

    # Check int values
    integer_vars_str = [
        "qmodel_calibration",
        "qmodel_calibration_new",
        "ptq_nbatch",
        "ptq_batchsize",
        "ptq_nouterloop",
        "ptq_ninnerloop",
    ]

    for integer_var_str in integer_vars_str:
        integer_var_default = default_config.get(integer_var_str)
        integer_var = config.get(integer_var_str, integer_var_default)
        # Check if integer was given as float (1.0 when it should be 1)
        if isinstance(integer_var, float) and integer_var.is_integer():
            config[integer_var_str] = int(integer_var)
            integer_var = int(integer_var)
        if not isinstance(integer_var, int):
            raise ValueError(f"{integer_var_str} = {integer_var} is not an integer")

    # Check fp values
    fp_vars_str = [
        "ptq_lrw",
        "ptq_lrcv_w",
        "ptq_lrcv_a",
    ]
    for fp_var_str in fp_vars_str:
        fp_var_default = default_config.get(fp_var_str)
        fp_var = config.get(fp_var_str, fp_var_default)
        # Check if float was given as an int (e.g. 1 when it should be 1.0)
        # NOTE: True/False qualifies as int.
        if isinstance(fp_var, int) and not isinstance(fp_var, bool):
            config[fp_var_str] = float(fp_var)
            fp_var = float(fp_var)
        if not isinstance(fp_var, float):
            raise ValueError(f"{fp_var_str} = {fp_var} is not a floating-point number")

    # Check iterable values
    iterable_vars_str = [
        "qskip_layer_name",
        "qspecial_layers",
        "qsinglesided_name",
        "ptqmod_to_be_optimized",
        "firstptqmodule",
        "params2optim",
        "clip_val_asst_percentile",
        "smoothq_scale_layers",
    ]
    for iterable_var_str in iterable_vars_str:
        iterable_var_default = default_config.get(iterable_var_str)
        iterable_var = config.get(iterable_var_str, iterable_var_default)
        if not hasattr(iterable_var, "__iter__"):
            raise ValueError(
                f"{iterable_var_str} = {iterable_var} is not an iterable object"
            )

    # Other values that require special settings

    # clip_val_asst is the percentile to use for calibration. TODO: consider renaming
    clip_val_asst_percentile_default = default_config.get("clip_val_asst_percentile")
    clip_val_asst_percentile = config.get(
        "clip_val_asst_percentile", clip_val_asst_percentile_default
    )
    if len(clip_val_asst_percentile) != 2:
        raise ValueError(
            f"clip_val_asst_percentile = {clip_val_asst_percentile} is not length 2"
        )
    val0 = clip_val_asst_percentile[0]
    val1 = clip_val_asst_percentile[1]

    # Check if either value is an int, when it should be a float (ie 1 when it should be 1.0)
    if isinstance(val0, int) and not isinstance(val0, bool):
        clip_val_asst_percentile[0] = float(val0)
        val0 = float(val0)
        config["clip_val_asst_percentile"] = clip_val_asst_percentile
    if isinstance(val1, int) and not isinstance(val1, bool):
        clip_val_asst_percentile[1] = float(val1)
        val1 = float(val1)
        config["clip_val_asst_percentile"] = clip_val_asst_percentile

    if not isinstance(val0, float):
        raise ValueError(
            f"clip_val_asst_percentile = {clip_val_asst_percentile} does not contain"
            " a float value at index 0"
        )

    if not isinstance(val1, float):
        raise ValueError(
            f"clip_val_asst_percentile = {clip_val_asst_percentile} "
            "does not contain a float value at index 1"
        )

    ptq_loss_func_settings = [
        "mse",
        "normalized_change",
        "ssim",
        "ssimlog",
        "ssimp0.2",
        "ssimp0.5",
        "ssimp2",
        "fisher_diag",
        "fisher_full",
        "adaround",
    ]
    ptq_loss_func = config.get("ptq_loss_func", "mse")
    if not ptq_loss_func in ptq_loss_func_settings:
        raise ValueError(
            f"ptq_loss_func = {ptq_loss_func} is not one of the following: "
            f"{ptq_loss_func_settings}"
        )

    ptq_coslr_settings = ["", "A", "W", "WA"]
    ptq_coslr = config.get("ptq_coslr", "")
    if not ptq_coslr in ptq_coslr_settings:
        raise ValueError(
            f"ptq_coslr = {ptq_coslr} is not one of the following: {ptq_coslr_settings}"
        )

    which2patch_contextmanager_settings = ["torch.bmm", "torch.matmul", None]
    which2patch_contextmanager = config.get("which2patch_contextmanager", None)
    if not which2patch_contextmanager in which2patch_contextmanager_settings:
        raise ValueError(
            f"which2patch_contextmanager = {which2patch_contextmanager} is not one of "
            f"the following: {which2patch_contextmanager_settings}"
        )

    smoothq_act_scale_path = config.get("smoothq_act_scale_path", None)
    if smoothq_act_scale_path and not smoothq_act_scale_path.endswith(".pt"):
        raise ValueError(f"{smoothq_act_scale_path=} is not a .pt checkpoint")

    # Check MX-related variables in mx_specs
    mx_specs = config.get("mx_specs", None)
    if mx_specs:
        # mx related modes for config:
        mx_spec_modes = [
            "fp8_e5m2",
            "fp8_e4m3",
            "fp4_e2m1",
            "fp4",
            "int8",
            "int4",
            "fp16",
            "float16",
            "bf16",
            "bfloat16",
        ]

        mx_specs_format_var_strs = {
            "w_elem_format",
            "a_elem_format",
            "w_elem_format_bp",
            "a_elem_format_bp",
            "a_elem_format_bp_ex",
            "a_elem_format_bp_os",
        }

        for format_var_str in mx_specs_format_var_strs:
            format_var = mx_specs[format_var_str]
            if not isinstance(format_var, str):
                raise ValueError(
                    f"mx_specs[{format_var_str}] = {format_var} is not a string"
                )
            if format_var not in mx_spec_modes:
                raise ValueError(
                    f"mx_specs[{format_var_str}] = {format_var} is not in one of the following: "
                    f"{mx_spec_modes}"
                )

        mx_spec_int_var_str_defaults = [
            ("scale_bits", 8),
            ("block_size", 32),
            ("bfloat", 16),
        ]
        mx_spec_int_var_values = {2, 4, 6, 8, 16, 32}

        for integer_var_str, integer_var_default in mx_spec_int_var_str_defaults:
            integer_var = mx_specs.get(integer_var_str, integer_var_default)
            # Check if integer was given as float (1.0 when it should be 1)
            if isinstance(integer_var, float) and integer_var.is_integer():
                mx_specs[integer_var_str] = int(integer_var)
                integer_var = int(integer_var)
            if not isinstance(integer_var, int):
                raise ValueError(
                    f"mx_specs[{integer_var_str}] = {integer_var} is not an integer"
                )
            if integer_var not in mx_spec_int_var_values:
                raise ValueError(
                    f"mx_specs[{integer_var_str}] = {integer_var} must be an integer in "
                    f"{mx_spec_int_var_values}"
                )

        mx_spec_bool_var_strs = {
            "mx_flush_fp32_subnorms",
            "bfloat_subnorms",
            "quantize_backprop",
            "softmax_exp2",
            "vec_use_exp2",
            "vec_use_recip",
            "custom_cuda",
        }
        for boolean_var_str in mx_spec_bool_var_strs:
            # assume default = False is not specified
            boolean_var = mx_specs.get(boolean_var_str, False)
            # Note: bool is a subclass of int, so we can't rely on isinstance
            # pylint: disable = unidiomatic-typecheck
            if type(boolean_var) is not bool:
                raise ValueError(
                    f"mx_specs[{boolean_var_str}] = {boolean_var} is not a boolean"
                )

        mx_spec_exp_var_strs = {
            "shared_exp_method",
        }
        mx_spec_exp_var_values = {"max", None}
        for exp_var_str in mx_spec_exp_var_strs:
            exp_var = mx_specs.get(exp_var_str, "max")
            if not isinstance(exp_var, str):
                raise ValueError(f"mx_specs[{exp_var_str}] = {exp_var} is not a string")
            if exp_var not in mx_spec_exp_var_values:
                raise ValueError(
                    f"mx_specs[{exp_var_str}] = {exp_var} is not in "
                    f"{mx_spec_exp_var_values}"
                )

        mx_spec_round_var_strs = {
            "round",
            "round_m",
            "round_weight",
            "round_output",
            "round_grad_weight",
            "round_grad_input",
            "round_mx_output",
            "round_mx_input_grad_input",
            "round_mx_weight_grad_input",
            "round_mx_grad_output_grad_input",
            "round_mx_input_grad_weight",
            "round_mx_grad_output_grad_weight",
        }
        mx_spec_round_var_values = {"nearest", "floor"}
        for round_var_str in mx_spec_round_var_strs:
            round_var = mx_specs.get(round_var_str, "nearest")
            if not isinstance(round_var, str):
                raise ValueError(
                    f"mx_specs[{round_var_str}] = {round_var} is not a string"
                )
            if round_var not in mx_spec_round_var_values:
                raise ValueError(
                    f"mx_specs[{round_var_str}] = {round_var} is not in"
                    f"{mx_spec_round_var_values}"
                )

        # If mapping is defined, check for MX  classes
        if available_packages["mx"]:
            # Local
            from fms_mo.modules.bmm import QBmmMX
            from fms_mo.modules.linear import QLinearMX

            mapping = config.get("mapping", None)

            # partial was used to wrap QLinearMX, will be an instance of partial
            # 1. can use .func pointer to find the original class
            # 2. QBmm is optional, could be partial(QBmmMX,) or QBmm
            if mapping is not None:
                if mapping[nn.Linear].func is not QLinearMX:
                    raise ValueError("MX mapping for nn.Linear is not QLinearMX")

                qbmm_map = mapping["matmul_or_bmm"]
                if bmm_mode_consistency > 0:
                    if getattr(qbmm_map, "func", None) is not QBmmMX:
                        raise ValueError("MX mapping for matmul_or_bmm is not QBmmMX")
                else:
                    if qbmm_map is not QBmm:
                        raise ValueError("Mapping for matmul_or_bmm is not QBmm")
