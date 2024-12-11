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
import json
import logging
import os
import warnings
from pathlib import Path
from typing import Any

# Third Party
import torch
from torch import nn

# First Party
from fms_mo.modules import QLSTM, QConv2d, QConvTranspose2d, QLinear

# import numpy as np # only used in experimental func


logger = logging.getLogger(__name__)


def config_defaults():
    """Create defaults for qconfig"""
    cfg_defaults = [
        # nbits vars
        ("nbits_a", 32),
        ("nbits_w", 32),
        ("nbits_a_alt", None),
        ("nbits_w_alt", None),
        ("nbits_a_qkv", None),
        ("nbits_w_qkv", None),
        ("nbits_bmm1", None),
        ("nbits_bmm2", None),
        ("nbits_kvcache", None),
        # qmodes vars
        ("qa_mode", "pact"),
        ("qw_mode", "sawb"),
        ("qa_qkv_mode", "pact"),
        ("qw_qkv_mode", "sawb"),
        ("bmm1_qm1_mode", "pact"),
        ("bmm1_qm2_mode", "pact"),
        ("bmm2_qm1_mode", "pact"),
        ("bmm1_qm2_mode", "pact"),
        # mode_calib vars
        ("qa_mode_calib", "percentile"),
        ("qw_mode_calib", "percentile"),
        # init_method vars
        ("a_init_method", "percentile"),
        ("w_init_method", "sawb"),
        # qmodel_calibration
        ("qmodel_calibration", 0),
        ("qmodel_calibration_new", 0),
        # Boolean vars
        ("qshortcutconv", False),
        ("q1stlastconv", False),
        ("qdw", False),
        ("qskipfpn", False),
        ("qkvsync", False),
        ("extend_act_range", False),
        ("plotsvg", False),
        # Iterable vars
        ("qskip_layer_name", []),
        ("qspecial_layers", {}),
        ("qsinglesided_name", []),
        ("clip_val_asst_percentile", [0.1, 99.9]),
        (
            "params2optim",
            {
                "W": [[] for _ in range(torch.cuda.device_count())],
                "cvs": [[] for _ in range(torch.cuda.device_count())],
            },
        ),
        # PTQ vars
        ("ptq_nbatch", 100),
        ("ptq_batchsize", 12),
        ("ptq_nouterloop", 20000),
        ("ptq_ninnerloop", 1),
        ("ptq_coslr", ""),
        ("ptq_lrw", 1e-05),
        ("ptq_lrcv_a", 0.001),
        ("ptq_lrcv_w", 0.001),
        ("ptq_freezecvs", False),
        ("ptq_qdrop", False),
        ("ptq_loss_func", "mse"),
        ("firstptqmodule", []),
        # Other vars
        ("which2patch_contextmanager", None),
    ]

    return cfg_defaults


def qconfig_init(recipe: str = None, args: Any = None):
    """Three possible ways to create qcfg:
    1. create a default qcfg
    2. load from a json
    3. parse the args
    NOTE: Content from higher number, e.g. arg parser, will override thier counterpart from lower
            numbers, e.g. json.

    Args:
    recipe: str. Recipe filename (json) that contains settings, if specified and exists. Will search
            cwd and fms_mo/recipes folder. ok to omit '.json' extension.
    args: argparser object that may contain relavant parameters.

    Important items in the config dictionary:
    nbits_[w|a]_alt: "_alt" stands for "alternative" -> the default prec for those "skipped" layers
                        e.g. usually the 1st/last layers are "skipped" and will NOT be swapped to
                        QLinear. But, if "nbits_x_alt = 8", they will.
    qmodel_calibration[_new]: set to non-zero will trigger calibration. "_new" means calibration
                                will happen during the first N calls of fwd path, better for long
                                training or fine-tuning that you don't mind losing the first N iters

    qlayer_name_pattern: allows partial or regex name matching, the layers satisfy the criteria will
                        be skipped. NOTE: tracing will be bypassed entirely if this arg is used
    qskip_layer_name: user can specify exact name to skip
    qspecial_layers: special case handling. user can specify any quant params for any given layer,
                     e.g. {'1st.conv':{'nbits_w':8,'qw_mode':'pact+sym'}, '2nd.layers':{...} }

    extend_act_range: symmetric act quantizers (maxsym, pactsym+, ...) to use full range, e.g.,
                      [-128, 127] instead [-127,127], TODO: should default to True?

    ptq_nbatch: total number of batches of data that will be fetched from loader for PTQ tuning
    ptq_batchsize: data used in PTQ tuning usually is fetched from loader directly, i.e. batchsize
                    is the unchanged from dataloader.batch_size. although it could be different if
                    needed, e.g. PTQ may allow larger bs due to only partial model tuning. But fine-
                    grain shuffling will be needed in that case.
    ptq_nouterloop: number of optimization "steps" in the PTQ outer loop. 1 outer loop uses 1 cached
                    data batch. when Nouter >= Nbatch, data will be re-used
    ptq_ninnerloop: number of "inner loop" for PTQ optimization. When 1 batch of data is fetched,
                    run (loss->loss.back->optim.step) this many times before fetching the next batch
                    NOTE: usually doesn't make big differences, hence, default to 1
    ptq_coslr: can be "", "W" or "A" or "WA", indicating which (or both) optimizer will use cosLR,
                otherwise use constantLR as default
    """

    qcfg = {}
    # 1. create a dict with default values
    qcfg["mapping"] = {
        nn.Conv2d: {"from": nn.Conv2d, "to": QConv2d, "otherwise": QConv2d},
        nn.ConvTranspose2d: {
            "from": nn.ConvTranspose2d,
            "to": QConvTranspose2d,
            "otherwise": QConvTranspose2d,
        },
        nn.Linear: {"from": nn.Linear, "to": QLinear, "otherwise": QLinear},
        nn.LSTM: {"from": nn.LSTM, "to": QLSTM, "otherwise": QLSTM},
    }
    # TODO: This could be further simplified. e.g. mapping["from class"] = "to class"
    #       "otherwise" is rarely used, and redundant "from" in the output dict

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
        cwd = Path().resolve()
        pkg_root = Path(__file__).parent.parent.resolve()
        file_in_cwd = cwd / recipe
        file_in_recipes = pkg_root / "recipes" / recipe
        file_in_recipes2 = pkg_root / "recipes" / f"{recipe}.json"
        temp_cfg = None

        if not recipe.endswith(".json") and file_in_recipes2.exists():
            qcfg_json = file_in_recipes2
        elif file_in_cwd.exists():
            qcfg_json = file_in_cwd
        elif file_in_recipes.exists():
            qcfg_json = file_in_recipes
        else:
            qcfg_json = None

        if qcfg_json:
            with open(qcfg_json, "r", encoding="utf-8") as openfile:
                temp_cfg = json.load(openfile)
            qcfg.update(temp_cfg)
            logger.info(
                f"Loaded settings from {qcfg_json} and updated the default values."
            )

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

    return qcfg


def has_non_serializable_object(anything):
    """
    Generalized recursive function looking for any non-serializable Python object
    Only types that are JSON serializable are None, primatives, tuples, lists, and dicts.
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


def serialize_config(config):
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


def remove_unwanted_from_config(config):
    """Remove deprecated items or things cannot be saved as text (json)"""
    unwanted_items = [
        "sweep_cv_percentile",
        "Qlist",
        "tb_writer",
        "mapping",
        "checkQerr_frequency",
        "newlySwappedModules",
        "force_calib_once",
        # if we keep the follwing LUTs, it will save the entire model
        "LUTmodule_name",
        "qkvsync_my_1st_sibling",
        "graph_in_out",
    ]
    len_before = len(config)
    dump = {k: config.pop(k) for k in unwanted_items if k in config}
    assert (
        len(config) + len(dump) == len_before
    ), "Inconsistency in config. Please check."
    return config, dump


def get_unwanted_defaults():
    """Add back those unserializable items if needed"""
    unwanted_items = [
        ("sweep_cv_percentile", False),
        ("tb_writer", None),
        (
            "mapping",
            {
                nn.Conv2d: {"from": nn.Conv2d, "to": QConv2d, "otherwise": QConv2d},
                nn.ConvTranspose2d: {
                    "from": nn.ConvTranspose2d,
                    "to": QConvTranspose2d,
                    "otherwise": QConvTranspose2d,
                },
                nn.Linear: {"from": nn.Linear, "to": QLinear, "otherwise": QLinear},
                nn.LSTM: {"from": nn.LSTM, "to": QLSTM, "otherwise": QLSTM},
            },
        ),
        ("checkQerr_frequency", False),
        ("newlySwappedModules", []),
        ("force_calib_once", False),
        # if we keep the follwing LUTs, it will save the entire model
        ("LUTmodule_name", {}),
    ]
    return unwanted_items


def add_required_defaults_to_config(config):
    """Recover "unserializable" items that are previously removed from config"""
    unwanted_items = get_unwanted_defaults()
    for key, default_val in unwanted_items:
        if key not in config:
            config[key] = default_val


def add_wanted_defaults_to_config(config):
    """Util function to add basic config defaults that are missing into a config
    if a wanted item is not in the config, add it w/ default value
    """
    wanted_items = config_defaults()
    for wanted_name, wanted_default_val in wanted_items:
        if wanted_name not in config:
            config[wanted_name] = wanted_default_val


def qconfig_save(qcfg, fname="qcfg.json"):
    """
    Try to save qcfg into a JSON file (or use .pt format if something really can't be text-only).
    For example, qcfg['mapping'] has some classes as keys and values, json won't work. We will try
    to remove unserializable items first.
    """

    # Remove deprecated/unwanted key,vals in config
    temp_qcfg, removed_items = remove_unwanted_from_config(qcfg)

    # Add back wanted defaults for any missing vars
    add_wanted_defaults_to_config(temp_qcfg)

    # Clean config of any unwanted key,vals not found in unwanted list
    temp_qcfg, removed_items2 = serialize_config(temp_qcfg)

    # Finally, check to ensure all values are valid before saving
    check_config(temp_qcfg)

    # Save config as json
    if os.path.isfile(fname):
        message = f"{fname} already exist, will overwrite."
        warnings.warn(message, UserWarning)
    with open(fname, "w", encoding="utf-8") as outfile:
        json.dump(temp_qcfg, outfile, indent=4)

    # restore original qcfg
    qcfg.update(removed_items)
    qcfg.update(removed_items2)


def qconfig_load(fname="qcfg.json"):
    """Read config in json format, work together with qconfig_save"""
    if os.path.isfile(fname):
        with open(fname, "r", encoding="utf-8") as openfile:
            config = json.load(openfile)

        # Add back wanted defaults for any missing vars
        add_wanted_defaults_to_config(config)
        add_required_defaults_to_config(config)

        # Ensure config has correct values before continuing
        check_config(config)

        return config

    logger.info(f"{fname} doesn't exist. cannot load the qcfg")


def check_config(config, model_dtype=None):
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

    # Check each for correct ranges
    for qa_mode_str in qa_modes_str:
        qa_mode = config.get(qa_mode_str, "pact+")
        if not qa_mode in qa_mode_settings:
            raise ValueError(
                f"{qa_mode_str} = {qa_mode} is not set to one of the following: "
                f"{qa_mode_settings}"
            )

    for qw_mode_str in qw_modes_str:
        qw_mode = config.get(qw_mode_str, "sawb+")
        if not qw_mode in qw_mode_settings:
            raise ValueError(
                f"{qw_mode_str} = {qw_mode} is not set to one of the following: "
                f"{qw_mode_settings}"
            )

    for bmm_mode_str in bmm_modes_str:
        bmm_mode = config.get(bmm_mode_str, "pactsym+")
        if not bmm_mode in bmm_mode_settings:
            raise ValueError(
                f"{bmm_mode_str} = {bmm_mode} is not set to one of the following: "
                f"{bmm_mode_settings}"
            )

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
    ]
    for boolean_var_str in boolean_vars_str:
        boolean_var = config.get(
            boolean_var_str, False
        )  # assume default = False is not specified
        if not isinstance(boolean_var, bool):
            raise ValueError(f"{boolean_var_str} = {boolean_var} is not a boolean")

    # Check int values
    integer_vars_str_default = [
        ("qmodel_calibration", 0),
        ("qmodel_calibration_new", 0),
        ("ptq_nbatch", 100),
        ("ptq_batchsize", 12),
        ("ptq_nouterloop", 20000),
        ("ptq_ninnerloop", 1),
    ]
    for integer_var_str, integer_var_default in integer_vars_str_default:
        integer_var = config.get(integer_var_str, integer_var_default)
        # Check if integer was given as float (1.0 when it should be 1)
        if isinstance(integer_var, float) and integer_var.is_integer():
            config[integer_var] = int(integer_var)
            fp_var = int(integer_var)
        if not isinstance(integer_var, int):
            raise ValueError(f"{integer_var_str} = {integer_var} is not an integer")

    # Check fp values
    fp_vars_str_default = [
        ("ptq_lrw", 1e-5),
        ("ptq_lrcv_w", 0.001),
        ("ptq_lrcv_a", 0.001),
    ]
    for fp_var_str, fp_var_default in fp_vars_str_default:
        fp_var = config.get(fp_var_str, fp_var_default)
        # Check if float was given as an int (e.g. 1 when it should be 1.0)
        # NOTE: True/False qualifies as int.
        if isinstance(fp_var, int) and not isinstance(fp_var, bool):
            config[fp_var_str] = float(fp_var)
            fp_var = float(fp_var)
        if not isinstance(fp_var, float):
            raise ValueError(f"{fp_var_str} = {fp_var} is not a floating-point number")

    # Check iterable values
    iterable_vars_str_default = [
        ("qskip_layer_name", ["pooler.dense"]),
        ("qspecial_layers", {}),
        ("qsinglesided_name", []),
        ("ptqmod_to_be_optimized", []),
        ("firstptqmodule", []),
        ("params2optim", {"W": [[]], "cvs": [[]]}),
        ("clip_val_asst_percentile", [0.1, 99.9]),
    ]
    for iterable_var_str, iterable_var_default in iterable_vars_str_default:
        iterable_var = config.get(iterable_var_str, iterable_var_default)
        if not hasattr(iterable_var, "__iter__"):
            raise ValueError(
                f"{iterable_var_str} = {iterable_var} is not an iterable object"
            )

    # Other values that require special settings

    # clip_val_asst is the percentile to use for calibration. TODO: consider renaming
    clip_val_asst_percentile = config[
        "clip_val_asst_percentile"
    ]  # already given default in iterable_var
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
