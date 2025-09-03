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
"""Main user interfacing functions, such as qmodel_prep()"""

# Standard
from pathlib import Path
import gc
import logging
import sys
import warnings

# Third Party
from torch import nn
import torch

# Local
from fms_mo.calib import qmodel_calib
from fms_mo.modules import QBmm_modules, QConv2d_modules, QLinear_modules, QLSTM_modules
from fms_mo.quant.quantizers import Qbypass
from fms_mo.utils.import_utils import available_packages
from fms_mo.utils.qconfig_utils import check_config, qconfig_save, set_mx_specs
from fms_mo.utils.utils import prepare_inputs

# import numpy as np # only used in experimental func


logger = logging.getLogger(__name__)


def update_optim_param_groups(net, optim, qcfg):
    """Step:
    0) confirm existing parameters in optimizer are all from net.named_params()
    1) collect clip_val parameters
    2) if they exist in any existing parameter_groups, remove them
    3) add new clipval_a (and clipval_w) group

    Args:
        net (nn.Modules): parameters in this model are going to be checked against optimizer
        optim (Any): optimizer to be analyzed and updated
        qcfg (dict): quantization config

    Returns:
        optimizer: updated optimizer with quant related parameters collected into a group.
    """
    setAllParams = {v for _, v in net.named_parameters()}  # this is "set comprehension"
    params_not_in_model = []
    for g in optim.param_groups:
        params_not_in_model += list(set(g["params"]) - setAllParams)
    if len(params_not_in_model) != 0:
        logger.warning(
            "Inconsistency between optimizer and model! \n"
            f"{len(params_not_in_model)} parameters found in optimizer but not in model "
        )

    clip_val_group = {
        "params": [p for n, p in net.named_parameters() if "clip_val" in n]
    }
    param_a, param_a_name = [], []
    param_w, param_w_name = [], []
    for name, param in net.named_parameters():
        if ".clip_val" in name:
            if "quantize_weight" in name:
                param_w.append(param)
                param_w_name.append(name)
            elif any(
                k in name
                for k in [
                    "quantize_feature",
                    "quantize_input",
                    "quantize_hidden",
                    "quantize_m",
                ]
            ):
                param_a.append(param)
                param_a_name.append(name)
    cv_group_w = {
        "params": param_w
    }  # for W clipvals, (if W407, may need to set LR differently)
    cv_group_a = {"params": param_a}  # for A clipvals (both clipval and clipvaln)

    setCVG = set(clip_val_group["params"])
    setCVGnames = set(param_a_name + param_w_name)
    Nparam_removed = 0
    pg2del = []
    nameChk = ["name", "names"]
    # NOTE: special case for EffecientDet, which has a key 'name' in the param_groups dict. doesn't
    #       seem useful and most other nets don't have it. if not used, do not add it to the dict.
    keyGroupName, keyIdvName = None, None
    for idx, g in enumerate(optim.param_groups):
        set_params = set(g["params"])
        for n in nameChk:
            if n in g:
                if isinstance(g[n], list):
                    # A list of param names. make sure user didn't pick "name" over "names"
                    keyIdvName = n
                elif isinstance(g[n], str):
                    # if not a list -> meant "group name". No need to update
                    keyGroupName = n

        if not setCVG.isdisjoint(set_params):
            inters_params = setCVG.intersection(set_params)
            new_set_params = set_params - setCVG
            logger.info(
                f"In param_group[{idx}], found {len(inters_params)} duplicate parameters"
            )
            if len(new_set_params) == 0:
                pg2del.append(idx)
            else:
                g["params"] = list(new_set_params)
                if keyIdvName:
                    g[keyIdvName] = list(set(g[keyIdvName]) - setCVGnames)

            Nparam_removed += len(inters_params)
    if len(pg2del) > 0:
        logger.info(
            f"Deleteing {len(pg2del)} param_group(s), "
            f"optimizer originally has {len(optim.param_groups)}"
        )
        pg2del.sort(reverse=True)
        # index of the group, delete from largest to smallest, otherwise will delete incorrect grps
        for pgi in pg2del:
            del optim.param_groups[pgi]
        logger.info(f"After deleteing, optimizer now has {len(optim.param_groups)}")
    logger.info(
        f"Total duplicate parameters removed {Nparam_removed}, clip_val_group has {len(setCVG)}"
    )

    cv_group_a["weight_decay"] = qcfg.get("pact_a_decay", 0.0)
    cv_group_w["weight_decay"] = qcfg.get("pact_w_decay", cv_group_a["weight_decay"])
    # NOTE: if pact decay for weight is not specified, use pact decay a might be ok
    param_names = [
        "lr",
        "momentum",
        "nesterov",
    ]
    str_a = f"decay={cv_group_a['weight_decay']}"
    str_w = f"decay={cv_group_w['weight_decay']}"
    for pn_i in param_names:
        if f"pact_a_{pn_i}" in qcfg:
            cv_group_a[pn_i] = qcfg[f"pact_a_{pn_i}"]
            str_a += f", {pn_i}={cv_group_a[pn_i]}"
        if f"pact_w_{pn_i}" in qcfg:
            cv_group_w[pn_i] = qcfg[f"pact_w_{pn_i}"]
            str_w += f", {pn_i}={cv_group_w[pn_i]}"

    if cv_group_a["params"]:
        # if any param is added to this dict, add a new group to optim
        if keyIdvName:
            cv_group_a[keyIdvName] = param_a_name
        if keyGroupName:
            cv_group_a[keyGroupName] = "clip_val_a"
        optim.add_param_group(cv_group_a)
        logger.info(f"New clipval_activation group added to optimizer: {str_a}")
    if cv_group_w["params"]:
        if keyIdvName:
            cv_group_w[keyIdvName] = param_w_name
        if keyGroupName:
            cv_group_a[keyGroupName] = "clip_val_w"
        optim.add_param_group(cv_group_w)
        logger.info(f"New clipval_weight group added to optimizer: {str_w}")

    return optim


def make_quant_module(module, curr_full_name, qcfg, verbose=False):
    """Create a Qmodule based on the provided nn.Module, e.g. nn.Linear -> QLinear. If input module
    is mappable, create a Qmodule and return, otherwise, return the original module. In the future,
    Qmodules need to have a .from_torch() or .from_nn() classmethod, and then this function will be
    greatly simplified.
    NOTE:
    1. This func will check qskip_layer_name before creating the Qmodule
    2. Qmodule will be created on "meta device" as a placeholder, which will skip params init and
        mem alloc, as weights and bias will be reassigned to module.weight/.bias right after

    Args:
        module (nn.Module): the module which Qmodule will be based on
        curr_full_name (str): derived from model.named_modules()
        qcfg (dict): quant_config
        verbose (bool, optional): whether to print details. Defaults to False.

    Returns:
        nn.Module: quantized module
    """
    mapping = qcfg.get("mapping")
    mappable_classes = [cls for cls in mapping.keys() if not isinstance(cls, str)]
    # if mapping is not defined, qmodel_prep should raise alarm before entering QAnyNet4
    qdw = qcfg.get("qdw", False)
    nbits_a = qcfg.get("nbits_a", 32)
    nbits_w = qcfg.get("nbits_w", 32)
    qa_mode = qcfg.get("qa_mode", "pact+")
    qw_mode = qcfg.get("qw_mode", "sawb+")

    # Check if MX has been set outside of qconfig_init without mx_specs being created
    if (
        available_packages["mx"]
        and "mx_specs" not in qcfg
        and (
            (qcfg["qa_mode"].startswith("mx_") and qcfg["qw_mode"].startswith("mx_"))
            or any(key.startswith("mx_") for key in qcfg.keys())
        )
    ):
        set_mx_specs(qcfg, use_mx=True)

    # check if on "black list" (need to be exact match), can be skipped or quantized those
    # to slightly higher "default" precision, or use qspecial_layers to have fine control
    if curr_full_name in qcfg["qskip_layer_name"]:
        nbits_a = qcfg.get("nbits_a_alt", None)
        nbits_w = qcfg.get("nbits_w_alt", None)
    if curr_full_name in qcfg["qspecial_layers"]:
        # this is for special case handling for any layers as user wants, eq qsilu
        # e.g. {'1st.conv':{'nbits_w':8,'qw_mode':'pact+sym'}, 'other.layers':{...} }
        qdict = qcfg["qspecial_layers"][curr_full_name]
        nbits_a = qdict.get("nbits_a", nbits_a)
        nbits_w = qdict.get("nbits_w", nbits_w)
        qa_mode = qdict.get("qa_mode", qa_mode)
        qw_mode = qdict.get("qw_mode", qw_mode)
        # NOTE: if any item is not defined, use current default

    if isinstance(module, tuple(mappable_classes)):
        base_params = {}
        if hasattr(module, "__constants__"):
            base_params = {k: getattr(module, k) for k in module.__constants__}
            base_params["bias"] = module.bias is not None
        base_params["device"] = "meta"

    module_output = module

    # If (W,A) is (32,8) or (8,32), one nbits = None ; Do not quantize in this case
    if nbits_a is None or nbits_w is None:
        if verbose:
            logger.info(
                f"Skip quantization of {curr_full_name} - nbits_a or nbits_w is None"
            )
        return module_output

    # For nn.Conv2d
    if isinstance(module, nn.Conv2d):
        if (
            module.__class__ != nn.Conv2d
        ):  # NOTE: isinstance() gives True for subclasses, too.
            logger.warning(
                f"{curr_full_name} {type(module)} seems to be a wrapper of Conv2d."
                "Please make sure it doesn't wrap BN and activation func."
                "Otherwise please create an equivalen QConv wrapper and change qcfg['mapping']."
            )

        QConv = mapping.get(nn.Conv2d, None)
        if QConv is None:
            if verbose:
                logger.info(
                    f"Skip quantization of {curr_full_name} - mapping of Conv2d is None"
                )
            return module_output  # None means no swap for this type

        base_params.pop(
            "output_padding"
        )  # will cause error if send this param to nn.Conv2d..
        if base_params["padding_mode"] != "zeros":
            logger.warning(
                f"{curr_full_name} padding mode = {base_params['padding_mode_i']}"
            )
        isDW = (module.in_channels == module.out_channels) and (
            module.groups == module.in_channels
        )
        # detect extra attributes for 'customized modules', add them to base_params and pass to
        # QConv altogether, make sure QConv has **kwargs so that it can receive/handle/ignore them
        typicalAttr = (
            dir(nn.Conv2d)
            + list(base_params.keys())
            + ["output_padding", "weight", "bias"]
        )
        extraAttr = [
            k for k in dir(module) if not k.startswith("_") and k not in typicalAttr
        ]
        for k in extraAttr:
            base_params[k] = getattr(module, k, None)

        # optional skip for DW layers
        if isDW and not qdw:
            nbits_a = qcfg.get("nbits_a_alt", None)
            nbits_w = qcfg.get("nbits_w_alt", None)

        # if either nbits_x_alt is None -> skip, otherwise, use alternative precision for this layer
        # e.g. 1st layer (W,A) can be (32,8)
        if nbits_a is None or nbits_w is None:
            if verbose:
                logger.info(f"skip quantization of {curr_full_name}")
        else:
            module_output = QConv(
                **base_params,
                # add new, necessary parameters below, try to utilize qcfg as much as u can,
                # e.g. clipvals... in the future, rely on Qmodule.from_torch()
                num_bits_feature=nbits_a,
                qa_mode=qa_mode,
                num_bits_weight=nbits_w,
                qw_mode=qw_mode,
                qcfg=qcfg,
                non_neg=(curr_full_name in qcfg["qsinglesided_name"]),
            )  # see findSingleSidedConv() for more details
            module_output.weight = (
                module.weight
            )  # don't forget to copy w and b tensors to the new module
            if base_params["bias"] is True:
                module_output.bias = module.bias

            # when adaR quantizer is init'ed, W has not been copied, need to call init delta here
            if "adaround" in qw_mode and nbits_w != 32:
                module_output.quantize_weight.to(module.weight.device)
                if "SAWB" not in qw_mode:
                    module_output.quantize_weight.init_delta(
                        module.weight, qw_mode, curr_full_name
                    )
                module_output.quantize_weight.init_alpha(module.weight)
                module_output.quantize_weight.soft_targets = True

    # For nn.ConvTranspose2d, basically the same as QConv
    elif isinstance(module, nn.ConvTranspose2d):
        if module.__class__ != nn.ConvTranspose2d:
            logger.warning(
                f"{curr_full_name} {type(module)} seems to be a wrapper of ConvTranspose2d."
                "Please make sure it doesn't wrap BN and activ func."
                "Otherwise please create an equivalen QConvT wrapper and change qcfg['mapping']."
            )

        QConvT = mapping.get(nn.ConvTranspose2d, None)
        if QConvT is None:
            if verbose:
                logger.info(
                    f"Skip quantization of {curr_full_name} - mapping of ConvTranspose2d is None"
                )
            return module_output  # None means no swap for this type

        if base_params["padding_mode"] != "zeros":
            logger.warning(
                f"{curr_full_name} padding mode = {base_params['padding_mode_i']}"
            )
        isDW = (module.in_channels == module.out_channels) and (
            module.groups == module.in_channels
        )
        # detect extra attributes for 'customized modules', add them to base_params and then pass to
        # QConv altogether. make sure QConv has **kwargs so that it can receive/handle/ignore them
        typicalAttr = (
            dir(nn.ConvTranspose2d)
            + list(base_params.keys())
            + ["output_padding", "weight", "bias"]
        )
        extraAttr = [
            k for k in dir(module) if not k.startswith("_") and k not in typicalAttr
        ]
        for k in extraAttr:
            base_params[k] = getattr(module, k, None)

        # optional skip for DW layers
        if isDW and not qdw:
            nbits_a = qcfg.get("nbits_a_alt", None)
            nbits_w = qcfg.get("nbits_w_alt", None)

        # if either nbits_x_alt is None -> skip, otherwise, use alternative precision for this layer
        # e.g. 1st layer (W,A) can be (32,8)
        if nbits_a is None or nbits_w is None:
            if verbose:
                logger.info(f"Skip quantization of {curr_full_name}")
        else:
            module_output = QConvT(
                **base_params,
                # add new, necessary parameters below, try to utilize qcfg as much as we can,
                # but in the future, Qmodules should have .from_torch() or .from_nn() method
                num_bits_feature=nbits_a,
                qa_mode=qa_mode,
                num_bits_weight=nbits_w,
                qw_mode=qw_mode,
                qcfg=qcfg,
                non_neg=(curr_full_name in qcfg["qsinglesided_name"]),
            )  # see findSingleSidedConv() for more details
            module_output.weight = module.weight
            # don't forget to copy w and b tensors to the new module
            if base_params["bias"] is True:
                module_output.bias = module.bias

    # For nn.Linear
    elif isinstance(module, nn.Linear):
        if module.__class__ != nn.Linear:
            if available_packages["compressed_tensors"]:
                # Third Party
                import compressed_tensors

            if isinstance(
                module, compressed_tensors.linear.compressed_linear.CompressedLinear
            ):
                pass
            else:
                logger.warning(
                    f"{curr_full_name} {type(module)} seems to be a wrapper of Linear."
                    "Please make sure it doesn't wrap BN and activ func. Otherwise"
                    "please create an equivalent Linear wrapper and change qcfg['mapping']."
                    )
        QLin = mapping.get(nn.Linear, None)
        if QLin is None:
            if verbose:
                logger.info(
                    f"Skip quantization of {curr_full_name} - mapping of Linear is None"
                )
            return module_output  # None means no swap for this type

        module_output = QLin(
            **base_params,
            num_bits_feature=nbits_a,
            qa_mode=qa_mode,
            num_bits_weight=nbits_w,
            qw_mode=qw_mode,
            qcfg=qcfg,
            non_neg=(curr_full_name in qcfg["qsinglesided_name"]),
        )
        module_output.weight = module.weight
        if base_params["bias"] is True:
            module_output.bias = module.bias
        # double check if there's any extra parameters in the old module, copied over if any
        new_params = dict(module_output.named_parameters())
        for n, p in module.named_parameters():
            if n not in new_params:
                module_output.register_parameter(n, p)
        new_buffs = dict(module_output.named_buffers())
        for n, b in module.named_buffers():
            if n not in new_buffs:
                module_output.register_buffer(n, b)

        # when adaR quantizer is init'ed, W was not copied, hence need to init delta here
        if "adaround" in qw_mode and nbits_w != 32:
            module_output.quantize_weight.to(module.weight.device)
            if "SAWB" not in qw_mode:
                module_output.quantize_weight.init_delta(
                    module.weight, qw_mode, curr_full_name
                )
            module_output.quantize_weight.init_alpha(module.weight)
            module_output.quantize_weight.soft_targets = True

        if qcfg["qkvsync"] and qcfg["qkvsync_my_1st_sibling"].get(module, None):
            Qmod_1st_sib = qcfg["qkvsync_my_1st_sibling"].get(module)
            if Qmod_1st_sib.__class__ == nn.Linear:
                # meaning first time run into this group (could be Q, K, or V), because
                # qcfg["qkvsync_my_1st_sibling"]'s "value" is not updated to Qlinear yet
                #  -> update the LUT's values
                qcfg["qkvsync_my_1st_sibling"].update(
                    {
                        k: module_output
                        for k, v in qcfg["qkvsync_my_1st_sibling"].items()
                        if v is module
                    }
                )
            else:
                Qattrs = [
                    attrb
                    for attrb in dir(Qmod_1st_sib)
                    if "quantize_" in attrb
                    and "calib_" not in attrb
                    and "_weight" not in attrb
                ]  # this covers quantize_features, _m1, _m2, don't sync _weights,
                for Qattr in Qattrs:
                    Quantizer_to_sync_to = getattr(Qmod_1st_sib, Qattr)
                    Quantizer_to_be_sync = getattr(module_output, Qattr)
                    if isinstance(Quantizer_to_sync_to, Qbypass):
                        continue  # skip sync'ing 32bit quantizers
                    cv = getattr(Quantizer_to_sync_to, "clip_val")
                    cvn = getattr(
                        Quantizer_to_sync_to, "clip_valn", None
                    )  # cv must exist, but cvn may not, hence default None

                    setattr(Quantizer_to_be_sync, "clip_val", cv)
                    if cvn:
                        setattr(Quantizer_to_be_sync, "clip_valn", cvn)

    # For nn.LSTM
    elif isinstance(module, nn.LSTM):
        if module.__class__ != nn.LSTM:
            logger.warning(
                f"{curr_full_name} {type(module)} seems to be a wrapper of LSTM."
                "Please make sure it doesn't wrap BN and activ func."
                "Otherwise please create an equivalen Linear wrapper and change qcfg['mapping']."
            )

        Qlstm = mapping.get(nn.LSTM, None)
        if Qlstm is None:
            if verbose:
                logger.info(
                    f"Skip quantization of {curr_full_name} - mapping of LSTM is None"
                )
            return module_output  # None means no swap for this type

        module_output = Qlstm(
            **base_params,
            num_bits_weight=qcfg["nbits_w_lstm"],
            qw_mode=qcfg["qw_mode_lstm"],
            num_bits_input=qcfg["nbits_i_lstm"],
            qi_mode=qcfg.get("qi_mode_lstm", qcfg["qa_mode_lstm"]),
            num_bits_hidden=qcfg["nbits_h_lstm"],
            qh_mode=qcfg.get("qh_mode_lstm", qcfg["qa_mode_lstm"]),
            align_zero=qcfg["align_zero"],
            qcfg=qcfg,
        )
        for k, v in module.named_parameters():
            if getattr(module, k, None):
                setattr(module_output, k, v)
        module_output._all_weights = module._all_weights

    return module_output


def q_any_net_5(model: nn.Module, qcfg: dict, verbose: bool = False):
    """Go through all model.named_modules(), try to create an equivalent Qlayer to replace each of
    the existing nn.layers.
    TODO: Check whether the new layer is on Qskip_layer list in make_quant_module(), why not here?

    Args:
        model (nn.Module): input model to be "prepared"
        qcfg (dict): quant config
        verbose (bool, optional): print debug info

    Returns:
        nn.Module: updated model is returned, but technically it's changed in place, users do not
                    need to rely on the return
    """
    # Third Party
    from torch.ao.quantization.utils import _parent_name
    from tqdm import tqdm

    total_modules = len(list(model.named_modules()))
    pbar = tqdm(
        model.named_modules(),
        total=total_modules,
        desc="Mapping modules to target Qmodules.",
    )
    for name, module in pbar:
        pbar.set_description(f"processing {name}")

        parent_module_name, curr_mod_name = _parent_name(name)
        new_module = make_quant_module(module, name, qcfg)
        parent_module = model.get_submodule(parent_module_name)

        if new_module is not module:
            parent_module.add_module(curr_mod_name, new_module)
            gc.collect()
            for r in gc.get_referrers(module):
                if isinstance(r, list):
                    logger.warning(
                        f"During swapping {name} module with quantizer, a 'list' in referrers"
                        f"was found !! {r if verbose else ''}"
                    )
                    logger.warning(
                        "Most likely somewhere in the forward() will utilize this list, so "
                        "list.replace(old_module, new_module) will be performed. PLEASE carefully "
                        "double check and make sure this is expected !!"
                    )
                    for i, e in enumerate(r):
                        r[i] = new_module if e is module else e

            if verbose:
                logger.info(f"Swap ({name}) from {type(module)} to {type(new_module)}")

    pbar.close()
    return model


quantized_modules = QBmm_modules + QConv2d_modules + QLinear_modules + QLSTM_modules


def has_quantized_module(model):
    """Check if model is already quantized - do not want to quantize twice if so"""
    return any(isinstance(m, quantized_modules) for m in model.modules())


def swap_qbmm(model: nn.Module, qcfg: dict):
    """Go through all model.named_modules(), try to create an equivalent
    Qbmm layer to replace each of the existing linear Bmm layers.

    Args:
        model (nn.Module): input model to be "prepared"
        qcfg (dict): quant config

    Returns: updated model is returned with the Qbmm added

    """

    # Local
    from fms_mo.modules import QBmm

    qcfg["which2patch_contextmanager"] = qcfg["bmm_prep"]["which2patch_contextmanager"]
    isbmm = qcfg["which2patch_contextmanager"] == "torch.bmm"
    for mod_name, line_nums in qcfg["bmm_prep"]["layers_with_bmm"].items():
        mod_bmm_happened = model.get_submodule(mod_name)
        for whichQBmm, ln in enumerate(line_nums, start=1):
            nbits = qcfg[f"nbits_bmm{whichQBmm}"]
            newQBmm = QBmm(
                num_bits_m1=max(nbits, 8) if whichQBmm == 2 else nbits,
                num_bits_m2=nbits,
                qm1_mode=qcfg[f"bmm{whichQBmm}_qm1_mode"],
                qm2_mode=qcfg[f"bmm{whichQBmm}_qm2_mode"],
                m1_unidirectional=(whichQBmm == 2),
                m1_bounded=(whichQBmm == 2),  # see Note 5
                m2_unidirectional=False,
                m2_bounded=False,
                replaceBmm=isbmm,
                qcfg=qcfg,
            )
            setattr(mod_bmm_happened, f"QBmm{ln}", newQBmm)


def qmodel_prep(
    model,
    dloader,
    qcfg,
    optimizer=None,
    ckpt_reload=None,
    prefwdproc=None,
    save_fname="temp_model.pt",
    Qcali=False,
    dev=None,
    use_dynamo=False,
    verbose=False,
    **kwargs,
):
    """Prepare a given PyTorch model for quantization process through three parts:

    PART I: module swapping

    First, determine which layer to quantize via two options:
    Option 1: user specifies layer name patterns "TO QUANTIZE" through qcfg["qlayer_name_pattern"]
              this will only perform name matching and bypass tracing !!! use carefully !!!
              this option will not support BMMs, since BMM are not layers. can't determine single or
              double-sided, either. have to assume double-sided for all.
              can be used together with qcfg['qskip_layer_name'] and qcfg['qspecial_layers']
    Option 2: trace the model with dynamo (or TorchScript) and identify candidates to quantize
              "model_analyzer" will set up things through qcfg which will be used later in QanyNet()

    PART II: Initialize clipvals

    pretrained model provides all the weights before qmodel_prep() through mechanism like HF's
    .from_pretrain( args.model_name_or_path ) BUT there's no quantization info, eg clipvals in ckpt.
    It's well known that a good initial guess is critical for quantization => we could EITHER:
    a) load clipvals from a previously trained/tuned ckpt, if you had one, OR
    b) run a "calibration" with a small amount of real data
    NOTE: case a) only works for "very simple cases". Complicated ckpt, e.g. state_dict is not at
         the upmost level in the ckpt file, USER needs to handle it carefully on their own
        An example of "complicated ckpt":
            torch.save({'model': model, 'optimizer': optimizer, 'other stuff': xxxm...})

    PART III: update optimizer

    Add new param_groups for clip_vals in the optimizer (and remove them from existing groups if
    exists). This way we can control LR, decay of quant params better.

    NOTE:
    1. "dloader" or "dataloadrt" could be either i) a real torch dataloader that we can fetch from,
        ii) a list of data, or iii) a data structure tha can be fed to model directly.
        To avoid confusion, our new convention is:
        a. User should always try to provide ONE 'ready-to-run' data, i.e. a data structure
           that can be fed to model() directly
        b. the only case user should provide A LIST OF 'ready-to-run' data is OLD CALIBRATION
        c. if NEW CALIBRATION and user provides a list of data -> assume confusion, extract 1st elem
        In short:
        if dloader is a list: make sure qmodel_calibration == len(this list)
        else: assume it's a ready-to-run data structure that model requires.

    2. qcfg will be attached to the model during 1st pass of tracing, which will cause deepcopy
        problem (if needed) later, we will pop some items from qcfg then add it back at the end
    3. if DP model, batchsize usually refers to (batch_size_per_device * N_gpus), which could be too
        much for tracing (on a single GPU), need to make sure it's properly sliced.
        For example: bs_DP = min(2, qcfg["batch_size"] // qcfg["world_size"])
    4. if DP model, need to unwrap before tracing then re-wrap afterward. will have problem if do
        module swapping directly on wrapped DP models
    5. safetensors ckpt is PREFERRED because it allows accessing individual tensors. So that when
        possible, we may only load the clipvals but not the weights. especially handy for LLMs.

    Args:
        model (nn.Module): model to be quantized
        dloader (Any): user provided data for tracing or calibration
        qcfg (dict): quant config
        optimizer (nn.optimizer, optional): optimizer for training process. will be updated to
                                            accommodate quantization parameters
        ckpt_reload (str, optional): file name to a quantized checkpoint, will try to reload the
                                    trained quant params after qmodel_prep is done.
        prefwdproc (callable, optional): sometimes data fetched from dataloader need extra
                                        processing before being fed to model. This func can help.
                                        model( prefwdproc(data_fetched_from_loader) )
        save_fname (str, optional): filename for saving tracing info. only used in TorchScript case
        Qcali (bool, optional): trigger for calibration. [To-be-obsoleted]
        dev (device, optional): target device.
        use_dynamo (bool, optional): select which tracer, Dynamo or TorchScript.
        verbose (bool, optional): print debug info

    Returns:
        nn.Module: quantized model ready for further PTQ/QAT
    """
    if qcfg["fp8_inference"]:
        if qcfg.get("QBmm"):
            swap_qbmm(model, qcfg)

        model = q_any_net_5(model, qcfg, verbose=False)
        return model

    sys.setrecursionlimit(4000)

    currDev = next(model.parameters()).device if dev is None else dev

    # Disable logger if not cpu or cuda:0 thread
    logger.disabled = currDev not in [torch.device("cpu"), torch.device("cuda:0")]

    tb_writer = qcfg.pop("tb_writer", None)

    # Check if model is already quantized
    if has_quantized_module(model):
        raise RuntimeError("Model to be quantized already has quantized module(s)")

    # Check config for bad values of important settings before consuming it
    model_dtype = next(model.parameters()).dtype
    check_config(qcfg, model_dtype)

    logger.info(f"--- Before model quantization --- \n {model}\n")
    qcfg["wasDPmodel"] = isinstance(model, nn.DataParallel)
    qcfg["wasDDPmodel"] = isinstance(model, torch.nn.parallel.DistributedDataParallel)
    qcfg["isRNNmodel"] = any(
        [qcfg["nbits_w_lstm"], qcfg["nbits_i_lstm"], qcfg["nbits_h_lstm"]]
    )
    qcfg["QBmm"] = not (
        qcfg["nbits_bmm1"] in [32, None]
        and qcfg["nbits_bmm2"] in [32, None]
        and qcfg["nbits_kvcache"] in [32, None]
    )

    if qcfg["wasDPmodel"] or qcfg["wasDDPmodel"]:
        DPorDDPdevices = model.device_ids
        model = model.module
    else:
        DPorDDPdevices = None

    qcfg["LUTmodule_name"] = {m: k for k, m in model.named_modules()}
    # TODO: This LUT is still being used, better to avoid this "global" way in qcfg and use local

    # ------ Determine which layer to quantize
    if qcfg["qlayer_name_pattern"] != []:
        # --- Option 1: rely on name matching, no graph tracing ---
        # Standard
        import re

        qskip_layer_name, QsinglesidedConvs = [], []
        mappable_classes = [
            cls for cls in qcfg["mapping"].keys() if not isinstance(cls, str)
        ]
        mappable_layers = [
            n
            for n, m in model.named_modules()
            if isinstance(m, tuple(mappable_classes))
        ]
        qskip_layer_name = set(mappable_layers)

        if isinstance(qcfg["qlayer_name_pattern"], str):
            qcfg["qlayer_name_pattern"] = [qcfg["qlayer_name_pattern"]]

        for pat in qcfg["qlayer_name_pattern"]:
            pcomp = re.compile(pat)
            matched = [name_i for name_i in mappable_layers if pcomp.match(name_i)]
            if verbose:
                logger.info(f"matched cases of pattern {pat}, {matched}")
            qskip_layer_name -= set(matched)
        qskip_layer_name = list(qskip_layer_name)
        logger.info(f"Layers that will not be quantized: {qskip_layer_name}")

    elif use_dynamo:
        # --- Option 2.1 trace the model with dynamo and find candidates to quantize
        # Local
        from fms_mo.fx.dynamo_utils import model_analyzer

        # TODO: need a more robust 'input parsing', similar to what we used in TS version
        if isinstance(dloader, torch.utils.data.DataLoader):
            sample_inp = next(iter(dloader))
        elif isinstance(dloader, list):
            Ncalib = max(qcfg["qmodel_calibration"], qcfg["qmodel_calibration_new"])
            if Ncalib > 0 and len(dloader) == Ncalib:
                sample_inp = dloader[0]
            else:
                sample_inp = dloader
        else:
            # assume user provides something ready-to-run
            sample_inp = dloader

        qskip_layer_name, QsinglesidedConvs = [], []
        sample_inp = prepare_inputs(currDev, sample_inp)
        model_analyzer(model, sample_inp, qcfg, plotsvg=kwargs.get("plotsvg", False))
        # NOTE: in this new model_analyzer, search results will be stored into
        #      qcfg['qskip_layer_name'] and qcfg['qsinglesided_name']
    else:
        # --- Option 2.2 trace the model with TorchScript
        # Local
        from fms_mo.utils.torchscript_utils import model_analyzer_ts

        qskip_layer_name, QsinglesidedConvs = model_analyzer_ts(
            model, dloader, qcfg, prefwdproc, save_fname, dev
        )

    # default Qxxx_name are [], use "append" to avoid overriding existing ones
    qcfg["qskip_layer_name"] += qskip_layer_name
    qcfg["qsinglesided_name"] += QsinglesidedConvs

    if "mapping" not in qcfg:
        raise RuntimeError(
            "Mapping dictionary not defined! Please double-check fms_mo_init()"
        )

    if verbose:
        logger.info(
            f"\nWill skip the following layers: \n {qcfg['qskip_layer_name']}\n"
        )
        logger.info(
            f"\nWill use single-sided Conv for: \n {qcfg['qsinglesided_name']}\n"
        )

    model = q_any_net_5(model, qcfg, verbose)

    model.to(currDev)
    qcfg["LUTmodule_name"].update(
        {m: k for k, m in model.named_modules()}
    )  # Qmodules added now

    # --- PART 2: Initialize clipvals, EITHER:
    #     a) load clipvals from a previously trained/tuned ckpt, if you had one, OR
    #     b) run a "calibration" with a small amount of real data

    #   --- Option a:
    if isinstance(ckpt_reload, str):
        fobj = Path(ckpt_reload)
        org_model_path = qcfg.get("model_name_or_path", "")
        need_to_load_weights = True
        if fobj.is_dir():
            need_to_load_weights = fobj != Path(org_model_path)
            ckpt_files = []
            for ext in ["safetensors", "pt", "bin"]:
                ckpt_files.extend(fobj.glob(f"*.{ext}"))
            assert (
                len(ckpt_files) > 0
            ), f"Cannot find any checkpoint files under {fobj} to reload."
            fobj = ckpt_files[
                0
            ]  # if more than 1 ckpt file, no preference, just pick the 1st (for now)
        elif fobj.is_file():
            need_to_load_weights = fobj.parent != Path(org_model_path)

        ckpt_state_dict = None
        w_shapes = {}
        file_ext = fobj.suffix
        if file_ext == ".bin":
            ckpt_state_dict = torch.load(fobj, map_location="cpu")
        elif file_ext == ".safetensors":
            # Third Party
            from safetensors import safe_open

            with safe_open(fobj, framework="pt", device="cpu") as f:
                if need_to_load_weights:
                    ckpt_state_dict = {key: f.get_tensor(key) for key in f.keys()}
                else:
                    ckpt_state_dict = {
                        key: f.get_tensor(key) for key in f.keys() if "clip_val" in key
                    }
                w_shapes = {
                    k: f.get_slice(k).get_shape() for k in f.keys() if "weight" in k
                }
        elif file_ext == ".pt":
            tmp_model = torch.load(fobj, map_location="cpu")
            if isinstance(tmp_model, nn.Module):
                ckpt_state_dict = tmp_model.state_dict()
            elif isinstance(tmp_model, dict):
                ckpt_state_dict = tmp_model

        if not ckpt_state_dict:
            raise RuntimeError(
                f"The provided checkpoint {ckpt_reload} has an unsupported format. Please check!"
            )
        if w_shapes == {}:
            w_shapes = {k: v.shape for k, v in ckpt_state_dict.items() if "weight" in k}

        # check model/ckpt perCh consistency, i.e. perCh vs perT for W
        for n, v in ckpt_state_dict.items():
            if n.endswith("quantize_weight.clip_val"):
                w_shape_ckpt = w_shapes[
                    n.replace("quantize_weight.clip_val", "weight")
                ]  # [out,in]
                is_cvw_vec_ckpt = (
                    v.shape[0] != 1
                )  # i.e, True -> cvw is a vector in ckpt -> perCh

                # case 1: model is perCh but clipval_W in ckpt to be loaded is a scalar
                #         => broadcast to model's shape
                if "perCh" in qcfg["qw_mode"] and not is_cvw_vec_ckpt:
                    logger.info(
                        f"Checkpoint w.clipval shape={v.shape} is inconsistent "
                        f"with w.shape {w_shape_ckpt}"
                    )
                    ckpt_state_dict[n] = v.expand(w_shape_ckpt[0])

                # case 2: model is not perCh but clipval_W in ckpt to be loaded is a vector
                #         => use max() to reduce cvw to scalar for perT
                elif "perCh" not in qcfg["qw_mode"] and is_cvw_vec_ckpt:
                    ckpt_state_dict[n] = torch.max(v)
                # TODO: how about perGroup?

        if qcfg["wasDPmodel"]:
            # if ckpt was saved when DP model is still wrapped, need to remove the prefix 'module.'
            ckpt_state_dict = {
                k.replace("module.", ""): v for k, v in ckpt_state_dict.items()
            }

        # make sure all items in ckpt_dict exist in model
        ckpt_keys_exist_in_model = [
            k for k in ckpt_state_dict if k in model.state_dict()
        ]
        ckpt_keys_extra = set(ckpt_state_dict.keys()) - set(ckpt_keys_exist_in_model)
        real_extra = [
            k for k in ckpt_keys_extra if not k.endswith("quantize_weight.clip_valn")
        ]
        if len(real_extra) > 0:
            message = (
                f"ckpt to-be-loaded has extra items that are not in the model: {real_extra}"
                + "\n           Please make sure this is the right ckpt."
            )
            warnings.warn(message, UserWarning)

        model.load_state_dict(ckpt_state_dict, strict=False)

    #   --- Option b): use calibration to get clip_vals
    elif Qcali or qcfg["qmodel_calibration"] > 0:
        logger.info(
            f"Calibration begins, will run {qcfg['qmodel_calibration']} passes."
        )
        qmodel_calib(model, qcfg, dataloader=dloader, prefwdproc=prefwdproc)
        # NOTE: by default qmodel_calib will not make a copy. if needed, use make_copy flag
    else:
        logger.info(
            "Please provide a valid quantized checkpoint or run calibration for best results."
        )

    model = model.to(currDev)

    # Now we can wrap DP/DDP model back, if it was a DP or DDP model.
    if qcfg["wasDPmodel"]:
        model = torch.nn.DataParallel(model, device_ids=DPorDDPdevices)
    if qcfg["wasDDPmodel"]:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=DPorDDPdevices
        )
    if qcfg["output_folder"] is None:
        qconfig_save(qcfg, fname="qcfg.json")
    else:
        qconfig_save(qcfg, fname=qcfg["output_folder"] + "/qcfg.json")
    qcfg["tb_writer"] = tb_writer

    logger.info(f"--- Quantized model --- \n{model}\n")

    # --- PART 3: update optimizer to add new param_groups for clip_vals
    if optimizer:
        optimizer = update_optim_param_groups(model, optimizer, qcfg)
    else:
        if verbose:
            logger.info(
                "If QAT is intended, please provide the optimizer to qmodel_prep(), "
                "or carefully handle the optimizer.param_group for clip_vals."
            )
        return model
