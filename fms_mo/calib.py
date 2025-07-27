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
from copy import deepcopy
from typing import Callable, Tuple, Union
import logging
import sys

# Third Party
from torch import nn
from transformers.tokenization_utils_base import BatchEncoding
import torch

# Local
from fms_mo.modules import QBmm, QConv2d, QConvTranspose2d, QLinear
from fms_mo.utils.utils import prepare_data_4_fwd, prepare_inputs

# import numpy as np # only used in experimental func


logger = logging.getLogger(__name__)


class ObserverPercentile(nn.Module):
    """Observer, to be attached as pre-forward-hook
    1. Conventionally people would say "99.9 percentile", for kthvalue() it means "0.999" *N_total
    2. Buffers are added to the module this observer is attached to, not the observer itself, so
       that clipvals can extract/reload easier later
    3. Only learnable W quantizers, e.g. PACT+sym, need to be initialized (obsrv_w_cv)
    4. We use "pre"-fwd-hook, so it will "set clipvals" before fwd(), i.e. directly apply to current
       output activations.
    """

    def __init__(self, module, key, percentile, tbw, doublesided, deviceID, **kwargs):
        super().__init__()
        self.key = key
        self.tb_writer = tbw
        self.doublesided = doublesided
        self.w_init_method = kwargs.get("w_init_method", "sawb")
        self.a_init_method = kwargs.get("a_init_method", "percentile")

        self.per = torch.tensor(percentile) * 0.01
        self.per_w = torch.tensor(kwargs.get("percentile_w", percentile)) * 0.01
        self.batch_tracked = 0

        module.register_buffer("obsrv_clipval", torch.zeros(1, device=deviceID))
        module.register_buffer("obsrv_w_clipval", torch.zeros(1, device=deviceID))
        if doublesided:
            module.register_buffer("obsrv_clipvaln", torch.zeros(1, device=deviceID))

    def __call__(self, module, inputs: torch.Tensor):
        """inputs is given as a tuple, usually (but not always) we only care about inputs[0]
        1. inplace update clipvals with fill_() on tempmodel, not the original model
        2. only need to calc W clipval on first call, since no updates for weights during calib
        3. still need to extract state_dict -> reload
        """
        with torch.no_grad():
            x = inputs[0].detach()  # TODO: still need detach() under no_grad context?

            symmetric = False
            if module.quantize_feature:
                # default to asymmetric clip_val computation
                # TODO: this misses symmetry of PACTPlusSym, PACT2Sym, and QFixSymmetric
                symmetric = not getattr(module.quantize_feature, "minmax", True)

            nelem = x.nelement()
            if self.a_init_method == "percentile":
                lower_k = int(self.per[0] * nelem)
                lower_per_cur_candidate = (
                    x.reshape(1, -1).kthvalue(lower_k).values.data[0]
                    if lower_k > 0
                    else x.min()
                )  # guard rail: tensors with very few elements could cause kthvalue(0) error
                upper_per_cur_candidate = (
                    x.reshape(1, -1).kthvalue(int(self.per[1] * nelem)).values.data[0]
                )
                if symmetric:
                    upper_per_cur = max(
                        upper_per_cur_candidate,
                        lower_per_cur_candidate.abs(),
                    )
                    lower_per_cur = -upper_per_cur
                else:
                    upper_per_cur = upper_per_cur_candidate
                    lower_per_cur = lower_per_cur_candidate
            elif symmetric:
                upper_per_cur = x.abs().max()
                lower_per_cur = -upper_per_cur
            else:
                lower_per_cur = x.min()
                upper_per_cur = x.max()

            module.obsrv_clipval = (
                module.obsrv_clipval * self.batch_tracked + upper_per_cur
            ) / (self.batch_tracked + 1)
            if module.quantize_feature and hasattr(module.quantize_feature, "clip_val"):
                # safeguard for for W4A32 or SAWB max for act
                module.quantize_feature.clip_val.fill_(module.obsrv_clipval.data[0])

            if self.doublesided:
                module.obsrv_clipvaln = (
                    module.obsrv_clipvaln * self.batch_tracked + lower_per_cur
                ) / (self.batch_tracked + 1)
                if module.quantize_feature and hasattr(
                    module.quantize_feature, "clip_valn"
                ):
                    module.quantize_feature.clip_valn.fill_(
                        module.obsrv_clipvaln.data[0]
                    )

            if self.batch_tracked == 0:
                if self.w_init_method == "sawb":
                    w_abs_mean = module.weight.abs().mean()
                    w_std_sawb = module.weight.pow(2).mean().sqrt()
                    module.obsrv_w_clipval = (
                        -12.80 * w_abs_mean + 12.68 * w_std_sawb
                    )  # this is SAWB formula for 403
                elif self.w_init_method == "3sigma":
                    w_mean = module.weight.mean()
                    w_std = module.weight.std()
                    module.obsrv_w_clipval = (
                        w_mean + 3 * w_std
                    )  # mean+3sigma, assuming ~Gaussian dist
                elif self.w_init_method == "max":
                    module.obsrv_w_clipval = torch.max(
                        -module.weight.min(), module.weight.max()
                    )
                elif self.w_init_method == "percentile":
                    nelem = module.weight.nelement()
                    module.obsrv_w_clipval = (
                        module.weight.reshape(1, -1)
                        .kthvalue(int(self.per[1] * nelem))
                        .values.data[0]
                    )

                if module.qw_mode == "pact+" and module.quantize_weight:
                    module.quantize_weight.clip_val.fill_(module.obsrv_w_clipval)

            self.batch_tracked += 1
            if self.tb_writer:
                self.tb_writer.add_histogram(self.key, x, self.batch_tracked)


class ObserverLSTM(nn.Module):
    """Observer, to be attached as pre-forward-hook
    1. Conventionally people would say "99.9 percentile", for kthvalue() it means "0.999" *N_total
    2. Buffers are added to the module this observer is attached to, not the observer itself, so
       that clipvals can extract/reload easier later
    3. Only learnable W quantizers, e.g. PACT+sym, need to be initialized (obsrv_w_cv)
    4. We use "pre"-fwd-hook, so it will "set clipvals" before fwd(), i.e. directly apply to current
       output activations.
    """

    def __init__(
        self, module, key, percentile, tbw, doublesided: bool, deviceID, **kwargs
    ):
        super().__init__()
        self.key = key
        self.tb_writer = tbw
        self.doublesided = doublesided
        self.w_init_method = kwargs.get("w_init_method", "sawb")
        self.a_init_method = kwargs.get("a_init_method", "percentile")
        self.device = deviceID

        self.per = torch.tensor(percentile) * 0.01
        self.per_w = torch.tensor(kwargs.get("percentile_w", percentile)) * 0.01
        self.batch_tracked = 0
        if isinstance(module, nn.LSTM):
            for layer in range(module.num_layers):
                self.register_buffer(
                    f"obsrv_i_lay{layer}_cv", torch.zeros(2, device=deviceID)
                )  # 2 for 2-sided, roughly bounded [-1, 1] but we could be more accurate.
                self.register_buffer(
                    f"obsrv_h_lay{layer}_cv", torch.zeros(2, device=deviceID)
                )
        else:
            raise TypeError(
                "This module is not LSTM. Please choose the correct observer."
            )

    def __call__(self, module, inputs):
        """for LSTM, inputs is given as a tuple, (inputs, hid)"""
        with torch.no_grad():
            x = inputs[0].detach()  # safeguard, avoid to trig autograd
            hid = inputs[1]

            for layer in range(module.num_layers):
                # handle inputs and hid together
                for act, name in [(x, "input"), (hid[layer], "hidden")]:
                    nelem = act.nelement()
                    if self.a_init_method == "percentile":
                        lower_k, upper_k = (
                            int(self.per[0] * nelem),
                            int(self.per[1] * nelem),
                        )
                        lower_per_cur = (
                            act.reshape(1, -1).kthvalue(lower_k).values.data[0]
                            if lower_k > 0
                            else act.min()
                        )
                        upper_per_cur = (
                            act.reshape(1, -1).kthvalue(upper_k).values.data[0]
                        )
                    else:
                        lower_per_cur = act.min()
                        upper_per_cur = act.max()
                    cvs = getattr(
                        self, f"obsrv_{name[0]}_lay{layer}_cv"
                    )  # [0] is cv [1] is cvn
                    setattr(
                        self,
                        f"obsrv_{name[0]}_lay{layer}_cv",
                        (
                            cvs * self.batch_tracked
                            + torch.tensor(
                                [upper_per_cur, lower_per_cur], device=self.device
                            )
                        )
                        / (self.batch_tracked + 1),
                    )

                    quantizer = getattr(module, f"quantize_{name}_layer{layer}", None)
                    if quantizer:  # safeguard for for W4A32
                        getattr(quantizer, "clip_val").fill_(
                            getattr(self, f"obsrv_{name[0]}_lay{layer}_cv")[0].item()
                        )
                        getattr(quantizer, "clip_valn").fill_(
                            getattr(self, f"obsrv_{name[0]}_lay{layer}_cv")[1].item()
                        )

            self.batch_tracked += 1
            if self.tb_writer:
                self.tb_writer.add_histogram(self.key, x, self.batch_tracked)


class ObserverGeneric(nn.Module):
    """Generic observer, to be attached to a nn.Module as pre-forward-hook
    1. Conventionally people would say "99.9 percentile", for kthvalue() it means "0.999" *N_total
    2. Buffers are added to the module this observer is attached to, not the observer itself, so
       that clipvals can extract/reload easier later
    3. Only learnable W quantizers, e.g. PACT+sym, need to be initialized (obsrv_w_cv)
    4. We use "pre"-fwd-hook, so it will "set clipvals" before fwd(), i.e. directly apply to current
       output activations.
    5. Convs and Linears have A+W while QBmm has 2 A's (inputs) to observe
    """

    def __init__(
        self, module, key, percentile, tbw, doublesided: bool, deviceID, **kwargs
    ):
        super().__init__()
        self.key = key
        self.tb_writer = tbw
        self.doublesided = doublesided
        self.w_init_method = kwargs.get("w_init_method", "sawb")
        self.a_init_method = kwargs.get("a_init_method", "percentile")
        self.device = deviceID
        # Local

        self.is_qbmm = isinstance(module, QBmm)

        self.per = torch.tensor(percentile) * 0.01
        self.per_w = torch.tensor(kwargs.get("percentile_w", percentile)) * 0.01
        self.batch_tracked = 0
        self.quantizer_attrs = [q for q in dir(module) if "quantize_" in q]
        for qa in self.quantizer_attrs:
            quantizer_i = getattr(module, qa)
            if hasattr(quantizer_i, "clip_val"):
                self.register_buffer(
                    f"obsrv_{qa}_cv",
                    torch.zeros_like(getattr(quantizer_i, "clip_val"), device=deviceID),
                )
            if hasattr(quantizer_i, "clip_valn"):
                self.register_buffer(
                    f"obsrv_{qa}_cvn",
                    torch.zeros_like(
                        getattr(quantizer_i, "clip_valn"), device=deviceID
                    ),
                )

    def __call__(self, module, inputs):
        """inputs is given as a tuple, usually (but not always) we only care about inputs[0]
        1. inplace update clipvals with fill_() on tempmodel, not the original model
        2. only need to calc W clipval on first call, since no updates for weights during calib
        3. still need to extract state_dict -> reload
        4. always assume W is symmetric, i.e. cvn = - cv
        """
        with torch.no_grad():
            if self.is_qbmm:
                matA = inputs[0].detach()
                matW = inputs[1].detach()
            else:
                matA = inputs[0].detach()
                matW = module.weight.detach()

            for k, v in self.named_buffers():
                if k.startswith("obsrv_") and "weight" not in k:
                    # --- calc act
                    matX = matW if "m2_" in k else matA
                    nelem = matX.nelement()
                    if self.a_init_method == "percentile":
                        lower_k = int(self.per[0] * nelem)
                        lower_per_cur = (
                            matX.reshape(1, -1).kthvalue(lower_k).values.data[0]
                            if lower_k > 0
                            else matX.min()
                        )
                        upper_per_cur = (
                            matX.reshape(1, -1)
                            .kthvalue(int(self.per[1] * nelem))
                            .values.data[0]
                        )
                    else:
                        lower_per_cur = matX.min()
                        upper_per_cur = matX.max()

                elif (
                    k.startswith("obsrv_") and "weight" in k and self.batch_tracked == 0
                ):
                    if self.w_init_method == "sawb":
                        w_abs_mean = matW.abs().mean()
                        w_std_sawb = matW.pow(2).mean().sqrt()
                        w_cv = (
                            -12.80 * w_abs_mean + 12.68 * w_std_sawb
                        )  # this is SAWB formula for 403
                    elif self.w_init_method == "3sigma":
                        w_mean = matW.mean()
                        w_std = matW.std()
                        w_cv = w_mean + 3 * w_std
                    elif self.w_init_method == "max":
                        w_cv = matW.abs().max()
                    else:
                        nelem = matW.nelement()
                        w_cv = (
                            matW.reshape(1, -1)
                            .kthvalue(int(self.per[1] * nelem))
                            .values.data[0]
                        )
                    upper_per_cur = w_cv
                    lower_per_cur = -w_cv
                else:
                    upper_per_cur, lower_per_cur = None, None

                if k.endswith("cvn"):
                    new_val = lower_per_cur
                    quantizer = getattr(module, k[6:-4])
                    # TODO: remove 'obsrv_' and '_cvn', better use .replace()
                    cvname = "clip_valn"
                else:
                    new_val = upper_per_cur
                    quantizer = getattr(module, k[6:-3])
                    # TODO: use .replace() instead
                    cvname = "clip_val"
                setattr(
                    self,
                    k,
                    (v * self.batch_tracked + new_val) / (self.batch_tracked + 1),
                )
                getattr(quantizer, cvname).fill_(getattr(self, k).data[0])

            self.batch_tracked += 1


def attach_observer_v2(
    model: torch.nn.Module,
    qcfg: dict,
    deviceID: str = "cuda",
    verbose: bool = False,
    **kwargs,
):
    """Simplified version, to attach observer to all Convs/Linear/bmm (calib for QLSTM is not very
    helpful), then return handles to all hooks, in case we need to remove them later

        Args:
        model (nn.Module): quantized model with QLinear, QConv, QBmm
        qcfg (dict): quant config
        deviceID (str, optional): device id. Defaults to "cuda".
        verbose (bool, optional): print debug info. Defaults to False.

    Returns:
        dict: {layername0:handle0, ...}
    """
    hook_handles = {}
    nbits_names = ["num_bits_feature", "num_bits_weight", "num_bits_m1", "num_bits_m2"]
    for name, mod in model.named_modules():
        if isinstance(mod, (QConv2d, QLinear, QBmm, QConvTranspose2d)):
            observer = ObserverGeneric if isinstance(mod, QBmm) else ObserverPercentile

            if any(getattr(mod, attr, None) not in [32, None] for attr in nbits_names):
                h = mod.register_forward_pre_hook(
                    observer(
                        mod,
                        name,
                        percentile=qcfg.get(
                            "calib_percentile",  # try to rename the param
                            qcfg["clip_val_asst_percentile"],
                        ),
                        tbw=kwargs.get("tbw", qcfg.get("tb_writer", None)),
                        w_init_method=qcfg.get(
                            "qw_mode_calib",  # try to unify the param names
                            qcfg.get("w_init_method", "sawb"),
                        ),
                        a_init_method=qcfg.get(
                            "qa_mode_calib", qcfg.get("a_init_method", "percentile")
                        ),
                        doublesided=not mod.non_neg
                        if hasattr(mod, "non_neg")
                        else True,
                        deviceID=deviceID,
                    )
                )
                hook_handles[name] = h
    if verbose:
        logger.info(
            f"Attached {len(hook_handles)} observers to the model for calibration purpose."
        )
    return hook_handles


def qmodel_calib(
    model,
    qcfg,
    dataloader=None,
    prefwdproc: Union[Callable[[Tuple], Tuple], str, None] = None,
    verbose=False,
    make_copy=False,
):
    """Calibrate quantized model, i.e. get better initial guess of quant params using a small
    number of training data. Basic flow as below:
    1) attach observers to Qmodules,
    2) run N iterations using user provided data,
    3) collect better estimates of clipvals,
    4) inject the new estimate of clip_vals back to Qmodules
    NOTE:
    1. qmodel_calib could be called manually by users or by qmodel_prep(). if later case and DP/DDP
       model in use, model could have been unwrapped already. we use call stack to confirm caller.
    2. In rare case user may want to keep model absolutely fresh, set make_copy=True can make a copy
       of the model and calibrate on the copy. Make sure 2 models can fit on GPU memory.
    3. careful about DP/DDP worldsize, batchsize, and etc... may need to manually all_reduce the
       clip_vals if multi-GPUs (DDP only).

    Args:
        model (nn.Module): model to be calibrated
        qcfg (dict): quant config
        dataloader (Any, optional): user provided data, could be as simple as a batch of
                                    "ready-to-run" data, a list of data, or a dataloader.
        prefwdproc (callable, optional): if the data fetched from dataloader needs further process
                                        before feeding to model, user can define this "pre-forward-
                                        process" function, which will be called in this way
                                            model( prefwdproc(data) ).
        verbose (bool, optional): print debug info.
        make_copy (bool, optional): calibrate on a clone instead of the original model.

    Returns:
        nn.Module: calibrated model
    """
    # Third Party
    import pandas as pd
    import torch.distributed as dist

    currDev = next(model.parameters()).device

    # Disable logger if not cpu or cuda:0 thread
    logger.disabled = currDev not in [torch.device("cpu"), torch.device("cuda:0")]

    if dataloader is None:
        logger.error(
            "Dataloader or a list of data is not provided. Cannot perform calibration!"
        )
        return model

    DPorDDPdevices = None
    if "qmodel_prep" not in sys._getframe().f_back.f_code.co_name:
        model.to(currDev)
        qcfg["wasDPmodel"] = qcfg.get("wasDPmodel", isinstance(model, nn.DataParallel))
        qcfg["wasDDPmodel"] = qcfg.get(
            "wasDDPmodel", isinstance(model, torch.nn.parallel.DistributedDataParallel)
        )

        if qcfg["wasDPmodel"] or qcfg["wasDDPmodel"]:
            DPorDDPdevices = model.device_ids
            model = model.module
        calledbyQmodelPrep = False
    else:
        calledbyQmodelPrep = True
        if verbose:
            logger.info(
                "Qcalib called by qmodel_prep, DP/DDP model unwrapping check is skipped."
            )

    if make_copy:
        tempmodel = deepcopy(model)
    else:
        tempmodel = model

    # Step 1: attach observers (pre_fwd_hooks) to Conv/Linear/bmm
    h_hooks = attach_observer_v2(tempmodel, qcfg=qcfg, deviceID=currDev)

    # Step 2: run Nbatches, observer will calculate the stats
    Nbatch = qcfg["qmodel_calibration"]
    if "perCh" not in qcfg["qw_mode"]:
        cv_sum_dict = {"layer": [], "value": []}
        for k, v in tempmodel.state_dict().items():
            if "clip" in k:
                cv_sum_dict["layer"].append(k)
                cv_sum_dict["value"].append(v.item())
        logger.info(f"Clipvals before calibration: \n{ pd.DataFrame(cv_sum_dict) }")

    with torch.no_grad():
        if "detectron2.modeling" in sys.modules:
            raise RuntimeError("Detectron2 is not supported for the moment")
        # --- special handling for RNN/LSTM ---
        if qcfg["isRNNmodel"]:
            # NOTE: probably need to adjust data_mb with dataloader
            hid = (
                torch.zeros(
                    qcfg["nlayers"] * (qcfg["bidirectional"] + 1),
                    qcfg["batch_size"],
                    qcfg["nhid"],
                ).to(currDev),
                torch.zeros(
                    qcfg["nlayers"] * (qcfg["bidirectional"] + 1),
                    qcfg["batch_size"],
                    qcfg["nhid"],
                ).to(currDev),
            )
            for i in range(0, Nbatch):
                data_mb, _ = dataloader()
                data_mb = data_mb.to(currDev)
                _, hid = tempmodel(data_mb, hid)
                logger.info(
                    f"Qmodel calibration (clip_val analysis) in progress: {i}/{Nbatch}"
                )
        # --- user provide a list of tensors or a list of dict, assume they are ready for forward
        elif isinstance(dataloader, list):
            assert Nbatch == len(
                dataloader
            ), "Length of provided data (list) != num of calibration."
            for i, data_mb in enumerate(dataloader):
                data_mb = prepare_inputs(currDev, data_mb)

                if isinstance(data_mb, (dict, BatchEncoding)):
                    tempmodel(**data_mb)
                elif isinstance(data_mb, tuple):
                    tempmodel(*data_mb)
                elif isinstance(data_mb, torch.Tensor):
                    tempmodel(data_mb)

                logger.info(
                    f"Qmodel calibration (clip_val analysis) in progress: {i}/{Nbatch}"
                )

        # --- assume it's normal torch.Dataloader or some kind of generator we can iter
        else:
            for i in range(0, Nbatch):
                data_mb = next(iter(dataloader))
                data_mb = prepare_data_4_fwd(data_mb, qcfg, prefwdproc, currDev)
                # TODO: review this prep_4_fwd(), possibly need updates
                tempmodel(*data_mb)

                logger.info(
                    f"Qmodel calibration (clip_val analysis) in progress: {i}/{Nbatch}"
                )

        cv_sum_dict = {"layer": [], "value": []}
        for k, v in tempmodel.state_dict().items():
            if "clip" not in k:
                continue

            if v.numel() > 1:
                k = k + "*"
                v = v.mean()
            cv_sum_dict["layer"].append(k)
            cv_sum_dict["value"].append(v.item())
        logger.info(
            f"Observed clipvals: ('*' if it's a vector) \n{ pd.DataFrame(cv_sum_dict) }"
        )

    # Step 3: extract new clip_vals, params and buffers, then remove handles if needed
    temp_new_clipvals = {
        k: v.to(currDev) for k, v in tempmodel.state_dict().items() if ".quantize_" in k
    }

    if make_copy:
        del tempmodel
    else:
        for h in h_hooks.values():  # handle_hooks dict is like {'name1':handle1,...}
            h.remove()

    if qcfg["world_size"] > 1 and qcfg["wasDDPmodel"]:
        # NOTE: if (was) DP model -> does not need all_reduce
        logger.info(
            f"Before all_reduce.\n global rank= {qcfg['global_rank']}\n, {temp_new_clipvals}"
        )
        for k, v in temp_new_clipvals.items():
            dist.all_reduce(v, op=dist.ReduceOp.SUM)
            try:
                temp_new_clipvals[k] = v / qcfg["world_size"]
            except RuntimeError:
                isTensor = isinstance(v, torch.Tensor)
                logger.info(
                    f"SKIP reducing parameter {k}   type: {type(v)}   "
                    f"size: {v.size() if isTensor else 'n/a'}   "
                    f"value: {v.item() if isTensor and v.dim() in [0, 1] else 'n/a'}"
                )
        logger.info(
            f"After all_reduce.\n global rank= {qcfg['global_rank']}\n, {temp_new_clipvals}"
        )
    else:
        logger.info("This model does not need all_reduce")

    # Step 4: Load modified state_dict back to the real model (DP not wrapped yet), only need to
    #         update those in the final model (in tmp_sd) and ignore extras (in tmp_new_clipvals)
    temp_model_sd = model.state_dict()
    for k in temp_model_sd.keys():
        if k.endswith("clip_val") or k.endswith("clip_valn"):
            assert (
                k in temp_new_clipvals
            ), f"Model inconsistency!! {k} exists in model but not in new clip_val dictionary"
            temp_model_sd[k] = temp_new_clipvals[k].reshape(
                temp_model_sd[k].size()
            )  # TODO: bandage sol'n, weight_clip_val shape inconsistency
    model.load_state_dict(temp_model_sd)

    # Last Step: if it was a DP or DDP model, wrap it back accordingly
    if not calledbyQmodelPrep:
        if qcfg["wasDPmodel"]:
            model = torch.nn.DataParallel(model, device_ids=DPorDDPdevices)
        if qcfg["wasDDPmodel"]:
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=DPorDDPdevices
            )

    return model
