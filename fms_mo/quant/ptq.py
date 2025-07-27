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
Post-Training Quantization (PTQ) functions

Class StraightThrough, function _fold_bn, fold_bn_into_conv, reset_bn, and
search_fold_and_remove_bn are modified from QDROP repo https://github.com/wimh966/QDrop


"""

# Standard
from functools import partial
from typing import Optional, Union
import logging
import math
import random
import sys

# Third Party
from tqdm import tqdm
import numpy as np
import pandas as pd

# from numpy.lib.function_base import iterable
import torch
import torch.nn as nn
import torch.nn.functional as F

# Local
from fms_mo.modules import QBmm, QLinear
from fms_mo.modules.conv import QConv2dPTQv2
from fms_mo.modules.linear import LinearFPxAcc, QLinearINT8Deploy
from fms_mo.quant.quantizers import (
    AdaRoundQuantizer,
    Qdynamic,
    get_activation_quantizer,
    lp_loss,
)
from fms_mo.utils.import_utils import available_packages
from fms_mo.utils.utils import move_to, patch_torch_bmm

logger = logging.getLogger(__name__)

try:
    # Third Party
    from piqa.piqa import SSIM  # ,MS_SSIM unused-import

    piqa_installed = True
except:
    piqa_installed = False


# TODO: this function is not used. Can be removed.
def set_seed(seed):
    """
    Set random seed
    Not all the reproducibility items are implemented,
    See https://pytorch.org/docs/stable/notes/randomness.html
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


# --------------------------------------------------------
# --------- PTQ-related util functions -------------------
# --------------------------------------------------------


class LinearTempDecay:
    def __init__(self, t_max=20000, warm_up=0.2, hold=1.0, b_range=(20, 2)):
        self.t_max = hold * t_max
        self.start_decay = warm_up * t_max
        # NOTE from warm_up to warm_up2, round_loss starts to work but no decay in beta and lambda
        # see PTQLossFunc for more details
        self.start_b = b_range[0]
        self.end_b = b_range[1]
        self.curr_beta = 0.0

    def __call__(self, t):
        # NOTE from warm_up to warm_up2, round_loss starts to work but no decay in beta and lambda
        if t < self.start_decay:
            self.curr_beta = self.start_b
        elif t > self.t_max:
            self.curr_beta = self.end_b
        else:
            rel_t = (t - self.start_decay) / (self.t_max - self.start_decay)
            self.curr_beta = self.end_b + (self.start_b - self.end_b) * max(
                0.0, (1 - rel_t)
            )
        return self.curr_beta


class CyclicTempDecay:
    def __init__(self, t_max=20000, warm_up=0.2, hold=1.0, b_maxmin=(20, 2), style="V"):
        self.t_max = hold * t_max
        self.start_decay = warm_up * t_max
        # Annealing only happens between [warmup, hold]*t_max, from max_b -> min_b -> max_b
        # e.g. usually no round_loss before 0.2*tmax (controlled by PTQloss_func),
        # here we still return b_max, and hold at b_max after 0.8*tmax
        self.max_b = b_maxmin[0]
        self.min_b = b_maxmin[1]
        assert self.max_b > self.min_b, "max_b is smaller than min_b, please check!"
        self.curr_beta = 0.0
        # style can be 'V', 'V*2', 'V*3'... 'cos*0.5', 'cos*2', 'cos' ... shape*N where
        # N means number of cycles, N could be 0.5
        if "*" not in style:
            style = style + "*1"
        self.style_cycles = style.split("*")
        self.period = (self.t_max - self.start_decay) // float(self.style_cycles[1])

    def __call__(self, t):
        # NOTE from warm_up to warm_up2, round_loss starts to work but no decay in beta and lambda
        if t < self.start_decay:
            self.curr_beta = self.max_b
        elif self.start_decay <= t < self.t_max:
            rel_t = ((t - self.start_decay) % self.period) / self.period
            if self.style_cycles[0] == "cos":
                self.curr_beta = (
                    self.min_b
                    + (self.max_b - self.min_b)
                    * (math.cos(math.pi * 2 * rel_t) + 1.0)
                    * 0.5
                )
            else:  # V-shape
                self.curr_beta = self.min_b + (self.max_b - self.min_b) * abs(
                    1.0 - 2 * rel_t
                )
        #        else: # case t > t_max
        #           # -> beta unchanged (hold), i.e. use last saved curr_beta
        return self.curr_beta


class PTQLossFunc(nn.Module):
    """
    Loss functions for PTQ block-sequential optimization.
    """

    def __init__(
        self,
        method="mse",
        Ntotal_iters=20000,
        layers=None,
        p=2.0,
        isOptimConv=False,
        adaR_anneal={
            "warmup": 0.1,
            "warmup2": 0.0,
            "hold": 0.9,
            "beta": {"range": (20, 2), "style": "linear"},
            "lambda": {"range": (0.01, 0.01), "style": "constant"},
        },
    ):
        super().__init__()
        self.method = method
        self.p = p
        self.count = 0
        self.Ntotal = Ntotal_iters
        self.layers = layers
        self.isOptimConv = isOptimConv
        self.warmup = adaR_anneal["warmup"]
        self.warmup2 = adaR_anneal["warmup2"]
        self.hold = adaR_anneal["hold"]
        self.loss_start = int(Ntotal_iters * self.warmup)
        self.reset_ReSig = None
        if self.warmup2 >= self.warmup:
            self.reset_ReSig = int(Ntotal_iters * self.warmup2)
        # NOTE, round_loss starts from warmup but decay could start from warmup2, controlled
        # by LinearTempDecay() and CyclicTempDecay() when using delayed-decay (warmup2 !=0),
        # we may further switch the formula, e.g. from 1ReSig to 3ReSig at decay_start

        self.beta = adaR_anneal["beta"]  # brecq's settings was (20, 2)
        self.lambda_eq21 = adaR_anneal["lambda"]

        if self.beta["style"] == "constant":
            self.beta_decay = lambda x: self.beta["range"][0]
        elif self.beta["style"] == "linear":
            self.beta_decay = LinearTempDecay(
                Ntotal_iters,
                warm_up=max(self.warmup, self.warmup2),
                hold=self.hold,
                b_range=self.beta["range"],
            )
        else:
            self.beta_decay = CyclicTempDecay(
                Ntotal_iters,
                warm_up=max(self.warmup, self.warmup2),
                hold=self.hold,
                b_maxmin=self.beta["range"],
                style=self.beta["style"],
            )
            # style can be 'cos','V','cos*0.5', 'cos*2','V*2'...

        if self.lambda_eq21["style"] == "constant":
            self.lambda_decay = lambda x: self.lambda_eq21["range"][0]
        elif self.lambda_eq21["style"] == "linear":
            self.lambda_decay = LinearTempDecay(
                Ntotal_iters,
                warm_up=max(self.warmup, self.warmup2),
                hold=self.hold,
                b_range=self.lambda_eq21["range"],
            )
        else:
            self.lambda_decay = CyclicTempDecay(
                Ntotal_iters,
                warm_up=max(self.warmup, self.warmup2),
                hold=self.hold,
                b_maxmin=self.lambda_eq21["range"],
                style=self.lambda_eq21["style"],
            )
        # if method not in ['mse','normalized_change','ssim','ssimlog','ssimp0.2',
        #                   'ssimp0.5','ssimp2','fisher_diag','fisher_full', 'adaround']:
        #     logger.info('!! PTQ Loss method not defined. Use "MSE" instead !!')
        #     self.method = 'mse'

    def __call__(
        self, im1, im2, grad=None, gt=None
    ):  # input im1, im2 are supposed to be q_out, fp_out
        self.count += 1

        if self.method == "mse":
            return F.mse_loss(im1, im2)

        if self.method in ["mae", "l1"]:
            return F.l1_loss(im1, im2)

        if self.method == "normalized_change":
            return torch.norm(im1 - im2) / torch.norm(im2)

        if "ssim" in self.method and piqa_installed:
            # inputs can have very different numerical range, one is the original fp tensor,
            # the other is clipvaln to clipval rescale to [0, 1] based on the larger range input,
            # clamp the smaller range tensor using the larger range tensor's min/max in
            # case of range inconsistency
            im_min = [im1.min(), im2.min()]
            im_max = [im1.max(), im2.max()]
            im_range = [im_max[0] - im_min[0], im_max[1] - im_min[1]]
            #        base_idx = 0 if im_range[0]>im_range[1] else 1
            loss_func = SSIM(n_channels=im1.shape[1], value_range=1).to(im1.device)
            #        im_scaled = [(im1-im_min[base_idx])/im_range[base_idx],
            #                      (im2-im_min[base_idx])/im_range[base_idx] ]
            im_scaled = [
                (im1 - im_min[0]) / im_range[0],
                (im2 - im_min[1]) / im_range[1],
            ]
            #        if im_min[base_idx]>im_min[1-base_idx] or im_max[base_idx]<im_max[1-base_idx]:
            #            im_scaled[1-base_idx] = torch.clamp(im_scaled[1-base_idx], 0.0, 1.0)
            ssimloss = 1.0 - loss_func(*im_scaled)
            loss = (
                torch.log(ssimloss)
                if self.method == "ssimlog"
                else torch.pow(ssimloss, 0.2)
                if self.method == "ssimp0.2"
                else torch.pow(ssimloss, 0.5)
                if self.method == "ssimp0.5"
                else torch.pow(ssimloss, 2)
                if self.method == "ssimp2"
                else {"mse": F.mse_loss(im1, im2), "0.01ssim": 0.01 * ssimloss}
                if self.method == "0.01ssim+mse"
                else {"mse": F.mse_loss(im1, im2), "0.1ssim": 0.1 * ssimloss}
                if self.method == "0.1ssim+mse"
                else ssimloss
            )  # 'ssim' or other 'ssimxxx' all default to simple form
            return loss

        if self.method == "fisher_diag":
            return ((im1 - im2).pow(2) * grad.pow(2)).sum(1).mean()

        if self.method == "fisher_full":
            a = (im1 - im2).abs()
            grad = grad.abs()
            batch_dotprod = torch.sum(a * grad, (1, 2, 3)).view(-1, 1, 1, 1)
            return (batch_dotprod * a * grad).mean() / 100

        if "adaround" in self.method:
            # default is mse + rounding loss as in the original paper
            round_loss = torch.tensor(0.0, device=im1.device)
            ssimloss = torch.tensor(0.0, device=im1.device)
            losses = {}
            if self.count > self.loss_start:
                # we can choose to anneal beta and lambda separately, or together
                b = self.beta_decay(self.count)
                lambda_eq21 = self.lambda_decay(
                    self.count
                )  # eq21 in adaround paper, use brecq's settings

                for l in self.layers:
                    if hasattr(l, "quantize_weight") and isinstance(
                        l.quantize_weight, AdaRoundQuantizer
                    ):
                        if self.count == self.reset_ReSig:
                            l.quantize_weight.reset_ReSig_param(3)

                        round_vals = (
                            l.quantize_weight.get_soft_targets()
                        )  # calc h from eq23, now support multimodal
                        if l.quantize_weight.multimodal:
                            round_vals = (
                                round_vals
                                + (round_vals < 0.0).to(torch.float32)
                                - (round_vals > 1.0).to(torch.float32)
                            )
                            # sine multi-modal f_reg
                            # round_loss += lambda_eq21 * ( torch.sin(round_vals * math.pi
                            #               ).abs().pow(b)).sum()
                        round_loss += (
                            lambda_eq21
                            * (1 - ((round_vals - 0.5).abs() * 2).pow(b)).sum()
                        )  # eq24

            if self.method == "adaroundKL":
                rec_loss = F.kl_div(
                    torch.log(im1 + 1e-6), im2 + 1e-6, reduction="batchmean"
                )
            elif self.method == "adaroundCos":
                rec_loss = torch.mean(
                    F.cosine_similarity(im1, im2)  # pylint: disable=not-callable
                )
            elif self.method == "adaroundL1":
                rec_loss = F.l1_loss(im1, im2)
            elif self.method.startswith("adaroundMonAll"):
                # monitor all losses, still use mse+round for total
                with torch.no_grad():
                    losses["l1"] = F.l1_loss(im1, im2)
                    losses["mse"] = F.mse_loss(
                        im1, im2
                    )  # backward will use "lp loss" instead of mse

                with torch.set_grad_enabled(self.method.endswith("_norm")):
                    losses["norm"] = torch.norm(im1 - im2) / torch.norm(im2)
                with torch.set_grad_enabled(self.method.endswith("_cos")):
                    losses["cos"] = 1.0 - torch.mean(
                        F.cosine_similarity(im1, im2)  # pylint: disable=not-callable
                    )
                with torch.set_grad_enabled(self.method.endswith("_ce")):
                    # ce loss only works for last layer, "im1" should be q_out, im2 won't be used,
                    # unless we want to check fp_out's ce loss
                    losses["qce"] = (
                        0.0 * F.cross_entropy(im1, gt) if gt is not None else None
                    )
                    # may need to adjust the weighting factor

                if self.method.endswith("_norm"):
                    rec_loss = losses["norm"]
                elif self.method.endswith("_cos"):
                    rec_loss = losses["cos"]
                elif self.method.endswith("_ce") and gt is not None:
                    rec_loss = losses[
                        "qce"
                    ]  # only last layer will have gt in input, others will default to lp_loss
                else:
                    rec_loss = lp_loss(im1, im2, p=self.p)

            else:
                # use brecq and qdrop's implementation
                rec_loss = lp_loss(im1, im2, p=self.p)

            losses["total"] = rec_loss + round_loss
            losses["reconstruct"] = rec_loss.detach()  # for tensorboard plot only
            losses["rounding"] = round_loss.detach()  # for tensorboard plot only
            return losses

        # method not defined!
        logger.info(f'PTQ Loss method {self.method} not defined. Use "MSE" instead.')
        return F.mse_loss(im1, im2)


class PTQHookRecInOut(nn.Module):
    """
    Post-hook to cache input (could be FP or Q) and output
    (FP only, set PTQ_mode to 'fp32_out' before running the hooks)
    """

    def __init__(self, qcfg, name=None, cls2rec=(nn.Conv2d), recInOnly=False):
        super().__init__()
        self.name = name
        self.qcfg = qcfg
        self.cls2rec = cls2rec
        self.rec_input_only = recInOnly

    def __call__(self, mod, x, output):
        # make sure this module/block's ptqmode is not 'q_out'
        submods = [m for m in mod.modules() if isinstance(m, self.cls2rec)]
        if any(sm.ptqmode == "q_out" for sm in submods):
            # don't record input/output if any of the submods has ptqmode =='q_out'
            return
        if len(x) > 1:
            # transformers has more than one input, e.g. masks, etc...
            self.qcfg["cached_input0"].append(x[0].detach().cpu())
            self.qcfg["cached_input1"].append(x[1].detach().cpu())
        else:
            self.qcfg["cached_input"].append(x[0].detach().cpu())

        if not self.rec_input_only:
            if isinstance(output, tuple):
                self.qcfg["cached_output"].append(output[0].detach().cpu())
            else:
                self.qcfg["cached_output"].append(output.detach().cpu())


class PTQHookRecInOutLMv2(nn.Module):
    """simplified version of recording hook for PTQ
    just record the entire input tuple, no matter how many inputs are there
    leave the special handling, e.g. reshape/cat/shuffling...etc, for later
    """

    def __init__(
        self,
        qcfg,
        name=None,
        cls2rec=(nn.Conv2d, nn.Linear),
        recInOnly=False,
        stop_after_rec=False,
        cache_dev="cuda",
    ):
        super().__init__()
        self.name = name
        self.qcfg = qcfg
        self.cls2rec = cls2rec
        self.rec_input_only = recInOnly
        self.num_valid_input = -1
        self.stop_after_rec = stop_after_rec
        self.cache_dev = cache_dev

    def __call__(self, mod, inputs, *args, **_kwargs):
        # make sure this module/block's ptqmode is not 'q_out'
        submods = [m for m in mod.modules() if isinstance(m, self.cls2rec)]
        if any(sm.ptqmode == "q_out" for sm in submods):
            # don't record input/output if any of the submods has ptqmode =='q_out'
            return
        # input should always be a tuple of tensors, but some could be None
        # check how many valid inputs are there
        if self.num_valid_input == -1:  # only check once then stick to it
            valid_inps = [inp is not None for inp in inputs]
            if False in valid_inps:
                self.num_valid_input = valid_inps.index(False)
            else:
                self.num_valid_input = len(valid_inps)  # if all True => all valid

        assert all(
            isinstance(inp_i, torch.Tensor) for inp_i in inputs[: self.num_valid_input]
        )
        # check available GPU memory, cache on GPU if possible:
        GPUmem_available, _GPUmem_total = torch.cuda.mem_get_info()
        # 1 block for SQUAD/BERT 500 batches*12/batch = ~10G
        if self.cache_dev == "cuda" and GPUmem_available / 1e9 > 20:
            cache_device = "cuda"
        else:
            cache_device = "cpu"

        self.qcfg["cached_input"].append(
            tuple(
                inp_i.detach().to(cache_device)
                for inp_i in inputs[: self.num_valid_input]
            )
        )

        # output could be a tuple of a single tensor or simply a tensor ?
        if not self.rec_input_only and "output" in args:
            output = args["output"]
            assert isinstance(output, (torch.Tensor, tuple))
            self.qcfg["cached_output"].append(
                output[0].detach().to(cache_device)
                if isinstance(output, tuple)
                else output.detach().to(cache_device)
            )
        assert not self.stop_after_rec


class HookRecPostQuantInOut(torch.nn.Module):
    """Another simplified hook to check post-quantized input/output, e.g. within +-127 for INT8."""

    def __init__(self, cache_dict={}, mod_name=None):
        super().__init__()
        self.cache_dict = cache_dict
        self.mod_name = mod_name
        name_split = mod_name.split(".")
        self.lay_idx = int(name_split[3])
        self.lay_key = name_split[6]

        self.cache_dev = "cpu"
        # prepare empty dict for later use
        self.cache_dict[mod_name] = {}
        self.fwd_mapping = {
            LinearFPxAcc: self.call_func_for_fpxacc,
            QLinear: self.call_func_for_qlinear,
            QLinearINT8Deploy: self.call_func_for_qlinear_int,
            torch.nn.Linear: self.call_func_for_nnlinear,
        }

    def call_func_for_fpxacc(self, mod, inputs, outputs, **_kwargs):
        raise NotImplementedError

    def call_func_for_qlinear(self, mod, inputs, outputs, **_kwargs):
        lay_idx = self.lay_idx
        lay_key = self.lay_key
        mod_name = self.mod_name
        cache_dict = self.cache_dict

        act_max = inputs[0].abs().amax(dim=[d for d in range(len(inputs[0].shape) - 1)])
        # mod.smoothq_act_scale
        w_max = mod.weight.abs().max(dim=0, keepdim=True)[0].clamp(min=1e-5)
        is_smq_layer = not torch.all(act_max == 0).item()
        # smoothQ scale = smoothq_act_scale**alpha / weight_scale**(1.0 - alpha)
        # smoothq_scale = mod.get_smoothq_scale(inputs[0])
        smoothq_scale = getattr(mod, "smq_scale", 1.0)
        # "smq_scale" only available in QLin_INT8

        with torch.no_grad():
            smoothed_inp = inputs[0] / smoothq_scale
            smoothed_w = mod.weight * smoothq_scale

            # this is assuming pertokenmax quantizer, NOTE calc quant scale after smoothing
            absmax = smoothed_inp.abs().max(dim=-1, keepdim=True)[0]
            qa_scale = absmax.clamp(min=1e-5) / 127
            qinput = torch.round(smoothed_inp / qa_scale).clamp(-127, 127)
            # should clamp to -128?
            if mod.qa_mode == "pertokenmax":
                # doesnt implement dequant=False yet, do it manually
                cva = mod.quantize_feature.clip_val
                qa_scale = cva.clamp(min=1e-5).div(127)
                qinput = smoothed_inp.div(qa_scale).round()
            else:
                mod.quantize_feature.dequantize = False
                qinput = mod.quantize_feature(smoothed_inp)
                mod.quantize_feature.dequantize = True

            # also record quantized, smoothed W in INT8, calc both maxperCh and SAWBperCh
            cvw = mod.quantize_weight.clip_val
            scale_w = cvw / 127
            mod.quantize_weight.dequantize = False
            qw = mod.quantize_weight(smoothed_w)
            mod.quantize_weight.dequantize = True

            # inputs is a tuple, QLinear only has 1 valid input
            cache_dict[mod_name]["input"] = inputs[0].to(self.cache_dev)
            cache_dict[mod_name]["cva"] = cva.to(self.cache_dev)
            cache_dict[mod_name]["cvw"] = cvw.to(self.cache_dev)
            cache_dict[mod_name]["smoothed_input"] = smoothed_inp.to(self.cache_dev)
            cache_dict[mod_name]["smoothed_weight"] = smoothed_w.to(self.cache_dev)
            cache_dict[mod_name]["qinput"] = qinput.to(self.cache_dev)
            # NOTE in INT8, *scales if need dQ
            cache_dict[mod_name]["qweight"] = qw.to(self.cache_dev)
            # torch.round(smoothed_w.T/scale_w).clamp(-127, 127).to(self.cache_dev)
            # cache_dict[mod_name]["qoutput"] = outputs.to(self.cache_dev)

    def call_func_for_qlinear_int(self, mod, inputs, outputs, **_kwargs):
        smoothq_scale = getattr(mod, "smq_scale", 1.0)
        mod_name = self.mod_name
        cache_dict = self.cache_dict
        with torch.no_grad():
            if mod.useDynMaxQfunc in [-1, -2]:
                qinput = mod.qa_dynamic_max_qfunc(inputs[0])
            elif mod.use_fake_zero_shift:
                qinput = mod.qa_dyn_max_fake_zero_shift(inputs[0])
            elif mod.usePTnativeQfunc:
                qinput = mod.qa_raw_qfunc(inputs[0])
            else:
                qinput = mod.qa_fmo_mo_qfunc(inputs[0])

            # inputs is a tuple, QLinear only has 1 valid input
            cache_dict[mod_name]["input"] = inputs[0].to(self.cache_dev)
            cache_dict[mod_name]["cva"] = mod.cvs[0].to(self.cache_dev)
            cache_dict[mod_name]["cvw"] = mod.cvs[2].to(self.cache_dev)
            cache_dict[mod_name]["qinput"] = qinput.to(self.cache_dev)
            cache_dict[mod_name]["qweight"] = mod.weight.to(self.cache_dev)

    def call_func_for_nnlinear(self, mod, inputs, outputs, **_kwargs):
        mod_name = self.mod_name
        cache_dict = self.cache_dict
        cache_dict[mod_name]["input"] = inputs[0].to(self.cache_dev)
        cache_dict[mod_name]["weight"] = mod.weight.to(self.cache_dev)

    def __call__(self, mod, inputs, outputs, **_kwargs):
        self.fwd_mapping[type(mod)](mod, inputs, outputs, **_kwargs)


class PTQHookRecQOut(nn.Module):
    """This hook is for ptq_loss_func == 'fisher_diag' and will temporarily hold the "Q_out" of the
    module"""

    def __init__(self, qcfg):
        super().__init__()
        self.qcfg = qcfg

    def __call__(self, mod, x, output):
        self.qcfg["cached_qout"] = (
            output  # hold only 1 and no detach() or cpu(), we need to do backward on this
        )


class PTQHookRecGrad(nn.Module):
    def __init__(self, qcfg):
        super().__init__()
        self.qcfg = qcfg

    def __call__(self, mod, grad_input, grad_output):
        self.qcfg["cached_grad_out"] = grad_output[
            0
        ]  # hold only 1 and no detach() or cpu(), we need to do backward on this


def update_train_or_ptq_mode(
    mod,
    ptqmode=None,
    set_mod_state="train",
    submod_names=["dummy"],
    class2change=(nn.Conv2d,),
):
    # ptqmode is either fp32_out or q_out
    #    layers = [mod] if isinstance(mod, class2change) else
    #       [c for c in mod.children() if isinstance(c, class2change)]
    layers = [
        c for c in mod.modules() if isinstance(c, class2change)
    ]  # m.modules() will include itself, too. should work for both layer-wise and block-wise
    for l in layers:
        # --- PTQ state --- if submod is specified (using str), like 'quantize_calib_feature'
        if ptqmode in ["fp32_out", "q_out"]:
            l.ptqmode = ptqmode

        for smname in submod_names:
            sm = getattr(l, smname, l)  # default to layer itself
            # --- module state --- regardless its original state,
            # just set it to train() or eval() when asked.
            if set_mod_state == "train":
                sm.train()
            elif set_mod_state == "eval":
                sm.eval()
            else:
                # keep PTQ.mode unchanged
                continue
    return layers  # in case u need to know which layers are changed


def ptq_mod_optim(m, layers, qcfg, optim_mode="both", **kwargs):
    #    from detectron2.utils.events import EventStorage
    """
    Block-wise PTQ optimization mainly for vision models.
    """
    curr_dev = kwargs["curr_dev"]
    loss_func = kwargs["loss_func"]
    mod_name = kwargs["mod_name"]
    Nsteps2acc = kwargs["Nsteps2acc"]
    isTransformers = hasattr(
        qcfg, "cached_input1"
    )  # will need special handling for transformers
    isLastFC = (
        isinstance(m, nn.Linear) and mod_name == "fc"
    )  # we may want to special handle the loss of last FC

    param_names = [[], [], [], []]  # 0=weights, 1=Qa, 2=Qw, 3=bias

    #    if qcfg['PTQ_freezeweight']: qcfg['ptq_lrw']=0.0
    ws, cva, cvw, biases = [], [], [], []
    for idx, l in enumerate(layers):
        if optim_mode != "Wonly":
            if hasattr(l, "quantize_feature"):
                if hasattr(l.quantize_feature, "clip_val"):
                    cva.append(l.quantize_feature.clip_val)
                    param_names[1].append(f"cv{idx}")
                if hasattr(l.quantize_feature, "clip_valn"):
                    cva.append(l.quantize_feature.clip_valn)
                    param_names[1].append(f"cvn{idx}")
                if hasattr(l.quantize_feature, "delta"):
                    cva.append(l.quantize_feature.delta)
                    param_names[1].append(f"delta{idx}")
            elif hasattr(l, "quantize_m1"):
                if hasattr(l.quantize_m1, "clip_val"):
                    cva.append(l.quantize_m1.clip_val)
                    param_names[1].append(f"cv{idx}")
                if hasattr(l.quantize_m2, "clip_val"):
                    cva.append(l.quantize_m2.clip_val)
                    param_names[1].append(f"cv{idx}")
                if hasattr(l.quantize_m1, "clip_valn"):
                    cva.append(l.quantize_m1.clip_valn)
                    param_names[1].append(f"cvn{idx}")
                if hasattr(l.quantize_m2, "clip_valn"):
                    cva.append(l.quantize_m2.clip_valn)
                    param_names[1].append(f"cvn{idx}")
            else:
                logger.info(f"Layer {l} has no trainable parameter for quantization")
        if optim_mode != "Aonly" and (not hasattr(l, "quantize_m1")):
            # sym BRECQ or PACT+ for weight
            if "brecq" in l.qw_mode:
                cvw.append(l.quantize_weight.delta)
                param_names[2].append(f"deltaW{idx}")
            if "adaround" in l.qw_mode:
                cvw.append(l.quantize_weight.alpha)
                param_names[2].append(f"alphaW{idx}")
            if "pact+" in l.qw_mode:
                cvw.append(l.quantize_weight.clip_val)
                param_names[2].append(f"cvW{idx}")
            if not qcfg["PTQ_freezeweight"]:
                ws.append(l.weight)
                param_names[0].append(f"W{idx}")
        if not hasattr(l, "quantize_m1"):
            if l.bias is not None and not qcfg["PTQ_freezebias"]:
                biases.append(l.bias)
                param_names[3].append(f"bias{idx}")

    optim_w = torch.optim.Adam(
        [
            {
                "params": ws,
                "lr": qcfg["ptq_lrw"],
            },  # default lr was 1e-5 for AdaQuant, BRECQ didn't optimize weights
            {"params": cvw, "lr": qcfg["ptq_lrcv_w"]},  # 1e-3 for BRECQ
            {"params": biases, "lr": 1e-3},
        ]
    )  # default is 1e-3 from AdaQuant
    # separate w and a optimizer as in QDROP
    optim_a = torch.optim.Adam(
        [
            {
                "params": cva,
                "lr": qcfg["ptq_lrcv_a"],
                "weight_decay": qcfg["pact_a_decay"],
            }
        ]
    )  # lr was 1e-1 or 1e-3 in AdaQuant, 4e-5 for BRECQ

    scheduler = []
    if "W" in qcfg["ptq_coslr"]:
        scheduler.append(
            torch.optim.lr_scheduler.CosineAnnealingLR(
                optim_w, T_max=qcfg["ptq_nouterloop"], eta_min=0.0
            )
        )
    if "A" in qcfg["ptq_coslr"]:
        scheduler.append(
            torch.optim.lr_scheduler.CosineAnnealingLR(
                optim_a, T_max=qcfg["ptq_nouterloop"], eta_min=0.0
            )
        )

    # NOTE typical shuffle is like
    # data_seq = torch.randperm(qcfg['ptq_nbatch']).repeat(
    #               qcfg['ptq_nouterloop']//qcfg['ptq_nbatch'] +1 )
    # fine-grained shuffling
    data_seq = [
        torch.randperm(qcfg["ptq_nbatch"] * qcfg["ptq_batchsize"])
        for _ in range(qcfg["ptq_nouterloop"] // qcfg["ptq_nbatch"] + 1)
    ]
    data_seq = torch.cat(data_seq).reshape([-1, qcfg["ptq_batchsize"]])

    pbar_desc = f"Phase 2.2: PTQ optimizing module {mod_name}. loss="
    pbar2 = tqdm(
        data_seq[: qcfg["ptq_nouterloop"]],
        desc=pbar_desc + "     ",
        leave=False,
        mininterval=5,
    )
    # prep for grad accum
    optim_w.zero_grad()
    optim_a.zero_grad()
    for i_outer, data_idx in enumerate(pbar2):
        # fetch the cached data
        if isTransformers:  # special handle for transformers
            inp = (
                qcfg["cached_input0"][data_idx].to(curr_dev),
                qcfg["cached_input1"][data_idx].to(curr_dev),
            )
        else:
            inp = qcfg["cached_input"][data_idx].to(curr_dev)
        fp_out = qcfg["cached_output"][data_idx].to(curr_dev)
        # gt will be only used for last FC layer, e.g. calc ce loss
        gt = qcfg["cached_lbls"][data_idx].to(curr_dev) if isLastFC else None

        # --- mask is for Qdrop only
        if qcfg["ptq_qdrop"]:
            dropout_mask_in = torch.bernoulli(
                torch.full_like(inp, 0.5)
            ).bool()  # FIXME use variable dropout rate as in NWQ?

        for j_inner in range(qcfg["ptq_ninnerloop"]):
            grad = None
            qcfg["cached_qout"] = []
            qcfg["cached_grad_out"] = []
            Niter = i_outer * qcfg["ptq_ninnerloop"] + j_inner

            if isTransformers:
                with patch_torch_bmm(qcfg):
                    q_out = m(*inp)
            else:
                q_out = m(inp)
                # if qcfg['cached_qout']==[] else qcfg['cached_qout']
                # # run module(input) if not cached already
            # --- Qdrop implemented here ---
            if qcfg["ptq_qdrop"]:
                # "inp" in the case of Qdrop is actually "all fp32" input
                # (i.e., all prev mods are set to fp32_out, not "sequential")
                qinp = qcfg["cached_qinput"][data_idx].to(
                    curr_dev
                )  # this is the real "qinput", where all previous modules are quantized
                mixed_inp = torch.where(dropout_mask_in, qinp, inp)
                q_out = m(mixed_inp)

            if isTransformers:
                PTQloss = loss_func(q_out[0], fp_out, grad)  # *loss_scaling_acc
                logger.info(f"Loss is {PTQloss}")
            else:
                PTQloss = loss_func(
                    q_out, fp_out, grad=grad, gt=gt
                )  # *loss_scaling_acc
                # only "fisher_diag" needs "grad", only "ce loss" needs gt

            loss4plot = {}
            # some loss func has more than 1 term (like mse+ssim),
            # will return a dict then we can plot each term in TB
            if isinstance(PTQloss, (dict)):
                loss4plot = {
                    k: v.item()
                    for k, v in PTQloss.items()
                    if isinstance(v, torch.Tensor)
                }
                PTQloss = (
                    PTQloss["total"]
                    if "total" in PTQloss
                    else torch.sum(PTQloss.values())
                )
            else:  # if only one term, plot it with the name of the loss, e.g. mse, ssim
                loss4plot[qcfg["ptq_loss_func"]] = PTQloss.item()

            PTQloss.backward()  # retain_graph=True if qcfg['ptq_ninnerloop']>1 else False)

            # accumulate grads over Nimgs2acc, usually 2 imgs per GPU, prefer to accum 16 imgs
            if (Niter + 1) % Nsteps2acc == 0 or (Niter + 1 == qcfg["ptq_nouterloop"]):
                optim_w.step()
                optim_w.zero_grad()
                optim_a.step()
                optim_a.zero_grad()

            # --- tensorboard output
            if qcfg["tb_writer"] and (
                (qcfg["ptq_ninnerloop"] == 1 and Niter % 100 == 0)
                or (qcfg["ptq_ninnerloop"] > 1 and i_outer % 10 == 0)
            ):
                # show loss on pbar
                pbar2.set_description(pbar_desc + f"{PTQloss:.3f}")
                # plot loss
                for k, v in loss4plot.items():
                    qcfg["tb_writer"].add_scalar(f"{mod_name}/PTQloss_{k}", v, Niter)

                # plot cv, delta, zp, alpha, and lr
                for k, v in m.named_buffers():
                    if any(kb in k for kb in ["delta", "zero_point", "clip_val"]):
                        if len(v.shape) > 0 and v.shape[0] > 1:  # perCh
                            qcfg["tb_writer"].add_histogram(f"{mod_name}/{k}", v, Niter)
                        else:
                            qcfg["tb_writer"].add_scalar(f"{mod_name}/{k}", v, Niter)

                for p, pname in zip(
                    optim_a.param_groups[0]["params"], param_names[1]
                ):  # cva
                    qcfg["tb_writer"].add_scalar(f"{mod_name}/{pname}", p, Niter)
                    qcfg["tb_writer"].add_scalar(
                        f"{mod_name}/LR_cv_a", optim_a.param_groups[0]["lr"], Niter
                    )
                for p, pname in zip(
                    optim_w.param_groups[0]["params"], param_names[0]
                ):  # weights
                    qcfg["tb_writer"].add_histogram(f"{mod_name}/{pname}", p, Niter)
                    qcfg["tb_writer"].add_scalar(
                        f"{mod_name}/LR_w", optim_w.param_groups[0]["lr"], Niter
                    )
                for p, pname in zip(
                    optim_w.param_groups[1]["params"], param_names[2]
                ):  # cvw
                    if "alpha" in pname:
                        qcfg["tb_writer"].add_histogram(f"{mod_name}/{pname}", p, Niter)
                    else:
                        qcfg["tb_writer"].add_scalar(f"{mod_name}/{pname}", p, Niter)
                    qcfg["tb_writer"].add_scalar(
                        f"{mod_name}/LR_cvw", optim_w.param_groups[1]["lr"], Niter
                    )
                if "adaround" in qcfg["qw_mode"]:
                    curr_beta = loss_func.beta_decay(loss_func.count)
                    qcfg["tb_writer"].add_scalar(
                        f"{mod_name}/AdaR_beta", curr_beta, Niter
                    )
                    for lidx, l in enumerate(layers):
                        if not hasattr(l, "quantize_m1"):
                            qcfg["tb_writer"].add_histogram(
                                f"{mod_name}/W{lidx}", l.weight, Niter
                            )
                            if hasattr(l.quantize_weight, "get_hard_targets"):
                                nzs = torch.count_nonzero(
                                    l.quantize_weight.get_soft_targets()
                                    - l.quantize_weight.get_hard_targets()
                                )
                                qcfg["tb_writer"].add_scalar(
                                    f"{mod_name}/W{lidx}_AdaR_nonzeros(soft-hard)%",
                                    nzs / l.weight.numel(),
                                    Niter,
                                )
                # almost never we will use bias in optimizer,
                # unless we do bn folding and optimize w and b both

        for s in scheduler:
            s.step()  # we set up scheduler based on Nouterloop, not inner
    #        if profiler: profiler.step() # for debug only

    # Once finish optimizing this module,
    # set all AdaR (if any) to real quantizer (soft_target = False)
    if "adaround" in qcfg["qw_mode"]:
        for l in layers:
            if not hasattr(l, "quantize_m1") and hasattr(
                l.quantize_weight, "soft_targets"
            ):
                l.quantize_weight.soft_targets = False

    return PTQloss


def calib_ptq_bn_tune(
    qcfg, model, loader, PTQmod_candidates, batch_size, pre_cache_func=None
):
    #    from detectron2.utils.events import EventStorage
    # from detectron2.layers import FrozenBatchNorm2d
    # --- Prep --- set up calib, PTQ post-fwd-hooks, can set up block-wise optimization as well

    if qcfg["PTQ_fold_BN"]:
        mods_folded = []
        search_fold_and_remove_bn(model, mods_folded)
        logger.info(f"--- Quantized model after BN folding--- \n {model}\n")
    else:
        BNmods = [
            m
            for k, m in model.named_modules()
            if isinstance(m, nn.BatchNorm2d) or "norm" in k
        ]

    # re-init alpha and delta for all adaR in case any changes in weights
    # weight changes could be due to a) bn folding or b) load pre-trained after qmodel_prep
    logger.info(
        " --- check and re-initialize AdaRound delta and alpha for all layers in PTQmod_candidates"
    )
    for m in PTQmod_candidates:
        # all the sub-modules, including quantizers, and m itself, will be included in m.modules()
        for sm in m.modules():
            if isinstance(sm, (QConv2dPTQv2, QLinear)) and "adaround" in sm.qw_mode:
                sm.quantize_weight.init_delta(sm.weight, sm.qw_mode)
                sm.quantize_weight.init_alpha(sm.weight)

    curr_dev = next(model.parameters()).device
    if qcfg["PTQ_keepBNfrozenDuringOptim"]:
        model.eval()
    torch.set_grad_enabled(False)

    # --- Phase 0 --- cache images

    stratified_loader = False
    loader_len = (
        len(loader.dataset.dataset.dataset)
        if "detectron2.modeling" in sys.modules
        else len(loader)
    )  # NOTE detectron2 needs special handling
    if (
        qcfg["ptq_nbatch"] > 0 and loader_len < batch_size * qcfg["ptq_nbatch"]
    ):  # if we need more than what loader has -> cache all
        # NOTE original training set has ~117000 images, stratified subset
        # will be a little larger than ptq_nbatch
        stratified_loader = True
        qcfg["ptq_nbatch"] = loader_len  # cache all images in the loader
        Nbatch_to_cache = qcfg["ptq_nbatch"]
    else:
        Nbatch_to_cache = max(
            qcfg["ptq_nbatch"], qcfg["qmodel_calibration_new"] + qcfg["BN_tune"]
        )
    #     cache images (to be placed on CPU mem)
    qcfg["cached_imgs"] = []
    qcfg["cached_lbls"] = []
    pbar = tqdm(
        loader, desc="Phase 0: PTQ caching images from loader", total=Nbatch_to_cache
    )
    for data_mb, _ in zip(pbar, range(Nbatch_to_cache)):
        if pre_cache_func is not None:
            imgs = pre_cache_func(data_mb)
            qcfg["cached_imgs"].append(imgs)
        else:
            imgs, lbls = data_mb  # unpack (imgs, lbls)
            qcfg["cached_imgs"].append(imgs)
            qcfg["cached_lbls"].append(lbls)
    Nimgs_per_batch = len(qcfg["cached_imgs"][0])
    # --- prep for fine-grained shuffling, cat [tensor(NCHW), tensor(NCHW), ...]
    # into a single tensor(Nbatch,NCHW)only works for same size imgs, e.g. ImgNet !!
    qcfg["cached_imgs"] = torch.stack(qcfg["cached_imgs"])
    # using torch.stack, final shape for cached_imgs = [num_batch, batchsize, C, H, W]
    # but in PTQoptim, "cached_input" will be 1) torch.cat into
    # [num_batch*batchsize, C,H,W] then 2) shuffled
    qcfg["cached_lbls"] = torch.cat(
        qcfg["cached_lbls"]
    )  # easier if we just torch.cat lables into shape of [num_batch*batchsize]

    # --- Phase 1 --- calibration of clip vals

    if qcfg["qmodel_calibration_new"] > 0:
        if not qcfg["PTQ_fold_BN"]:
            BNmean = [m.running_mean.mean() for m in BNmods]
            logger.info(
                "Before calibration, (BN running mean).abs().mean() =",
                torch.stack(BNmean).abs().mean(),
            )

        # set all QConv2d.Qdynamic under model to training mode, so that they
        # will calc and update clip_vals
        update_train_or_ptq_mode(model, set_mod_state="train", class2change=Qdynamic)
        # this func does the following things:
        # 1) make a list of m and its children if they are instances of class2change
        # 2) set those layers' ptqmode to the given mode, if 'PTQmod=xxx' is specified and is
        #       in ['fp32_out', 'q_out']
        # 3) for each layer, set layer.train() or layer.eval() if set_mod_state is specified.

        pbar = tqdm(
            qcfg["cached_imgs"],
            desc="Phase 1: calibration",
            total=qcfg["qmodel_calibration_new"],
        )
        for data_mb, Niters in zip(pbar, range(qcfg["qmodel_calibration_new"])):
            if isinstance(data_mb, torch.Tensor):
                data_mb = data_mb.to(
                    curr_dev
                )  # usually detectron2 will move data to device for us

            model(
                data_mb
            )  # just fwd(), no need to save outputs. kwargs for yolo test is just augment=True

            # record clipvals
            cv_sum_table = {}
            Qmods = {
                k: m
                for k, m in model.named_modules()
                if isinstance(m, (QConv2dPTQv2, QLinear))
            }
            for modname, m in Qmods.items():
                cv_sum_table[modname] = [
                    None,
                    None,
                    None,
                    None,
                ]  # will store "cv_a, cvn_a, cv_w, cvn_w"
                Qparams = {k: v for k, v in m.named_parameters() if "quantize_" in k}
                for k, v in Qparams.items():
                    if "alpha" not in k:
                        var_name = k.split("quantize_")[1]
                        var_idx = ("weight" in var_name) * 2 + (
                            "clip_valn" in var_name or "zero_point" in var_name
                        )
                        cv_sum_table[modname][var_idx] = v.item()
                        qcfg["tb_writer"].add_scalar(f"{modname}/{var_name}", v, Niters)
                    else:
                        # special handle for adaround, delta and zp are buffers, not parameters,
                        # use mean() in case perCh
                        cv_sum_table[modname][2] = m.quantize_weight.delta.mean().item()
                        cv_sum_table[modname][3] = (
                            m.quantize_weight.zero_point.mean().item()
                        )
                        qcfg["tb_writer"].add_scalar(
                            f"{modname}/delta", m.quantize_weight.delta.mean(), Niters
                        )

        pd.options.display.float_format = "{:.4f}".format
        dfCV = pd.DataFrame(cv_sum_table).T
        dfCV.columns = (
            ["cv_a", "cvn_a", "cv_w", "cvn_w"]
            if qcfg["qw_mode"] != "adaround"
            else ["cv_a", "cvn_a", "w_delta", "w_zp"]
        )
        logger.info(dfCV)

    if not qcfg["PTQ_fold_BN"]:
        BNmeanAfterCalib = {
            k: m.running_mean.mean()
            for k, m in model.named_modules()
            if isinstance(m, (nn.BatchNorm2d,))  # FrozenBatchNorm2d,
        }
        logger.info(
            f"After calibration {qcfg['qmodel_calibration_new']},"
            "(BN running mean).abs().mean() =",
            torch.stack(list(BNmeanAfterCalib.values())).abs().mean(),
        )

    # --- Phase 2 --- PTQ

    if (
        qcfg["ptq_nbatch"] > 0 and qcfg["ptq_nouterloop"] > 0
    ):  # default Ninner = 1 if not specified
        Nsteps2acc = max(
            qcfg.get("PTQ_Nimgs2acc", Nimgs_per_batch) // Nimgs_per_batch, 1
        )
        loss_scaling_acc = 1.0 / Nsteps2acc
        Ntotal_iters = qcfg["ptq_nouterloop"] * qcfg["ptq_ninnerloop"]
        Nouter_new = math.ceil(
            qcfg["ptq_nouterloop"] / np.lcm(Nsteps2acc, qcfg["ptq_nbatch"])
        ) * np.lcm(Nsteps2acc, qcfg["ptq_nbatch"])

        if stratified_loader:
            logger.info(
                f"Using stratified dataloader, Nouterloop is adjusted from"
                f"{qcfg['ptq_nouterloop']} to {Nouter_new}"
            )
            qcfg["ptq_nouterloop"] = Nouter_new

        # in detectron2, only model.train() will output losses,
        # otherwise only output predictions (instances)
        if "fisher" in qcfg["ptq_loss_func"]:
            model.train()

        torch.set_grad_enabled(True)
        pbar = tqdm(PTQmod_candidates, desc="Phase 2.1: PTQ optimization")
        qcfg[
            "cached_output_model"
        ] = []  # this is for fp32 model output, only needed for 'fisher_dias' loss func
        previously_optimized_mods = []
        for m in pbar:
            layers = update_train_or_ptq_mode(
                m,
                ptqmode="fp32_out",
                set_mod_state="train",
                class2change=(QConv2dPTQv2, QLinear, QBmm),
            )
            # When we set QConv2dPTQv2 (i.e., DetQConv2d) to train(), it will
            # make FrozenBN layer.training (under DetQConv2d) True
            for subm in m.modules():  # make sure all BN is under correct mode
                if (
                    isinstance(subm, nn.BatchNorm2d)  # FrozenBatchNorm2d)
                    and qcfg["PTQ_keepBNfrozenDuringOptim"]
                ):
                    subm.eval()  # set them back to eval() again
                # otherwise, BN will be updated

            # this input is from all prev modules being quantized
            qcfg["cached_input"] = []
            # this is the output when this module is set to fp32_out (this module isn't quantized)
            qcfg["cached_output"] = []
            mod_name = qcfg["LUTmodule_name"][m]

            h = m.register_forward_hook(
                PTQHookRecInOut(
                    qcfg, name=mod_name, cls2rec=(QConv2dPTQv2, QLinear, QBmm)
                )
            )

            pbar2 = tqdm(
                qcfg["cached_imgs"],
                desc=f"Phase 2.1: PTQ caching intermediate input/output for {mod_name}.",
                leave=False,
                mininterval=5,
                miniters=int(len(qcfg["cached_imgs"]) / 50),
                maxinterval=200,
            )
            # with EventStorage(0) as storage: # FIXME this EventStorage is specific for detectron2
            for data_mb in pbar2:
                if isinstance(data_mb, torch.Tensor):
                    data_mb = data_mb.to(
                        curr_dev
                    )  # usually detectron2 will move data to device for us
                fpout_model = model(
                    data_mb
                )  # cached_input and _output will be recorded by the hook
                if "fisher" in qcfg["ptq_loss_func"] and len(
                    qcfg["cached_output_model"]
                ) < len(qcfg["cached_imgs"]):
                    qcfg["cached_output_model"].append(
                        sum(fpout_model.values()).detach()
                    )  # no need to put it on cpu, as this is one scalar only
            # --- Qdrop implemented here
            if qcfg["ptq_qdrop"]:
                # need to record "real" FP32 input, i.e. set PTQ_mode in
                # "all previous modules" back to fp32_out
                qcfg["cached_qinput"] = qcfg[
                    "cached_input"
                ]  # this existing input is recorded with all previous modules being quantized
                qcfg["cached_input"] = []
                h.remove()
                h = m.register_forward_hook(
                    PTQHookRecInOut(
                        qcfg,
                        name=mod_name,
                        cls2rec=(QConv2dPTQv2, QLinear),
                        recInOnly=True,
                    )
                )  # cached_input only this time

                for mm in previously_optimized_mods:
                    # careful, "m" is being used, don't mess it up
                    # with EventStorage(0) as storage:
                    # # FIXME this EventStorage is specific for detectron2
                    update_train_or_ptq_mode(
                        mm, ptqmode="fp32_out", class2change=(QConv2dPTQv2, QLinear)
                    )

                # cached_input now is calc'ed from ptqmode=fp32
                for data_mb in pbar2:
                    model(data_mb)

                for mm in previously_optimized_mods:
                    update_train_or_ptq_mode(
                        mm, ptqmode="q_out", class2change=(QConv2dPTQv2, QLinear)
                    )

            for l in layers:
                if not qcfg["ptq_qdrop"]:
                    del l.W_fp  # release some unused memory
                l.ptqmode = "q_out"
            h.remove()

            # --- prep for fine-grained shuffling, cat [tensor(NCHW), tensor(NCHW), ...]
            # into a single tensor(N*Nbatch,CHW)
            if qcfg["ptq_qdrop"]:
                qcfg["cached_qinput"] = torch.cat(qcfg["cached_qinput"])
            qcfg["cached_input"] = torch.cat(qcfg["cached_input"])
            qcfg["cached_output"] = torch.cat(qcfg["cached_output"])

            # 2-2. start to optimize this module
            argdict = {
                "curr_dev": curr_dev,
                "mod_name": mod_name,
                "loss_func": PTQLossFunc(
                    qcfg["ptq_loss_func"],
                    Ntotal_iters=Ntotal_iters,
                    layers=layers,
                    adaR_anneal=qcfg[
                        "PTQ_adaR_annealing"
                    ],  # defines warmup, hold, beta, lambda, etc...
                    isOptimConv=not isinstance(m, nn.Linear),
                ),
                # default is MSE, can choose others. warmup and beta are for adaR only
                "loss_scaling_acc": loss_scaling_acc,
                "Nsteps2acc": Nsteps2acc,
                # assume PTQ_candidates list is ordered
                "opt_lastFC": isinstance(m, (QLinear, nn.Linear))
                and m is PTQmod_candidates[-1]
                and qcfg.get("PTQ_lastFC", False),
            }
            if qcfg.get("PTQ_brecq_style", None):
                # 1st round: WxA32
                # this mode will force QConv2d to bypass activation quantizer
                for l in layers:
                    l.ptqmode = "A32_brecq"
                ptq_mod_optim(
                    m, layers, qcfg, optim_mode="Wonly", **argdict
                )  # optim_mode can be ['Wonly', 'Aonly','both'...]
                # 2nd round WxAy
                for l in layers:
                    l.ptqmode = "q_out"
                ptq_mod_optim(m, layers, qcfg, optim_mode="Aonly", **argdict)
            else:
                ptq_mod_optim(
                    m, layers, qcfg, optim_mode="both", **argdict
                )  # , profiler=p
                #   exit()

            # finally, add current block/module to "finished" list
            previously_optimized_mods.append(m)

    if not qcfg["PTQ_fold_BN"]:
        BNmeanAfterPTQ = {
            k: m.running_mean.mean()
            for k, m in model.named_modules()
            if isinstance(m, (nn.BatchNorm2d,))  # FrozenBatchNorm2d,
        }
        dfBNmean = pd.DataFrame(
            {
                k: [BNmeanAfterCalib[k].cpu(), BNmeanAfterPTQ[k].cpu()]
                for k in BNmeanAfterCalib.keys()
            },
            index={"after calib", "after PTQ"},
        )
        logger.info(dfBNmean.T)

    # --- Phase 3 --- BN tuning

    if not qcfg["PTQ_fold_BN"] and qcfg["BN_tune"] > 0:
        BNmeanb4 = [m.running_mean.mean() for m in BNmods]
        torch.set_grad_enabled(False)
        # because YOLO has a special case for model.train(),
        # we have to manually change individual BN layers
        for m in BNmods:
            m.train()

        # NOTE if ptq_nbatch>0, we will use the image data we cached, so
        # if qcfg['BN_tune'] > qcfg['ptq_nbatch'], we will reuse images again
        data_seq = torch.randperm(qcfg["ptq_nbatch"]).repeat(
            math.ceil(qcfg["BN_tune"] / qcfg["ptq_nbatch"])
        )
        for i in tqdm(
            range(qcfg["BN_tune"]), desc="Phase 3: BN tuning", total=qcfg["BN_tune"]
        ):
            data_mb = qcfg["cached_imgs"][data_seq[i]].to(curr_dev)
            model(data_mb)

            BNmeanafter = [m.running_mean.mean() for m in BNmods]
            meanBNchange = torch.stack(BNmeanb4) - torch.stack(BNmeanafter)
            logger.info(
                f"After {i} iters of BN tuning, delta(BN running mean).abs().mean() =",
                meanBNchange.abs().mean(),
            )

    model.eval()


def update_ptq_mode(mod, ptqmode=None, class2change=(nn.Conv2d,)):
    """
    Set ptqmode to be either fp32_out or q_out
    """
    layers = [c for c in mod.modules() if isinstance(c, class2change)]
    for l in layers:
        # --- PTQ state --- if submod is specified (using str), like 'quantize_calib_feature'
        if ptqmode in ["fp32_out", "q_out"]:
            l.ptqmode = ptqmode


def remove_wfp(mod, class2change=(nn.Conv2d,)):
    """
    Remove full precision weight copies in PTQ
    """
    layers = [c for c in mod.modules() if isinstance(c, class2change)]
    for l in layers:
        l.W_fp = None


def get_layers(mod, layer2get=(nn.LayerNorm,)):
    """
    Get sub_layers in a module
    """
    layers = [c for c in mod.modules() if isinstance(c, layer2get)]
    return layers


def update_train_or_eval_mode(
    mod,
    set_mod_state="train",
    class2change=(nn.Conv2d,),
):
    """
    Change module mode to train or eval.
    """
    layers = [c for c in mod.modules() if isinstance(c, class2change)]
    for l in layers:
        if set_mod_state == "train":
            l.train()
        elif set_mod_state == "eval":
            l.eval()
        else:
            continue


def ptq_mod_optim_lm(_model, m, layers, qcfg, optim_mode="both", **kwargs):
    """
    Block-sequential PTQ optimization mainly for small size models such as Bert/Roberta.
    """
    curr_dev = kwargs["curr_dev"]
    loss_func = kwargs["loss_func"]
    mod_name = kwargs["mod_name"]
    Nsteps2acc = kwargs["Nsteps2acc"]
    fine_grain_shuffle = kwargs["fg_shuffling"]

    param_names = [[], [], [], []]  # 0=weights, 1=Qa, 2=Qw, 3=bias

    ws, cva, cvw, biases = [], [], [], []
    parm_shortname = {
        "clip_val": "cv",
        "clip_valn": "cvn",
        "delta": "delta",
    }  # LUT for short names
    for idx, l in enumerate(layers):
        if optim_mode != "Wonly":
            # collect activation related parameters so we can put them into optimizer later
            for qtzr_name in ["quantize_feature", "quantize_m1", "quantize_m2"]:
                quantizer = getattr(l, qtzr_name, None)
                if quantizer is None:
                    continue

                for attr_name in ["clip_val", "clip_valn", "delta"]:
                    quant_attr = getattr(quantizer, attr_name, None)
                    if quant_attr is None:
                        continue

                    shortname = parm_shortname[attr_name]
                    qtzr_name_ext = qtzr_name.split("_")[1]
                    if qtzr_name_ext in ["m1", "m2"]:
                        shortname = qtzr_name_ext + shortname

                    cva.append(quant_attr)
                    param_names[1].append(f"{shortname}{idx}")

        if optim_mode != "Aonly" and (not hasattr(l, "quantize_m1")):
            # sym BRECQ or PACT+ for weight
            if "brecq" in l.qw_mode:
                cvw.append(l.quantize_weight.delta)
                param_names[2].append(f"deltaW{idx}")
            if "adaround" in l.qw_mode:
                cvw.append(l.quantize_weight.alpha)
                param_names[2].append(f"alphaW{idx}")
            if "pact+" in l.qw_mode:
                cvw.append(l.quantize_weight.clip_val)
                param_names[2].append(f"cvW{idx}")
            # NOTE SAWB has no trainable parameters
            if not qcfg.get("PTQ_freezeweight", False):
                ws.append(l.weight)
                param_names[0].append(f"W{idx}")
        if not hasattr(l, "quantize_m1"):
            if l.bias is not None and not qcfg.get("PTQ_freezebias", False):
                biases.append(l.bias)
                param_names[3].append(f"bias{idx}")

    # --- user could customize PTQ parameters in qcfg["qspecial_layers"]
    if mod_name in qcfg["qspecial_layers"]:
        cus_ptq_params = {
            k: v
            for k, v in qcfg["qspecial_layers"][mod_name].items()
            if k.startswith("PTQ_")
        }
        backup_ptq_params = {k: qcfg[k] for k in cus_ptq_params if k in qcfg}
        extra_ptq_params_2_del = [k for k in cus_ptq_params if k not in qcfg]
        qcfg.update(cus_ptq_params)

    optim_w = torch.optim.Adam(
        [
            {
                "params": ws,
                "lr": qcfg["ptq_lrw"],
            },
            {"params": cvw, "lr": qcfg["ptq_lrcv_w"]},
            {"params": biases, "lr": qcfg.get("PTQ_LRbias_w", 1e-3)},
        ]
    )
    # separate w and a optimizer as in QDROP
    optim_a = torch.optim.Adam(
        [
            {
                "params": cva,
                "lr": qcfg["ptq_lrcv_a"],
                "weight_decay": qcfg["pact_a_decay"],
            }
        ]
    )

    scheduler = []
    if "W" in qcfg["ptq_coslr"]:
        scheduler.append(
            torch.optim.lr_scheduler.CosineAnnealingLR(
                optim_w, T_max=qcfg["ptq_nouterloop"], eta_min=0.0
            )
        )
    if "A" in qcfg["ptq_coslr"]:
        scheduler.append(
            torch.optim.lr_scheduler.CosineAnnealingLR(
                optim_a, T_max=qcfg["ptq_nouterloop"], eta_min=0.0
            )
        )

    # fine-grained shuffling is not recommended
    if fine_grain_shuffle:
        data_seq = [
            torch.randperm(qcfg["ptq_nbatch"] * qcfg["ptq_batchsize"])
            for _ in range(qcfg["ptq_nouterloop"] // qcfg["ptq_nbatch"] + 1)
        ]
        data_seq = torch.cat(data_seq).reshape([-1, qcfg["ptq_batchsize"]])
    else:
        data_seq = [
            torch.randperm(qcfg["ptq_nbatch"])
            for _ in range(qcfg["ptq_nouterloop"] // qcfg["ptq_nbatch"] + 1)
        ]
        data_seq = torch.cat(data_seq)  # will be a 1D tensor, simply batch_index

    pbar_desc = f"Phase 2.2: PTQ optimizing module {mod_name}. loss="
    pbar2 = tqdm(
        data_seq[: qcfg["ptq_nouterloop"]],
        desc=pbar_desc + "     ",
        leave=False,
        mininterval=5,
    )
    # prep for grad accum
    optim_w.zero_grad()
    optim_a.zero_grad()
    for i_outer, data_idx in enumerate(pbar2):
        if fine_grain_shuffle:
            inp = []
            for cached_inp_i in qcfg[
                "cached_input"
            ]:  # a list of [input0_new, input1_new, ...]
                inp.append(cached_inp_i[data_idx].to(curr_dev))
            len_inp = len(inp)
            inp = tuple(inp)  # turn a list into a tuple
            assert len(inp) == len_inp  # sometimes tuple() can mess up torch.Tensors
        else:
            # without fine-grain shuffling, cached input is just [b1, b2, b3 ...]
            # where b1 = (Tensor0, Tensor1, ...)
            inp = tuple(inp_i.to(curr_dev) for inp_i in qcfg["cached_input"][data_idx])

        # Note: unused-variable
        # if len(qcfg["attn_mask"]) > 0:
        #     attn_mask = qcfg["attn_mask"][data_idx].to(curr_dev)

        fp_out = qcfg["cached_output"][data_idx].to(curr_dev)
        # --- mask is for Qdrop only
        # Note: unused-variable
        # if qcfg["ptq_qdrop"]:
        #     dropout_mask_in = torch.bernoulli(torch.full_like(inp, 0.5)).bool()

        for j_inner in range(qcfg["ptq_ninnerloop"]):
            grad = None
            qcfg["cached_qout"] = []
            qcfg["cached_grad_out"] = []
            Niter = i_outer * qcfg["ptq_ninnerloop"] + j_inner

            with patch_torch_bmm(qcfg):
                q_out = m(*inp)

            # --- Qdrop implemented here ---
            if qcfg["ptq_qdrop"]:
                raise NotImplementedError

            if isinstance(q_out, tuple):
                PTQloss = loss_func(q_out[0], fp_out, grad)
            else:
                PTQloss = loss_func(q_out, fp_out, grad)

            loss4plot = {}
            if isinstance(PTQloss, (dict)):
                loss4plot = {k: v.item() for k, v in PTQloss.items()}
                PTQloss = (
                    PTQloss["total"]
                    if "total" in PTQloss
                    else torch.sum(PTQloss.values())
                )
            else:
                loss4plot[qcfg["ptq_loss_func"]] = PTQloss.item()

            PTQloss.backward()
            if (Niter + 1) % Nsteps2acc == 0 or (Niter + 1 == qcfg["ptq_nouterloop"]):
                optim_w.step()
                optim_w.zero_grad()
                optim_a.step()
                optim_a.zero_grad()

            # --- standard and tensorboard output
            isOutput = i_outer % 100 == 0

            if isOutput:
                # show loss on pbar
                pbar2.set_description(pbar_desc + f"{PTQloss:.6f}")

            if available_packages["tensorboard"]:
                # Third Party
                from torch.utils.tensorboard import SummaryWriter

                if isinstance(qcfg["tb_writer"], SummaryWriter) and isOutput:
                    scalars2log = {}
                    hist2log = {}

                    for k, v in loss4plot.items():  # plot loss
                        scalars2log[f"{mod_name}/PTQloss_{k}"] = v
                    for k, v in m.named_buffers():  # plot cv, delta, zp, alpha, and lr
                        if any(kb in k for kb in ["delta", "zero_point", "clip_val"]):
                            if len(v.shape) > 0 and v.shape[0] > 1:  # perCh
                                hist2log[f"{mod_name}/{k}"] = v
                            else:
                                scalars2log[f"{mod_name}/{k}"] = v
                    for p, pname in zip(
                        optim_a.param_groups[0]["params"], param_names[1]
                    ):  # cva
                        scalars2log[f"{mod_name}/{pname}"] = p.item()
                        scalars2log[f"{mod_name}/LR_cv_a"] = optim_a.param_groups[0][
                            "lr"
                        ]
                    for p, pname in zip(
                        optim_w.param_groups[0]["params"], param_names[0]
                    ):  # weights
                        hist2log[f"{mod_name}/{pname}"] = p
                        scalars2log[f"{mod_name}/LR_w"] = optim_w.param_groups[0]["lr"]
                    for p, pname in zip(
                        optim_w.param_groups[1]["params"], param_names[2]
                    ):  # cvw
                        if "alpha" in pname:
                            hist2log[f"{mod_name}/{pname}"] = p
                        else:
                            scalars2log[f"{mod_name}/{pname}"] = p.item()
                        scalars2log[f"{mod_name}/LR_cvw"] = optim_w.param_groups[1][
                            "lr"
                        ]
                    if "adaround" in qcfg["qw_mode"]:
                        scalars2log[f"{mod_name}/AdaR_beta"] = (
                            loss_func.temp_decay.curr_beta
                        )
                        for lidx, l in enumerate(layers):
                            if not hasattr(l, "quantize_m1"):
                                hist2log[f"{mod_name}/W{lidx}"] = l.weight

                    # write every in one shot will mess up the folder, better write them one by one
                    for n, v in scalars2log.items():
                        qcfg["tb_writer"].add_scalar(n, v, Niter)
                    for n, v in hist2log.items():
                        qcfg["tb_writer"].add_histogram(n, v, Niter)

        for s in scheduler:
            s.step()  # we set up scheduler based on Nouterloop, not inner

    # --- restore default PTQ parameters in qcfg
    if mod_name in qcfg["qspecial_layers"]:
        qcfg.update(backup_ptq_params)
        for k in extra_ptq_params_2_del:
            del qcfg[k]

    # Once finish optimizing this module, set all AdaR (if any)
    # to real quantizer (soft_target = False)
    if "adaround" in qcfg["qw_mode"]:
        for l in layers:
            if not hasattr(l, "quantize_m1") and hasattr(
                l.quantize_weight, "soft_targets"
            ):
                l.quantize_weight.soft_targets = False

    return PTQloss


def calib_PTQ_lm(
    qcfg,
    model,
    loader,
    PTQmod_candidates,
    fine_grain_shuffle=False,
    pre_cache_func=None,
):
    """
    PTQ for smaller LM
    args:
        qcfg
        model
        loader: dataloader or list of tensors
        PTQmod_candidates: a list of nn.Modules that we want to run PTQ on,
            can be Qlayers or blocks of Qlayers
        fine_grain_shuffle: allow mixing data between different batches to create new batch of data
            need to pad_to_max so that every batch has the same length, NOT RECOMMENDED.
        pre_cache_func: in case we need to do some additional process before sending data to model,
            e.g. slicing, extract only part of the dict...

        A few other parameters are stored in qcfg currently:
        qcfg['ptq_nbatch']: how many iterations of PTQ we want to run
    """
    curr_dev = next(model.parameters()).device

    # --- Phase 0 --- cache input tokens/sequence
    Ntotal_rows_data = len(loader) * qcfg["loader.batchsize"]
    if not fine_grain_shuffle:
        assert qcfg["loader.batchsize"] == qcfg["ptq_batchsize"]
        Nbatch_to_cache = qcfg["ptq_nbatch"]
    elif (
        qcfg["ptq_nbatch"] > 0
        and Ntotal_rows_data < qcfg["ptq_batchsize"] * qcfg["ptq_nbatch"]
    ):
        # if we need more than what loader has -> cache all
        raise RuntimeError(
            "Not enough data in the loader, should not happen for LM PTQ. Please check."
        )
    else:
        # fine-grain shuffling is not recommended for LM PTQ, due to variable seq_len in each batch
        Nbatch_to_cache = int(
            qcfg["ptq_batchsize"] * qcfg["ptq_nbatch"] / qcfg["loader.batchsize"]
        )

    qcfg["cached_data_from_loader"] = []
    pbar = tqdm(
        loader, desc="Phase 0: PTQ caching data from loader", total=Nbatch_to_cache
    )
    for data_mb, _ in zip(pbar, range(Nbatch_to_cache)):
        if pre_cache_func is not None:
            data_mb = pre_cache_func(data_mb)
        qcfg["cached_data_from_loader"].append(data_mb)

    # --- Phase 2 --- PTQ
    if not qcfg.get("cali_only", False):
        if qcfg["ptq_nbatch"] > 0 and qcfg["ptq_nouterloop"] > 0:
            Nsteps2acc = qcfg.get("PTQ_Nimgs2acc", 1)
            loss_scaling_acc = 1.0
            Ntotal_iters = qcfg["ptq_nouterloop"] * qcfg["ptq_ninnerloop"]

            torch.set_grad_enabled(True)
            pbar = tqdm(PTQmod_candidates, desc="Phase 2.1: PTQ optimization")
            qcfg["cached_output_model"] = []
            previously_optimized_mods = []
            for m in pbar:
                sub_layers = update_train_or_ptq_mode(
                    m,
                    ptqmode="fp32_out",
                    set_mod_state="train",
                    class2change=(QLinear, QBmm),
                )
                qcfg["cached_input"] = []
                qcfg["cached_output"] = []
                qcfg["attn_mask"] = []
                mod_name = qcfg["LUTmodule_name"][m]

                h = m.register_forward_hook(
                    PTQHookRecInOutLMv2(qcfg, name=mod_name, cls2rec=(QLinear, QBmm))
                )
                pbar2 = tqdm(
                    qcfg["cached_data_from_loader"],
                    desc=f"Phase 2.1: PTQ caching intermediate input/output for {mod_name}.",
                    leave=False,
                    mininterval=5,
                )
                for data_mb in pbar2:
                    data_mb = move_to(data_mb, curr_dev)
                    with torch.no_grad():
                        if isinstance(data_mb, (list, tuple)):
                            qcfg["attn_mask"].append(data_mb[1])
                            model(
                                data_mb[0], enc_mask=data_mb[1]
                            )  # for (FM roberta), input is a 1d tensor, unsqueeze to batch-dim
                        else:
                            model(**data_mb)

                for l in sub_layers:
                    if not qcfg["ptq_qdrop"] and isinstance(l, QLinear):
                        del l.W_fp  # release some unused memory
                    l.ptqmode = "q_out"
                h.remove()

                if fine_grain_shuffle:
                    # only when doing fine-grain shuffling we need to blend + cat the batched inputs
                    temp_cached_input = []
                    for i in range(len(qcfg["cached_input"][0])):
                        temp_cached_input.append(
                            torch.cat([b[i] for b in qcfg["cached_input"]], dim=0)
                        )
                        # this may double the CPU memory usage, be careful
                    qcfg["cached_input"] = temp_cached_input

                    # assume we already remove the tuple for output in the hook,
                    # it's a list of tensors now
                    qcfg["cached_output"] = torch.cat(qcfg["cached_output"], dim=0)

                    if qcfg["attn_mask"] != []:
                        qcfg["attn_mask"] = torch.cat(qcfg["attn_mask"])

                # 2-2. start to optimize this module
                argdict = {
                    "curr_dev": curr_dev,
                    "mod_name": mod_name,
                    "loss_func": PTQLossFunc(
                        qcfg["ptq_loss_func"],
                        Ntotal_iters=Ntotal_iters,
                        layers=sub_layers,
                    ),
                    "loss_scaling_acc": loss_scaling_acc,
                    "Nsteps2acc": Nsteps2acc,
                    "fg_shuffling": fine_grain_shuffle,
                    # NOTE assume PTQ_candidates list is ordered
                    "opt_lastFC": isinstance(m, (QLinear, nn.Linear))
                    and m is PTQmod_candidates[-1]
                    and qcfg.get("PTQ_lastFC", False),
                }
                if qcfg.get("PTQ_brecq_style", None):
                    # BRECQ style means optim W and A separately
                    # 1st round: WxA32
                    for l in sub_layers:
                        # this mode will force QConv2d to bypass activation quantizer
                        l.ptqmode = "A32_brecq"
                    ptq_mod_optim_lm(
                        model, m, sub_layers, qcfg, optim_mode="Wonly", **argdict
                    )
                    # 2nd round WxAy
                    for l in sub_layers:
                        l.ptqmode = "q_out"
                    ptq_mod_optim_lm(
                        model, m, sub_layers, qcfg, optim_mode="Aonly", **argdict
                    )
                else:
                    ptq_mod_optim_lm(
                        model, m, sub_layers, qcfg, optim_mode="both", **argdict
                    )
                # finally, add current block/module to "finished" list
                previously_optimized_mods.append(m)


def ptq_mod_llm_optim(m, sub_layers, qcfg, optim_mode="both", **kwargs):
    curr_dev = kwargs["curr_dev"]
    loss_func = kwargs["loss_func"]
    mod_name = kwargs["mod_name"]
    Nsteps2acc = kwargs["Nsteps2acc"]
    isTransformers = qcfg.get(
        "isTransformers"
    )  # will need special handling for transformers

    param_names = [[], [], [], []]  # 0=weights, 1=Qa, 2=Qw, 3=bias

    ws, cva, cvw, biases = [], [], [], []
    for idx, l in enumerate(sub_layers):
        if optim_mode != "Wonly":
            if hasattr(l, "quantize_feature"):
                if hasattr(l.quantize_feature, "clip_val"):
                    cva.append(l.quantize_feature.clip_val)
                    param_names[1].append(f"cv{idx}")
                if hasattr(l.quantize_feature, "clip_valn"):
                    cva.append(l.quantize_feature.clip_valn)
                    param_names[1].append(f"cvn{idx}")
                if hasattr(l.quantize_feature, "delta"):
                    cva.append(l.quantize_feature.delta)
                    param_names[1].append(f"delta{idx}")
            elif hasattr(l, "quantize_m1"):
                if hasattr(l.quantize_m1, "clip_val"):
                    cva.append(l.quantize_m1.clip_val)
                    param_names[1].append(f"m1cv{idx}")
                if hasattr(l.quantize_m2, "clip_val"):
                    cva.append(l.quantize_m2.clip_val)
                    param_names[1].append(f"m2cv{idx}")
                if hasattr(l.quantize_m1, "clip_valn"):
                    cva.append(l.quantize_m1.clip_valn)
                    param_names[1].append(f"m1cvn{idx}")
                if hasattr(l.quantize_m2, "clip_valn"):
                    cva.append(l.quantize_m2.clip_valn)
                    param_names[1].append(f"m2cvn{idx}")
            else:
                logger.info(f"Layer {l} has no trainable parameter for quantization")
        if optim_mode != "Aonly" and (not hasattr(l, "quantize_m1")):
            if hasattr(l, "qw_mode"):  # to separate from LN layers
                if "brecq" in l.qw_mode:
                    cvw.append(l.quantize_weight.delta)
                    param_names[2].append(f"deltaW{idx}")
                if "adaround" in l.qw_mode:
                    cvw.append(l.quantize_weight.alpha)
                    param_names[2].append(f"alphaW{idx}")
                if "pact+" in l.qw_mode:
                    cvw.append(l.quantize_weight.clip_val)
                    param_names[2].append(f"cvW{idx}")
            if not qcfg["PTQ_freezeweight"]:
                ws.append(l.weight)
                param_names[0].append(f"W{idx}")
        if not hasattr(l, "quantize_m1"):
            if l.bias is not None and not qcfg["PTQ_freezebias"]:
                biases.append(l.bias)
                param_names[3].append(f"bias{idx}")
    optim_w = torch.optim.Adam(
        [
            {
                "params": ws,
                "lr": qcfg["ptq_lrw"],
            },
            {"params": cvw, "lr": qcfg["ptq_lrcv_w"]},
            {"params": biases, "lr": 1e-3},
        ]
    )
    optim_a = torch.optim.Adam(
        [
            {
                "params": cva,
                "lr": qcfg["ptq_lrcv_a"],
                "weight_decay": qcfg["pact_a_decay"],
            }
        ]
    )
    scheduler = []
    scheduler_step = math.ceil(qcfg["ptq_nouterloop"] / Nsteps2acc)
    if "W" in qcfg["ptq_coslr"]:
        scheduler.append(
            torch.optim.lr_scheduler.CosineAnnealingLR(
                optim_w, T_max=scheduler_step, eta_min=0.0
            )
        )
    if "A" in qcfg["ptq_coslr"]:
        scheduler.append(
            torch.optim.lr_scheduler.CosineAnnealingLR(
                optim_a, T_max=scheduler_step, eta_min=0.0
            )
        )

    data_seq = [
        torch.randperm(qcfg["ptq_nbatch"] * qcfg["ptq_batchsize"])
        for _ in range(qcfg["ptq_nouterloop"] // qcfg["ptq_nbatch"] + 1)
    ]
    data_seq = torch.cat(data_seq).reshape([-1, qcfg["ptq_batchsize"]])

    pbar_desc = f"Phase 2.3: PTQ optimizing module {mod_name}. loss="
    pbar2 = tqdm(
        data_seq[: qcfg["ptq_nouterloop"]],
        desc=pbar_desc + "     ",
        leave=False,
        mininterval=5,
    )
    optim_w.zero_grad()
    optim_a.zero_grad()
    for i_outer, data_idx in enumerate(pbar2):
        if isTransformers:
            if "FMS" in qcfg["model_type"]:  # for FMS models
                inp = {
                    "x": qcfg["cached_input"][data_idx].to(curr_dev),
                    "mask": qcfg["cached_mask"].to(curr_dev),
                    "rel_pos_bias": qcfg["cached_pos_bias"].to(curr_dev),
                }
            else:
                inp = (qcfg["cached_input"][data_idx].to(curr_dev),)
                if qcfg["cached_alibi"]:
                    kw_inp = {
                        "alibi": qcfg["cached_alibi"][data_idx].to(curr_dev),
                        "attention_mask": qcfg["cached_mask"][data_idx].to(curr_dev),
                    }
                else:
                    kw_inp = {
                        "attention_mask": None
                        if qcfg["cached_mask"] == []
                        else qcfg["cached_mask"][data_idx].to(curr_dev),
                        "position_ids": qcfg["position_ids"][data_idx].to(curr_dev),
                    }
        update_ptq_mode(m, ptqmode="fp32_out", class2change=(QLinear, QBmm))
        with torch.no_grad():
            if i_outer + 1 < qcfg["qmodel_calibration_new"]:
                # ensure bmm layers are calibrated during the first cali samples
                with patch_torch_bmm(qcfg):
                    if "FMS" in qcfg["model_type"]:
                        fp_out = m(**inp)[0]
                    else:
                        fp_out = m(*inp, **kw_inp)[0]
            else:
                # when caching, donot call QBmm
                if "FMS" in qcfg["model_type"]:
                    fp_out = m(**inp)[0]
                else:
                    fp_out = m(*inp, **kw_inp)[0]

        for j_inner in range(qcfg["ptq_ninnerloop"]):
            grad = None
            qcfg["cached_qout"] = []
            qcfg["cached_grad_out"] = []
            Niter = i_outer * qcfg["ptq_ninnerloop"] + j_inner
            PTQloss = 0
            if (Niter + 1) > qcfg["qmodel_calibration_new"]:
                # skip optimization for calibration steps.
                update_ptq_mode(m, ptqmode="q_out", class2change=(QLinear, QBmm))
                if isTransformers and (not qcfg["decoder_arch"]):  # for encoder models
                    with patch_torch_bmm(qcfg):
                        q_out = m(*inp, **kw_inp)
                elif qcfg["decoder_arch"]:
                    with patch_torch_bmm(qcfg):
                        if "FMS" in qcfg["model_type"]:
                            q_out = m(**inp)
                        else:
                            q_out = m(*inp, **kw_inp)
                else:
                    q_out = m(inp)
                if isinstance(q_out, tuple):
                    PTQloss = loss_func(q_out[0], fp_out, grad)
                else:
                    PTQloss = loss_func(q_out, fp_out, grad)

                loss4plot = {}
                if isinstance(PTQloss, (dict)):
                    loss4plot = {k: v.item() for k, v in PTQloss.items()}
                    PTQloss = (
                        (PTQloss["total"] / Nsteps2acc)
                        if "total" in PTQloss
                        else torch.sum(PTQloss.values())
                    )
                else:
                    loss4plot[qcfg["ptq_loss_func"]] = PTQloss.item()
                    PTQloss = PTQloss / Nsteps2acc

                PTQloss.backward()
                if (Niter + 1) % Nsteps2acc == 0 or (
                    Niter + 1 == qcfg["ptq_nouterloop"]
                ):
                    optim_w.step()
                    optim_w.zero_grad()
                    optim_a.step()
                    optim_a.zero_grad()

                    for s in scheduler:
                        s.step()

                # --- tensorboard output
                if qcfg["tb_writer"] and (
                    (qcfg["ptq_ninnerloop"] == 1 and Niter % 10 == 0)
                    or (qcfg["ptq_ninnerloop"] > 1 and i_outer % 10 == 0)
                ):
                    pbar2.set_description(pbar_desc + f"{PTQloss:.3f}")
                    for k, v in loss4plot.items():
                        qcfg["tb_writer"].add_scalar(
                            f"{mod_name}/PTQloss_{k}", v, Niter
                        )

                    for k, v in m.named_buffers():
                        if any(kb in k for kb in ["delta", "zero_point", "clip_val"]):
                            if len(v.shape) > 0 and v.shape[0] > 1:  # perCh
                                qcfg["tb_writer"].add_histogram(
                                    f"{mod_name}/{k}", v, Niter
                                )
                            else:
                                qcfg["tb_writer"].add_scalar(
                                    f"{mod_name}/{k}", v, Niter
                                )

                    for p, pname in zip(
                        optim_a.param_groups[0]["params"], param_names[1]
                    ):  # cva
                        qcfg["tb_writer"].add_scalar(
                            f"{mod_name}/{pname}", p.float(), Niter
                        )
                        qcfg["tb_writer"].add_scalar(
                            f"{mod_name}/LR_cv_a", optim_a.param_groups[0]["lr"], Niter
                        )
                    for p, pname in zip(
                        optim_w.param_groups[0]["params"], param_names[0]
                    ):  # weights
                        qcfg["tb_writer"].add_histogram(
                            f"{mod_name}/{pname}", p.float(), Niter
                        )
                        qcfg["tb_writer"].add_scalar(
                            f"{mod_name}/LR_w", optim_w.param_groups[0]["lr"], Niter
                        )
                    for p, pname in zip(
                        optim_w.param_groups[1]["params"], param_names[2]
                    ):  # cvw
                        if "alpha" in pname:
                            qcfg["tb_writer"].add_histogram(
                                f"{mod_name}/{pname}", p.float(), Niter
                            )
                        else:
                            qcfg["tb_writer"].add_scalar(
                                f"{mod_name}/{pname}", p.float(), Niter
                            )
                        qcfg["tb_writer"].add_scalar(
                            f"{mod_name}/LR_cvw", optim_w.param_groups[1]["lr"], Niter
                        )
                    if "adaround" in qcfg["qw_mode"]:
                        qcfg["tb_writer"].add_scalar(
                            f"{mod_name}/AdaR_beta",
                            loss_func.temp_decay.curr_beta,
                            Niter,
                        )
                        for lidx, l in enumerate(sub_layers):
                            if not hasattr(l, "quantize_m1"):
                                qcfg["tb_writer"].add_histogram(
                                    f"{mod_name}/W{lidx}", l.weight.float(), Niter
                                )
    if "adaround" in qcfg["qw_mode"]:
        for l in sub_layers:
            if not hasattr(l, "quantize_m1") and hasattr(
                l.quantize_weight, "soft_targets"
            ):
                l.quantize_weight.soft_targets = False

    return PTQloss


class RunModule(nn.Module):
    """
    Run model fwd until finishing input module.
    """

    def __init__(self, module, qcfg):
        super().__init__()
        self.qcfg = qcfg
        self.module = module

    def forward(self, inp, **kwargs):
        self.qcfg["cached_block0_input"].append(inp.cpu())
        self.qcfg["cache_id"] += 1
        for kw_org, kw_qcfg in self.qcfg["kw_to_cache"].items():
            if kw_qcfg not in self.qcfg:
                self.qcfg[kw_qcfg] = []
            v = kwargs.get(kw_org, None)
            if v is not None:
                self.qcfg[kw_qcfg].append(move_to(v, "cpu"))
        raise ValueError


class RunFMModule(nn.Module):
    """
    Run model fwd until finishing input module.
    """

    def __init__(self, module, qcfg):
        super().__init__()
        self.qcfg = qcfg
        self.module = module

    def forward(self, **kwargs):
        self.qcfg["cached_block0_input"][self.qcfg["cache_id"]] = kwargs["x"].cpu()
        self.qcfg["cache_id"] += 1
        for kw_org, kw_qcfg in self.qcfg["kw_to_cache"]:
            if kw_qcfg not in self.qcfg:
                self.qcfg[kw_qcfg] = []
            v = kwargs.get(kw_org, None)
            if v is not None:
                self.qcfg[kw_qcfg].append(v.cpu())

        raise ValueError


def get_blocks(model, model_type=None):
    """
    For a given model.config.model_type, return
        1) blocks,
        2) embedding,
        3) position embedding (if any),
        4) embedding layernom (if any),
        5) final layernorm, and
        6) final lm_head

    TODO: Double check if FMS config has model_type attribute, and if it's the same as HF.
    """

    SAME_ARCH = {
        "neo": "granite",
        "mixtral": "llama",
        "nemotron": "llama",
    }
    BLOCK_MAP = {
        "bloom": (
            "transformer.h",
            "transformer.word_embeddings",
            None,
            "transformer.word_embeddings_layernorm",
            "transformer.ln_f",
            "lm_head",
        ),
        "opt": (
            "model.decoder.layers",
            "model.decoder.embed_tokens",
            "model.decoder.embed_positions",
            "model.decoder.final_layer_norm",
            None,
            "lm_head",
        ),
        "granite_gptbigcode": (
            "transformer.h",
            "transformer.wte",
            "transformer.wpe",
            None,
            "transformer.ln_f",
            "lm_head",
        ),
        "granite": (
            "model.layers",
            "model.embed_tokens",
            "model.rotary_emb",
            None,
            "model.norm",
            "lm_head",
        ),
        "llama": (
            "model.layers",
            "model.embed_tokens",
            "model.rotary_emb",
            None,
            "model.norm",
            "lm_head",
        ),
        "graniteFMS": (
            "base_granite.dec_process",
            "base_granite.shared.emb",
            None,
            "base_granite.shared.head",
            None,
            "head",
        ),
    }

    if model_type == None:
        if hasattr(model, "config") and hasattr(model.config, "model_type"):
            model_type = model.config.model_type

    assert model_type, "Please provide a valid model_type."
    assert (
        model_type in SAME_ARCH or model_type in BLOCK_MAP
    ), f"Unknown model type {model_type}. Please define block mapping in get_block()."

    if model_type in SAME_ARCH:
        model_type = SAME_ARCH[model_type]

    return [model.get_submodule(b) if b else None for b in BLOCK_MAP[model_type]]


def cpu_cali(model, dloader, qcfg):
    """
    Whole model calibration on CPU
    """
    update_train_or_ptq_mode(model, set_mod_state="train", class2change=Qdynamic)
    pbar = tqdm(
        dloader,
        desc="Phase 1.1: calibration on cpu",
        total=qcfg["qmodel_calibration_new"],
    )
    for data_mb, Niters in zip(pbar, range(qcfg["qmodel_calibration_new"])):
        # data_mb should on cpu, model is currently on cpu
        with patch_torch_bmm(qcfg):
            model(**data_mb)
        # record clipvals
        cv_sum_table = {}
        Qmods = {
            k: m for k, m in model.named_modules() if isinstance(m, (QLinear, QBmm))
        }
        for modname, m in Qmods.items():
            cv_sum_table[modname] = [
                None,
                None,
                None,
                None,
            ]  # will store "cv_a, cvn_a, cv_w, cvn_w"
            Qparams = {k: v for k, v in m.named_parameters() if "quantize_" in k}
            for k, v in Qparams.items():
                if "alpha" not in k:
                    var_name = k.split("quantize_")[1]
                    var_idx = ("weight" in var_name) * 2 + (
                        "clip_valn" in var_name or "zero_point" in var_name
                    )
                    cv_sum_table[modname][var_idx] = v.item()
                    if qcfg["tb_writer"]:
                        qcfg["tb_writer"].add_scalar(f"{modname}/{var_name}", v, Niters)
                else:
                    cv_sum_table[modname][2] = m.quantize_weight.delta.mean().item()
                    cv_sum_table[modname][3] = (
                        m.quantize_weight.zero_point.mean().item()
                    )
                    if qcfg["tb_writer"]:
                        qcfg["tb_writer"].add_scalar(
                            f"{modname}/delta", m.quantize_weight.delta.mean(), Niters
                        )

    pd.options.display.float_format = "{:.4f}".format
    dfCV = pd.DataFrame(cv_sum_table).T
    dfCV.columns = (
        ["cv_a", "cvn_a", "cv_w", "cvn_w"]
        if qcfg["qw_mode"] != "adaround"
        else ["cv_a", "cvn_a", "w_delta", "w_zp"]
    )
    logger.info(dfCV)


def cache_block0_inputs(
    model, dloader, qcfg, blocks, emb=None, emb_pos=None, emb_ln=None, dev="cpu"
):
    """
    To cache the input to the first transformer block. Basically a "forward_pre_hook"
    NOTE, change caching from tensor to list to allow varying input length, slightly
    increase memeory due to mask and alibi.
    """
    emb = emb.to(dev)
    if emb_pos is not None:
        emb_pos.to(dev)
    if emb_ln is not None:
        emb_ln = emb_ln.to(dev)
    blocks[0] = blocks[0].to(dev)
    # move block0 to GPU and excuting fwd() until finish block0
    if "fms" in qcfg["model_type"]:
        qcfg["kw_to_cache"] = {
            "mask": "cached_mask",
            "rel_pos_bias": "cached_pos_bias",
        }
        blocks[0] = RunFMModule(blocks[0], qcfg)
    else:
        # latest transformers requires pos_ids to be fed into fwd()
        qcfg["kw_to_cache"] = {
            "attention_mask": "cached_mask",
            "alibi": "cached_alibi",
            "position_ids": "position_ids",
            "position_embeddings": "position_embeddings",
        }
        blocks[0] = RunModule(blocks[0], qcfg)

    # clear up old cache, if exists.
    qcfg["cached_block0_input"] = []
    qcfg["cache_id"] = 0
    for kw in qcfg["kw_to_cache"].values():
        if kw in qcfg:
            qcfg[kw] = []

    if isinstance(dloader, torch.utils.data.DataLoader):
        pbar = tqdm(
            dloader, desc="Phase 0: Caching block0 inputs", total=qcfg["ptq_nbatch"]
        )
        for data_mb, _ in zip(pbar, range(qcfg["ptq_nbatch"])):
            try:
                data_mb = move_to(data_mb, dev)
                # only input of block0, no need qbmm
                # need this, otherwise, the previous Layernorm grad will be saved to CPU
                with torch.no_grad():
                    model(**data_mb)
            except ValueError:
                pass
    else:
        n_samples = qcfg.get("n_samples", qcfg["ptq_nbatch"])

        for i in tqdm(range(n_samples), desc="Caching block0 input for evaluation..."):
            try:
                data_mb = dloader["input_ids"][
                    :, (i * qcfg["seq_len"]) : ((i + 1) * qcfg["seq_len"])
                ].to(dev)
                with torch.no_grad():
                    model(data_mb)
            except ValueError:
                pass

    # convert block[0] a standard module, then send back to cpu.
    blocks[0] = blocks[0].module
    blocks[0] = blocks[0].cpu()
    emb = emb.cpu()
    if emb_pos is not None:
        emb_pos = emb_pos.cpu()
    if emb_ln is not None:
        emb_ln = emb_ln.cpu()
    torch.cuda.empty_cache()


def get_optimized_outputs(m, qcfg, dev="cpu"):
    """
    Write the optimized outputs back to cached inputs.
    """
    update_ptq_mode(m, ptqmode="fp32_out", class2change=(QLinear, QBmm))
    for i in range(qcfg["ptq_nbatch"]):
        with torch.no_grad():
            if "FMS" in qcfg["model_type"]:
                qcfg["cached_input"][i] = m(
                    x=qcfg["cached_input"][i].unsqueeze(0).to(dev),
                    mask=qcfg["cached_mask"].to(dev),
                    rel_pos_bias=qcfg["cached_pos_bias"].to(dev),
                )[0].cpu()
            else:
                if qcfg["cached_alibi"]:
                    qcfg["cached_input"][i] = m(
                        qcfg["cached_input"][i].unsqueeze(0).to(dev),
                        attention_mask=qcfg["cached_mask"][i].unsqueeze(0).to(dev),
                        alibi=qcfg["cached_alibi"][i].unsqueeze(0).to(dev),
                    )[0].cpu()
                else:
                    qcfg["cached_input"][i] = m(
                        qcfg["cached_input"][i].to(dev),
                        attention_mask=None
                        if qcfg["cached_mask"] == []
                        else qcfg["cached_mask"][i].to(dev),
                        position_ids=qcfg["position_ids"][i].to(dev),
                    )[0].cpu()


def ptq_llm_1gpu(qcfg, model, dloader, local_rank):
    """
    block-wise quantization optimization for LLM using 1 GPU
    """
    dev = "cuda:" + local_rank  # cuda:0 is used for PTQ
    qcfg["batch_size"] = 1  # for dataloading, always use batch_size of 1
    qcfg["loader_len"] = len(dloader)
    qcfg["dtype"] = next(iter(model.parameters())).dtype
    qcfg["hidden_size"] = model.config.hidden_size
    assert (
        qcfg["loader_len"] == qcfg["ptq_nbatch"]
    ), "set batch_size=1 and PTQ samples== Nbatches"
    # --- Phase 0 cache the inputs of the block0---
    model.config.use_cache = False
    blocks, emb, emb_pos, emb_ln, _, _ = get_blocks(model, model.config.model_type)
    cache_block0_inputs(
        model, dloader, qcfg, blocks, emb=emb, emb_pos=emb_pos, emb_ln=emb_ln, dev=dev
    )
    logger.info("Done, caching inputs to block0 for PTQ optimization")

    # --- Phase 1.1 --- calibration of clip vals on CPU
    if qcfg.get("qmodel_calibration_new", 0) > 0 and qcfg.get("cpu_cali", False):
        cpu_cali(model, dloader, qcfg)
        logger.info("Finish cablibration on whole model.")

    # --- Phase 1 --- PTQ
    if (
        qcfg["ptq_nbatch"] > 0 and qcfg["ptq_nouterloop"] > 0
    ):  # default Ninner = 1 if not specified
        Nsteps2acc = qcfg["PTQ_Nimgs2acc"]
        loss_scaling_acc = 1.0
        Ntotal_iters = qcfg["ptq_nouterloop"] * qcfg["ptq_ninnerloop"]

        torch.set_grad_enabled(True)
        pbar = tqdm(blocks, desc="Phase 1.1: PTQ optimization")
        qcfg["cached_input"] = [
            inp.clone().detach() for inp in qcfg["cached_block0_input"]
        ]
        for m in pbar:
            m.to(dev)
            if qcfg["PTQ_freezeLN"]:
                update_train_or_eval_mode(
                    m, set_mod_state="eval", class2change=(nn.LayerNorm)
                )
            else:
                update_train_or_eval_mode(
                    m, set_mod_state="train", class2change=(nn.LayerNorm)
                )
            update_ptq_mode(m, ptqmode="fp32_out", class2change=(QLinear, QBmm))
            sub_layers = get_layers(m, layer2get=(QLinear, QBmm))

            qcfg["cached_output"] = []
            mod_name = qcfg["LUTmodule_name"][m]

            # 2-2. start to optimize this module
            argdict = {
                "curr_dev": dev,
                "mod_name": mod_name,
                "loss_func": PTQLossFunc(
                    qcfg["ptq_loss_func"],
                    Ntotal_iters=Ntotal_iters,
                    layers=sub_layers,
                ),
                "loss_scaling_acc": loss_scaling_acc,
                "Nsteps2acc": Nsteps2acc,
                # assume PTQ_candidates list is ordered
                "opt_lastFC": isinstance(m, (QLinear, nn.Linear))
                and m is blocks[-1]
                and qcfg.get("PTQ_lastFC", False),
            }
            # NOTE, clip_val calibration is done at begining of optimization for calibration_new
            # times different from whole model calibration on cpu,
            # the statistics of each calibration are from previous optimized blocks
            ptq_mod_llm_optim(m, sub_layers, qcfg, optim_mode="both", **argdict)

            # recorde the optimized output and write to input
            get_optimized_outputs(m, qcfg, dev=dev)
            logger.info(
                f"Done, write optimized output of {mod_name} in FP32 to cached_inputs"
            )

            update_ptq_mode(m, ptqmode="q_out", class2change=(QLinear, QBmm))
            remove_wfp(m, class2change=QLinear)
            m.cpu()
            torch.cuda.empty_cache()

        logger.info("All blocks are optimized")


def freeze_layers(m, layer_list):
    """
    Freeze select layers from optimization.
    """
    for name, module in m.named_modules():
        if isinstance(module, (QLinear,)):
            if any(x in name for x in layer_list):
                module.eval()


@torch.no_grad()
def calibration_llm_1GPU(qcfg, model, dloader):
    """Calibration for large models that can not fit on 1 GPU."""

    model.train()
    dev = "cuda"
    qcfg["batch_size"] = 1
    qcfg["dtype"] = next(iter(model.parameters())).dtype
    qcfg["n_samples"] = qcfg["qmodel_calibration_new"]

    assert "model_type" in qcfg, "Unknown model type. please check before proceed."
    # --- Phase 0 cache the inputs of the block0---
    model.config.use_cache = False
    blocks, emb, emb_pos, emb_ln, _, _ = get_blocks(model, qcfg["model_type"])
    cache_block0_inputs(
        model,
        dloader,
        qcfg,
        blocks,
        emb=emb,
        emb_pos=emb_pos,
        emb_ln=emb_ln,
        dev="cpu",
    )
    logger.info("Done, caching inputs to block0 for calibration")

    # --- Phase 1 --- compute blocks and last linear layer
    pbar = tqdm(blocks, desc="calibration: compute blocks")
    qcfg["cached_input"] = [inp.clone().detach() for inp in qcfg["cached_block0_input"]]
    for m in pbar:
        m.to(dev)
        for i in range(qcfg["n_samples"]):
            if qcfg["cached_alibi"]:
                cached_inp_prev_lay = qcfg["cached_input"][i].unsqueeze(0).to(dev)
                data_mb = {
                    "attention_mask": qcfg["cached_mask"][i].unsqueeze(0).to(dev),
                    "alibi": qcfg["cached_alibi"][i].unsqueeze(0).to(dev),
                }
            else:
                cached_inp_prev_lay = qcfg["cached_input"][i].to(dev)
                data_mb = {
                    "attention_mask": qcfg["cached_mask"][i].to(dev)
                    if len(qcfg["cached_mask"]) > 0
                    else None,
                    "position_ids": qcfg["position_ids"][i].to(dev),
                }

            with torch.no_grad(), patch_torch_bmm(qcfg):
                qcfg["cached_input"][i] = m(cached_inp_prev_lay, **data_mb)[0].cpu()

        m.cpu()
        torch.cuda.empty_cache()

    logger.info("All blocks are calibrated")


@torch.no_grad()
def calibration_llm_1GPU_v2(qcfg, model, dloader):
    """
    Improved version of Calibration for large language models that can not fit on 1 GPU with new
    (built-in) calibration mechanism.
    NOTE:
    1. Calibration only, NO update to weights!
    2. Rely on a alternative "pre fwd hook" to cache all possible inputs.
    3. As calibration usually cache a small number of data only, no need to move each batch back and
        forth between GPU and CPU.
    """

    model.train()
    dev = "cuda"
    qcfg["batch_size"] = 1
    qcfg["dtype"] = next(iter(model.parameters())).dtype
    qcfg["n_samples"] = min(qcfg["ptq_nbatch"], qcfg["qmodel_calibration_new"])

    assert "model_type" in qcfg, "Unknown model type. please check before proceed."
    assert isinstance(
        dloader, torch.utils.data.DataLoader
    ), "Please provide a valid dataloader."
    # --- Phase 0 cache the inputs of the block0---
    model.config.use_cache = False
    blocks, emb, emb_pos, emb_ln, _, _ = get_blocks(model, qcfg["model_type"])

    cache_block0_inputs(
        model,
        dloader,
        qcfg,
        blocks,
        emb=emb,
        emb_pos=emb_pos,
        emb_ln=emb_ln,
        dev="cpu",
    )
    logger.info("Done, caching inputs to block0 for calibration")

    # --- Phase 1 --- compute blocks and last linear layer
    pbar = tqdm(
        blocks, desc="Phase 1: Calibration for each block", position=0, leave=True
    )
    qcfg["cached_input"] = [
        inp.clone().detach().to(dev) for inp in qcfg["cached_block0_input"]
    ]
    kw_to_use = {
        kw_org: kw_new
        for kw_org, kw_new in qcfg["kw_to_cache"].items()
        if len(qcfg[kw_new]) == len(qcfg["cached_input"])
    }
    for _num_block, m in enumerate(pbar):
        m.to(dev)
        for i in tqdm(
            range(qcfg["n_samples"]), desc="number of samples", position=1, leave=False
        ):
            if qcfg["cached_alibi"]:
                cached_inp_prev_lay = qcfg["cached_input"][i].unsqueeze(0).to(dev)
                data_mb = {
                    "attention_mask": qcfg["cached_mask"][i].unsqueeze(0).to(dev),
                    "alibi": qcfg["cached_alibi"][i].unsqueeze(0).to(dev),
                }
            else:
                cached_inp_prev_lay = qcfg["cached_input"][i]
                data_mb = {
                    kw_org: move_to(qcfg[kw_new][i], dev)
                    for kw_org, kw_new in kw_to_use.items()
                }

            with patch_torch_bmm(qcfg):
                qcfg["cached_input"][i] = m(cached_inp_prev_lay, **data_mb)[0]

        m.cpu()
        torch.cuda.empty_cache()

    logger.info("All blocks are calibrated")


@torch.no_grad()
def activation_stats(name, tensor, act_scales):
    # TODO if 'QBmm' in name: reshape the tensor.
    hidden_dim = tensor.shape[-1]
    tensor = tensor.view(-1, hidden_dim).abs().detach()
    coming_max = torch.max(tensor, dim=0)[0].float().cpu()
    if name in act_scales:
        act_scales[name] = torch.max(act_scales[name], coming_max)
    else:
        act_scales[name] = coming_max
    return act_scales


@torch.no_grad()
def input_stats_hook(m, x, _y, name, act_scales):
    if isinstance(x, tuple):
        if isinstance(m, (QBmm,)):
            x = x[1].detach()
        else:
            x = x[0].detach()
    activation_stats(name, x, act_scales)


@torch.no_grad()
def get_act_scales(
    model,
    dloader,
    qcfg: dict,
    device: Optional[Union[str, torch.device]] = None,
):
    """Compute smoothquant activation scales of quantized linear layers.
    Model and examples are moved to selected device, if provided.
    """

    model.eval()

    if device is None:
        device = next(model.parameters()).device
    else:
        logger.info(
            f"Moving model to {device} to compute smoothquant activation scales"
        )
        model.to(device)

    act_scales = {}
    qcfg["sample_id"] = 0
    hooks = []
    for name, m in model.named_modules():
        if isinstance(m, (nn.Linear, QLinear, QBmm)):
            hooks.append(
                m.register_forward_hook(
                    partial(input_stats_hook, name=name, act_scales=act_scales)
                )
            )

    n_samples = 100
    pbar = tqdm(dloader, desc="Calibrating for activation scale", total=n_samples)

    for data_mb, _ in zip(pbar, range(n_samples)):
        qcfg["sample_id"] += 1
        data_mb = move_to(data_mb, device)
        if (
            qcfg["nbits_bmm1"] < 32
            or qcfg["nbits_bmm2"] < 32
            or qcfg["nbits_kvcache"] < 32
        ):
            with patch_torch_bmm(qcfg):
                model(**data_mb)
        else:
            model(**data_mb)

    for h in hooks:
        h.remove()

    return act_scales


@torch.no_grad()
def get_module_act_scales(m, block_idx, qcfg, act_scales):
    """
    To get the act scales from a module, such as a transformer block.
    """
    dev = next(m.parameters()).device
    hooks = []
    qcfg["sample_id"] = 0
    prefix = "model.layers." + str(block_idx) + "."
    for name, layer in m.named_modules():
        if isinstance(layer, (nn.Linear,)):
            hooks.append(
                layer.register_forward_hook(
                    partial(
                        input_stats_hook,
                        name=prefix + name,
                        act_scales=act_scales,
                    )
                )
            )

    for i in range(len(qcfg["cached_input"])):
        if qcfg["cached_alibi"]:
            qcfg["cached_input"][i] = m(
                qcfg["cached_input"][i].unsqueeze(0).to(dev),
                attention_mask=qcfg["cached_mask"][i].unsqueeze(0).to(dev),
                alibi=qcfg["cached_alibi"][i].unsqueeze(0).to(dev),
            )[0].cpu()
        else:
            kwargs = {
                kw_org: move_to(qcfg[kw_qcfg][i], dev) if qcfg[kw_qcfg] != [] else None
                for kw_org, kw_qcfg in qcfg["kw_to_cache"].items()
            }
            qcfg["cached_input"][i] = m(
                qcfg["cached_input"][i].to(dev),
                **kwargs,
            )[0].cpu()
    for h in hooks:
        h.remove()

    return act_scales


@torch.no_grad()
def get_act_scales_1gpu(model, dloader, qcfg):
    """
    get activation blocks on 1gpu for very large models that cannot fit in 1gpu
    """
    dev = "cuda"
    qcfg["batch_size"] = 1
    qcfg["loader_len"] = len(dloader)
    qcfg["dtype"] = next(iter(model.parameters())).dtype
    qcfg["hidden_size"] = model.config.hidden_size

    assert "model_type" in qcfg, "Unknown model type. please check before proceed."
    assert (
        qcfg["loader_len"] >= qcfg["ptq_nbatch"]
    ), "Please make sure dataloader has enough data needed for PTQ (ie. check qcfg['ptq_nbatch'])."
    # --- Phase 0 cache the inputs of the block0---
    blocks, emb, emb_pos, emb_ln, _, _ = get_blocks(model, qcfg["model_type"])
    cache_block0_inputs(
        model.cpu(),
        dloader,
        qcfg,
        blocks,
        emb=emb,
        emb_pos=emb_pos,
        emb_ln=emb_ln,
        dev="cpu",
    )
    logger.info("Done, caching inputs to block0")

    pbar = tqdm(blocks, desc="get activation scales from each block")
    qcfg["cached_input"] = [inp.clone().detach() for inp in qcfg["cached_block0_input"]]

    act_scales = {}
    for block_idx, m in enumerate(pbar):
        m.to(dev)
        get_module_act_scales(m, block_idx, qcfg, act_scales)
        m.cpu()
        torch.cuda.empty_cache()

    logger.info("Finish getting act_scales for all blocks")
    return act_scales


def dq_llm(model, scale, qcfg):
    """
    This function is used to do direct quantization, i.e using small amount of data to
    calibrate clip_vals, but no weight updates.
    The algos are similar to smoothquant.
    """

    for name, module in model.named_modules():
        if isinstance(module, (QLinear,)):
            if any(x in name for x in qcfg["smoothq_scale_layers"]):
                module.set_act_scale(scale[name])
                logger.info(
                    f"Apply layer {name} with activation scales (10)"
                    f"of {module.smoothq_act_scale[:10]}"
                )
    return model


def set_quantizers(model, qcfg, new_qa_mode):
    """
    This function is used to reset the quantizers, precisions and so on for certain layers,
    after qmodel_prep()
    currently use to set activation quantizers for QLinear layers
    """

    for name, module in model.named_modules():
        if isinstance(module, (QLinear,)):
            if any(x in name for x in qcfg["layer_reset_quantizer"]):
                module.quantize_feature = get_activation_quantizer(
                    qa_mode=new_qa_mode,
                    nbits=module.num_bits_feature,
                    clip_val=module.act_clip_init_val,
                    clip_valn=module.act_clip_init_valn,
                    non_neg=module.non_neg,
                    align_zero=module.align_zero,
                    extend_act_range=module.extend_act_range,
                    use_PT_native_Qfunc=module.use_PT_native_Qfunc,
                )

                logger.info(
                    f"change layer {name} act quantizer to {module.quantize_feature}"
                )
    return model


class StraightThrough(nn.Module):
    """
    Use to set bn module to straight-through.
    """

    def __init__(self, num_features, eps):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.register_buffer("weight", torch.ones(num_features))
        self.register_buffer("bias", torch.zeros(num_features))
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features) - eps)

    def forward(self, x):
        return x


def _fold_bn(conv_module, bn_module, bn_affine=None):
    w = conv_module.weight.data
    y_mean = bn_module.running_mean
    y_var = bn_module.running_var
    safe_std = torch.sqrt(y_var + bn_module.eps)
    w_view = (conv_module.out_channels, 1, 1, 1)
    if bn_affine or bn_module.affine:
        weight = w * (bn_module.weight / safe_std).view(w_view)
        beta = bn_module.bias - bn_module.weight * y_mean / safe_std
        if conv_module.bias is not None:
            bias = bn_module.weight * conv_module.bias / safe_std + beta
        else:
            bias = beta
    else:
        weight = w / safe_std.view(w_view)
        beta = -y_mean / safe_std
        if conv_module.bias is not None:
            bias = conv_module.bias / safe_std + beta
        else:
            bias = beta
    return weight, bias


def fold_bn_into_conv(conv_module, bn_module, bn_affine=None):
    w, b = _fold_bn(conv_module, bn_module, bn_affine)
    if conv_module.bias is None:
        conv_module.bias = nn.Parameter(b)
    else:
        conv_module.bias.data = b
    conv_module.weight.data = w
    # set bn running stats
    bn_module.running_mean = bn_module.bias.data
    bn_module.running_var = bn_module.weight.data**2


def reset_bn(module: nn.BatchNorm2d):
    """
    Function not currently used.
    """
    if module.track_running_stats:
        module.running_mean.zero_()
        module.running_var.fill_(1 - module.eps)
        # we do not reset numer of tracked batches here
    if module.affine:
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)


try:
    # Local
    from fms_mo.modules.conv import DetQConv2d

    BNofInteret = (
        nn.BatchNorm2d,
        nn.BatchNorm1d,
    )  # FrozenBatchNorm2d)
    AbsorbLayers = (nn.Conv2d, nn.Linear, DetQConv2d)
    bn_affine = True  # FrozenBN doesn't have .affine property
except:
    BNofInteret = (nn.BatchNorm2d, nn.BatchNorm1d)
    AbsorbLayers = (nn.Conv2d, nn.Linear)


def search_fold_and_remove_bn(model, mod_folded):
    model.eval()
    prev = None
    for n, m in model.named_children():
        if isinstance(m, BNofInteret) and isinstance(prev, AbsorbLayers):
            fold_bn_into_conv(prev, m)
            # set the bn module to straight through
            setattr(model, n, StraightThrough(m.num_features, m.eps))
            mod_folded.append(prev)  # make a dict for those folded Convs/Linears

        elif isinstance(m, AbsorbLayers):
            prev = m
            # Added for detectron2 style, i.e. norm is a child of Conv2d
            for n_child, m_child in m.named_children():
                if isinstance(m_child, BNofInteret):
                    BN_name, BN_mod = n_child, m_child
                    fold_bn_into_conv(prev, BN_mod, bn_affine=bn_affine)
                    setattr(
                        m, BN_name, StraightThrough(BN_mod.num_features, BN_mod.eps)
                    )
                    mod_folded.append(prev)
                    # there should be only 1 BN under Conv in detectron2
                    break
        else:
            prev = search_fold_and_remove_bn(m, mod_folded)
    return prev


# ---------------------------------------------
