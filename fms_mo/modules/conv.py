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

"""Quantization convolution modules"""
# pylint: disable=arguments-renamed

# Standard
import logging

# Third Party
from torch import nn
import torch
import torch.nn.functional as F

# Local
from fms_mo.quant.quantizers import (
    PACT,
    HardPrune,
    Qbypass,
    Qdynamic,
    get_activation_quantizer,
    get_weight_quantizer,
    mask_conv2d_kij,
)

logger = logging.getLogger(__name__)


class QConv2d(nn.Conv2d):
    """
    A wrapper for the  quantization of nn.Conv2d layers.

    Layer weights and input activation can be quantized to low-precision integers through popular
    quantization methods.

    Bias is not quantized.

    Supports both non-negtive activations (after relu) and symmetric/nonsymmetric 2-sided
    activations (after sum, swish, silu ..)

    Attributes:
        num_bits_feature   : precision for activations
        num_bits_weight    : precision for weights
        qa_mode            : quantizers for activation quantization. Options:PACT, CGPACT, PACT+,
                                LSQ+, DoReFa, and etc
        qw_mode            : quantizers for weight quantization. Options: SAWB, SAWB+, PACT, CGPACT,
                                PACT+sym, LSQ+, DoReFa, AdaRound.
        act_clip_init_val  : initialization value for activation clip_val on positive side
        act_clip_init_valn : initialization value for activation clip_val on negative  side
        w_clip_init_val    : initialization value for weight clip_val on positive side
                                (None for SAWB)
        w_clip_init_valn   : initialization value for weight clip_val on negative side
                                (None for SAWB)
        non_neg            : if True, call one-side activation quantizer
        align_zero         : if True, preserve zero, i.e align zero to an integer level

    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        num_bits_feature=32,
        qa_mode=None,
        num_bits_weight=32,
        qw_mode=None,
        **kwargs,
    ):
        """
        Initializes the QConv2d layer.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_size (int or tuple): Size of the convolutional kernel.
            stride (int or tuple, optional): Stride of the convolution. Defaults to 1.
            padding (int or tuple, optional): Padding added to all sides of the input.
                                                Defaults to 0.
            dilation (int or tuple, optional): Spacing between kernel elements. Defaults to 1.
            groups (int, optional): Number of blocked connections from input channels to
                                            output channels. Defaults to 1.
            bias (bool, optional): If True, adds a learnable bias to the output. Defaults to True.
            num_bits_feature (int, optional): Number of bits for feature quantization.
                                                Defaults to 32.
            qa_mode (str, optional): Activation quantization mode. Defaults to None.
            num_bits_weight (int, optional): Number of bits for weight quantization. Defaults to 32.
            qw_mode (str, optional): Weight quantization mode. Defaults to None.
            **kwargs (dict): Additional keyword arguments.
        """
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            device=kwargs.get("device", "cuda"),
        )
        try:
            qcfg = kwargs.pop("qcfg")
        except KeyError:
            logger.error("qcfg was not included in the keyword arguments for QConv2d!")
        self.num_bits_feature = num_bits_feature
        self.num_bits_weight = num_bits_weight
        self.qa_mode = qa_mode
        self.qw_mode = qw_mode

        self.qa_mode_calib = kwargs.get(
            "qa_mode_calib",
            qcfg.get("qa_mode_calib", "max" if num_bits_feature == 8 else "percentile"),
        )
        self.qw_mode_calib = kwargs.get(
            "qw_mode_calib",
            qcfg.get("qw_mode_calib", "max" if num_bits_feature == 8 else "percentile"),
        )
        self.act_clip_init_val = kwargs.get(
            "act_clip_init_val", qcfg.get("act_clip_init_val", 8.0)
        )
        self.act_clip_init_valn = kwargs.get(
            "act_clip_init_valn", qcfg.get("act_clip_init_valn", -8.0)
        )
        self.w_clip_init_val = kwargs.get(
            "w_clip_init_val", qcfg.get("w_clip_init_val", 1.0)
        )
        self.w_clip_init_valn = kwargs.get(
            "w_clip_init_valn", qcfg.get("w_clip_init_valn", -1.0)
        )

        self.non_neg = kwargs.get("non_neg", qcfg.get("non_neg", False))
        self.align_zero = kwargs.get("align_zero", qcfg.get("align_zero", True))
        self.extend_act_range = kwargs.get(
            "extend_act_range", qcfg.get("extend_act_range", False)
        )
        self.register_buffer(
            "calib_counter", torch.tensor(qcfg.get("qmodel_calibration_new", 0))
        )  # Counters has to be buffer in case DP is used.
        self.register_buffer(
            "num_module_called", torch.tensor(0)
        )  # A counter to record how many times this module has been called

        self.qcfg = qcfg

        self.calib_iterator = []  # To simplify update of clipvals in forward()
        self.quantize_feature = Qbypass()
        self.quantize_calib_feature = Qbypass()
        if self.num_bits_feature not in [32, 16]:
            self.quantize_feature = get_activation_quantizer(
                self.qa_mode,
                nbits=self.num_bits_feature,
                clip_val=self.act_clip_init_val,
                clip_valn=self.act_clip_init_valn,
                non_neg=self.non_neg,
                align_zero=self.align_zero,
                extend_act_range=bool(self.extend_act_range),
            )
            if self.calib_counter > 0:
                self.quantize_calib_feature = Qdynamic(
                    self.num_bits_feature,
                    qcfg,
                    non_neg=self.non_neg,
                    align_zero=self.align_zero,
                    qmode=self.qa_mode_calib,
                    quantizer2sync=self.quantize_feature,
                )

        self.quantize_weight = Qbypass()
        self.quantize_calib_weight = Qbypass()
        if self.num_bits_weight not in [32, 16]:
            self.quantize_weight = get_weight_quantizer(
                self.qw_mode,
                nbits=self.num_bits_weight,
                clip_val=self.w_clip_init_val,
                clip_valn=self.w_clip_init_valn,
                align_zero=self.align_zero,
                w_shape=self.weight.shape,
            )
            if self.calib_counter > 0:
                self.quantize_calib_weight = (
                    self.quantize_weight
                    if self.qw_mode in ["sawb", "sawb+", "max"]
                    else Qdynamic(
                        self.num_bits_weight,
                        qcfg,
                        non_neg=False,
                        align_zero=True,
                        qmode=self.qw_mode_calib,
                        symmetric=True,
                        quantizer2sync=self.quantize_weight,
                    )
                )
                # For non-learnable quantizers, use the real quantizer as the calib
                # quantizer directly

        if self.qw_mode == "oldsawb":
            logger.info(
                "Please consider using the new SAWB quantizer. 'oldsawb' mode is "
                "not supported anymore."
            )
        self.mask = None
        self.mask_type = qcfg.get("mask_type", "kij")
        self.update_type = qcfg.get("update_type", "hard")
        self.prune_group = qcfg.get("prune_group", 4)
        self.prune_ratio = qcfg.get("prune_ratio", 0.0)
        self.prune_mix = qcfg.get("prune_mix", False)
        self.in_threshold = qcfg.get("in_threshold", 128)
        w_size = self.weight.shape
        if (
            self.prune_mix
            and self.prune_ratio == 0.75
            and w_size[1] <= self.in_threshold
        ):
            self.prune_ratio = 0.50
        self.p_inplace = qcfg.get("p_inplace", False)
        if self.prune_ratio != 0.0:
            self.get_mask()

    def get_mask(self):
        """
        Gets the mask for the weight tensor.

        By default, uses hard pruning. The mask is stored in the `mask` attribute.

        Returns:
            torch.Tensor: The mask tensor.
        """
        if self.mask_type == "kij":
            self.mask = mask_conv2d_kij(
                self.weight, group=self.prune_group, prune_ratio=self.prune_ratio
            )
        else:
            self.mask = None

    def get_prune_ratio(self):
        """
        Calculates the prune ratio of the mask.

        Returns:
            float: The prune ratio of the mask.
        """
        mask = self.mask.reshape(-1)
        return torch.sum(mask) * 1.0 / mask.shape[0]

    def forward(self, x):
        """
        Forward function for the quantized convolutional layer.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, out_height, out_width).
        """
        # pylint: disable = access-member-before-definition
        if self.calib_counter > 0:
            with torch.no_grad():
                qinput = self.quantize_calib_feature(x)
                qweight = self.quantize_calib_weight(self.weight)
                # If quantizer2sync is specified in init(), clipvals will be synched (inside Qdyn)
                # automatically

            self.calib_counter -= 1
            if self.calib_counter == 0:
                self.quantize_calib_feature = self.quantize_calib_weight = (
                    self.calib_counter
                ) = None
            # This should prevent calib_counter to be saved in ckpt, which will override future
            # fp32init if exists.

        else:
            qinput = self.quantize_feature(x)
            # By default self.update_type == 'hard' pruning.
            if self.mask is not None:
                pweight = HardPrune.apply(
                    self.weight, self.mask.to(self.weight.device), self.p_inplace
                )
                qweight = self.quantize_weight(pweight)
            else:
                qweight = self.quantize_weight(self.weight)

        qbias = self.bias  # Bias is not quantized

        # pylint: disable=not-callable
        output = F.conv2d(
            qinput,
            qweight,
            qbias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )

        self.num_module_called += 1

        return output

    @classmethod
    def from_torch(cls, conv2d, qcfg, **kwargs):
        """Converts a torch.nn.Conv2d module to a quantized counterpart.

        Args:
            cls (class): The class of the quantized convolutional layer.
            conv2d (torch.nn.Conv2d): The original floating-point convolutional layer.
            qcfg (dict): A dictionary containing the quantization configuration parameters.
            **kwargs (dict): Additional keyword arguments.

        Returns:
            torch.nn.Module: The quantized convolutional layer.
        """
        qconv2d = cls(
            in_channels=conv2d.in_channels,
            out_channels=conv2d.out_channels,
            kernel_size=conv2d.kernel_size,
            stride=conv2d.stride,
            padding=conv2d.padding,
            dilation=conv2d.dilation,
            groups=conv2d.groups,
            bias=conv2d.bias is not None,
            num_bits_feature=qcfg["nbits_a"],
            qa_mode=qcfg["qa_mode"],
            num_bits_weight=qcfg["nbits_w"],
            qw_mode=qcfg["qw_mode"],
            qcfg=qcfg,
        )
        device = kwargs.get("device", next(conv2d.parameters()).device)
        qconv2d.weight = nn.Parameter(conv2d.weight)
        if conv2d.bias is not None:
            qconv2d.bias = nn.Parameter(conv2d.bias)
        return qconv2d.to(device)

    def __repr__(self):
        """Return a string that represents the quantized convolutional layer."""
        str_padding = "" if self.padding[0] == 0 else f",padding{self.padding}"
        str_group = "" if self.groups == 1 else f",groups={self.groups}"
        str_bias = "" if self.bias is not None else f",bias={self.bias is not None}"
        str_prune = (
            ""
            if self.mask is None
            else f", p_rate={self.prune_ratio}, p_group={self.prune_group}"
        )
        str_quantizer = ",QntzerW,A="
        str_quantizer += (
            ""
            if self.num_bits_weight == 32
            else f"{self.quantize_weight.__repr__().split('(')[0]},"
        )
        str_quantizer += (
            ""
            if self.num_bits_feature == 32
            else f"{self.quantize_feature.__repr__().split('(')[0]}"
        )
        return (
            f"{self.__class__.__name__}({self.in_channels},{self.out_channels},kernel "
            f"{self.kernel_size},stride{self.stride}"
            + str_padding
            + str_group
            + str_bias
            + str_prune
            + f", Nbits_W,A={self.num_bits_weight},{self.num_bits_feature}, "
            f"uniDir={self.non_neg} {str_quantizer})"
        )


class QConv2dPTQ(nn.Conv2d):
    """
    A wrapper for the  quantization of nn.Conv2d layers.
    Layer weights and input activation can be quantized to low-precision integers through popular
    quantization methods.
    Bias is not quantized.
    Support both non-negtive activations (after relu) and symmetric/nonsymmetric 2-sided activations
    (after sum, swish, silu ..)

    Attributes:
        num_bits_feature   : precision for activations
        num_bits_weight    : precision for weights
        qa_mode            : quantizers for activation quantization. Options:PACT, CGPACT, PACT+,
                                LSQ+, DoReFa.
        qw_mode            : quantizers for weight quantization. Options: SAWB, OlDSAWB, PACT,
                                CGPACT, PACT+, LSQ+, DoReFa.
        act_clip_init_val  : initialization value for activation clip_val on positive side
        act_clip_init_valn : initialization value for activation clip_val on negative  side
        w_clip_init_val    : initialization value for weight clip_val on positive side
                                (None for SAWB)
        w_clip_init_valn   : initialization value for weight clip_val on negative side
                                (None for SAWB)
        non_neg            : if True, call one-side activation quantizer
        align_zero         : if True, preserve zero, i.e align zero to an integer level

    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        num_bits_feature=32,
        qa_mode=None,
        num_bits_weight=32,
        qw_mode=None,
        **kwargs,
    ):
        """
        Initializes the QConv2dPTQ layer.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_size (int or tuple): Size of the convolutional kernel.
            stride (int or tuple, optional): Stride of the convolution. Defaults to 1.
            padding (int or tuple, optional): Padding added to all sides of the input.
                                                Defaults to 0.
            dilation (int or tuple, optional): Spacing between kernel elements. Defaults to 1.
            groups (int, optional): Number of blocked connections from input channels to
                                            output channels. Defaults to 1.
            bias (bool, optional): If True, adds a learnable bias to the output. Defaults to True.
            num_bits_feature (int, optional): Number of bits for feature quantization.
                                                Defaults to 32.
            qa_mode (str, optional): Activation quantization mode. Defaults to None.
            num_bits_weight (int, optional): Number of bits for weight quantization. Defaults to 32.
            qw_mode (str, optional): Weight quantization mode. Defaults to None.
            **kwargs (dict): Additional keyword arguments.
        """
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
        )
        qcfg = kwargs.pop("qcfg")
        self.num_bits_feature = num_bits_feature
        self.num_bits_weight = num_bits_weight
        self.qa_mode = qa_mode
        self.qw_mode = qw_mode

        self.qa_mode_calib = kwargs.get(
            "qa_mode_calib",
            qcfg.get("qa_mode_calib", "max" if num_bits_feature == 8 else "percentile"),
        )
        self.qw_mode_calib = kwargs.get(
            "qw_mode_calib",
            qcfg.get("qw_mode_calib", "max" if num_bits_feature == 8 else "percentile"),
        )
        self.act_clip_init_val = kwargs.get(
            "act_clip_init_val", qcfg.get("act_clip_init_val", 8.0)
        )
        self.act_clip_init_valn = kwargs.get(
            "act_clip_init_valn", qcfg.get("act_clip_init_valn", -8.0)
        )
        self.w_clip_init_val = kwargs.get(
            "w_clip_init_val", qcfg.get("w_clip_init_val", 1.0)
        )
        self.w_clip_init_valn = kwargs.get(
            "w_clip_init_valn", qcfg.get("w_clip_init_valn", -1.0)
        )

        self.non_neg = kwargs.get("non_neg", qcfg.get("non_neg", False))
        self.align_zero = kwargs.get("align_zero", qcfg.get("align_zero", True))
        self.extend_act_range = kwargs.get(
            "extend_act_range", qcfg.get("extend_act_range", False)
        )
        self.register_buffer(
            "calib_counter", torch.tensor(qcfg.get("qmodel_calibration_new", 0))
        )  # Counters has to be buffer in case DP is used.
        self.register_buffer(
            "ptq_calibration_counter", torch.tensor(qcfg.get("ptq_calibration", 0))
        )
        # dynCalib and PTQ cannot be used together, we need to run dynQ first to estimate the
        # init clipvals before PTQ
        self.ptqmode = "simple"  # ['simple', 'cache', 'fp32_out', 'qin_qout', None]
        if self.ptq_calibration_counter > 0:
            self.W_fp = None
            self.withinPTQblock = False

        self.register_buffer(
            "num_module_called", torch.tensor(-1)
        )  # A counter to record how many times this module has been called

        self.qcfg = qcfg

        self.calib_iterator = []  # To simplify update of clipvals in forward()
        self.quantize_feature = Qbypass()
        self.quantize_calib_feature = Qbypass()
        if self.num_bits_feature not in [32, 16]:
            self.quantize_feature = get_activation_quantizer(
                self.qa_mode,
                nbits=self.num_bits_feature,
                clip_val=self.act_clip_init_val,
                clip_valn=self.act_clip_init_valn,
                non_neg=self.non_neg,
                align_zero=self.align_zero,
            )
            if self.calib_counter > 0:
                self.quantize_calib_feature = Qdynamic(
                    self.num_bits_feature,
                    qcfg,
                    non_neg=self.non_neg,
                    align_zero=self.align_zero,
                    qmode=self.qa_mode_calib,
                    quantizer2sync=self.quantize_feature,
                )

        self.quantize_weight = Qbypass()
        self.quantize_calib_weight = Qbypass()
        if self.num_bits_weight not in [32, 16]:
            self.quantize_weight = get_weight_quantizer(
                self.qw_mode,
                nbits=self.num_bits_weight,
                clip_val=self.w_clip_init_val,
                clip_valn=self.w_clip_init_valn,
                align_zero=self.align_zero,
            )
            if self.calib_counter > 0:
                self.quantize_calib_weight = (
                    self.quantize_weight
                    if self.qw_mode in ["sawb", "sawb+", "max"]
                    else Qdynamic(
                        self.num_bits_weight,
                        qcfg,
                        non_neg=False,
                        align_zero=True,
                        qmode=self.qw_mode_calib,
                        symmetric=True,
                        quantizer2sync=self.quantize_weight,
                    )
                )
                # For non-learnable quantizers, use the real quantizer as the calib
                # quantizer directly

        if self.qw_mode == "oldsawb":
            logger.info(
                "Please consider using new SAWB quantizer. 'oldsawb' mode is not supported anymore."
            )

    def forward(self, x):
        self.num_module_called += 1

        # pylint: disable = access-member-before-definition
        if (self.calib_counter and not self.qcfg["temp_disable_calib"]) or self.qcfg[
            "force_calib_once"
        ]:
            with torch.no_grad():
                # Calib counter >0 => call calib_quantizer (Qdynamic), which will record clipvals
                # automatically based on input range
                qinput = self.quantize_calib_feature(x)
                qweight = self.quantize_calib_weight(self.weight)
                # Given quantizer2sync is specified, clipvals will be synched automatically
            if not self.qcfg[
                "force_calib_once"
            ]:  # Turn this flag off until the end of model forward()
                self.calib_counter -= 1

            if (
                self.calib_counter == 0 and self.ptq_calibration_counter == 0
            ):  # Keep the calibQ for PTQ if needed (will use force_calib_once in PTQ)
                # [optional] this should release some unused memory
                self.quantize_calib_feature = self.quantize_calib_weight = (
                    self.calib_counter
                ) = None

        elif (
            self.ptq_calibration_counter
            and self.ptqmode == "simple"
            and not self.qcfg["temp_disable_PTQ"]
        ):  # slice->conv->stack
            # 1st module:
            #   if 1st iter -> record the original batch_size.
            # (batch_size will be doubled after this) and set firstptqmodule=True
            #  need to consider DP/DDP mode, each device has its own "1st module" and org_batch_size
            curr_dev = x.device
            if (
                curr_dev not in self.qcfg["org_batch_size"]
            ):  # 1st time run into this 1st module for this device
                self.qcfg["org_batch_size"][curr_dev] = len(x)
                self.qcfg["firstptqmodule"].append(self)
            if (
                self in self.qcfg["firstptqmodule"]
            ):  # Everytime run into this 1st module, for all devices
                q_in = x.detach()
                fp_in = x.detach()
            else:
                Nbs = self.qcfg["org_batch_size"][curr_dev]
                q_in = x[Nbs:] if self.withinPTQblock else x[Nbs:].detach()
                fp_in = x[:Nbs].detach()

            # 1st time this module is run, e.g. the 5th QConv module in the 1st iter
            if self.W_fp is None:
                # Make a copy of W only first time this module runs PTQ, assuming weight
                # is initialized by fp32 already
                # pylint: disable=not-callable
                self.W_fp = self.weight.detach().clone()
                self.weight.requires_grad = (
                    True  # Some models prefer to set requires_grad to False by default
                )
                self.qcfg["params2optim"]["W"][curr_dev.index].append(self.weight)
                # FIXME: Need to consider learnable weight quantizers as well
                if not self.qcfg["ptq_freezecvs"]:
                    for param in [
                        "clip_val",
                        "clip_valn",
                        "delta",
                    ]:  # 'delta' is for brecq quantizer only
                        if hasattr(self.quantize_feature, param):
                            v = getattr(self.quantize_feature, param)
                            v.requires_grad = True
                            self.qcfg["params2optim"]["cvs"][curr_dev.index].append(v)
                # See details in fwd-hook regarding how the optimizer is set up

            # pylint: disable=not-callable
            q_out = F.conv2d(
                self.quantize_feature(q_in),
                self.quantize_weight(self.weight),
                self.bias,
                self.stride,
                self.padding,
                self.dilation,
                self.groups,
            )
            with torch.no_grad():
                # pylint: disable=not-callable
                fp32_out = F.conv2d(
                    fp_in,
                    self.W_fp,
                    self.bias,
                    self.stride,
                    self.padding,
                    self.dilation,
                    self.groups,
                )

            # NOTE: PTQ counter will be decreased in the PTQhook when inner optim is done
            return torch.cat([fp32_out, q_out])

        else:
            qinput = self.quantize_feature(x)
            qweight = self.quantize_weight(self.weight)

        qbias = self.bias  # bias not quantized

        # pylint: disable=not-callable
        return F.conv2d(
            qinput,
            qweight,
            qbias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )

    def __repr__(self):
        """Return a string that represents the PTQ quantized convolutional layer."""
        str_padding = "" if self.padding[0] == 0 else f",padding{self.padding}"
        str_group = "" if self.groups == 1 else f",groups={self.groups}"
        str_bias = "" if self.bias is not None else f",bias={self.bias is not None}"
        str_quantizer = ",QntzerW,A="
        str_quantizer += (
            ""
            if self.num_bits_weight == 32
            else f"{self.quantize_weight.__repr__().split('(')[0]},"
        )
        str_quantizer += (
            ""
            if self.num_bits_feature == 32
            else f"{self.quantize_feature.__repr__().split('(')[0]}"
        )
        return (
            f"{self.__class__.__name__}({self.in_channels},{self.out_channels},kernel "
            f"{self.kernel_size},stride{self.stride}"
            + str_padding
            + str_group
            + str_bias
            + f", Nbits_W,A={self.num_bits_weight},{self.num_bits_feature}, "
            f"uniDir={self.non_neg} {str_quantizer})"
        )


class QConv2dPTQv2(nn.Conv2d):
    """
    A wrapper for the  quantization of nn.Conv2d layers.
    Layer weights and input activation can be quantized to low-precision integers through popular
    quantization methods. Bias is not quantized.
    Supports both non-negtive activations (after relu) and symmetric/nonsymmetric 2-sided
    activations (after sum, swish, silu ..)

    Attributes:
        num_bits_feature   : precision for activations
        num_bits_weight    : precision for weights
        qa_mode            : quantizers for activation quantization. Options:PACT, CGPACT, PACT+,
                                LSQ+, DoReFa.
        qw_mode            : quantizers for weight quantization. Options: SAWB, OlDSAWB, PACT,
                                CGPACT, PACT+, LSQ+, DoReFa.
        act_clip_init_val  : initialization value for activation clip_val on positive side
        act_clip_init_valn : initialization value for activation clip_val on negative  side
        w_clip_init_val    : initialization value for weight clip_val on positive side
                                (None for SAWB)
        w_clip_init_valn   : initialization value for weight clip_val on negative side
                                (None for SAWB)
        non_neg            : if True, call one-side activation quantizer
        align_zero         : if True, preserve zero, i.e align zero to an integer level

    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        num_bits_feature=32,
        qa_mode=None,
        num_bits_weight=32,
        qw_mode=None,
        **kwargs,
    ):
        """
        Initializes the QConv2dPTQv2 layer.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_size (int or tuple): Size of the convolutional kernel.
            stride (int or tuple, optional): Stride of the convolution. Defaults to 1.
            padding (int or tuple, optional): Padding added to all sides of the input.
                                                Defaults to 0.
            dilation (int or tuple, optional): Spacing between kernel elements. Defaults to 1.
            groups (int, optional): Number of blocked connections from input channels to
                                            output channels. Defaults to 1.
            bias (bool, optional): If True, adds a learnable bias to the output. Defaults to True.
            num_bits_feature (int, optional): Number of bits for feature quantization.
                                                Defaults to 32.
            qa_mode (str, optional): Activation quantization mode. Defaults to None.
            num_bits_weight (int, optional): Number of bits for weight quantization. Defaults to 32.
            qw_mode (str, optional): Weight quantization mode. Defaults to None.
            **kwargs (dict): Additional keyword arguments.
        """
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
        )
        qcfg = kwargs.pop("qcfg")
        self.num_bits_feature = num_bits_feature
        self.num_bits_weight = num_bits_weight
        self.qa_mode = qa_mode
        self.qw_mode = qw_mode

        self.qa_mode_calib = kwargs.get(
            "qa_mode_calib",
            qcfg.get("qa_mode_calib", "max" if num_bits_feature == 8 else "percentile"),
        )
        self.qw_mode_calib = kwargs.get(
            "qw_mode_calib",
            qcfg.get("qw_mode_calib", "max" if num_bits_feature == 8 else "percentile"),
        )
        self.act_clip_init_val = kwargs.get(
            "act_clip_init_val", qcfg.get("act_clip_init_val", 8.0)
        )
        self.act_clip_init_valn = kwargs.get(
            "act_clip_init_valn", qcfg.get("act_clip_init_valn", -8.0)
        )
        self.w_clip_init_val = kwargs.get(
            "w_clip_init_val", qcfg.get("w_clip_init_val", 1.0)
        )
        self.w_clip_init_valn = kwargs.get(
            "w_clip_init_valn", qcfg.get("w_clip_init_valn", -1.0)
        )

        self.non_neg = kwargs.get("non_neg", qcfg.get("non_neg", False))
        self.align_zero = kwargs.get("align_zero", qcfg.get("align_zero", True))
        self.extend_act_range = kwargs.get(
            "extend_act_range", qcfg.get("extend_act_range", False)
        )
        self.register_buffer(
            "calib_counter", torch.tensor(qcfg.get("qmodel_calibration_new", 0))
        )  # counters has to be buffer in case DP is used.
        self.register_buffer(
            "ptq_calibration_counter", torch.tensor(qcfg.get("ptq_calibration", 0))
        )
        # dynCalib and PTQ cannot be used together, we need to run dynQ first to estimate the
        # init clipvals before PTQ
        self.ptqmode = "qout"  # ['fp32_out', 'qout', None]
        self.W_fp = None

        self.register_buffer("num_module_called", torch.tensor(-1))

        self.qcfg = qcfg

        self.calib_iterator = []  # To simplify update of clipvals in forward()
        self.quantize_feature = Qbypass()
        self.quantize_calib_feature = Qbypass()
        if self.num_bits_feature not in [32, 16]:
            self.quantize_feature = get_activation_quantizer(
                self.qa_mode,
                nbits=self.num_bits_feature,
                clip_val=self.act_clip_init_val,
                clip_valn=self.act_clip_init_valn,
                non_neg=self.non_neg,
                align_zero=self.align_zero,
            )
            if self.calib_counter > 0:
                self.quantize_calib_feature = Qdynamic(
                    self.num_bits_feature,
                    qcfg,
                    non_neg=self.non_neg,
                    align_zero=self.align_zero,
                    qmode=self.qa_mode_calib,
                    quantizer2sync=self.quantize_feature,
                )

        self.quantize_weight = Qbypass()
        self.quantize_calib_weight = Qbypass()
        if self.num_bits_weight not in [32, 16]:
            self.quantize_weight = get_weight_quantizer(
                self.qw_mode,
                nbits=self.num_bits_weight,
                clip_val=self.w_clip_init_val,
                clip_valn=self.w_clip_init_valn,
                align_zero=self.align_zero,
            )
            if self.calib_counter > 0:
                self.quantize_calib_weight = (
                    self.quantize_weight
                    if any(m in self.qw_mode for m in ["sawb", "max", "adaround"])
                    else Qdynamic(
                        self.num_bits_weight,
                        qcfg,
                        non_neg=False,
                        align_zero=True,
                        qmode=self.qw_mode_calib,
                        symmetric=True,
                        quantizer2sync=self.quantize_weight,
                    )
                )
                # For non-learnable quantizers, use the real quantizer as the calib
                # quantizer directly

        if self.qw_mode == "oldsawb":
            logger.info(
                "Please consider using new SAWB quantizer. 'oldsawb' mode is not supported anymore."
            )

    def forward(self, x):
        self.num_module_called += 1

        # pylint: disable = access-member-before-definition
        if (
            torch.is_tensor(self.calib_counter)
            and (
                torch.is_nonzero(self.calib_counter)
                and not self.qcfg["temp_disable_calib"]
            )
            or self.qcfg["force_calib_once"]
        ):
            with torch.no_grad():
                # Calib counter >0 => call calib_quantizer (Qdynamic), which will record clipvals
                # automatically based on input range
                qinput = self.quantize_calib_feature(x)
                qweight = self.quantize_calib_weight(self.weight)
            if not self.qcfg[
                "force_calib_once"
            ]:  # Turn this flag off until the end of forward()
                self.calib_counter -= 1

            if self.calib_counter == 0 and self.ptq_calibration_counter == 0:
                # Keep the calibQ for PTQ if needed (will use force_calib_once in PTQ)
                # [optional] this should release some unused memory
                self.quantize_calib_feature = self.quantize_calib_weight = (
                    self.calib_counter
                ) = None

        elif self.ptqmode == "fp32_out":
            # ie, 1st time this module is run, clone the FP32 weights, assuming weight is
            # initialized by fp32 already
            if self.W_fp is None:
                # pylint: disable=not-callable
                self.W_fp = self.weight.detach().clone()
                self.weight.requires_grad = True
            # pylint: disable=not-callable
            return F.conv2d(
                x,
                self.W_fp,
                self.bias,
                self.stride,
                self.padding,
                self.dilation,
                self.groups,
            )

        else:
            qinput = self.quantize_feature(x) if self.ptqmode != "A32_brecq" else x
            qweight = self.quantize_weight(self.weight)

        qbias = self.bias  # bias is not quantized

        # pylint: disable=not-callable
        return F.conv2d(
            qinput,
            qweight,
            qbias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )

    def __repr__(self):
        """Return a string that represents the PTQ quantized convolutional layer."""
        str_padding = "" if self.padding[0] == 0 else f",padding{self.padding}"
        str_group = "" if self.groups == 1 else f",groups={self.groups}"
        str_bias = "" if self.bias is not None else f",bias={self.bias is not None}"
        str_quantizer = ",QntzerW,A="
        str_quantizer += (
            ""
            if self.num_bits_weight == 32
            else f"{self.quantize_weight.__repr__().split('(')[0]},"
        )
        str_quantizer += (
            ""
            if self.num_bits_feature == 32
            else f"{self.quantize_feature.__repr__().split('(')[0]}"
        )
        return (
            f"{self.__class__.__name__}({self.in_channels},{self.out_channels},kernel "
            f"{self.kernel_size},stride{self.stride}"
            + str_padding
            + str_group
            + str_bias
            + f", Nbits_W,A={self.num_bits_weight},{self.num_bits_feature}, "
            f"uniDir={self.non_neg} {str_quantizer})"
        )


class DetQConv2d(QConv2dPTQv2):
    """
    A wrapper around :class:`fms_mo.modules.QConv2d` for detectron2 to support empty
    inputs and more features.

    NOTE: inherit "QConv2dPTQv2" should work for PTQ and non-PTQ cases. Inherit QConv2d for debug
    """

    def __init__(self, *args, **kwargs):
        """
        Extra keyword arguments supported in addition to those in `torch.nn.Conv2d`:

        Args:
            norm (nn.Module, optional): a normalization layer
            activation (callable(Tensor) -> Tensor): a callable activation function

        It assumes that norm layer is used before activation.
        """
        norm = kwargs.pop("norm", None)
        activation = kwargs.pop("activation", None)
        super().__init__(*args, **kwargs)

        self.norm = norm
        self.activation = activation

    def forward(self, x):
        # Torchscript does not support SyncBatchNorm yet
        # https://github.com/pytorch/pytorch/issues/40507
        # and we skip these codes in torchscript given:
        # 1. currently we only support torchscript in evaluation mode
        # 2. features needed by exporting module to torchscript are added in PyTorch 1.6 or
        # later version, `Conv2d` in these PyTorch versions has already supported empty inputs.
        if not torch.jit.is_scripting():
            if x.numel() == 0 and self.training:
                # https://github.com/pytorch/pytorch/issues/12013
                assert not isinstance(
                    self.norm, torch.nn.SyncBatchNorm
                ), "SyncBatchNorm does not support empty inputs!"

        x = super().forward(x)  # call QConv2d's forward first
        if self.norm is not None:
            x = self.norm(x)
        if self.activation is not None:
            x = self.activation(x)
        return x

    def __repr__(self):
        """Return a string that represents the detectron quantized convolutional layer."""
        expr = (
            f"{self.__class__.__name__}({self.in_channels},{self.out_channels},kernel "
            f"{self.kernel_size},stride{self.stride},padding{self.padding}, "
            f"bias={self.bias is not None},"
            f"Nbits_A,W={self.num_bits_feature}, {self.num_bits_weight})"
        )
        if self.norm is not None:
            expr += f"\n    (norm): {self.norm}, "
        if self.activation is not None:
            expr += f"\n    (act): {self.activation}, "
        return expr


class QConvTranspose2d(nn.ConvTranspose2d):
    """
    For some U-net up-sampling.
    A wrapper for the quantization of nn.ConvTranspose2d layers.

    Layer weights and input activation can be quantized to low-precision integers through popular
    quantization methods.

    Bias is not quantized.

    Supports both non-negtive activations (after relu) and symmetric/nonsymmetric 2-sided
    activations (after sum, swish, silu ..)

    Attributes:
        num_bits_feature   : precision for activations
        num_bits_weight    : precision for weights
        qa_mode            : quantizers for activation quantization. Options:PACT, CGPACT, PACT+,
                                LSQ+, DoReFa.
        qw_mode            : quantizers for weight quantization. Options: SAWB, OlDSAWB, PACT,
                                CGPACT, PACT+, LSQ+, DoReFa.
        act_clip_init_val  : initialization value for activation clip_val on positive side
        act_clip_init_valn : initialization value for activation clip_val on negative  side
        w_clip_init_val    : initialization value for weight clip_val on positive side
                                (None for SAWB)
        w_clip_init_valn   : initialization value for weight clip_val on negative side
                                (None for SAWB)
        non_neg            : if True, call one-side activation quantizer
        align_zero         : if True, preserve zero, i.e align zero to an integer level

    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        output_padding=0,
        dilation=1,
        groups=1,
        bias=True,
        num_bits_feature=32,
        qa_mode=None,
        num_bits_weight=32,
        qw_mode=None,
        **kwargs,
    ):
        """
        Initializes the quantized convolutional layer.

        Args:
            in_channels (int): Number of channels in the input image.
            out_channels (int): Number of channels produced by the convolution.
            kernel_size (int or tuple): Size of the convolving kernel.
            stride (int or tuple, optional): Stride of the convolution. Defaults to 1.
            padding (int or tuple, optional): Zero-padding added to both sides of the input.
                                                Defaults to 0.
            output_padding (int or tuple, optional): Additional size added to the output shape.
                                                Defaults to 0.
            dilation (int or tuple, optional): Spacing between kernel elements. Defaults to 1.
            groups (int, optional): Number of blocked connections from input channels to output
                                                channels. Defaults to 1.
            bias (bool, optional): If ``True``, adds a learnable bias to the output.
                                                Defaults to ``True``.
            num_bits_feature (int, optional): Number of bits for feature quantization.
                                                Defaults to 32.
            qa_mode (str, optional): Quantization mode for feature. Defaults to None.
            num_bits_weight (int, optional): Number of bits for weight quantization.
                                                Defaults to 32.
            qw_mode (str, optional): Quantization mode for weight. Defaults to None.
            **kwargs (dict): Other keyword arguments.
        """
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        qcfg = kwargs.pop("qcfg")
        self.num_bits_feature = num_bits_feature
        self.num_bits_weight = num_bits_weight
        self.qa_mode = qa_mode
        self.qw_mode = qw_mode

        self.qa_mode_calib = kwargs.get(
            "qa_mode_calib",
            qcfg.get("qa_mode_calib", "max" if num_bits_feature == 8 else "percentile"),
        )
        self.qw_mode_calib = kwargs.get(
            "qw_mode_calib",
            qcfg.get("qw_mode_calib", "max" if num_bits_feature == 8 else "percentile"),
        )
        self.act_clip_init_val = kwargs.get(
            "act_clip_init_val", qcfg.get("act_clip_init_val", 8.0)
        )
        self.act_clip_init_valn = kwargs.get(
            "act_clip_init_valn", qcfg.get("act_clip_init_valn", -8.0)
        )
        self.w_clip_init_val = kwargs.get(
            "w_clip_init_val", qcfg.get("w_clip_init_val", 1.0)
        )
        self.w_clip_init_valn = kwargs.get(
            "w_clip_init_valn", qcfg.get("w_clip_init_valn", -1.0)
        )

        self.non_neg = kwargs.get("non_neg", qcfg.get("non_neg", False))
        self.align_zero = kwargs.get("align_zero", qcfg.get("align_zero", True))
        self.extend_act_range = kwargs.get(
            "extend_act_range", qcfg.get("extend_act_range", False)
        )
        self.register_buffer(
            "calib_counter", torch.tensor(qcfg.get("qmodel_calibration_new", 0))
        )
        self.register_buffer("num_module_called", torch.tensor(0))

        self.qcfg = qcfg

        self.calib_iterator = []
        self.quantize_feature = Qbypass()
        self.quantize_calib_feature = Qbypass()
        if self.num_bits_feature not in [32, 16]:
            self.quantize_feature = get_activation_quantizer(
                self.qa_mode,
                nbits=self.num_bits_feature,
                clip_val=self.act_clip_init_val,
                clip_valn=self.act_clip_init_valn,
                non_neg=self.non_neg,
                align_zero=self.align_zero,
            )
            if self.calib_counter > 0:
                self.quantize_calib_feature = Qdynamic(
                    self.num_bits_feature,
                    qcfg,
                    non_neg=self.non_neg,
                    align_zero=self.align_zero,
                    qmode=self.qa_mode_calib,
                )
                if hasattr(self.quantize_feature, "clip_val"):
                    self.calib_iterator.append(
                        ("quantize_feature", "quantize_calib_feature", "clip_val")
                    )
                if hasattr(self.quantize_feature, "clip_valn"):
                    self.calib_iterator.append(
                        ("quantize_feature", "quantize_calib_feature", "clip_valn")
                    )

        self.quantize_weight = Qbypass()
        self.quantize_calib_weight = Qbypass()
        if self.num_bits_weight not in [32, 16]:
            self.quantize_weight = get_weight_quantizer(
                self.qw_mode,
                nbits=self.num_bits_weight,
                clip_val=self.w_clip_init_val,
                clip_valn=self.w_clip_init_valn,
                align_zero=self.align_zero,
            )
            if self.calib_counter > 0:
                self.quantize_calib_weight = (
                    self.quantize_weight
                    if self.qw_mode in ["sawb", "sawb+", "max"]
                    else Qdynamic(
                        self.num_bits_weight,
                        qcfg,
                        non_neg=False,
                        align_zero=True,
                        qmode=self.qw_mode_calib,
                        symmetric=True,
                    )
                )
                # for non-learnable quantizers, use real quantizer as the calib quantizer directly
                if hasattr(self.quantize_weight, "clip_val"):
                    self.calib_iterator.append(
                        ("quantize_weight", "quantize_calib_weight", "clip_val")
                    )
                if hasattr(self.quantize_weight, "clip_valn"):
                    self.calib_iterator.append(
                        ("quantize_weight", "quantize_calib_weight", "clip_valn")
                    )

        if self.qw_mode == "oldsawb":
            logger.info(
                "Please consider using new SAWB quantizer. 'oldsawb' mode is not supported anymore."
            )

    def forward(self, x):
        """
        Forward pass of the quantized convolution transpose layer.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor.
        """
        if self.calib_counter > 0:
            with torch.no_grad():
                qinput = self.quantize_calib_feature(x)
                qweight = self.quantize_calib_weight(self.weight)
                for realQattr, calibQattr, cvi in self.calib_iterator:
                    realQ, calibQ = getattr(self, realQattr), getattr(self, calibQattr)
                    getattr(realQ, cvi).fill_(getattr(calibQ, cvi).item())

            self.calib_counter -= 1
            if self.calib_counter == 0:
                # [optional] this should release the memory
                self.quantize_calib_feature = self.quantize_calib_weight = None

        else:
            qinput = self.quantize_feature(x)
            qweight = self.quantize_weight(self.weight)

        qbias = self.bias  # Bias not quantized
        # pylint: disable=not-callable
        output = F.conv_transpose2d(
            qinput,
            qweight,
            qbias,
            stride=self.stride,
            padding=self.padding,
            output_padding=self.output_padding,
            dilation=self.dilation,
            groups=self.groups,
        )

        self.num_module_called += 1

        return output

    @classmethod
    def from_torch(cls, convtranspose2d, qcfg, **kwargs):
        """Converts a torch convolution transpose layer to a quantized convolution transpose layer.

        Args:
            cls (class): The class of the quantized convolution transpose layer to be created.
            convtranspose2d (torch.nn.ConvTranspose2d): The torch convolution transpose layer to
                        be converted.
            qcfg (dict): A dictionary containing the quantization configuration parameters.
            **kwargs (dict): Additional keyword arguments to be passed to the quantized convolution
                        transpose layer constructor.

        Returns:
            QConvTranspose2d: The quantized convolution transpose layer created from the
                            torch convolution transpose layer.
        """
        qconvtranspose2d = cls(
            in_channels=convtranspose2d.in_channels,
            out_channels=convtranspose2d.out_channels,
            kernel_size=convtranspose2d.kernel_size,
            stride=convtranspose2d.stride,
            padding=convtranspose2d.padding,
            output_padding=convtranspose2d.output_padding,
            dilation=convtranspose2d.dilation,
            groups=convtranspose2d.groups,
            bias=convtranspose2d.bias is not None,
            num_bits_feature=qcfg["nbits_a"],
            qa_mode=qcfg["qa_mode"],
            num_bits_weight=qcfg["nbits_w"],
            qw_mode=qcfg["qw_mode"],
            qcfg=qcfg,
        )
        device = kwargs.get("device", next(convtranspose2d.parameters()).device)
        qconvtranspose2d.weight = nn.Parameter(convtranspose2d.weight)
        if convtranspose2d.bias is not None:
            qconvtranspose2d.bias = nn.Parameter(convtranspose2d.bias)
        return qconvtranspose2d.to(device)

    def __repr__(self):
        """Return a string that represents the quantized convolutional transposed layer."""
        str_padding = "" if self.padding[0] == 0 else f",padding{self.padding}"
        str_group = "" if self.groups == 1 else f",groups={self.groups}"
        str_bias = "" if self.bias is not None else f",bias={self.bias is not None}"
        str_quantizer = ",QntzerW,A="
        str_quantizer += (
            ""
            if self.num_bits_weight == 32
            else f"{self.quantize_weight.__repr__().split('(')[0]},"
        )
        str_quantizer += (
            ""
            if self.num_bits_feature == 32
            else f"{self.quantize_feature.__repr__().split('(')[0]}"
        )
        return (
            f"{self.__class__.__name__}({self.in_channels},{self.out_channels},kernel "
            f"{self.kernel_size},stride{self.stride}"
            + str_padding
            + str_group
            + str_bias
            + f", Nbits_W,A={self.num_bits_weight},{self.num_bits_feature}, uniDir={self.non_neg} "
            f"{str_quantizer})"
        )


# ------------------------------------------------------------------------------
# ----- The following wrappers are for torch FX CPU lowering only (FBGEMM) -----
# ----- NOTE: do not use them directly in QAT, backward is not defined     -----
# ------------------------------------------------------------------------------


UINT8_TO_INT8_ADJ = 128


class QConv2d_cutlass_i8i32_nt(nn.Conv2d):
    """
    A new QConv2d class for testing external kernels specifically for int8
    NOTE: Need to add functionality for torch.quint8, currently assumes torch.int8.
    """

    @classmethod
    def from_fms_mo(cls, fms_mo_qconv2d, **kwargs):
        """Converts a QuantizedConv2d module to a cutlass QConv2d_cutlass_i8i32_nt module.

        Args:
            cls (class): The class of the QConv2d_cutlass_i8i32_nt module.
            fms_mo_qconv2d (nn.Module): The QuantizedConv2d module to be converted.
            **kwargs (dict): Additional keyword arguments.

        Returns:
            nn.Module: The converted QConv2d_cutlass_i8i32_nt module.
        """
        assert (
            fms_mo_qconv2d.num_bits_feature == 8 and fms_mo_qconv2d.num_bits_weight == 8
        ), "Please check nbits setting!"

        target_device = kwargs.get(
            "target_device", next(fms_mo_qconv2d.parameters()).device
        )

        qconv2d_cutlass = cls(
            fms_mo_qconv2d.in_channels,
            fms_mo_qconv2d.out_channels,
            fms_mo_qconv2d.kernel_size,
            fms_mo_qconv2d.stride,
            fms_mo_qconv2d.padding,
        )

        target_feature_bits, target_weight_bits = torch.int8, torch.int8

        with torch.no_grad():
            qconv2d_cutlass.quantize_feature = fms_mo_qconv2d.quantize_feature
            qconv2d_cutlass.quantize_weight = fms_mo_qconv2d.quantize_weight

            if not hasattr(qconv2d_cutlass.quantize_feature, "clip_valn"):
                if isinstance(qconv2d_cutlass.quantize_feature, PACT):
                    logger.warning(
                        f"Using PACT {qconv2d_cutlass.quantize_feature} object, "
                        " will assume clip_valn =0"
                    )
                    qconv2d_cutlass.quantize_feature.clip_valn = 0
                else:
                    logger.warning(
                        f"clip_valn doesn't exist for {qconv2d_cutlass.quantize_feature} "
                        "object, will assume clip_valn = -clip_val"
                    )
                    qconv2d_cutlass.quantize_feature.clip_valn = (
                        -qconv2d_cutlass.quantize_feature.clip_val
                    )

            input_scale = (
                qconv2d_cutlass.quantize_feature.clip_val
                - qconv2d_cutlass.quantize_feature.clip_valn
            ) / (2**qconv2d_cutlass.quantize_feature.num_bits - 1)
            input_zero_point = torch.round(
                -qconv2d_cutlass.quantize_feature.clip_valn / input_scale
            ).to(target_feature_bits)

            qconv2d_cutlass.register_buffer("input_scale", input_scale)
            qconv2d_cutlass.register_buffer("input_zp", input_zero_point)

            # NOTE: Store for debug purposes
            qconv2d_cutlass.float_weight = fms_mo_qconv2d.weight.float()

            # If we use PT native Qfunc, may not work with fp16, hence the use of weight.float()
            qconv2d_cutlass.quantize_weight.dequantize = False
            # Convert weight from FP16 to INT8 quantized values
            # NOTE: The position of this line is important, it must occur before w_scale computation
            # as it may updated the clip_val and clip_valn values.
            quantized_weights = (
                qconv2d_cutlass.quantize_weight(fms_mo_qconv2d.weight.float())
                .to(target_weight_bits)
                .to(memory_format=torch.channels_last)
            )

            if not hasattr(qconv2d_cutlass.quantize_weight, "clip_valn"):
                logger.warning(
                    f"clip_valn does not exist for {qconv2d_cutlass.quantize_weight} "
                    "object, will assume clip_valn = -clip_val"
                )
                qconv2d_cutlass.quantize_weight.clip_valn = (
                    -qconv2d_cutlass.quantize_weight.clip_val
                )
            w_scale = (
                qconv2d_cutlass.quantize_weight.clip_val
                - qconv2d_cutlass.quantize_weight.clip_valn
            ) / (2**qconv2d_cutlass.quantize_weight.num_bits - 2)
            w_zp = torch.zeros_like(w_scale, dtype=target_weight_bits)

            qconv2d_cutlass.register_buffer("w_scale", w_scale.float())
            qconv2d_cutlass.register_buffer("w_zp", w_zp)

        qconv2d_cutlass.weight = nn.Parameter(
            quantized_weights,
            requires_grad=False,
        )
        if fms_mo_qconv2d.bias is None:
            qconv2d_cutlass.bias = nn.Parameter(
                fms_mo_qconv2d.bias, requires_grad=False
            )

        return qconv2d_cutlass.to(target_device)

    def _get_name(self):
        """
        Returns the name of the QConv2d_cutlass_i8i32_nt as a string.
        """
        return "QuantizedConv2d_cutlassi8i32"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x comes in as torch.float32 or torch.float16
        """
        with torch.no_grad():
            # Store initial input dtype to return output dtype consistent with original
            input_dtype = x.dtype

            # Step 1: Q activation
            self.quantize_feature.dequantize = False
            x = self.quantize_feature(x)

            # The purpose is to go from uint8 to int8
            x = x - UINT8_TO_INT8_ADJ

            # Set back to int8 as it is required by the custom kernel
            x = x.to(torch.int8).to(memory_format=torch.channels_last)

            output_int8 = torch.ops.cutlass.conv2di8i32nt(
                x, self.weight, self.stride[0], self.padding[0], self.dilation[0]
            )

            # Create torch of ones from image:
            x_ones = torch.ones(
                x.shape[-2] - 2 * self.padding[0],
                x.shape[-1] - 2 * self.padding[0],
                device=x.device,
            )
            # Create padding around the tensor above if there is padding
            if self.padding[0] > 0:
                x_ones_padded = F.pad(x_ones, pad=2 * self.padding)
            else:
                x_ones_padded = x_ones

            # Expand the tensor dimensions
            x_ones_padded = x_ones_padded.expand(
                (
                    1,
                    x.shape[1],
                )
                + x_ones_padded.shape
            )

            padding_correction_term = torch.ops.aten.conv2d(
                x_ones_padded,
                self.weight.float(),
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
            )

            # NOTE: If torch.uint8 is not used, we assume input_zp is not applied hence
            # no adjustments needed.
            output = (
                self.w_scale
                * self.input_scale
                * (
                    output_int8
                    - padding_correction_term * (self.input_zp - UINT8_TO_INT8_ADJ)
                )
            )
        return output.to(input_dtype)


# KEEP THIS AT END OF FILE - classes must be declared
QConv2d_modules = (
    QConv2d,
    DetQConv2d,
    QConv2dPTQ,
    QConv2dPTQv2,
    QConvTranspose2d,
    QConv2d_cutlass_i8i32_nt,
)


def isinstance_qconv2d(module):
    """
    Checks if the given module is one of the available quantized convolution classes.

    Args:
        module (nn.Module): The module to check.

    Returns:
        bool: True if the module is a quantized convolution class, False otherwise.
    """
    return isinstance(module, QConv2d_modules)
