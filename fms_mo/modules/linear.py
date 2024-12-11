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

"""Quantization linear modules"""

# pylint: disable=arguments-renamed

# Standard
import json
import logging

# Third Party
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

# First Party
from fms_mo.custom_ext_kernels.utils import pack_vectorized
from fms_mo.quant.quantizers import (HardPrune, Qbypass, Qdynamic,
                                     get_activation_quantizer,
                                     get_weight_quantizer, mask_fc_kij)

logger = logging.getLogger(__name__)


class QLinear(nn.Linear):
    """docstring for QLinear_pact.
    A wrapper for the  quantization of linear (aka. affine or fc) layers
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
        w_clip_init_valn   : initialization value for weight clip_val on negative  side
                                (None for SAWB)
        non_neg            : if True, call one-side activation quantizer
        align_zero         : if True, preserve zero, i.e align zero to an integer level
    """

    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        num_bits_feature=32,
        qa_mode=None,
        num_bits_weight=32,
        qw_mode=None,
        **kwargs,
    ):
        """
        Initializes the quantized linear layer.

        Args:
            in_features (int): Number of input features.
            out_features (int): Number of output features.
            bias (bool, optional): Whether to include a bias term. Defaults to True.
            num_bits_feature (int, optional): Number of bits for feature quantization.
                                                Defaults to 32.
            qa_mode (str, optional): Quantization mode for feature. Defaults to None.
            num_bits_weight (int, optional): Number of bits for weight quantization.
                                                Defaults to 32.
            qw_mode (str, optional): Quantization mode for weight. Defaults to None.
            **kwargs (dict): Additional keyword arguments.
        """

        super().__init__(
            in_features, out_features, bias, device=kwargs.get("device", "cuda")
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
        self.fp8_use_subnormal = kwargs.get(
            "fp8_use_subnormal", qcfg.get("fp8_use_subnormal", False)
        )
        self.register_buffer(
            "calib_counter", torch.tensor(qcfg.get("qmodel_calibration_new", 0))
        )  # Counters has to be buffer in case DP is used.
        self.register_buffer(
            "num_module_called", torch.tensor(0)
        )  # A counter to record how many times this module has been called
        self.ptqmode = "qout"  # ['fp32_out', 'qout', None]
        self.W_fp = None
        self.use_PT_native_Qfunc = kwargs.get(
            "use_PT_native_Qfunc", qcfg.get("use_PT_native_Qfunc", False)
        )

        self.perGp = kwargs.get("qgroup", qcfg.get("qgroup", None))
        self.qcfg = qcfg

        self.calib_iterator = []
        # To simplify update of clipvals in forward()
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
                use_PT_native_Qfunc=self.use_PT_native_Qfunc,
                use_subnormal=self.fp8_use_subnormal,
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
                perGp=self.perGp,
                use_subnormal=self.fp8_use_subnormal,
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
        # For non-learnable quantizers, use the real quantizer as the calib quantizer directly

        if self.qw_mode == "oldsawb":
            logger.info(
                "Please consider using new SAWB quantizer. 'oldsawb' mode is not supported anymore."
            )
        self.smoothq = qcfg.get("smoothq", False)
        if self.smoothq:
            self.register_buffer("smoothq_act_scale", torch.zeros(w_size[1]))
            self.register_buffer(
                "smoothq_alpha",
                torch.tensor([qcfg.get("smoothq_alpha", 0.5)], dtype=torch.float32),
            )

    def forward(self, x):
        """
        Forward pass of the layer.

        Args:
            x (Tensor): Input tensor of shape (batch_size, in_features).

        Returns:
            Tensor: Output tensor of shape (batch_size, out_features).
        """
        if self.smoothq:
            scale = self.get_smoothq_scale(x)
        else:
            scale = torch.tensor([1.0]).to(x.dtype).to(x.device)

        # pylint: disable = access-member-before-definition
        if self.calib_counter:
            with torch.no_grad():
                qinput = self.quantize_calib_feature(x / scale)
                qweight = self.quantize_calib_weight(self.weight * scale)
            self.calib_counter -= 1
            if self.calib_counter == 0:
                self.quantize_calib_feature = None
                self.quantize_calib_weight = None
                self.calib_counter = None  # [optional] this should release the memory

        elif self.ptqmode == "fp32_out":
            if self.W_fp is None:
                # i.e., 1st time this module is run, clone the FP32 weights, assuming weight is
                # initialized by fp32 already
                # pylint: disable=not-callable
                self.W_fp = self.weight.detach().clone()
                self.weight.requires_grad = (
                    True  # Some models prefer to set requires_grad to False by default
                )

            # pylint: disable=not-callable
            return F.linear(x, self.W_fp, self.bias)
        else:
            qinput = self.quantize_feature(x / scale)
            # Default self.update_type == 'hard' pruning.
            if self.mask is not None:
                pweight = HardPrune.apply(
                    self.weight, self.mask.to(self.weight.device), self.p_inplace
                )
                qweight = self.quantize_weight(pweight)
            else:
                qweight = self.quantize_weight(self.weight * scale)

        qbias = self.bias

        # pylint: disable=not-callable
        output = F.linear(qinput, qweight, qbias)

        self.num_module_called += 1

        return output

    def get_mask(self):
        """
        Gets the mask for the weight tensor.

        By default, uses hard pruning. The mask is stored in the `mask` attribute.

        Returns:
            torch.Tensor: The mask tensor.
        """
        if self.mask_type == "kij":
            self.mask = mask_fc_kij(
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

    def set_act_scale(self, act_scale):
        """Sets the activation scale for smooth quantization.

        Args:
            act_scale (torch.Tensor): The activation scale to be set.
                It should have the same number of channels as the weight tensor.
        """
        assert (
            act_scale.shape[0] == self.weight.shape[1]
        ), "scale applies to per-channel"
        self.smoothq_act_scale.copy_(act_scale)

    def get_smoothq_scale(self, x):
        """
        Calculate the smoothQ scale for a given input tensor x.

        Args:
            x: The input tensor for which to calculate the smoothQ scale.

        Returns:
            smoothq_scale: The calculated smoothQ scale for the input tensor x.
        """
        if self.smoothq_act_scale.sum().item() == 0.0:
            smoothq_scale = torch.tensor([1.0]).to(x.dtype).to(x.device)
        else:
            weight_scale = self.weight.abs().max(dim=0, keepdim=True)[0].clamp(min=1e-5)
            if isinstance(self.smoothq_alpha, torch.Tensor):
                alpha = self.smoothq_alpha.item()
            else:
                alpha = self.smoothq_alpha
            smoothq_scale = (
                (self.smoothq_act_scale.pow(alpha) / weight_scale.pow(1.0 - alpha))
                .clamp(min=1e-5)
                .to(x.dtype)
            )
        return smoothq_scale

    def __repr__(self):
        """
        Returns a string representation of the quantized linear layer.
        """
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
        str_quantizer += (
            ""
            if self.mask is None
            else f", p_rate={self.prune_ratio}, p_group={self.prune_group}, "
        )
        return (
            f"{self.__class__.__name__}({self.in_features},{self.out_features}, "
            f"Nbits_W,A={self.num_bits_weight},{self.num_bits_feature}{str_quantizer})"
        )


# ------------------------------------------------------------------------------
# ----- The following wrappers are for torch FX CPU lowering only (FBGEMM) -----
# ----- NOTE: do not use them directly in QAT, backward is not defined     -----
# ------------------------------------------------------------------------------
class QLinearFPout(torch.ao.nn.quantized.Linear):
    """
    A new QLinear class for fbgemm lowering, not for generic QAT/PTQ use (no backward)

    Original "torch.ao.nn.quantized.Linear" is designed to
    find pattern   Q->(dQ->ref Linear->Q)->dQ   then
    swap to        Q->     QLinear       ->dQ
    which means 1) this QLinear takes INT8 as input AND output INT8
                2) this QLinear needs to know output scale/zp, which is usually unavailable
                        from fms_mo models

    Here we utilize another native backend function to
    find pattern   (Q->dQ->ref Linear)->
    swap to        QLinearFPout         ->dQ

    """

    @classmethod
    def from_reference(cls, ref_qlinear, input_scale, input_zero_point):
        r"""Creates a (fbgemm/qnnpack) quantized module from a reference quantized module

        Args:
            ref_qlinear (Module): a reference quantized linear module, either produced by
                        torch.ao.quantization utilities or provided by the user
            input_scale (float): scale for input Tensor
            input_zero_point (int): zero point for input Tensor
            NOTE: scale/zp are from input node
        """
        qlinear = cls(
            ref_qlinear.in_features,
            ref_qlinear.out_features,
        )
        qweight = ref_qlinear.get_quantized_weight()
        # CPU/FBGEMM doesn't support perCh
        if ref_qlinear.weight_qscheme in [
            torch.per_channel_symmetric,
            torch.per_channel_affine,
        ]:
            qscale_perT = max(qweight.q_per_channel_scales())
            qweight = torch.quantize_per_tensor(
                qweight.dequantize(), qscale_perT, 0, torch.qint8
            )
        qlinear.set_weight_bias(qweight.cpu(), ref_qlinear.bias.cpu())

        qlinear.scale = float(input_scale)
        qlinear.zero_point = int(input_zero_point)
        return qlinear

    def _get_name(self):
        """
        Returns the name of the QuantizedLinear_FPout as a string.
        """
        return "QuantizedLinear_FPout"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the layer.

        Args:
            x (Tensor): Input tensor of shape (batch_size, in_features).

        Returns:
            Tensor: Output tensor of shape (batch_size, out_features).
        """
        return torch.ops.quantized.linear_with_input_q_dq_qweight_dq_output_fp32(
            x, self.scale, self.zero_point, self._packed_params._packed_params
        )


class QLinearDebug(nn.Linear):
    """
    A new QLinear class for debugging lowering, no backward

    Here we assume the graph has Q/dQ nodes already, and try to absorb those nodes into this QLinear
    here we use FP32 native backend function, but external kernels can be used, too
    find pattern   (Q->dQ->ref Linear) ->
    swap to        (QLinearDebug ) ->

    """

    @classmethod
    def from_reference(cls, ref_qlinear, input_scale, input_zero_point):
        r"""Creates a quantized linear module from a reference quantized module
        Args:
            ref_qlinear (Module): a reference quantized linear module, either produced by
                                torch.ao.quantization utilities or provided by the user
            input_scale (float): scale for input Tensor
            input_zero_point (int): zero point for input Tensor
            NOTE: scale/zp are for input activation
        """

        nnlinear = cls(ref_qlinear.in_features, ref_qlinear.out_features)

        nnlinear.register_buffer("input_scale", input_scale)
        nnlinear.register_buffer("input_zp", input_zero_point)
        was_fp16 = False
        if ref_qlinear.weight.dtype == torch.float16:
            was_fp16 = True
            ref_qlinear.float()
        nnlinear.weight = nn.Parameter(
            ref_qlinear.get_weight(), requires_grad=False
        )  # this is Q(w).dQ()
        nnlinear.bias = nn.Parameter(ref_qlinear.bias, requires_grad=False)
        if was_fp16:
            ref_qlinear.half()
            nnlinear.half()
        return nnlinear

    @classmethod
    def from_fms_mo(cls, fms_mo_qlinear, **kwargs):
        """
        Converts a QLinear module to QLinearDebug.

        Args:
            cls: The class of the QLinearModule to be created.
            fms_mo_qlinear: The QLinear module to be converted.
            kwargs: Additional keyword arguments.

        Returns:
            A QLinearDebug object initialized with the weights and biases from the
                QLinear module.
        """
        assert fms_mo_qlinear.num_bits_feature in [
            4,
            8,
        ] and fms_mo_qlinear.num_bits_weight in [4, 8], "Please check nbits setting!"

        target_device = kwargs.get(
            "target_device", next(fms_mo_qlinear.parameters()).device
        )
        qlinear_cublas = cls(fms_mo_qlinear.in_features, fms_mo_qlinear.out_features)
        qlinear_cublas.input_dtype = (
            torch.quint8
        )  # Assume input to fms_mo QLinear is always asym

        with torch.no_grad():
            Qa = fms_mo_qlinear.quantize_feature
            input_scale = (Qa.clip_val - Qa.clip_valn) / (2**Qa.num_bits - 1)
            input_zero_point = torch.round(-Qa.clip_valn / input_scale).to(torch.int)
            qlinear_cublas.register_buffer("input_scale", input_scale)
            qlinear_cublas.register_buffer("input_zp", input_zero_point)

            Qw = fms_mo_qlinear.quantize_weight
            w_scale = Qw.clip_val * 2 / (2**Qw.num_bits - 2)
            w_zp = torch.zeros_like(w_scale, dtype=torch.int)
            qlinear_cublas.register_buffer("w_scale", w_scale.float())
            qlinear_cublas.register_buffer("w_zp", w_zp)

        qlinear_cublas.weight = nn.Parameter(
            Qw(fms_mo_qlinear.weight), requires_grad=False
        )
        qlinear_cublas.bias = nn.Parameter(fms_mo_qlinear.bias, requires_grad=False)

        return qlinear_cublas.to(target_device)

    def _get_name(self):
        """
        Returns the name of the QLinear_Debug as a string.
        """
        return "QuantizedLinear_Debug"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the layer.

        Args:
            x (Tensor): Input tensor of shape (batch_size, in_features).

        Returns:
            Tensor: Output tensor of shape (batch_size, out_features).
        """
        with torch.no_grad():
            x = torch.clamp(
                (x / self.input_scale + self.input_zp).round(), 0, 255
            )  # map to int
            x = (x - self.input_zp) * self.input_scale  # deQ
            x = super().forward(x)
        return x


class QLinearW4A32Debug(nn.Linear):
    """
    Here we assume the graph does not have Q/dQ nodes, since A32,
    so all we need is to dQ the W
    """

    @classmethod
    def from_reference(cls, ref_qlinear):
        r"""Creates a quantized linear module from a reference quantized module
        Args:
            ref_qlinear (Module): a reference quantized linear module
        """
        qlinear = cls(ref_qlinear.in_features, ref_qlinear.out_features)

        org_dtype = ref_qlinear.weight.dtype
        qlinear.weight = nn.Parameter(
            ref_qlinear.float().get_weight().to(org_dtype), requires_grad=False
        )  # This is Q(w).dQ()
        qlinear.bias = nn.Parameter(ref_qlinear.bias.to(org_dtype), requires_grad=False)
        return qlinear

    @classmethod
    def from_fms_mo(cls, fms_mo_qlinear):
        """
        Converts a QLinear module to QLinearW4A32Debug.

        Args:
            cls: The class of the QLinearModule to be created.
            fms_mo_qlinear: The QLinear module to be converted.

        Returns:
            A QLinearW4A32Debug object initialized with the weights and biases from the
                QLinear module.
        """
        qlinear = cls(fms_mo_qlinear.in_features, fms_mo_qlinear.out_features)

        # If your model is half(), ref_linear and PT native quant func won't work,
        # need to convert to .float() first
        org_dtype = fms_mo_qlinear.weight.dtype
        fms_mo_qlinear.float()
        qlinear.weight = nn.Parameter(
            fms_mo_qlinear.quantize_weight(fms_mo_qlinear.weight).to(org_dtype),
            requires_grad=False,
        )  # This is Q(w).dQ()
        qlinear.bias = nn.Parameter(
            fms_mo_qlinear.bias.to(org_dtype), requires_grad=False
        )
        return qlinear

    def _get_name(self):
        """
        Returns the name of the QLinearW4A32Debug as a string.
        """
        return "QuantizedLinear_W4A32_Debug"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the layer.

        Args:
            x (Tensor): Input tensor of shape (batch_size, in_features).

        Returns:
            Tensor: Output tensor of shape (batch_size, out_features).
        """
        with torch.no_grad():
            # Could use cublas fp16 kernel, but make sure it accept out_feat < 16 cases
            x = super().forward(x)
        return x


class NNLinearCublasDebug(nn.Linear):
    """
    A Linear class for debugging FP32, no Quantization, simply swap nn.Linear with this,
    which calls cublas gemm instead of F.linear
    """

    @classmethod
    def from_float(cls, nnlinear):
        """
        Converts a floating point neural network layer to a cublas neural network layer.

        Args:
            cls (class): The class NNLinearCublasDebug.
            nnlinear (nn.Linear): The floating point Linear layer to be converted.

        Returns:
            nnlinear_cu (cls): The converted NNLinearCublasDebug.
        """
        nnlinear_cu = cls(nnlinear.in_features, nnlinear.out_features)
        nnlinear_cu.weight = nn.Parameter(
            nnlinear.weight.float(), requires_grad=False
        )  # force to use F32
        nnlinear_cu.bias = nn.Parameter(nnlinear.bias.float(), requires_grad=False)
        return nnlinear_cu

    def _get_name(self):
        """
        Returns the name of the NNLinearCublasDebug as a string.
        """
        return "Linear_cublas_fp32"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the layer.

        Args:
            x (Tensor): Input tensor of shape (batch_size, in_features).

        Returns:
            Tensor: Output tensor of shape (batch_size, out_features).
        """
        # Check shape before GEMM
        if len(x.shape) == 3 and len(self.weight.shape) == 2:  # Batched input
            re_shape = (-1, x.shape[2])
            tar_shape = tuple(x.shape[:2]) + (self.weight.shape[0],)  # W is transposed
            x = x.reshape(re_shape)
        elif len(x.shape) == len(self.iweight.shape) == 2:  # 2D
            tar_shape = (x.shape[0], self.weight.shape[0])  # W is transposed
        else:
            raise RuntimeError("Input dimension to Linear is not 2D or batched 2D")

        with torch.no_grad():
            x = torch.ops.mygemm.gemm_nt_f32(x, self.weight)  # fp32 only
            x = x.reshape(tar_shape) + self.bias
        return x


class QLinearINT8Deploy(nn.Linear):
    """
    A QLinear class for lowering test, no backward
    weight is stored in torch.int8 (or could use int32 for gptq?)
    also need to override forward to make it   Q->Linear->dQ    on the graph.
                                (as opposed to Q->dQ->Linear)
    """

    @classmethod
    def from_fms_mo(cls, fms_mo_qlinear, **kwargs):
        """
        Converts a QLinear module to QLinearINT8Deploy.

        Args:
            cls: The class of the QLinearModule to be created.
            fms_mo_qlinear: The QLinear module to be converted.

        Returns:
            A QLinearINT8Deploy object initialized with the weights and biases from the
                QLinear module.
        """
        assert all(
            getattr(fms_mo_qlinear, a_or_w) in [4, 8]
            for a_or_w in ["num_bits_feature", "num_bits_weight"]
        ), "Please check nbits setting!"

        target_device = kwargs.get(
            "target_device",
            kwargs.get("device", next(fms_mo_qlinear.parameters()).device),
        )
        fms_mo_w_dtype = fms_mo_qlinear.weight.dtype
        qlin_int = cls(
            fms_mo_qlinear.in_features,
            fms_mo_qlinear.out_features,
            bias=fms_mo_qlinear.bias is not None,
            device=target_device,
        )
        # Make sure to register an Op for integer matmul, could be real INT matmul or emulation
        qcfg = getattr(fms_mo_qlinear, "qcfg", {})
        qlin_int.useINTkernel = qcfg.get("useINTkernel", True)
        qlin_int.usePTnativeQfunc = kwargs.get("use_PT_native_Qfunc", False)
        qlin_int.acc24minmax = (int(-(2**24) / 2 + 1), int(2**24 / 2))
        qlin_int.simi24toi16 = kwargs.get("simi24toi16", False)
        qlin_int.acc_dtype = torch.float16
        qlin_int.nbits_a = fms_mo_qlinear.num_bits_feature  # only support INT8 for now
        qlin_int.nbits_w = fms_mo_qlinear.num_bits_weight

        with torch.no_grad():
            Qa = fms_mo_qlinear.quantize_feature
            Qw = fms_mo_qlinear.quantize_weight
            a_cv, a_cvn = Qa.clip_val.item(), Qa.clip_valn.item()
            w_cv = Qw.clip_val.item()
            # NOTE: Keep w transposed to prevent confusion
            Qw.dequantize = False
            w_int8 = Qw(
                fms_mo_qlinear.weight.float()
            )  # Qw.clipval should have been updated after this
            qlin_int.weight = nn.Parameter(
                w_int8.to(torch.int8), requires_grad=False
            )  # NOTE: Needs INT W stored as FP...

            if qlin_int.usePTnativeQfunc:
                input_scale = torch.tensor(
                    [(a_cv - a_cvn) / (2**qlin_int.nbits_a - 1)], device=target_device
                )
                input_zero_point = torch.round(-a_cvn / input_scale).to(torch.int)
                w_scale = torch.tensor([w_cv * 2 / (2**qlin_int.nbits_w - 2)])
            else:
                # fms_mo formula is a bit different from conventional PT formula
                quant_scale = (2**qlin_int.nbits_a - 1) / torch.tensor(
                    [a_cv - a_cvn], device=target_device
                )
                quant_stepsize = 1.0 / quant_scale
                quant_zero_point = torch.round(a_cvn * quant_scale)
                input_scale = quant_stepsize
                input_zero_point = -quant_zero_point
                quant_w_scale = (2**qlin_int.nbits_a - 2) / torch.tensor(
                    [w_cv * 2], device=target_device
                )
                w_scale = 1.0 / quant_w_scale
                qlin_int.register_buffer("quant_scale", quant_scale)
                qlin_int.register_buffer("quant_stepsize", quant_stepsize)
                qlin_int.register_buffer("quant_zero_point", quant_zero_point)
            w_zp = torch.zeros_like(w_scale, dtype=torch.int)

            qlin_int.register_buffer("input_scale", input_scale)
            qlin_int.register_buffer("input_zp", input_zero_point)
            qlin_int.register_buffer("w_scale", w_scale)
            qlin_int.register_buffer("w_zp", w_zp)
            # Store original cv_a and cv_w (in python floats, not tensors), and sq scales
            # for later verification
            qlin_int.cvs = [Qa.clip_val.item(), Qa.clip_valn.item(), Qw.clip_val.item()]

            corr_term = (
                (input_zero_point - 128)
                * (w_int8.sum(dim=1))
                * w_scale.float()
                * input_scale.float()
            )
            # dim=1 because w_int is in [out,in], after sum shape=[out,], same as w_scale and bias.
            # (zp-128)*w_int8.sum(dim=1) can be >> fp16.max, use fp32 scales
            # to make sure dtype is large enough
            qlin_int.register_buffer("corr_term", corr_term.half())  # [DEBUG only]
            if fms_mo_qlinear.bias is not None:
                qlin_int.bias = nn.Parameter(
                    (fms_mo_qlinear.bias - corr_term).to(fms_mo_w_dtype),
                    requires_grad=False,
                )
                qlin_int.org_model_has_bias = True
            else:
                qlin_int.register_buffer("bias", -corr_term.to(fms_mo_w_dtype))
                qlin_int.org_model_has_bias = False

        qlin_int.register_buffer("Qa_clip_val", Qa.clip_val.detach())
        qlin_int.register_buffer(
            "Qa_clip_valn", Qa.clip_valn.detach()
        )  # TODO: case for PACT?
        qlin_int.register_buffer(
            "Qw_clip_val", Qw.clip_val.detach()
        )  # asym W quantizer may have clipvaln

        qlin_int.set_matmul_op()

        return qlin_int.to(target_device)

    @classmethod
    def from_torch_iW(cls, nnlin_iW, prec, a_cv, a_cvn, w_cv, zero_shift, **kwargs):
        """Converts a torch.nn.Linear module to a QLinearINT8Deploy.

        Args:
            cls (class): The class of the QLinearINT8Deploy to be created.
            nnlin_iW (torch.nn.Linear): The original torch.nn.Linear module.
            prec (str): The precision of the quantized weights, must be "int8".
            a_cv (float): The activation CV of the input tensor.
            a_cvn (float): The activation CV of the input tensor's negative part.
            w_cv (float): The weight CV of the weights tensor.
            zero_shift (float or str): The zero shift value. If a string,
                    it should be a JSON-formatted list of floats.
            **kwargs: Additional keyword arguments.

        Returns:
            QLinearINT8Deploy: The converted QLinearINT8Deploy.
        """
        assert prec == "int8", "Only support INT8 for now."

        target_device = kwargs.get(
            "target_device", kwargs.get("device", next(nnlin_iW.parameters()).device)
        )

        qlinear_iW = cls(
            nnlin_iW.in_features,
            nnlin_iW.out_features,
            bias=nnlin_iW.bias is not None,
            device=target_device,
        )

        qlinear_iW.nbits_a = 8  # Only support INT8 for now
        qlinear_iW.nbits_w = 8
        qlinear_iW.acc_dtype = torch.float16
        qlinear_iW.usePTnativeQfunc = kwargs.get("use_PT_native_Qfunc", True)
        qlinear_iW.useINTkernel = True
        qlinear_iW.weight = nn.Parameter(
            nnlin_iW.weight.to(torch.int8), requires_grad=False
        )
        qlinear_iW.acc24minmax = (int(-(2**24) / 2 + 1), int(2**24 / 2))
        qlinear_iW.simi24toi16 = kwargs.get("simi24toi16", False)

        with torch.no_grad():
            if qlinear_iW.usePTnativeQfunc:
                input_scale = torch.Tensor(
                    [(a_cv - a_cvn) / (2**qlinear_iW.nbits_a - 1)]
                )
                input_zero_point = torch.round(-a_cvn / input_scale).to(torch.int)
                w_scale = torch.Tensor([w_cv * 2 / (2**qlinear_iW.nbits_w - 2)])
            else:
                # fms_mo formula is a bit different from conventional PT formula
                quant_scale = (2**qlinear_iW.nbits_a - 1) / torch.Tensor([a_cv - a_cvn])
                quant_stepsize = 1.0 / quant_scale
                quant_zero_point = torch.round(a_cvn * quant_scale)
                input_scale = quant_stepsize
                input_zero_point = -quant_zero_point
                quant_w_scale = (2**qlinear_iW.nbits_a - 2) / torch.Tensor([w_cv * 2])
                w_scale = 1.0 / quant_w_scale
                qlinear_iW.register_buffer("quant_scale", quant_scale)
                qlinear_iW.register_buffer("quant_stepsize", quant_stepsize)
                qlinear_iW.register_buffer("quant_zero_point", quant_zero_point)
            w_zp = torch.zeros_like(w_scale, dtype=torch.int)

            qlinear_iW.register_buffer("input_scale", input_scale)
            qlinear_iW.register_buffer("input_zp", input_zero_point)
            qlinear_iW.register_buffer("w_scale", w_scale)
            qlinear_iW.register_buffer("w_zp", w_zp)
            # Store original cv_a and cv_w (in python floats, not tensors), and sq scales
            # for later verification
            qlinear_iW.cvs = [a_cv, a_cvn, w_cv]

            if isinstance(zero_shift, str):
                zero_s = torch.Tensor(json.loads(zero_shift))
            else:  # Symmetrical case has no zero_shift
                zero_s = torch.Tensor([zero_shift])
            corr_term = (input_zero_point - 128) * zero_s * w_scale * input_scale
            # NOTE: This term may be calculated in 'double',
            #     need to use >= fp32 here to make sure dtype is large enough (fp16 could overflow)
            qlinear_iW.register_buffer("corr_term", corr_term)  # [DEBUG only]
            qlinear_iW.register_buffer("zero_shift", zero_s)  # [DEBUG only]
            if nnlin_iW.bias is not None:
                qlinear_iW.bias = nn.Parameter(
                    (nnlin_iW.bias - corr_term.to(target_device)).to(
                        qlinear_iW.acc_dtype
                    ),
                    requires_grad=False,
                )
                qlinear_iW.org_mod_has_bias = True
            else:
                qlinear_iW.register_buffer("bias", -corr_term.to(qlinear_iW.acc_dtype))
                qlinear_iW.org_mod_has_bias = False

        qlinear_iW.set_matmul_op()

        return qlinear_iW.to(target_device)

    def qa_pt_qfunc_wrapped(self, x):
        """
        Activation quantizer for deployment

        torch.quantizer_per_tensor() with a wrapper, registered in imatmul_ops_reg(), return int8
            can be traced, look simpler on graph, PT func is faster than raw formula if you do not
            want to use torch.compile()

        Args:
            x (Tensor): Input tensor to be quantized.

        Returns:
            Tensor: Quantized tensor with values in the range [-128, 127].
        """
        return torch.ops.fms_mo.q_per_t_sym(
            x.float(), self.input_scale, self.input_zp - 128
        )

    def qa_pt_quant_func(self, x):
        """
        Quantizes the input tensor x to 8-bit integer values for deployment using
        torch.quantizer_per_tensor() without a wrapper

        Args:
            x (Tensor): Input tensor to be quantized.

        Returns:
            Tensor: Quantized tensor with values in the range [-128, 127].
        """
        return torch.quantize_per_tensor(
            x.float(), self.input_scale, self.input_zp - 128, torch.qint8
        ).int_repr()

    def qa_raw_qfunc(self, x):
        """
        Quantizes the input tensor x to 8-bit integer values using raw formula, slower if not
        torch.compiled

        Args:
            x (Tensor): Input tensor to be quantized.

        Returns:
            Tensor: Quantized tensor with values in the range [-128, 127].
        """
        x = torch.clamp((x / self.input_scale + self.input_zp - 128).round(), -128, 127)
        return x.to(torch.int8)

    def qa_fmo_mo_qfunc(self, x):
        """
        Quantizes the input tensor x to 8-bit integer values.

        Args:
            x (Tensor): Input tensor to be quantized.

        Returns:
            Tensor: Quantized tensor with values in the range [-128, 127].
        """
        x = (
            torch.round(
                x.clamp(self.cvs[1], self.cvs[0]) / self.quant_stepsize
                - self.quant_zero_point
            )
            - 128
        )
        return x.to(torch.int8)

    def iaddmm_int(self, bias, m1, m2):
        """
        Performs integer matrix multiplication with optional addition of a bias term.

        NOTE: if use 2 layers of wrapper, i.e. iaddmm->imatmul->kernel, will be slower
                q_iaddmm_dq calls raw func with only 1 wrapper, might be better
        NOTE: m1=x, m2=W.t(), both are INT, dQ are included in fms_mo.iaddmm

        Args:
            bias: The bias tensor to be added to the result.
            m1: The first input tensor.
            m2: The second input tensor.

        Returns:
            The result of the integer matrix multiplication with the bias added.
        """

        if self.usePTnativeQfunc:
            m1 = self.qa_raw_qfunc(m1)
        else:
            m1 = self.qa_fmo_mo_qfunc(m1)

        if self.simi24toi16:
            chunk_size = 99999
            idx = list(range(0, m1.shape[1], chunk_size))
            Nchunk = len(idx)
            idx.append(m1.shape[1])
            fp16_out = torch.zeros(
                (m1.shape[0], m2.shape[1]), dtype=torch.float16, device=m1.device
            )
            for i in range(Nchunk):
                imm_out = torch.ops.fms_mo.imatmul(
                    m1[:, idx[i] : idx[i + 1]], m2[idx[i] : idx[i + 1], :]
                )
                imm_out = imm_out.clamp(self.acc24minmax[0], self.acc24minmax[1])
                imm_out = torch.bitwise_right_shift(imm_out + 128, 8)
                imm_out = imm_out.to(torch.int16)
                fp16_out += imm_out.to(torch.float16)

            return (
                fp16_out * (256 * self.input_scale * self.w_scale).to(torch.float16)
                + bias
            ).to(self.acc_dtype)
        # The safest casting, i32 -> f32
        imm_out = torch.ops.fms_mo.imatmul(m1, m2)
        return (
            imm_out.float() * (self.input_scale * self.w_scale).to(torch.float16) + bias
        ).to(self.acc_dtype)

    def iaddmm_FP(self, bias, m1, m2):
        """
        Performs a matrix multiplication of matrices `m1` and `m2`
        with addition of `bias`. Matrix dimensions are expected to be
        compatible (see `torch.addmm()`).

        Args:
            bias (Tensor): the additive bias tensor
            m1 (Tensor): the first matrix to be multiplied
            m2 (Tensor): the second matrix to be multiplied

        Returns:
            Tensor: the result of the matrix multiplication with addition of bias
        """
        m2 = m2.to(m1.dtype)
        return torch.addmm(bias, m1, m2)

    def set_matmul_op(self):
        """
        Sets the matmul operator for the quantized linear module.

        If `useINTkernel` is True and CUDA is available, it will use the INT kernel
        for integer matrix multiplication. Otherwise, it will use the FP kernel.

        If the operator has already been set, it will do nothing.
        """
        if self.useINTkernel and not torch.cuda.is_available():
            logger.warning(
                "Cannot set useINTkernel=True when CUDA is not available. "
                "Fallback to useINTkernel=False"
            )
            self.useINTkernel = False

        if hasattr(torch.ops, "fms_mo") and hasattr(torch.ops.fms_mo, "imatmul"):
            # imatmul already registered, e.g. when swapping the 2nd QLinear
            self.imatmul = torch.ops.fms_mo.imatmul
            self.iaddmm = self.iaddmm_int if self.useINTkernel else self.iaddmm_FP
        else:
            # When swapping the first QLinear, need to register our custom Op and choose the kernel
            # First Party
            from fms_mo.custom_ext_kernels.utils import (
                cutlass_ops_load_and_reg, imatmul_ops_reg)

            if self.useINTkernel:  # will use real imatmul
                cutlass_ops_load_and_reg()
                # Third Party
                import cutlass_mm  # this module will only be available after calling reg()

                imm_func = cutlass_mm.run
            else:
                imm_func = torch.matmul

            imatmul_ops_reg(self.useINTkernel, imm_func)
            self.imatmul = torch.ops.fms_mo.imatmul
            self.iaddmm = self.iaddmm_int if self.useINTkernel else self.iaddmm_FP

    def _get_name(self):
        """
        Returns the name of the QLinearINT8Deploy as a string.
        """
        return "QLinear_INT8"

    def extra_repr(self) -> str:
        """
        Returns an alternative string representation of the object
        """
        return (
            f"in={self.in_features}, out={self.out_features}, bias={self.bias is not None}, "
            f"useINTkernel={self.useINTkernel}"
        )

    def __getstate__(self):
        """
        Returns a dictionary representing the object's state.
        This method is used by the pickle module to serialize the object.

        Copy the object's state from self.__dict__ which contains all our instance attributes.
        Always use the dict.copy() to avoid modifying the original state.
        """
        state = self.__dict__.copy()
        del state["imatmul"]  # Remove the unpicklable entries.
        return state

    def __setstate__(self, state):
        """
        Sets the state of the object. Restore instance attributes (i.e., filename and line number).
        Use set_matmul_op to restore the previously unpicklable object and make sure the Op is
        registered already

        Args:
            state (dict): The state dictionary containing the instance attributes.
        """
        self.__dict__.update(state)

        self.set_matmul_op()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the layer.

        Args:
            x (Tensor): Input tensor of shape (batch_size, in_features).

        Returns:
            Tensor: Output tensor of shape (batch_size, out_features).
        """

        with torch.no_grad():
            # Q, imatmul, add bias/corr, dQ, reshape should be all taken care of in the iaddmm
            # simplify to either real iaddmm or iadd_FP, one-liner here but graph will differ
            # NOTE: imatmul should be like matmul, and self.W should stay [out,in]
            #       which will need correct dims, i.e. [m,k]@[k,n], hence W.t()
            org_dtype = x.dtype
            re_shape = (-1, x.shape[-1])
            tar_shape = tuple(x.shape[:-1]) + (
                self.weight.shape[0],
            )  # W.shape=[out,in]

            x = self.iaddmm(self.bias, x.view(re_shape), self.weight.t()).reshape(
                tar_shape
            )

        return x.to(org_dtype)


class QLinearCublasI8I32NT(nn.Linear):
    """
    A new QLinear class for testing external kernels,
    similar to CPU lowering, will absorb scales and zp and etc
    need to store 1) INT W and 2) input scales, zps, 3) bias 4) correction term
    """

    @classmethod
    def from_reference(
        cls, ref_qlinear, input_scale, input_zero_point, input_dtype, forceSymA=False
    ):
        """
        Converts a reference QLinear module into a Cublas counterpart.

        Args:
            cls (class): The class of the Cublas module to create.
            ref_qlinear (QLinear): The reference QLinear module to convert.
            input_scale (torch.Tensor): The input scale tensor.
            input_zero_point (torch.Tensor): The input zero point tensor.
            input_dtype (torch.dtype): The data type of the input tensor.
            forceSymA (bool, optional): Whether to force symmetric quantization for the activation.
                                Defaults to False.

        Returns:
            torch.nn.Module: The converted Cublas module.
        """
        qlinear_cublas = cls(ref_qlinear.in_features, ref_qlinear.out_features)
        qlinear_cublas.forceSymA = forceSymA
        qlinear_cublas.input_dtype = input_dtype
        qlinear_cublas.register_buffer("input_scale", input_scale)
        qlinear_cublas.register_buffer("input_zp", input_zero_point)

        qweight = (
            ref_qlinear.float().get_quantized_weight()
        )  # 1) qint8, 2) W for linear is already transposed
        if ref_qlinear.weight_qscheme in [
            torch.per_channel_symmetric,
            torch.per_channel_affine,
        ]:
            qlinear_cublas.register_buffer(
                "w_scale", qweight.q_per_channel_scales().float()
            )  # dtype was fp64
            # NOTE: w_scale is 1D but should be in feat_out dim, W is [out, in] => need unsqueeze(1)
            #  if mul with W directly
            qlinear_cublas.register_buffer(
                "w_zp", qweight.q_per_channel_zero_points()
            )  # dtype=int64
        else:
            raise RuntimeError("QLinear_cublas only supports perCh for now. ")

        qlinear_cublas.weight = nn.Parameter(
            qweight.int_repr(), requires_grad=False
        )  # dtype will be torch.int8
        qlinear_cublas.bias = nn.Parameter(ref_qlinear.bias, requires_grad=False)

        # cublas only support int8*int8, int8*uint8 is not allowed. we have 2 options
        if input_dtype == torch.quint8:
            if forceSymA:
                # option 1: adjust scale and make it symmetric
                a_min = -input_zero_point * input_scale
                a_max = (255 - input_zero_point) * input_scale
                a_max_new = max(abs(a_min), abs(a_max))
                a_scale_new = (2 * a_max_new) / (255 - 2)
                qlinear_cublas.input_scale.copy_(a_scale_new)
                qlinear_cublas.input_zp.copy_(0)
                qlinear_cublas.input_dtype = torch.qint8
            else:
                # option 2: use a correction term and combine with bias
                corr_term = (
                    qlinear_cublas.w_scale
                    * input_scale
                    * (input_zero_point - 128)
                    * qlinear_cublas.weight.sum(dim=1)
                )
                corr_term = corr_term.to(ref_qlinear.bias.dtype)
                # correction term and bias should be of same shape, can combine them
                qlinear_cublas.bias = nn.Parameter(
                    ref_qlinear.bias - corr_term, requires_grad=False
                )

        return qlinear_cublas

    @classmethod
    def from_fms_mo(cls, fms_mo_qlinear, **kwargs):
        """
        Converts a QLinear module to a Cublas QLinear module.

        Args:
            cls: The class of the Cublas QLinear module to be created.
            fms_mo_qlinear: The QLinear module to be converted.
            kwargs: Additional keyword arguments for the Cublas QLinear module.

        Returns:
            A Cublas QLinear module equivalent from the QLinear module.
        """
        assert fms_mo_qlinear.num_bits_feature in [
            4,
            8,
        ] and fms_mo_qlinear.num_bits_weight in [4, 8], "Please check nbits setting!"

        target_device = kwargs.get(
            "target_device", next(fms_mo_qlinear.parameters()).device
        )
        qlinear_cublas = cls(fms_mo_qlinear.in_features, fms_mo_qlinear.out_features)
        qlinear_cublas.input_dtype = (
            torch.quint8
        )  # assume input to fms_mo QLinear is always asym

        with torch.no_grad():
            Qa = fms_mo_qlinear.quantize_feature
            input_scale = (Qa.clip_val - Qa.clip_valn) / (2**Qa.num_bits - 1)
            input_zero_point = torch.round(-Qa.clip_valn / input_scale).to(torch.int)
            qlinear_cublas.register_buffer("input_scale", input_scale)
            qlinear_cublas.register_buffer("input_zp", input_zero_point)

            Qw = fms_mo_qlinear.quantize_weight
            w_scale = Qw.clip_val * 2 / (2**Qw.num_bits - 2)
            w_zp = torch.zeros_like(w_scale, dtype=torch.int)
            qlinear_cublas.register_buffer("w_scale", w_scale.float())
            qlinear_cublas.register_buffer("w_zp", w_zp)

        # if we use PT native Qfunc, may not work with fp16, hence the use of weight.float()
        Qw.dequantize = False
        qlinear_cublas.weight = nn.Parameter(
            Qw(fms_mo_qlinear.weight.float()).to(torch.int8), requires_grad=False
        )
        qlinear_cublas.bias = nn.Parameter(fms_mo_qlinear.bias, requires_grad=False)

        # cublas only support int8*int8, int8*uint8 is not allowed. we have 2 options
        # option 2: use a correction term and combine with bias
        corr_term = (
            qlinear_cublas.w_scale
            * input_scale
            * (input_zero_point - 128)
            * qlinear_cublas.weight.sum(dim=1)
        )
        corr_term = corr_term.to(fms_mo_qlinear.bias.dtype)
        # correction term and bias should be of same shape, can combine them
        qlinear_cublas.bias = nn.Parameter(
            fms_mo_qlinear.bias - corr_term, requires_grad=False
        )

        return qlinear_cublas.to(target_device)

    def _get_name(self):
        """
        Returns the name of the QLinearCublasI8I32NT as a string.
        """
        return "QuantizedLinear_cublasi8i32"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward function for the QLinear layer.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_size).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_size).
        """
        with torch.no_grad():
            in_dtype = x.dtype
            # step 1: Q activation
            # NOTE: usually we use asym activ, but cublas_gemm_i8i32 only supports sym...
            if self.input_dtype == torch.quint8:
                x = torch.clamp((x / self.input_scale + self.input_zp).round(), 0, 255)
                x = x.to(torch.int16) - 128
                x = x.to(torch.int8)
            else:
                x = torch.clamp(
                    (x / self.input_scale + self.input_zp).round(), -127, 127
                ).to(torch.int8)

            # step 2: gemm
            if len(x.shape) == 3 and len(self.weight.shape) == 2:  # batched input
                re_shape = (-1, x.shape[2])
                tar_shape = (
                    x.shape[0],
                    x.shape[1],
                    self.weight.shape[0],
                )  # W is transposed
                x = x.reshape(re_shape)
            elif len(x.shape) == len(self.weight.shape) == 2:  # 2D
                tar_shape = (x.shape[0], self.weight.shape[0])
            else:
                raise RuntimeError("Input dimension to QLinear is not 2D or batched 2D")

            x = torch.ops.mygemm.gemm_nt_i8i32(
                x, self.weight
            )  # only support torch.int8, (sym)
            x = x.reshape(tar_shape)

            # step 3: dQ and add bias, NOTE: zp_corr is included in the bias already
            x = self.input_scale * self.w_scale * x + self.bias
        return x.to(in_dtype)


class QLinearCutlassI8I32NT(QLinearCublasI8I32NT):
    """
    A QLinear class for running int8 with cutlass
    """

    def _get_name(self):
        """
        Returns the name of the QLinearCutlassI8I32NT as a string.
        """
        return "QuantizedLinear_cutlassi8i32"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward function of the quantized linear layer.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_features).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_features).
        """
        with torch.no_grad():
            in_dtype = x.dtype
            # step 1: Q activation
            # NOTE: usually we use asym activ, but cublas_gemm_i8i32 only supports sym...
            if self.input_dtype == torch.quint8:
                x = torch.clamp((x / self.input_scale + self.input_zp).round(), 0, 255)
                x = x.to(torch.int16) - 128
                x = x.to(torch.int8)
            else:
                x = torch.clamp(
                    (x / self.input_scale + self.input_zp).round(), -127, 127
                ).to(torch.int8)

            # step 2: gemm, input x could be 2D or 3D (batched 2D)
            tar_shape = x.shape[:-1] + (self.weight.shape[0],)  # W is transposed
            x = x.view(-1, x.shape[-1])

            x = torch.ops.cutlass_gemm.i8i32nt(
                x, self.weight.t()
            )  # this func takes [m,k] and [k,n] with NT mem layout
            x = x.reshape(tar_shape)

            # step 3: dQ and add bias, NOTE: zp_corr is included in the bias already
            x = self.input_scale * self.w_scale * x + self.bias
        return x.to(in_dtype)


try:
    # Third Party
    from auto_gptq.nn_modules.qlinear.qlinear_exllama import \
        QuantLinear as QLinearExllamaV1
    from auto_gptq.nn_modules.qlinear.qlinear_exllamav2 import \
        QuantLinear as QLinearExllamaV2
    from auto_gptq.nn_modules.qlinear.qlinear_exllamav2 import \
        ext_gemm_half_q_half
    from exllama_kernels import prepare_buffers, set_tuning_params
    from transformers.pytorch_utils import Conv1D

    class QLinearExv1WI4AF16(QLinearExllamaV1):
        """
        A QLinear class for testing Exllama W4A16 external kernels,
        1) activation is FP16, there will be no Q/dQ node on the graph
        2) need to store INT4 W in a special packed format
        """

        @classmethod
        def from_fms_mo(cls, fms_mo_qlinear, **kwargs):
            """
            Converts a QLinear module to QLinearExv1WI4AF16.

            Args:
                cls: The class of the QLinearModule to be created.
                fms_mo_qlinear: The QLinear module to be converted.
                kwargs: Additional keyword arguments.

            Returns:
                A QLinearExv1WI4AF16 object initialized with the weights and biases from the
                    QLinear module.
            """
            assert (
                fms_mo_qlinear.num_bits_feature == 32
                and fms_mo_qlinear.num_bits_weight == 4
            ), "Please check nbits setting!"

            target_device = kwargs.get("target_device", "cuda:0")
            fms_mo_qlinear.cpu()
            qlinear_ex = cls(
                bits=4,
                group_size=kwargs.get(
                    "group_size", -1
                ),  # default -1 -> in_feat -> perCh
                infeatures=fms_mo_qlinear.in_features,
                outfeatures=fms_mo_qlinear.out_features,
                bias=isinstance(
                    fms_mo_qlinear.bias, torch.Tensor
                ),  # if True, only allocates, later in pack() will assign the values
            )

            # exllama QLinear will converts float() to half() if needed
            Qw = fms_mo_qlinear.quantize_weight
            w_scale = Qw.clip_val * 2 / (2**Qw.num_bits - 2)
            if len(w_scale.shape) == 1:
                # pack() is expecting scale and zero in [out, n_group], as Linear.W.shape is
                # [out, in]
                w_scale = w_scale.unsqueeze(1)
            w_zp = (
                torch.ones_like(w_scale) * 8
            )  # This kernel needs to pack in uint, use zp to shift [-8, 7] to [0, 15]

            assert (
                len(w_scale) == fms_mo_qlinear.out_features
            ), " Option other than perCh for QLinear_ft has not been implemented yet. "

            qlinear_ex.pack(fms_mo_qlinear, w_scale, w_zp)
            qlinear_ex.eval().to(target_device)
            max_inner_outer_dim = max(
                fms_mo_qlinear.in_features, fms_mo_qlinear.out_features
            )
            max_dq_buffer_size = qlinear_ex.infeatures * qlinear_ex.outfeatures
            max_input_len = 2 * qlinear_ex.infeatures
            buffers = {
                "temp_state": torch.zeros(
                    (max_input_len, max_inner_outer_dim),
                    dtype=torch.float16,
                    device=target_device,
                ),
                "temp_dq": torch.zeros(
                    (1, max_dq_buffer_size), dtype=torch.float16, device=target_device
                ),
            }

            prepare_buffers(target_device, buffers["temp_state"], buffers["temp_dq"])

            # Default from exllama
            matmul_recons_thd = 8
            matmul_fused_remap = False
            matmul_no_half2 = False
            set_tuning_params(matmul_recons_thd, matmul_fused_remap, matmul_no_half2)

            return qlinear_ex

        def extra_repr(self) -> str:
            """
            Returns an alternative string representation of the object
            """
            return (
                f"in={self.infeatures}, out={self.outfeatures}, bias={self.bias is not None}, "
                f"group_size={self.group_size}"
            )

        def forward(self, x):
            """
            Forward pass of the layer. Matrix multiplication, returns x @ q4"

            Args:
                x (Tensor): Input tensor of shape (batch_size, in_features).

            Returns:
                Tensor: Output tensor of shape (batch_size, out_features).
            """
            with torch.no_grad():
                x = torch.ops.autogptq_gemm.exv1_i4f16(x.half(), self.q4, self.width)

            if self.bias is not None:
                x.add_(self.bias)
            return x

        def pack(self, linear, scales, zeros, g_idx=None):
            """
            Minor correction from the original pack function

            Args:
                linear (nn.Linear): The linear layer to be packed.
                scales (torch.Tensor): The scales to be used for quantization.
                zeros (torch.Tensor): The zeros to be used for quantization.
                g_idx (torch.Tensor, optional): The group indices.
                                    Defaults to None.
            """
            W = linear.weight.data.clone()
            if isinstance(linear, nn.Conv2d):
                W = W.flatten(1)
            if isinstance(linear, Conv1D):
                W = W.t()

            self.g_idx = g_idx.clone() if g_idx is not None else self.g_idx

            scales = scales.t().contiguous()
            zeros = zeros.t().contiguous()
            scale_zeros = zeros * scales
            self.scales = scales.clone().half()
            if linear.bias is not None:
                self.bias = linear.bias.clone().half()

            intweight = []
            for idx in range(self.infeatures):
                intweight.append(
                    torch.round(
                        (W[:, idx] + scale_zeros[self.g_idx[idx]])
                        / self.scales[self.g_idx[idx]]
                    ).to(torch.int)[:, None]
                )
            intweight = torch.cat(intweight, dim=1)
            intweight = intweight.t().contiguous().clamp(0, 15)
            intweight = intweight.numpy().astype(np.uint32)

            i = 0
            row = 0
            qweight = np.zeros(
                (intweight.shape[0] // 32 * self.bits, intweight.shape[1]),
                dtype=np.uint32,
            )
            while row < qweight.shape[0]:
                if self.bits in [4]:
                    for j in range(i, i + (32 // self.bits)):
                        qweight[row] |= intweight[j] << (self.bits * (j - i))
                    i += 32 // self.bits
                    row += 1
                else:
                    raise NotImplementedError("Only 4 bits are supported.")

            qweight = qweight.astype(np.int32)
            self.qweight = torch.from_numpy(qweight)

            zeros -= 1
            zeros = zeros.numpy().astype(np.uint32)
            qzeros = np.zeros(
                (zeros.shape[0], zeros.shape[1] // 32 * self.bits), dtype=np.uint32
            )
            i = 0
            col = 0
            while col < qzeros.shape[1]:
                if self.bits in [4]:
                    for j in range(i, i + (32 // self.bits)):
                        qzeros[:, col] |= zeros[:, j] << (self.bits * (j - i))
                    i += 32 // self.bits
                    col += 1
                else:
                    raise NotImplementedError("Only 4 bits are supported.")

            qzeros = qzeros.astype(np.int32)
            self.qzeros = torch.from_numpy(qzeros)

    class QLinearExv2WI4AF16(QLinearExllamaV2):
        """
        A QLinear class for testing Exllama W4A16 external kernels,
        1) activation is FP16, there will be no Q/dQ node on the graph
        2) need to store INT4 W in a special packed format
        """

        @classmethod
        def from_fms_mo(cls, fms_mo_qlinear, **kwargs):
            """
            Converts a QLinear module to QLinearExv2WI4AF16.

            Args:
                cls: The class of the QLinearModule to be created.
                fms_mo_qlinear: The QLinear module to be converted.
                kwargs: Additional keyword arguments.

            Returns:
                A QLinearExv2WI4AF16 object initialized with the weights and biases from the
                    QLinear module.
            """
            assert (
                fms_mo_qlinear.num_bits_feature == 32
                and fms_mo_qlinear.num_bits_weight == 4
            ), "Please check nbits setting!"

            target_device = kwargs.get("target_device", "cuda:0")
            fms_mo_qlinear.cpu()
            qlinear_ex = cls(
                bits=4,
                group_size=kwargs.get(
                    "group_size", -1
                ),  # default -1 -> in_feat -> perCh
                infeatures=fms_mo_qlinear.in_features,
                outfeatures=fms_mo_qlinear.out_features,
                bias=isinstance(
                    fms_mo_qlinear.bias, torch.Tensor
                ),  # if True, only allocates, later in pack() will assign the values
            )

            # exllama QLinear will convert float() to half() if needed
            Qw = fms_mo_qlinear.quantize_weight
            w_scale = Qw.clip_val * 2 / (2**Qw.num_bits - 2)
            if len(w_scale.shape) == 1:
                # pack() expects scale and zero in [out, n_group], as Linear.W.shape is [out, in]
                w_scale = w_scale.unsqueeze(1)
            w_zp = (
                torch.ones_like(w_scale) * 8
            )  # This kernel needs to pack in uint, use zp to shift [-8, 7] to [0, 15]

            # Assert w_scale.shape[1] == fms_mo_qlinear.out_features,' Option other than perCh for
            # QLinear_ft has not been implemented yet. '
            qweight, qzeros, scales = pack_vectorized(
                fms_mo_qlinear, w_scale, w_zp, qlinear_ex.g_idx, device=target_device
            )

            qlinear_ex.qweight = qweight
            qlinear_ex.qzeros = qzeros
            qlinear_ex.scales = scales
            qlinear_ex.bias = (
                fms_mo_qlinear.bias.clone().half()
                if fms_mo_qlinear is not None
                else None
            )
            qlinear_ex.eval().to(target_device)

            if kwargs.get(
                "useInductor", False
            ):  # anything other than False or None will use torch wrapped version
                qlinear_ex.extOp = torch.ops.autogptq_gemm.exv2_i4f16
            else:
                qlinear_ex.extOp = ext_gemm_half_q_half

            return qlinear_ex

        def extra_repr(self) -> str:
            """
            Returns an alternative string representation of the object
            """
            return (
                f"in={self.infeatures}, out={self.outfeatures}, bias={self.bias is not None}, "
                f"group_size={self.group_size}"
            )

        def forward(self, x, force_cuda=False):
            """
            Forward pass of the layer.

            Args:
                x (Tensor): Input tensor of shape (batch_size, in_features).
                force_cuda (bool, optional): Whether to force the tensor to be moved to CUDA.
                                                Defaults to False.

            Returns:
                Tensor: Output tensor of shape (batch_size, out_features).
            """
            with torch.no_grad():
                x = self.extOp(x.half(), self.q_handle, self.outfeatures, force_cuda)

                if self.bias is not None:
                    x.add_(self.bias)
                return x

except ModuleNotFoundError:
    logger.warning(
        "AutoGPTQ is not properly installed. "
        "QLinearExv1WI4AF16 and QLinearExv2WI4AF16 wrappers will not be available."
    )

QLinear_modules = (
    QLinear,
    QLinearFPout,
    QLinearDebug,
    QLinearW4A32Debug,
    QLinearINT8Deploy,
    QLinearCublasI8I32NT,
    QLinearCutlassI8I32NT,
)


def isinstance_qlinear(module):
    """
    Checks if the given module is one of the available quantized linear classes.

    Args:
        module (nn.Module): The module to check.

    Returns:
        bool: True if the module is a quantized linear class, False otherwise.
    """
    return isinstance(module, QLinear_modules)
