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
from torch import nn
import numpy as np
import torch
import torch.nn.functional as F

# Local
from fms_mo.custom_ext_kernels.utils import pack_vectorized
from fms_mo.quant.quantizers import (
    SAWB,
    HardPrune,
    Qbypass,
    Qdynamic,
    get_activation_quantizer,
    get_weight_quantizer,
    mask_fc_kij,
)
from fms_mo.utils.import_utils import available_packages

if available_packages["triton"]:
    # Local
    from fms_mo.custom_ext_kernels.triton_kernels import (
        tl_matmul_chunk_truncate as tl_matmul,
    )

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

        Note:
            scales could be of higher precision than x or W, need to make sure qinput.dtype after
            Qa(x/scale) are consistent with x. Same for W
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
                qa_mode_calib = (
                    self.qa_mode_calib + "sym"
                    if self.qa_mode.endswith("sym")
                    else self.qa_mode_calib
                )
                self.quantize_calib_feature = Qdynamic(
                    self.num_bits_feature,
                    qcfg,
                    non_neg=self.non_neg,
                    align_zero=self.align_zero,
                    qmode=qa_mode_calib,
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
        if self.calib_counter > 0:
            with torch.no_grad():
                qinput = self.quantize_calib_feature(x / scale)
                qweight = self.quantize_calib_weight(self.weight * scale)
            self.calib_counter -= 1
            if self.calib_counter == 0:
                self.quantize_calib_feature = None
                self.quantize_calib_weight = None

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
            qinput = self.quantize_feature(x / scale).to(x.dtype)
            # Default self.update_type == 'hard' pruning.
            if self.mask is not None:
                pweight = HardPrune.apply(
                    self.weight, self.mask.to(self.weight.device), self.p_inplace
                )
                qweight = self.quantize_weight(pweight)
            else:
                qweight = self.quantize_weight(self.weight * scale).to(
                    self.weight.dtype
                )

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
            (experimental)
            use_int_kernel: choose from ['cutlass', 'triton', False], "cutlass" kernel is faster,
                            "triton" supports chunky truncation, "False" fallbacks to torch.matmul
            max_acc_bits: usually INT matmul accumulate in INT32, but some HW could have different
                            design, such as using INT24 accumulator, which will saturate at
                            (-2**(acc_bit-1) +1, 2**(acc_bit-1) )
            truncate_lsb: some HW may apply truncation on least-significant bits (LSBs) of the
                            accumulated partial sum
            chunk_size: some HW may have specific chunk size (BLOCK SIZE, especially in k-dim) for
                        the reason to avoid overflow/underflow problem. This can be simulated using
                        PyTorch (break a matmul into serial smaller matmuls, slow) or Triton kernel
            useDynMaxQfunc: [-1, -2] indicates reduce_dim, 0< val <64 indicates artificial
                        zero-shift, False -> use normal static quantization.

        Returns:
            A QLinearINT8Deploy object initialized with the weights and biases from the
                QLinear module.
        """
        assert all(
            getattr(fms_mo_qlinear, a_or_w) in [4, 8]
            for a_or_w in ["num_bits_feature", "num_bits_weight"]
        ), "Please check nbits setting!"

        tar_dev = kwargs.get(
            "target_device",
            kwargs.get("device", next(fms_mo_qlinear.parameters()).device),
        )
        fms_mo_w_dtype = fms_mo_qlinear.weight.dtype
        qlin_int = cls(
            fms_mo_qlinear.in_features,
            fms_mo_qlinear.out_features,
            bias=fms_mo_qlinear.bias is not None,
            device="meta",  # init on tar_dev is unnecessary
        )
        # Make sure to register an Op for integer matmul, could be real INT matmul or emulation
        qcfg = getattr(fms_mo_qlinear, "qcfg", {})
        qlin_int.use_int_kernel = kwargs.get(
            "use_int_kernel", qcfg.get("use_int_kernel", "cutlass")
        )
        qlin_int.usePTnativeQfunc = kwargs.get("use_PT_native_Qfunc", False)
        qlin_int.useDynMaxQfunc = kwargs.get("use_dynamic_max_act_Qfunc", False)
        qlin_int.useSymAct = (
            "sym" in fms_mo_qlinear.qa_mode
            or fms_mo_qlinear.qa_mode in ["pertokenmax", "max"]
            # these are the symmetric quantizers with no "sym" in their names
        )
        qlin_int.max_acc_bits = kwargs.get("max_acc_bits", 32)
        qlin_int.accminmax = (
            -(1 << (qlin_int.max_acc_bits - 1)),
            (1 << (qlin_int.max_acc_bits - 1)) - 1,
        )
        qlin_int.truncate_lsb = kwargs.get("truncate_lsb", 0)
        qlin_int.chunk_size = kwargs.get("chunk_size", 100000)
        qlin_int.acc_dtype = torch.float16
        qlin_int.nbits_a = fms_mo_qlinear.num_bits_feature  # only support INT8 for now
        qlin_int.nbits_w = fms_mo_qlinear.num_bits_weight
        w_levels = 2**qlin_int.nbits_w - 2
        a_levels = 2**qlin_int.nbits_a - 1 - qlin_int.useSymAct

        with torch.no_grad():
            Qa = fms_mo_qlinear.quantize_feature
            Qw = fms_mo_qlinear.quantize_weight
            # if no calibration has been run before swapping, clipvals stored in Qw will be the
            # ones from ckpt or default. But if want to experiment with new quantizers different
            # from the ckpt, need to run one quantizer.fwd() to update the clipvals.
            # NOTE currently it will recalc by default, but user can choose to turn it off
            if kwargs.get("recalc_clipvals", True):
                Qw(fms_mo_qlinear.weight)
            w_cv = Qw.clip_val
            a_cv = getattr(Qa, "clip_val", torch.tensor(8.0, device=tar_dev))
            a_cvn = getattr(Qa, "clip_valn", torch.tensor(-8.0, device=tar_dev))
            # Store original cv_a and cv_w in python floats (instead of tensors) will be more
            # accurate, but not compatible for per-ch and per-token.
            qlin_int.cvs = [a_cv, a_cvn, w_cv]  # TODO remove the need of this?

            # prepare smoothQuant scale, = (smQ_a_scale ^ alpha)/(smQ_w_scale ^ (1-alpha) )
            smoothq_scale = torch.tensor([1.0], device=tar_dev, dtype=fms_mo_w_dtype)
            if getattr(fms_mo_qlinear, "smoothq", False):
                smoothq_a_scale = fms_mo_qlinear.smoothq_act_scale
                smoothq_w_scale = (
                    fms_mo_qlinear.weight.abs()
                    .max(dim=0, keepdim=True)[0]
                    .clamp(min=1e-5)
                )
                smoothq_alpha = fms_mo_qlinear.smoothq_alpha
                if torch.all(smoothq_a_scale != 0).item():
                    smoothq_scale = (
                        (
                            smoothq_a_scale**smoothq_alpha
                            / smoothq_w_scale ** (1.0 - smoothq_alpha)
                        )
                        .clamp(min=1e-5)
                        .to(smoothq_a_scale.dtype)
                    )

            # could trigger Qw.clipval re-calc for SAWB here, if needed
            input_scale = torch.tensor(1.0, device=tar_dev)
            w_scale = w_cv * 2 / w_levels
            qlin_int.use_fake_zero_shift = False
            if qlin_int.useDynMaxQfunc in [-1, -2]:
                input_zero_point = torch.tensor(
                    128 - qlin_int.useSymAct, device=tar_dev
                )
            elif 0 < qlin_int.useDynMaxQfunc < 65:
                # introduce fake zero-shift, input_scale will be calc dynamically
                qlin_int.use_fake_zero_shift = True
                input_zero_point = torch.tensor(qlin_int.useDynMaxQfunc, device=tar_dev)
            elif qlin_int.usePTnativeQfunc:
                input_scale = torch.tensor([(a_cv - a_cvn) / a_levels], device=tar_dev)
                input_zero_point = torch.round(-a_cvn / input_scale)
            else:
                # fms_mo formula is a bit different from conventional PT formula
                quant_scale = a_levels / torch.tensor([a_cv - a_cvn], device=tar_dev)
                quant_stepsize = 1.0 / quant_scale
                quant_zero_point = torch.round(a_cvn * quant_scale)
                input_scale = quant_stepsize
                input_zero_point = -quant_zero_point
                quant_w_scale = w_levels / (w_cv * 2)
                w_scale = 1.0 / quant_w_scale
                qlin_int.register_buffer("quant_scale", quant_scale)
                qlin_int.register_buffer("quant_stepsize", quant_stepsize)
                qlin_int.register_buffer("quant_zero_point", quant_zero_point)
            w_zp = torch.zeros_like(w_scale, dtype=torch.int)

            input_zero_point = input_zero_point.to(torch.int)  # note 2 in pre-compute
            qlin_int.register_buffer("input_scale", input_scale)
            qlin_int.register_buffer("input_zp", input_zero_point)
            qlin_int.register_buffer("w_scale", w_scale)
            qlin_int.register_buffer("w_zp", w_zp)
            qlin_int.register_buffer("smoothq_scale", smoothq_scale)

            # NOTE:
            # 1. Keep W transposed to prevent confusion, hence (W.t()/scale).t()
            # 2. only a few quantizer have .dequantize working correctly, e.g. SAWB
            # 3. smooth_quant factor is included in the W here, will also include it in the forward
            if isinstance(Qw, SAWB):
                Qw.dequantize = False
                w_int8 = Qw(fms_mo_qlinear.weight.float() * smoothq_scale)
            else:
                w_int8 = (
                    torch.round((fms_mo_qlinear.weight * smoothq_scale).t() / w_scale)
                    .clamp(-w_levels / 2, w_levels / 2)
                    .t()
                )
            w_int8 = w_int8.to(
                torch.int
            )  # stored as int32 as correction term needs sum()
            qlin_int.weight = nn.Parameter(w_int8.to(torch.int8), requires_grad=False)

            # Pre-compute the "correction term" for zero-shift for asym activation quantizers
            # NOTE:
            # 1. sym act should have corr_term=0, unless we want to introduce fake zero-shift
            # 2. sum to reduce dim=1 because w_int is in [out,in], after sum shape=[out,], same as
            #    w_scale (per-Ch) and bias.
            # 3. calc INT part, i.e. (zp-128)*w_int8.sum(dim=1), first in INT32. because it can be
            #    >> fp16.max (~65535 only) easily, make sure not to cast INT32 to FP16 during calc,
            #    simply cast scales to FP32
            # 4. for the "fake zero-shift case", input_scale will be max/(127-fake_zero_shift)
            #    instead of max/127, see qa_dyn_max_fake_zero_shift()
            # 5. Combine correction term into linear.bias for non-dynamic cases. For dyn quant,
            #    input_scale is a placehold for now and will be calc'ed on the fly later.
            if qlin_int.useSymAct:
                corr_term_int = 0
                if qlin_int.use_fake_zero_shift:
                    # one exception, fake zero-shift
                    corr_term_int = input_zero_point * (w_int8.sum(dim=1))
            else:
                corr_term_int = (input_zero_point - 128) * (w_int8.sum(dim=1))

            qlin_int.register_buffer(
                "corr_term", corr_term_int * w_scale.float() * input_scale.float()
            )  # keep in FP32, cast at the end

            qlin_int.org_model_has_bias = fms_mo_qlinear.bias is not None
            # Combine correction term into linear.bias when possible. NOTE the magnitude of these 2
            # terms could vary a lot. use fp32 in case of underflow and lose accuracy.
            if qlin_int.org_model_has_bias:
                new_bias = fms_mo_qlinear.bias.float() - qlin_int.corr_term
            else:
                new_bias = -qlin_int.corr_term

            if qlin_int.use_fake_zero_shift:
                # dyn sym act but with fake zp, remove corr_term from bias
                new_bias += qlin_int.corr_term

            delattr(qlin_int, "bias")
            # sometimes reg_buffer() is unhappy about existing bias
            qlin_int.register_buffer("bias", new_bias.to(fms_mo_w_dtype))

        # redundant variables to be cleaned up
        # qlin_int.register_buffer("Qa_clip_val", Qa.clip_val.detach())
        # qlin_int.register_buffer("Qa_clip_valn", Qa.clip_valn.detach())
        # qlin_int.register_buffer("Qw_clip_val", Qw.clip_val.detach())

        qlin_int.set_matmul_op()

        return qlin_int.to(tar_dev)

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
        qlinear_iW.acc_dtype = kwargs.get("acc_dtype", torch.float)
        qlinear_iW.usePTnativeQfunc = kwargs.get("use_PT_native_Qfunc", True)
        qlinear_iW.use_int_kernel = kwargs.get(
            "use_int_kernel", "triton" if available_packages["triton"] else False
        )
        qlinear_iW.weight = nn.Parameter(
            nnlin_iW.weight.to(torch.int8), requires_grad=False
        )
        qlinear_iW.max_acc_bits = kwargs.get("max_acc_bits", 32)
        qlinear_iW.accminmax = (
            -(1 << (qlinear_iW.max_acc_bits - 1)),
            (1 << (qlinear_iW.max_acc_bits - 1)) - 1,
        )
        qlinear_iW.truncate_lsb = kwargs.get("truncate_lsb", False)
        qlinear_iW.chunk_size = kwargs.get("chunk_size", 100000)

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
            x.float(), self.input_scale, self.input_zp - 128 + self.useSymAct
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
            x.float(),
            self.input_scale,
            self.input_zp - 128 + self.useSymAct,
            torch.qint8,
        ).int_repr()

    def qa_raw_qfunc(self, x):
        """
        Quantizes the input tensor x to 8-bit integer values using raw formula, slower if not
        torch.compiled
        """
        x = torch.clamp(
            (x / self.input_scale + self.input_zp - 128 + self.useSymAct).round(),
            -128,
            127,
        )
        return x.to(torch.int8)

    def qa_fmo_mo_qfunc(self, x):
        """
        Quantizes the input tensor x to 8-bit integer values. Note that old fms-mo formula clamps
        before rounds, as opposed to typical torch formula that rounds before clamps.
        (See qa_raw_qfunc() above.)
        """
        x = torch.round(
            x.clamp(self.cvs[1], self.cvs[0]) / self.quant_stepsize
            - self.quant_zero_point
        ) - (128 - self.useSymAct)
        return x.to(torch.int8)

    def qa_dynamic_max_qfunc(self, x):
        """
        Symmetric dynamic quantizer, same as QDynMax, which allows per-token or per-channel.
        This quantizer will not use self.input_scale but instead will update it every time.
        NOTE
        1. self.input_scale.shape should be (x.shape[-2], ) if reduce_dim == -1 and (, x.shape[-1])
            for reduce_dim == -2.
        2. input_scale should be be broadcasted correctly together with W_scale (e.g. if per-Ch) at
            final output step, i.e. imm_out*(a_scale*w_scale)*...
        """
        amax = x.abs().max(dim=self.useDynMaxQfunc, keepdim=True)[0]
        levels = 2 ** (self.nbits_a - 1) - 1
        self.cvs[0] = amax
        self.cvs[1] = -amax
        self.input_scale = amax.clamp(min=1e-5).div(levels)
        return torch.round(x / self.input_scale).to(torch.int8)

    def qa_dyn_max_fake_zero_shift(self, x):
        """Dynamic max quantizer with fake zero-shift in order to accommodate "zero-centered"
        activations. "partial" correction term has been pre-computed in from_fms_mo() but still need
        to multiply input_scale. (Assuming per-tensor, can shift left or right)
        """
        amax = x.abs().max()
        levels = 2 ** (self.nbits_a - 1) - 1 - self.input_zp
        self.cvs[0] = amax
        self.cvs[1] = -amax
        self.input_scale = amax.clamp(min=1e-5) / levels
        xq = torch.round(x / self.input_scale) + self.input_zp
        return xq.to(torch.int8)

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

        if self.useDynMaxQfunc in [-1, -2]:
            m1 = self.qa_dynamic_max_qfunc(m1)
        elif self.use_fake_zero_shift:
            m1 = self.qa_dyn_max_fake_zero_shift(m1)
        elif self.usePTnativeQfunc:
            m1 = self.qa_raw_qfunc(m1)
        else:
            m1 = self.qa_fmo_mo_qfunc(m1)

        # NOTE simulate chunk behavior in pytorch is serial and slow, use triton when possible
        if m1.shape[1] > self.chunk_size and self.use_int_kernel != "triton":
            idx = list(range(0, m1.shape[1], self.chunk_size))
            Nchunk = len(idx)
            idx.append(m1.shape[1])
            accumulator = torch.zeros(
                (m1.shape[0], m2.shape[1]),
                dtype=torch.int,
                device=m1.device,  # cast float16 if needed
            )
            trun_scale = 1
            if self.truncate_lsb > 0:
                round_bit = 1 << (self.truncate_lsb - 1)
                trun_scale = 1 << self.truncate_lsb

            for i in range(Nchunk):
                imm_out = torch.ops.fms_mo.imatmul(
                    m1[:, idx[i] : idx[i + 1]], m2[idx[i] : idx[i + 1], :]
                )
                if self.max_acc_bits < 32:
                    imm_out = imm_out.clamp(self.accminmax[0], self.accminmax[1])
                if self.truncate_lsb > 0:
                    imm_out = torch.bitwise_right_shift(
                        imm_out + round_bit, self.truncate_lsb
                    )
                    # could cast to smaller data type to further simulate HW behavior, for example,
                    # if HW truncates 8b from both sides of i32 accumulator, the remaining data can
                    # be cast to i16 to be more realistic. pay attention to overflow handling
                accumulator += imm_out  # .to(torch.float16) if needed

            return (
                accumulator
                * (trun_scale * self.input_scale * self.w_scale)  # .to(torch.float16)
                + bias
            ).to(self.acc_dtype)  # safest casting would be i32 -> f32

        imm_out = torch.ops.fms_mo.imatmul(m1, m2)

        updated_bias = bias
        if self.use_fake_zero_shift:
            # Do NOT change the stored self.corr_term and self.bias
            updated_bias = bias - self.input_scale * self.corr_term

        # cast to fp16 could be modified based on real HW behavior/design
        return (
            imm_out.float() * (self.input_scale * self.w_scale).to(torch.float16)
            + updated_bias
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
        if self.useDynMaxQfunc in [-1, -2]:
            m1 = self.qa_dynamic_max_qfunc(m1)
        elif self.usePTnativeQfunc:
            m1 = self.qa_raw_qfunc(m1)
        else:
            m1 = self.qa_fmo_mo_qfunc(m1)

        return torch.matmul(m1 * self.input_scale, m2 * self.w_scale) + bias

    def set_matmul_op(self):
        """
        Sets the matmul operator for the quantized linear module.

        If `use_int_kernel` is True and CUDA is available, it will use the INT kernel
        for integer matrix multiplication. Otherwise, it will use the FP kernel.

        If the operator has already been set, it will do nothing.
        """
        if self.use_int_kernel and not torch.cuda.is_available():
            logger.warning(
                "Cannot set use_int_kernel=True when CUDA is not available. "
                "Fallback to use_int_kernel=False"
            )
            self.use_int_kernel = False

        if hasattr(torch.ops, "fms_mo") and hasattr(torch.ops.fms_mo, "imatmul"):
            # imatmul already registered, e.g. when swapping the 2nd QLinear
            self.imatmul = torch.ops.fms_mo.imatmul
            self.iaddmm = self.iaddmm_int if self.use_int_kernel else self.iaddmm_FP
        else:
            # When swapping the first QLinear, need to register our custom Op and choose the kernel
            # Standard
            from functools import partial

            # Local
            from fms_mo.custom_ext_kernels.utils import (
                cutlass_ops_load_and_reg,
                imatmul_ops_reg,
            )

            if self.use_int_kernel == "triton" and available_packages["triton"]:
                # will use real imatmul written in triton
                imm_func = partial(
                    tl_matmul,
                    chunk_trun_bits=self.truncate_lsb,
                    chunk_size=self.chunk_size,
                )

            elif self.use_int_kernel == "cutlass" and available_packages["cutlass"]:
                # will use real imatmul written in cutlass
                cutlass_ops_load_and_reg()
                # Third Party
                import cutlass_mm  # this module will only be available after calling reg()

                imm_func = cutlass_mm.run
            else:
                imm_func = torch.matmul

            imatmul_ops_reg(self.use_int_kernel, imm_func)
            self.imatmul = torch.ops.fms_mo.imatmul
            self.iaddmm = self.iaddmm_int if self.use_int_kernel else self.iaddmm_FP

    def _get_name(self):
        """
        Returns the name of the QLinearINT8Deploy as a string.
        """
        return "QLinear_INT8"

    def extra_repr(self) -> str:
        """
        Returns an alternative string representation of the object
        """
        repr_str = (
            f"in={self.in_features}, out={self.out_features}, bias={self.bias is not None}, "
            f"int_kernel={self.use_int_kernel}"
        )
        if self.truncate_lsb > 0 or self.max_acc_bits < 32:
            repr_str += f", acc_bits={self.max_acc_bits}, trun_lsb={self.truncate_lsb}"
        return repr_str

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

            if torch.all(self.smoothq_scale != 1).item():
                x = x.view(re_shape) / self.smoothq_scale
            else:
                x = x.view(re_shape)

            x = self.iaddmm(self.bias, x, self.weight.t()).reshape(tar_shape)

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


gptq_available = (
    available_packages["gptqmodel"]
    and available_packages["gptqmodel_exllama_kernels"]
    and available_packages["gptqmodel_exllamav2_kernels"]
)

if gptq_available:
    # Third Party
    from gptqmodel.nn_modules.qlinear.exllama import (
        ExllamaQuantLinear as QLinearExllamaV1,
    )
    from gptqmodel.nn_modules.qlinear.exllamav2 import (
        ExllamaV2QuantLinear as QLinearExllamaV2,
    )
    from gptqmodel.nn_modules.qlinear.exllamav2 import ext_gemm_half_q_half
    from gptqmodel_exllama_kernels import prepare_buffers, set_tuning_params
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
                x = torch.ops.gptqmodel_gemm.exv1_i4f16(x.half(), self.q4, self.width)

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
                qlinear_ex.extOp = torch.ops.gptqmodel_gemm.exv2_i4f16
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


class LinearFuncFPxFwdBwd(torch.autograd.Function):
    """Linear function using FP24 accumulation, experimental only.
    Input and weights can be fp16, bf16, or fp32. W.shape = [out, in].
    W and bias could be of different dtype from input, will cast before calling
    triton kernel. This triton kernel will always use fp32 accumulation, then
    truncate/rounded last 8 or 16 or 20 bits (from LSB side).
    Modified from microxcaling Linear.
    """

    @staticmethod
    def forward(
        ctx,
        x,
        weight,
        bias=None,
        trun_bits=0,
        chunk_size=16,
        fp8_dyn=False,
        clamp_acc_to_dl16=False,
    ):
        assert x.dtype in [torch.float, torch.bfloat16, torch.float16]
        # input can be 2D or 3D, need to reshape before tl_matmul
        org_dtype = x.dtype
        target_shape_output = x.shape[:-1] + (weight.shape[0],)
        x = x.reshape(-1, x.shape[-1])

        if bias is not None:
            ctx.has_bias = True
            ctx.bias_dtype = bias.dtype
        else:
            ctx.has_bias = False

        ctx.save_for_backward(x, weight)  # x, W are saved in their original dtype
        ctx.trun_bits = trun_bits
        ctx.chunk_size = chunk_size
        ctx.fp8_dyn = fp8_dyn
        ctx.clamp_acc_to_dl16 = clamp_acc_to_dl16
        ctx.fp8_e4m3_max = torch.finfo(torch.float8_e4m3fn).max
        ctx.fp8_e5m2_max = torch.finfo(torch.float8_e5m2).max
        ctx.dl8_min = 0.0087890625

        x_scale = torch.tensor(1.0, device=x.device, dtype=org_dtype)
        w_scale = x_scale.clone()
        if fp8_dyn:
            # use Q/dQ simulation for now, meaning still compute in fp16/bf16
            # if choose per_token for input, use per_channel for W
            # (W saved as [out, in], reduce inCh-dim, => reduce_dim=1)
            reduce_dim = None if fp8_dyn == "per_tensor" else 1
            x_scale = (
                x.abs().amax(dim=reduce_dim, keepdim=True) / ctx.fp8_e4m3_max
            ).clamp(min=1e-5)
            w_scale = (
                weight.abs().amax(dim=reduce_dim, keepdim=True) / ctx.fp8_e4m3_max
            ).clamp(min=1e-5)

            x = (x / x_scale).to(torch.float8_e4m3fn).to(torch.float32)
            weight = (weight / w_scale).to(torch.float8_e4m3fn).to(torch.float32)
            if clamp_acc_to_dl16:
                # at this point, x and W are clamped to PT's FP8 range (2^-9 to 448). But since DL8
                # doesn't support subnorm like PyTorch, need to flush subnorms to 0 BEFORE descaling
                x.masked_fill_(x.abs() < ctx.dl8_min, 0)
                weight.masked_fill_(weight.abs() < ctx.dl8_min, 0)

        # triton kernel assumes 2D inputs and cast the return to input.dtype
        output = (
            (
                tl_matmul(
                    x,
                    weight.t(),
                    chunk_trun_bits=trun_bits,
                    chunk_size=chunk_size,
                    clamp_acc_to_dl16=clamp_acc_to_dl16,
                )
                * x_scale
                * w_scale.t()
            )
            .to(org_dtype)
            .reshape(target_shape_output)
        )

        if bias is not None:
            output = output + bias.to(org_dtype)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        # load context, should be bf16 already, x should be 2D already
        x, weight = ctx.saved_tensors  # x, W are saved in original dtype
        trun_bits = ctx.trun_bits
        chunk_size = ctx.chunk_size
        out_dim = weight.shape[0]
        in_dim = weight.shape[1]
        dtype_input = x.dtype
        # input and output could be 3D tl_matmul only takes 2D.
        target_shape_grad_input = grad_output.shape[:-1] + (in_dim,)
        grad_output_2D = grad_output.reshape(-1, out_dim).to(dtype_input)

        x_scale = torch.tensor(1.0, device=x.device, dtype=dtype_input)
        w_scale = x_scale.clone()
        if ctx.fp8_dyn:
            reduce_dim = None if ctx.fp8_dyn == "per_tensor" else 1
            x_scale = x.abs().amax(dim=reduce_dim) / ctx.fp8_e5m2_max
            w_scale = weight.abs().amax(dim=reduce_dim) / ctx.fp8_e5m2_max
            # always assume perT in this case
            grad_out_scale = grad_output_2D.abs().amax(dim=None) / ctx.fp8_e5m2_max

            x = (x / x_scale).to(torch.float8_e5m2).to(torch.float)
            weight = (weight / w_scale).to(torch.float8_e5m2).to(torch.float)
            grad_output_2D = (
                (grad_output_2D / grad_out_scale).to(torch.float8_e5m2).to(torch.float)
            )
            if ctx.clamp_acc_to_dl16:
                # flush subnorm numbers to 0 as DL8 doesn't support it
                x.masked_fill_(x.abs() < ctx.dl8_min, 0)
                weight.masked_fill_(weight.abs() < ctx.dl8_min, 0)
                grad_output_2D.masked_fill_(grad_output_2D.abs() < ctx.dl8_min, 0)

        # Compute grad_weight, shape = [out, in]
        # NOTE: this triton kernel requires A matrix to be contiguous
        grad_weight = (
            tl_matmul(
                grad_output_2D.transpose(0, 1).contiguous(),
                x,
                chunk_trun_bits=trun_bits,
                chunk_size=chunk_size,
                clamp_acc_to_dl16=ctx.clamp_acc_to_dl16,
            )
            * grad_out_scale.t()
            * x_scale
        ).to(weight.dtype)
        # Compute grad_input in 2D then reshape to target shape, could be 3D or 2D
        grad_input = (
            (
                tl_matmul(
                    grad_output_2D,
                    weight,
                    chunk_trun_bits=trun_bits,
                    chunk_size=chunk_size,
                    clamp_acc_to_dl16=ctx.clamp_acc_to_dl16,
                )
                * grad_out_scale
                * w_scale
            )
            .to(dtype_input)
            .reshape(target_shape_grad_input)
        )

        if not ctx.has_bias:
            grad_bias = None
        else:
            grad_bias = grad_output_2D.sum(0).to(ctx.bias_dtype)

        return grad_input, grad_weight, grad_bias, None, None, None, None


class LinearFPxAcc(torch.nn.Linear):
    """Linear layer wrapper that can simulate the HW behavior of LSB truncation on FP accumulation.
    Some HW may have options to allow FP matmul engine to accumulate in precision lower than FP32,
    such as accumulate in TF32 or even BF16. According to Nvidia doc, ~7-10x speed up with minor
    accuracy trade-off. This supports both FWD and BWD.
    Ref:
    1. https://developer.nvidia.com/blog/accelerating-ai-training-with-tf32-tensor-cores/
    2. PyTorch's "torch.backends.cuda.matmul.allow_tf32"
    """

    @classmethod
    def from_nn(cls, nnlin, trun_bits=0, **kwargs):
        """Converts a torch.nn.Linear module to a LinearFPxAcc, which supports accumulation in
        reduced precision FPx, where x < 32.

        Args:
            cls (class): The class to be created.
            nnlin (torch.nn.Linear): The original torch.nn.Linear module.
            trun_bits (int): truncate [0 to 22] LSBs from FP32 accumulation.
            dynamic_fp8: whether to use dynamic quantization for fp8 activations, available options
                        are ["per_tensor", "per_token", False]
            clamp_acc_to_dl16: clamp local accumulator into DL16 range, to simulate the effect of
                                this special dtype
            **kwargs: Additional keyword arguments.

        Returns:
            LinearFPxAcc: The converted linear layer.
        """

        target_device = kwargs.get(
            "target_device", kwargs.get("device", next(nnlin.parameters()).device)
        )

        lin24acc = cls(
            nnlin.in_features,
            nnlin.out_features,
            bias=nnlin.bias is not None,
            device="meta",
        )

        lin24acc.weight = nnlin.weight
        lin24acc.trun_bits = trun_bits
        lin24acc.chunk_size = kwargs.get("chunk_size", False)
        lin24acc.fp8_dyn = kwargs.get("dynamic_fp8", False)
        lin24acc.clamp_acc_to_dl16 = kwargs.get("clamp_acc_to_dl16", False)

        if nnlin.bias is not None:
            lin24acc.bias = nnlin.bias
        return lin24acc.to(target_device)

    def forward(self, inputs):
        # This Linear Class will cast to BF16 before matmul and return FP32
        return LinearFuncFPxFwdBwd.apply(
            inputs,
            self.weight,
            self.bias,
            self.trun_bits,
            self.chunk_size,
            self.fp8_dyn,
            self.clamp_acc_to_dl16,
        )

    def extra_repr(self) -> str:
        """
        Returns an alternative string representation of the object.
        """
        repr_str = f"{self.in_features},{self.out_features}"
        if self.bias is not None:
            repr_str += f",bias={self.bias is not None}"
        if self.trun_bits > 0:
            repr_str += f",trun_bits={self.trun_bits}"
        if self.fp8_dyn:
            repr_str += f",fp8_dyn={self.fp8_dyn}"
        if self.clamp_acc_to_dl16:
            repr_str += ",use_DL16_acc"
        repr_str += f",chunk_size={self.chunk_size}"
        return repr_str


class LinearFuncINT8FwdFP32Bwd(torch.autograd.Function):
    """[Experimental] Linear autograd function using INT matmul/accumulation to simulate HW behavior
    during QAT, in order to adjust weights for specific HW design.
    Args:
        activation x: FP tensor, need to be reshaped to 2D and quantized to INT8.
        weight: FP 2D tensor, W.shape = [out, in].
        bias: bias from original Linear, does not include INT ZP correction term yet.
    NOTE:
    1. main purpose is to utilize triton INT kernel to simulate MSB/LSB truncation in FWD.
    2. BWD simply uses torch.matmul.
    3. *Max per-Ch* for weights and *dynamic max per-Token* for activations.
    """

    @staticmethod
    def forward(
        ctx,
        x,
        weight,
        bias=None,
        lsb_trun_bits=0,
        chunk_size=64,
        max_acc_bits=32,
    ):
        assert x.dtype in [torch.float, torch.bfloat16, torch.float16]
        # input can be 2D or 3D, need to reshape before tl_matmul
        org_dtype = x.dtype
        target_shape_output = x.shape[:-1] + (weight.shape[0],)
        x = x.reshape(-1, x.shape[-1])

        if bias is not None:
            ctx.has_bias = True
            ctx.bias_dtype = bias.dtype
        else:
            ctx.has_bias = False

        ctx.save_for_backward(x, weight)  # x, W are saved in their original dtype

        # max per_token for input -> reduce_dim = -1
        # per_channel for W but W.shape = [out, in] -> reduce_dim = -1
        # sym activation -> correction term = 0
        x_scale = x.abs().amax(dim=-1, keepdim=True) / 127
        w_scale = weight.abs().amax(dim=-1, keepdim=True) / 127

        x_i8 = torch.round(x / x_scale).to(torch.int8)
        w_i8 = torch.round(weight / w_scale).to(torch.int8)

        # triton kernel accepts 2d int8 then return int32
        output = tl_matmul(
            x_i8,
            w_i8.t(),
            chunk_trun_bits=lsb_trun_bits,
            chunk_size=chunk_size,
            max_acc_bits=max_acc_bits,
        )
        output = (
            (output.to(torch.float) * x_scale * w_scale.t())
            .reshape(target_shape_output)
            .to(org_dtype)
        )
        if bias is not None:
            output = output + bias.to(org_dtype)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        # load x and W from context, x is 2D already. no quant.
        # option 1: use compute dtype = x.dtype
        # option 2: compute in fp32 for best results.
        x, weight = ctx.saved_tensors  # x, W are saved in original dtype
        out_dim = weight.shape[0]
        in_dim = weight.shape[1]
        dtype_grad = x.dtype  # torch.float
        # grad_input and grad_output could be 3D as x
        target_shape_grad_input = grad_output.shape[:-1] + (in_dim,)
        grad_output_2D = grad_output.reshape(-1, out_dim).to(dtype_grad)

        # Compute grad_weight, shape = [out, in]
        grad_weight = torch.matmul(
            grad_output_2D.transpose(0, 1).contiguous(),
            x.to(dtype_grad),
        ).to(weight.dtype)
        # Compute grad_input in 2D then reshape to target shape, could be 3D or 2D
        grad_input = (
            torch.matmul(
                grad_output_2D,
                weight.to(dtype_grad),
            )
            .reshape(target_shape_grad_input)
            .to(x.dtype)
        )

        if not ctx.has_bias:
            grad_bias = None
        else:
            grad_bias = grad_output_2D.sum(0).to(ctx.bias_dtype)

        return grad_input, grad_weight, grad_bias, None, None, None


class QLinearINT8Train(torch.nn.Linear):
    """QLinear layer wrapper that simulates INT8 HW behavior, e.g. MSB/LSB truncation, in forward
    and FP32 in backward.
    """

    @classmethod
    def from_fms_mo(cls, nnlin, lsb_trun_bits=0, **kwargs):
        """Converts a torch.nn.Linear or QLinear module to QLinearINT8Train

        Args:
            cls (class): The class to be created.
            nnlin (torch.nn.Linear or QLinear): The Linear module to be converted.
            lsb_trun_bits (int): INT8 LSB truncation, [0 to 16].
            chunk_size (int): usually >= 64 for INT8, based on HW design.
            max_acc_bits: accumulator max bits, <=32, based on HW design.

        Returns:
            LinearFPxAcc: The converted linear layer.
        """

        target_device = kwargs.get(
            "target_device", kwargs.get("device", next(nnlin.parameters()).device)
        )

        lin_int8fwd = cls(
            nnlin.in_features,
            nnlin.out_features,
            bias=nnlin.bias is not None,
            device="meta",  # target_device,
        )

        lin_int8fwd.weight = nnlin.weight
        lin_int8fwd.lsb_trun_bits = lsb_trun_bits
        lin_int8fwd.chunk_size = kwargs.get("chunk_size", 64)
        lin_int8fwd.max_acc_bits = kwargs.get("max_acc_bits", 32)

        if nnlin.bias is not None:
            lin_int8fwd.bias = nnlin.bias
        return lin_int8fwd.to(target_device)

    def forward(self, inputs):
        return LinearFuncINT8FwdFP32Bwd.apply(
            inputs,
            self.weight,
            self.bias,
            self.lsb_trun_bits,
            self.chunk_size,
            self.max_acc_bits,
        )

    def extra_repr(self) -> str:
        """Returns an alternative string representation of the object."""
        repr_str = f"{self.in_features},{self.out_features}"
        repr_str += f",bias={self.bias is not None},chunk_size={self.chunk_size}"
        if self.lsb_trun_bits > 0:
            repr_str += f",lsb_trun={self.lsb_trun_bits}"
        if self.max_acc_bits < 32:
            repr_str += f",max_acc_bits={self.max_acc_bits}"
        return repr_str


if available_packages["mx"]:
    # Third Party
    # pylint: disable = import-error
    from mx.elemwise_ops import quantize_elemwise_op
    from mx.linear import linear as mx_linear
    from mx.specs import apply_mx_specs, mx_assert_test

    # import mx  # defaults to import all classes

    mx_specs_default = {
        "w_elem_format": "fp8_e4m3",
        "a_elem_format": "fp8_e4m3",
        "block_size": 32,
        "bfloat": 16,
        "custom_cuda": True,
        # For quantization-aware finetuning, do backward pass in FP32
        "quantize_backprop": False,
    }

    class QLinearMX(torch.nn.Linear):
        """Modified from mx.linear class. Only mildly changed init() and add extra_repr.
        1. Add **kwargs to receive extra (unused) params passed from qmodel_prep
        2. pass device to super.init, i.e. nn.Linear's
        """

        def __init__(
            self,
            in_features,
            out_features,
            bias=True,
            mx_specs=None,
            name=None,
            **kwargs,
        ):
            mx_assert_test(mx_specs)
            self.mx_none = mx_specs is None

            self.name = name
            self.prequantized_weights = False
            self.mx_specs = apply_mx_specs(mx_specs)
            super().__init__(
                in_features, out_features, bias, device=kwargs.get("device", "cuda")
            )

        def apply_mx_specs(self, mx_specs):
            """Unchanged."""
            mx_assert_test(mx_specs)
            self.mx_none = mx_specs is None
            self.mx_specs = apply_mx_specs(mx_specs)

        def append_name(self, postfix):
            """Unchanged."""
            self.name += postfix

        def prequantize_weights(self):
            """Unchanged."""
            # Can't prequantize if not using bfloat weights
            if self.mx_none:
                return

            assert (
                self.mx_specs["round"] == "even"
            ), "Bfloat round should be 'even' for prequantizing weights."
            assert (
                torch.cuda.is_bf16_supported()
            ), "Current device does not support bfloat16"
            assert self.mx_specs[
                "bfloat_subnorms"
            ], "Bfloat_subnorms should be set to True for prequantizing weights."
            assert (
                self.mx_specs["bfloat"] == 16
            ), "Only Bfloat16 is supported for prequantizing weights."

            with torch.no_grad():
                self.weight.data = quantize_elemwise_op(
                    self.weight.data,
                    mx_specs=self.mx_specs,
                    round=self.mx_specs["round_weight"],
                ).to(torch.bfloat16)

                if self.bias is not None:
                    self.bias.data = quantize_elemwise_op(
                        self.bias.data,
                        mx_specs=self.mx_specs,
                        round=self.mx_specs["round_weight"],
                    ).to(torch.bfloat16)

            self.prequantized_weights = True

        def forward(self, inputs):
            """Unchanged."""
            if self.mx_none:
                return super().forward(inputs)

            if self.prequantized_weights:
                assert (
                    not self.training
                ), "Cannot use prequantized weights when training!"

            return mx_linear(
                input=inputs,
                weight=self.weight,
                bias=self.bias,
                mx_specs=self.mx_specs,
                prequantized_weights=self.prequantized_weights,
                name=self.name,
            )

        def extra_repr(self) -> str:
            repr_str = (
                f"in={self.in_features},out={self.out_features},"
                f"w_fmt={self.mx_specs['w_elem_format']},a_fmt={self.mx_specs['a_elem_format']},"
                f"blk_size={self.mx_specs['block_size']}"
            )
            return repr_str


# KEEP THIS AT END OF FILE - classes must be declared
QLinear_modules = (
    QLinear,
    QLinearFPout,
    QLinearDebug,
    QLinearW4A32Debug,
    QLinearINT8Deploy,
    QLinearCublasI8I32NT,
    QLinearCutlassI8I32NT,
)
if available_packages["mx"]:
    QLinear_modules += (QLinearMX,)

if gptq_available:
    QLinear_modules += (
        QLinearExllamaV1,
        QLinearExllamaV2,
        QLinearExv1WI4AF16,
        QLinearExv2WI4AF16,
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
