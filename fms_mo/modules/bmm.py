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

"""Quantization Batch Matrix-Multiplication (BMM) modules"""

# Third Party
from torch import nn
import torch

# Local
from fms_mo.quant.quantizers import Qbypass, Qdynamic, get_activation_quantizer
from fms_mo.utils.import_utils import available_packages


class QBmm(nn.Module):
    """
    A wrapper for the  quantization of BMM or MM operation in transformer.
    """

    def __init__(
        self,
        num_bits_m1=32,
        qm1_mode=None,
        num_bits_m2=32,
        qm2_mode=None,
        m1_unidirectional=False,
        m2_unidirectional=False,
        m1_bounded=False,
        m2_bounded=False,
        replaceBmm=True,
        **kwargs,
    ):
        """
        Initializes the quantized bmm module.

        NOTE: priority = kwargs > qcfg > default

        Args:
            num_bits_m1 (int): Number of bits for the first matrix in the BMM operation.
                                Defaults to 32.
            qm1_mode (str): Quantization mode for the first matrix. Defaults to None.
            num_bits_m2 (int): Number of bits for the second matrix in the BMM operation.
                                Defaults to 32.
            qm2_mode (str): Quantization mode for the second matrix. Defaults to None.
            m1_unidirectional (bool): Whether the first matrix is uni-directional or bi-directional.
                                Defaults to False.
            m2_unidirectional (bool): Whether the second matrix is uni- or bi-directional.
                                Defaults to False.
            m1_bounded (bool): Whether the first matrix is bounded. Defaults to False.
            m2_bounded (bool): Whether the second matrix is bounded. Defaults to False.
            replaceBmm (bool): Whether to replace the original BMM operation with the quantized
                                BMM operation. Defaults to True.
        """
        super().__init__()
        qcfg = kwargs.pop("qcfg")

        self.replaceBmm = replaceBmm
        self.num_bits_m1 = num_bits_m1
        self.nbits_kvcache = kwargs.get(
            "nbits_kvcache",
            num_bits_m2
            if qcfg.get("nbits_kvcache", None) is None
            else qcfg.get("nbits_kvcache"),
        )
        self.num_bits_m2 = self.nbits_kvcache
        # Uni-directional or bi-dir, will be determined by graph searching
        self.m1_unidirectional = m1_unidirectional
        self.m2_unidirectional = m2_unidirectional
        # Bounded input can use max in calibration and/or real training
        self.m1_bounded = m1_bounded
        self.m2_bounded = m2_bounded
        self.qm1_mode = qm1_mode
        self.qm2_mode = qm2_mode

        self.m1_clip_init_val = kwargs.get(
            "m1_clip_init_val", qcfg.get("m1_clip_init_val", 1.0)
        )
        self.m1_clip_init_valn = kwargs.get(
            "m1_clip_init_valn", qcfg.get("m1_clip_init_valn", -1.0)
        )
        self.m2_clip_init_val = kwargs.get(
            "m2_clip_init_val", qcfg.get("m2_clip_init_val", 1.0)
        )
        self.m2_clip_init_valn = kwargs.get(
            "m2_clip_init_valn", qcfg.get("m2_clip_init_valn", -1.0)
        )
        self.qa_mode_calib = kwargs.get(
            "qa_mode_calib", qcfg.get("qa_mode_calib", "percentile")
        )

        self.non_neg = kwargs.get("non_neg", qcfg.get("non_neg", False))
        self.align_zero = kwargs.get("align_zero", qcfg.get("align_zero", True))
        self.sym_7bits_bmm = kwargs.get(
            "sym_7bits_bmm", qcfg.get("sym_7bits_bmm", False)
        )
        self.extend_act_range = kwargs.get(
            "extend_act_range", qcfg.get("extend_act_range", False)
        )
        self.fp8_use_subnormal = kwargs.get(
            "fp8_use_subnormal", qcfg.get("fp8_use_subnormal", False)
        )
        self.register_buffer(
            "calib_counter", torch.tensor(qcfg.get("qmodel_calibration_new", 0))
        )  # Has to be buffer in case DP is used.
        self.register_buffer(
            "num_module_called", torch.tensor(0)
        )  # A counter to record how many times this module has been called

        self.qcfg = qcfg
        self.symmetric = False
        if self.sym_7bits_bmm:
            if self.m1_unidirectional:
                self.num_bits_m1 -= 1
            elif self.m2_unidirectional:
                self.num_bits_m2 -= 1
            else:
                self.symmetric = True
                assert "sym" in qm1_mode and "sym" in qm2_mode, (
                    "should use one of the symmetric quantizers (pactsym, maxsym, pactsym+) "
                    "for qbmm modules"
                )

        self.calib_iterator = []  # To simplify update of clipvals in forward()
        self.quantize_m1 = Qbypass()
        self.quantize_calib_m1 = Qbypass()
        if self.num_bits_m1 not in [32, 16]:
            self.quantize_m1 = get_activation_quantizer(
                self.qm1_mode if (not m1_bounded or "fp8" in qm1_mode) else "minmax",
                nbits=self.num_bits_m1,
                clip_val=self.m1_clip_init_val,
                clip_valn=self.m1_clip_init_valn,
                non_neg=self.m1_unidirectional,
                align_zero=self.align_zero,
                extend_act_range=bool(self.extend_act_range),
                use_subnormal=self.fp8_use_subnormal,
            )
            if self.calib_counter > 0:
                self.quantize_calib_m1 = Qdynamic(
                    self.num_bits_m1,
                    qcfg,
                    non_neg=self.m1_unidirectional,
                    quantizer2sync=self.quantize_m1,
                    align_zero=self.align_zero,
                    qmode="max" if m1_bounded else self.qa_mode_calib,
                    symmetric=self.symmetric,
                )

        self.quantize_m2 = Qbypass()
        self.quantize_calib_m2 = Qbypass()
        if self.num_bits_m2 not in [32, 16]:
            self.quantize_m2 = get_activation_quantizer(
                self.qm2_mode if (not m2_bounded or "fp8" in qm2_mode) else "minmax",
                nbits=self.num_bits_m2,
                clip_val=self.m2_clip_init_val,
                clip_valn=self.m2_clip_init_valn,
                non_neg=self.m2_unidirectional,
                align_zero=self.align_zero,
                extend_act_range=bool(self.extend_act_range),
                use_subnormal=self.fp8_use_subnormal,
            )
            if self.calib_counter > 0:
                self.quantize_calib_m2 = Qdynamic(
                    self.num_bits_m2,
                    qcfg,
                    non_neg=self.m2_unidirectional,
                    quantizer2sync=self.quantize_m2,
                    align_zero=self.align_zero,
                    qmode="max" if m2_bounded else self.qa_mode_calib,
                    symmetric=self.symmetric,
                )

    def forward(self, m1, m2):
        """
        Forward pass of the quantized bmm module.

        Args:
            m1 (torch.Tensor): Input tensor m1.
            m2 (torch.Tensor): Input tensor m2.

        Returns:
            torch.Tensor: Output tensor after quantized bmm.
        """
        # pylint: disable = access-member-before-definition
        if self.calib_counter > 0:
            with torch.no_grad():
                qm1 = self.quantize_calib_m1(m1)
                qm2 = self.quantize_calib_m2(m2)

            self.calib_counter -= 1
            if self.calib_counter == 0:
                self.quantize_calib_m1 = self.quantize_calib_m2 = self.calib_counter = (
                    None
                )

        else:
            qm1 = self.quantize_m1(m1)
            qm2 = self.quantize_m2(m2)

        if self.replaceBmm:
            # The Op we want to replace is a bmm, so we call torch.matmul() (or inf loop will occur)
            output = torch.matmul(qm1, qm2)
        else:
            # The Op we want to replace is a matmul, so we call torch.bmm() instead
            # BMM only take 3d tensors, reshape 4d if needed
            org_batch_header = qm1.shape[:2]
            if len(qm1.shape) > 3:
                qm1 = qm1.reshape([-1, qm1.shape[-2], qm1.shape[-1]])
            if len(qm2.shape) > 3:
                qm2 = qm2.reshape([-1, qm2.shape[-2], qm2.shape[-1]])
            output = torch.bmm(qm1, qm2)
            output = output.reshape([*org_batch_header, *output.shape[1:]])

        self.num_module_called += 1

        return output

    def __repr__(self):
        """
        Returns a string representation of the quantized bmm layer.
        """
        str_quantizer = ",QntzerW,A="
        str_quantizer += (
            ""
            if self.num_bits_m1 == 32
            else f"{self.quantize_m1.__repr__().split('(')[0]},"
        )
        str_quantizer += (
            ""
            if self.num_bits_m2 == 32
            else f"{self.quantize_m2.__repr__().split('(')[0]}"
        )
        return (
            f"{self.__class__.__name__}(Nbits_m1,m2={self.num_bits_m1}, "
            f"{self.num_bits_m2}{str_quantizer})"
        )


if available_packages["mx"]:
    # pylint: disable = import-error
    # Third Party
    from mx.elemwise_ops import quantize_elemwise_op
    from mx.matmul_precision import set_matmul_precision
    from mx.mx_ops import quantize_mx_op
    from mx.specs import apply_mx_specs, get_backwards_mx_specs, mx_assert_test
    import numpy as np

    class MatMulFunction(torch.autograd.Function):
        """Modified from mx.matmul.MatMulFunc. Try to control m1/m2 in a compatible way with fms-mo.
        Matches functionality of torch.matmul. Attempts to broadcast
        outmost dims if in1 and in2 have the same number of dims.
            in1: (..., out_rows, features)
            in2: (..., features, out_cols)
            out: (..., out_rows, out_cols)
        Otherwise, it expects the following shapes:
            in1: (..., out_rows, features)
            in2: (features, out_cols)
            out: (..., out_rows, out_cols)
        """

        @staticmethod
        def forward(ctx, in1, in2, bias, mx_specs, qm1_mode, qm2_mode, replaceBmm):
            ctx.qm1_mode = qm1_mode
            ctx.qm2_mode = qm2_mode
            ctx.replaceBmm = replaceBmm

            qin1_elem_format = qm1_mode
            qin2_elem_format = qm2_mode

            bf_in1 = quantize_elemwise_op(
                in1, mx_specs=mx_specs, round=mx_specs["round_output"]
            )
            bf_in2 = quantize_elemwise_op(
                in2, mx_specs=mx_specs, round=mx_specs["round_output"]
            )

            if bias is not None:
                bf_bias = quantize_elemwise_op(
                    bias, mx_specs=mx_specs, round=mx_specs["round_weight"]
                )

                ctx.bias_shape = list(bias.shape)
            else:
                bf_bias = None
                ctx.bias_shape = None

            if mx_specs["quantize_backprop"]:
                ctx.save_for_backward(bf_in1, bf_in2)
            else:
                ctx.save_for_backward(in1, in2)

            # quantize along the dot product dimension
            qin1 = quantize_mx_op(
                bf_in1,
                mx_specs,
                elem_format=qin1_elem_format,
                axes=[-1],
                round=mx_specs["round_mx_output"],
            )
            qin2 = quantize_mx_op(
                bf_in2,
                mx_specs,
                elem_format=qin2_elem_format,
                axes=[-2],
                round=mx_specs["round_mx_output"],
            )

            with set_matmul_precision(qin1, qin2, qin1_elem_format, qin2_elem_format):
                # If the matmul Op to be replaced is torch.bmm => call torch.matmul(). Or if the Op
                # to be replaced is torch.matmul => call torch.bmm instead. (or inf loop will occur)
                if replaceBmm:
                    out = torch.matmul(qin1, qin2)
                else:
                    # BMM only take 3d tensors, reshape 4d if needed
                    org_batch_header = qin1.shape[:2]
                    m1_reshape_to = [-1, qin1.shape[-2], qin1.shape[-1]]
                    m2_reshape_to = [-1, qin2.shape[-2], qin2.shape[-1]]
                    if len(qin1.shape) > 3:
                        qin1 = qin1.reshape(m1_reshape_to)
                    if len(qin2.shape) > 3:
                        qin2 = qin2.reshape(m2_reshape_to)
                    out = torch.bmm(qin1, qin2)
                    out_reshape_to = [*org_batch_header, *out.shape[1:]]
                    out = out.reshape(out_reshape_to)
                    ctx.reshape_3d = [m1_reshape_to, m2_reshape_to]

            out = quantize_elemwise_op(
                out, mx_specs=mx_specs, round=mx_specs["round_output"]
            )

            if bias is not None:
                out = out + bf_bias
                out = quantize_elemwise_op(
                    out, mx_specs=mx_specs, round=mx_specs["round_output"]
                )

            ctx.mx_specs = get_backwards_mx_specs(mx_specs)
            return out

        @staticmethod
        def backward(ctx, grad_out):
            """
            For a matmul in "wa" mode, the fwd and bwd matmuls configs are:
                FWD wt x act: w x a
                BWD wt x grad: w x a
                BWD act x grad: a x a <-- no mixed precision!
            """
            qin1_elem_format = ctx.qm1_mode
            qin2_elem_format = ctx.qm2_mode

            in1, in2 = ctx.saved_tensors

            grad_out = quantize_elemwise_op(
                grad_out,
                mx_specs=ctx.mx_specs,
                round=ctx.mx_specs["round_grad_input"],
            )

            #####################################################
            # perform madtile operation for grad_in1, grad_in2
            #####################################################
            qin1 = quantize_mx_op(
                in1,
                ctx.mx_specs,
                elem_format=qin1_elem_format,
                axes=[-2],
                round=ctx.mx_specs["round_mx_input_grad_input"],
            )
            qin2 = quantize_mx_op(
                in2,
                ctx.mx_specs,
                elem_format=qin2_elem_format,
                axes=[-1],
                round=ctx.mx_specs["round_mx_input_grad_input"],
            )

            # quantize along out_cols
            qgrad_out1 = quantize_mx_op(
                grad_out,
                ctx.mx_specs,
                elem_format=ctx.mx_specs["a_elem_format_bp_os"],
                axes=[-1],
                round=ctx.mx_specs["round_mx_grad_output_grad_input"],
            )
            # quantize along out_rows
            qgrad_out2 = quantize_mx_op(
                grad_out,
                ctx.mx_specs,
                elem_format=ctx.mx_specs["a_elem_format_bp_os"],
                axes=[-2],
                round=ctx.mx_specs["round_mx_grad_output_grad_input"],
            )

            m1_org_shape = qin1.shape
            m2_org_shape = qin2.shape
            out_reshape_to = [-1, grad_out.shape[-2], grad_out.shape[-1]]
            if ctx.replaceBmm:
                # simple case, no reshape needed
                torch_matmul = torch.matmul
            else:
                torch_matmul = torch.bmm
                m1_reshape_to, m2_reshape_to = ctx.reshape_3d
                # reshape every tensors to 3d so that bmm can work on them
                # NOTE input shapes could be 3d 4d or even 2d, focus on dim -1 and -2
                # m1 = [..., out_r, feat]
                # m2 = [..., feat, out_c]
                # out and grad_out after bmm should be [..., out_r, out_c]
                # In backward
                # grad_in1 = grad_out@m2.t(-1, -2) = [..., out_r, out_c]@[..., out_c, feat]
                #          = [..., out_r, feat], i.e. shape of m1
                # grad_in2 = m1.t(-1, -2)@grad_out = [..., feat, out_r]@[..., out_r, out_c]
                #          = [..., feat, out_c], i.e. shape of m2
                qin1 = qin1.reshape(m1_reshape_to)
                qin2 = qin2.reshape(m2_reshape_to)
                qgrad_out1 = qgrad_out1.reshape(out_reshape_to)
                qgrad_out2 = qgrad_out1.reshape(out_reshape_to)

            # compute grad_in1 and grad_in2
            with set_matmul_precision(
                qgrad_out1, qin2, ctx.mx_specs["a_elem_format_bp_os"], qin2_elem_format
            ):
                grad_in1 = torch_matmul(qgrad_out1, qin2.transpose(-1, -2))
                grad_in1 = grad_in1.to(m1_org_shape)

            with set_matmul_precision(
                qin1, qgrad_out2, qin1_elem_format, ctx.mx_specs["a_elem_format_bp_os"]
            ):
                grad_in2 = torch_matmul(qin1.transpose(-1, -2), qgrad_out2)
                grad_in2 = grad_in2.to(m2_org_shape)

            # element-wise quantize for grad_in1 and grad_in2
            grad_in1 = quantize_elemwise_op(
                grad_in1,
                mx_specs=ctx.mx_specs,
                round=ctx.mx_specs["round_grad_input"],
            )
            grad_in2 = quantize_elemwise_op(
                grad_in2,
                mx_specs=ctx.mx_specs,
                round=ctx.mx_specs["round_grad_input"],
            )

            #####################################################
            # Compute grad_bias
            #####################################################
            if ctx.bias_shape is None:
                grad_bias = None
            else:
                inner_size = grad_out.shape[-1]
                assert np.prod(ctx.bias_shape) == inner_size
                grad_bias = grad_out.reshape(-1, inner_size).sum(0)
                grad_bias = grad_bias.reshape(ctx.bias_shape)

                grad_bias = quantize_elemwise_op(
                    grad_bias,
                    mx_specs=ctx.mx_specs,
                    round=ctx.mx_specs["round_grad_weight"],
                )

            return (grad_in1, grad_in2, grad_bias, None, None, None, None)

    class QBmmMX(QBmm):
        """Wrapper for MX. Modified based on mx.matmul."""

        def __init__(
            self,
            **kwargs,
        ):
            # NOTE: 1. qm1_mode, qm2_mode will be saved by super().init()
            #       2. "mx_" prefix in qmX_mode should stil be in place in kwargs. will remove the
            #           prefix when stored in class properties
            mx_specs = kwargs.pop("mx_specs")
            mx_assert_test(mx_specs)
            self.mx_none = mx_specs is None

            self.mx_specs = apply_mx_specs(mx_specs)
            # always init super as 32b, to avoid QBmm trying to get_w_quantizer of non-exist "mx_xx"
            # quantizers. Actual bits is still controlled by MX (based on qm1_ and qm2_mode)
            kwargs["num_bits_m1"] = 32
            kwargs["num_bits_m2"] = 32
            kwargs["nbits_kvcache"] = 32
            super().__init__(**kwargs)

            self.qm1_mode = self.qm1_mode.replace("mx_", "")
            self.qm2_mode = self.qm2_mode.replace("mx_", "")

        def apply_mx_specs(self, mx_specs):
            """Borrow from mx without changes."""
            mx_assert_test(mx_specs)
            self.mx_none = mx_specs is None
            self.mx_specs = apply_mx_specs(mx_specs)

        def forward(self, m1, m2, bias=None):
            """Fwd/Bwd pass of the quantized bmm module using mx. Backward should work, too.
            Can be used for addmm() is bias is provided.
            """
            mx_specs = self.mx_specs
            mx_assert_test(mx_specs)
            mx_specs = apply_mx_specs(mx_specs)

            return MatMulFunction.apply(
                m1, m2, bias, mx_specs, self.qm1_mode, self.qm2_mode, self.replaceBmm
            )

        def __repr__(self) -> str:
            repr_str = (
                f"{self.__class__.__name__}(m1={self.qm1_mode},m2={self.qm2_mode},"
                f"blk_size={self.mx_specs['block_size']})"
            )
            return repr_str


# ------------------------------------------------------------------------------
# ----- The following wrappers are for torch FX CPU lowering only (FBGEMM) -----
# ----- NOTE: do not use them directly in QAT, backward is not defined     -----
# ------------------------------------------------------------------------------


class QMatmulDebug(nn.Module):
    """
    To lower torch.matmul (QBmm during fms_mo training)
    find patter:    m1 -> Q1 -> dQ -> torch.matmul     (NOTE, node.op is 'call_function')
                    m2 -> Q2 -> dQ  /
    swap to:        m1 -> QMatmulDebug                (this is a 'call_module' node)
                    m2  /
    using similar backend func as used by QLinearFPout
    since linear = x@(W.T)+b -> we need m1@m2 only, so we will use W=m2.T
    let m1 = "activation" (asym), m2.T = "weight", set bias=None to avoid unnecessary compute
    assuming m1.shape = [m, n], m2.shape = [n, k]
    """

    def __init__(
        self,
        scale_m1,
        zp_m1,
        dtype_m1=torch.quint8,
        scale_m2=1.0,
        zp_m2=0,
        dtype_m2=torch.qint8,
    ):
        super().__init__()
        self.register_buffer("scale_m1", scale_m1)
        self.register_buffer("scale_m2", scale_m2)
        self.register_buffer("zp_m1", zp_m1)
        self.register_buffer("zp_m2", zp_m2)
        self.dtype_m1 = dtype_m1
        self.dtype_m2 = dtype_m2
        self.int_range_m1 = (
            (-128, 127) if dtype_m1 == torch.qint8 else (0, 255)
        )  # Assume either qint8 or quint8, nothing else
        self.int_range_m2 = (
            (-128, 127) if dtype_m2 == torch.qint8 else (0, 255)
        )  # Assume either qint8 or quint8, nothing else
        # because m2 is 'weight' has to be qint8 with zp = 0

    def _get_name(self):
        """
        Returns the name of the QMatmulDebug as a string.
        """
        return "QMatmulDebug"

    def extra_repr(self):
        """
        Returns an alternative string representation of the object
        """
        return (
            f"scale1={self.scale_m1:.3f}, zp1={self.zp_m1}, scale2={self.scale_m2:.3f}, "
            f"zp2={self.zp_m2}"
        )

    def forward(self, m1: torch.Tensor, m2: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the quantized matrix multiplication debug module.

        Args:
            m1 (torch.Tensor): Input tensor m1.
            m2 (torch.Tensor): Input tensor m2.

        Returns:
            torch.Tensor: Output tensor after quantized matrix multiplication.
        """
        in_dtype = m1.dtype
        with torch.no_grad():
            m1 = torch.clamp(
                (m1 / self.scale_m1).round() + self.zp_m1, *self.int_range_m1
            )  # map to int
            m1 = (m1 - self.zp_m1) * self.scale_m1  # deQ
            m2 = torch.clamp(
                (m2 / self.scale_m2).round() + self.zp_m2, *self.int_range_m2
            )
            m2 = (m2 - self.zp_m2) * self.scale_m2  # deQ
            # NOTE: Our batched int gemm impl still has bug... use fp16 for debug for now
            out = torch.matmul(m1.half(), m2.half())
            return out.to(in_dtype)


class QBmmINT8Deploy(nn.Module):
    """
    A fake QBmm class for lowering test, convert m1 and m2 to INT then call torch.matmul then dQ
    NOTE: 1. in QBmm, we treat m1 as activations (always asym), m2 as weights (always sym)
          2. Haven't implemented any batched CUTLASS matmul, hence use torch.matmul to simulate
          3. a special case for m1=softmax_out, m1_unidirectional will be True and nbits_m1=7
            => meaning we still want to use INT8 but use (0, 127) to represent FP32 (0, 1)
          3. make sure the forward() is doing   Q->Linear->dQ    on the graph.
                                                (as opposed to Q->dQ->Linear)
    """

    @classmethod
    def from_fms_mo(cls, fms_mo_qbmm, **kwargs):
        """
        Converts a quantized BMM model.

        Args:
            cls (class): The class of the quantized model.
            fms_mo_qbmm (Module): QBMM module.
            kwargs (dict): Additional keyword arguments.

        Returns:
            Module: The quantized model with fake-quantization.
        """
        assert (
            fms_mo_qbmm.num_bits_m1 in [7, 8] and fms_mo_qbmm.num_bits_m2 == 8
        ), "Only support 8bit QBmm!"

        target_device = kwargs.get(
            "target_device", kwargs.get("device", next(fms_mo_qbmm.parameters()).device)
        )
        qbmm_int = cls().to(target_device)
        qbmm_int.num_bits_m1 = fms_mo_qbmm.num_bits_m1
        qbmm_int.num_bits_m2 = fms_mo_qbmm.num_bits_m2
        qcfg = getattr(fms_mo_qbmm, "qcfg", None)
        qbmm_int.use_int_kernel = False  # always False until int kernel is implemented
        qbmm_int.use_PT_native_Qfunc = qcfg["use_PT_native_Qfunc"] if qcfg else False

        with torch.no_grad():
            Qa = fms_mo_qbmm.quantize_m1
            Qw = fms_mo_qbmm.quantize_m2
            # may be trained as sym (-127,127) but can use extended range (-128,127)
            cvn_ext_ratio = 128 / 127
            if qbmm_int.num_bits_m1 == 7 or "sym" in fms_mo_qbmm.qm1_mode:
                Qa_cvn = -Qa.clip_val
            else:
                Qa_cvn = getattr(Qa, "clip_valn", torch.zeros_like(Qa.clip_val))
            m1_scale = (Qa.clip_val - Qa_cvn) / (2**Qa.num_bits - 1)
            m2_scale = Qw.clip_val * 2 / (2**Qw.num_bits - 2)
            qbmm_int.register_buffer("m1_scale", m1_scale)
            qbmm_int.register_buffer(
                "m1_zp", torch.round(-Qa_cvn / m1_scale).to(torch.int)
            )
            qbmm_int.register_buffer("m2_scale", m2_scale)
            qbmm_int.register_buffer(
                "m2_zp", torch.zeros_like(m2_scale, dtype=torch.int)
            )

            qbmm_int.register_buffer("m1_clip_val", Qa.clip_val.detach())
            qbmm_int.register_buffer("m1_clip_valn", Qa_cvn.detach() * cvn_ext_ratio)
            qbmm_int.register_buffer("m2_clip_val", Qw.clip_val.detach())
            # Always symmetrical
            qbmm_int.register_buffer(
                "m2_clip_valn", -Qw.clip_val.detach() * cvn_ext_ratio
            )

        return qbmm_int.to(target_device)

    def qfunc_pt(self, x, scale, zp):
        """
        Quantizes a tensor using per-tensor quantization with a given scale and zero point,
                    without a wrapper

        Args:
            x (Tensor): The tensor to be quantized.
            scale (float): The scale factor for quantization.
            zp (int): The zero point for quantization.

        Returns:
            Tensor: The quantized tensor.
        """
        return torch.quantize_per_tensor(x.float(), scale, zp, torch.qint8).int_repr()

    def qfunc_raw(self, x, scale, zp):
        """
        Quantizes a tensor using the "raw" formula, slower if not torch.compiled

        Args:
            x (Tensor): The tensor to quantize.
            scale (float): The scale factor for quantization.
            zp (int): The zero point offset for quantization.

        Returns:
            Tensor: The quantized tensor.
        """
        return torch.clamp((x / scale + zp).round(), -128, 127).to(torch.int8)

    def _get_name(self):
        """
        Returns the name of the QBmmINT8Deploy as a string.
        """
        return "QBmmINT8Deploy"

    def extra_repr(self) -> str:
        """
        Returns an alternative string representation of the object
        """
        return (
            f"nbits_m1,m2={self.num_bits_m1},{self.num_bits_m2}, "
            f"use_int_kernel={self.use_int_kernel}"
        )

    def forward(self, m1: torch.Tensor, m2: torch.Tensor) -> torch.Tensor:
        """
        Performs matrix multiplication between two tensors m1 and m2,
            taking into account int8 quantization.

        Args:
            m1: A tensor of shape [b, m, k].
            m2: A tensor of shape [b, k, n].

        Returns:
            A tensor of shape [b, m, n] representing the result of the matrix multiplication.
        """

        # Assuming [b,m,k]@[b,k,n]->[b,m,n]
        with torch.no_grad():
            # Assume asym, Qfunc will cast to int8, hence zp - 128
            m1_i8 = self.qfunc_raw(m1, self.m1_scale, self.m1_zp - 128)
            m2_i8 = self.qfunc_raw(m2, self.m2_scale, self.m2_zp)
            corr_term = (
                self.m1_scale.float()
                * (self.m1_zp - 128)
                * m2_i8.sum(dim=0)
                * self.m2_scale.float()
            )
            x = (
                torch.matmul(m1_i8.half(), m2_i8.half()) * self.m1_scale * self.m2_scale
                - corr_term
            )

        return x.to(m1.dtype)


# KEEP THIS AT END OF FILE - classes must be declared
QBmm_modules = (
    QBmm,
    QMatmulDebug,
    QBmmINT8Deploy,
)
if available_packages["mx"]:
    QBmm_modules += (QBmmMX,)


def isinstance_qbmm(module):
    """
    Checks if the given module is one of the available quantized bmm classes.

    Args:
        module (nn.Module): The module to check.

    Returns:
        bool: True if the module is a quantized bmm class, False otherwise.
    """
    return isinstance(module, QBmm_modules)
