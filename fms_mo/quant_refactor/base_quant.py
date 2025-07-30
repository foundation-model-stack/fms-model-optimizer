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
Base Quantizer classes to be inherited.

Implements:
    Qscheme
    Quantizer
"""

# Standard
from dataclasses import dataclass

# Third Party
import torch


@dataclass
class Qscheme:
    """
    To make it similar to torch.qscheme,
    accept torch.per_tensor_affine, torch.per_channel_symmetric, ...
    Convert it into 2 attributes, q_unit and symmetric
    also allows perGrp as q_unit here

    Raises:
        RuntimeError: New PyTorch qscheme found.  Need to update.
        RuntimeError: perCh or perGrp was selected without specifying Nch, Ngrp.
        RuntimeError: qscheme is not allowed, or could be a typo.
    """

    q_unit: str = "perT"
    q_unit_allowed = ["perT", "perCh", "perGrp", "perTok"]  # NOTE 'perch' not allowed
    symmetric: bool = True
    # for PACT kind quantizer only, clipvaln always fixed at 0
    single_sided: bool = False
    # for symmetric quantizers: reduce qlevels to 2**b - 2 ; some special cases don't use this!
    qlevel_lowering: bool = True

    def __init__(
        self,
        unit: str,
        symmetric: bool = True,
        single_sided: bool = False,
        qlevel_lowering: bool = True,
        Nch: int = None,
        Ngrp: int = None,
        NperGrp: int = None,
        axis: int = None,
    ):
        """
        Init Qscheme

        Args:
            unit (str): Type of quantization.
            symmetric (bool, optional): Specify if clip values are symmetric. Defaults to True.
            Nch (int, optional): Number of channels. Defaults to None.
            Ngrp (int, optional): Number of groups. Defaults to None.
            single_sided (bool, optional): Specify if clip values are positive. Defaults to False.
            qlevel_lowering (bool, optional): Specify lowering of quantized levels.
                Defaults to True.
            axis (int, optional): Specify which tensor dimension to quantize indiviually.
                Defaults to 0.

        Raises:
            RuntimeError: New PyTorch qscheme found.  Need to update.
            RuntimeError: perCh or perGrp was selected without specifying Nch, Ngrp.
            RuntimeError: qscheme is not allowed, or could be a typo.
        """
        # Init perCh/perGrp vars to none incase they won't be set
        self.Nch = None
        self.Ngrp = None
        self.NperGrp = None
        self.axis = None
        if isinstance(unit, torch.qscheme):
            if "per_channel" in str(unit):
                self.q_unit = "perCh"
            elif "per_tensor" in str(unit):
                self.q_unit = "perT"
            else:
                raise RuntimeError("New torch.qscheme! Need to update Qscheme!")
            self.symmetric = "symmetric" in str(unit)
            self.single_sided = single_sided
            self.qlevel_lowering = qlevel_lowering

        elif unit in self.q_unit_allowed:
            self.q_unit = unit
            self.symmetric = symmetric
            if unit == "perCh":
                if Nch is not None and issubclass(type(Nch), int):
                    assert Nch > 0, "Provided Nch is negative"
                    self.Nch = Nch
                else:
                    raise RuntimeError("perCh was selected without specifying Nch.")
                if axis is not None and issubclass(type(axis), int):
                    self.axis = axis
                else:
                    raise RuntimeError(
                        "perCh was selected without specifying channel axis dimension."
                    )
            elif unit == "perGrp":
                if issubclass(type(NperGrp), int):
                    assert NperGrp > 0, "Provided NperGrp is negative"
                    self.NperGrp = NperGrp
                elif issubclass(type(Ngrp), int):
                    assert Ngrp > 0, "Provided Ngrp is negative"
                    self.Ngrp = Ngrp
                else:
                    raise RuntimeError(
                        "perGrp was selected without specifying Ngrp or NperGrp."
                    )

            self.single_sided = single_sided
            self.qlevel_lowering = qlevel_lowering
        else:
            raise RuntimeError("qscheme is not allowed, or could be a typo.")

    def __repr__(self):
        """
        String representation of Qscheme

        Returns:
            str: String of class
        """
        q_uint_str = f"qunit={self.q_unit}"
        symmetric_str = f", symmetric={self.symmetric}"
        Nch_str = f", Nch={self.Nch}, axis={self.axis}" if self.Nch is not None else ""
        Ngrp_str = f", NperGrp={self.Ngrp}" if self.Ngrp is not None else ""
        NperGrp_str = f", Ngrp={self.NperGrp}" if self.NperGrp is not None else ""
        single_sided_str = f", single_sided={self.single_sided}"
        qlevel_lowering_str = f", qlevel_lowering={self.qlevel_lowering}"
        return (
            f"{self.__class__.__name__}({q_uint_str}{symmetric_str}{Nch_str}"
            f"{Ngrp_str}{NperGrp_str}{single_sided_str}{qlevel_lowering_str})"
        )


class Quantizer(torch.nn.Module):
    """
    Base class for quantizers
    basic use should be:
        qtensor = quantizer(tensor_fp)

    quantization formula should be:
        qtensor = torch.round( tensor_fp/scale + zero_point )
    ,where
        scale = (clip_upper - clip_lower)/ (2**nbits -1 (or -2))
        zero_point =  torch.round(-clip_lower/scale).to(torch.int)


    1. quantizers need to be compatible with training =>
        make sure output qtensor is of datatype fp32, not qint8 or int8
    2. make sure it works for both activation and weight, e.g. PACT+ and SAWB, minmax...

    Eccential args:
        num_bits:
        dequantize:
        perCh and perGrp: for weight quantizer only

    For more info:
    pytorch.org/blog/quantization-in-practice/#affine-and-symmetric-quantization-schemes
    """

    def __init__(
        self,
        num_bits: torch.Tensor,
        dequantize: bool = True,
        qscheme: Qscheme = torch.per_tensor_symmetric,
        use_PT_native_Qfunc: bool = False,
        # --- the following flags should be deprecated
        inplace: bool = False,
        align_zero: bool = True,
        clipSTE: bool = True,  # whether to clip grad outside the [cvn, cv] ranges
        **kwargs,
    ):
        """
        Init Quantizer Class

        Args:
            num_bits (torch.Tensor): Number of bit for quantization.
            dequantize (bool, optional): Return dequantized or int tensor. Defaults to True.
            qscheme (Qscheme, optional): Quantization scheme.
                Defaults to Qscheme(unit="perT", symmetric=True).
            inplace (bool, optional): _description_. Defaults to False.
            align_zero (bool, optional): _description_. Defaults to True.
            clipSTE (bool, optional): _description_. Defaults to True.
        """
        super().__init__()
        self.num_bits = num_bits
        self.dequantize = dequantize
        self.qscheme = (
            qscheme
            if isinstance(qscheme, Qscheme)
            else Qscheme(qscheme)
            if isinstance(qscheme, torch.qscheme)
            else Qscheme(*qscheme)
            if isinstance(qscheme, tuple)
            else Qscheme(unit="perT", symmetric=True)
        )
        self.use_PT_native_Qfunc = use_PT_native_Qfunc

        # --- derived attr, just for convenience
        self.perCh = self.qscheme.q_unit == "perCh"
        self.perGrp = self.qscheme.Ngrp if self.qscheme.q_unit == "perGrp" else None
        self.inplace = inplace
        self.align_zero = align_zero
        self.clipSTE = clipSTE

        temp_clipvals = (
            torch.ones(self.qscheme.Nch) if self.perCh else torch.Tensor([1.0])
        )
        self.register_parameter("clip_val", torch.nn.Parameter(temp_clipvals.clone()))
        # Keep clip_valn as positive 1.0 to allow simpler multiplication with
        #   negative numbers (clip_valn.data *= clip_valn)
        if self.qscheme.symmetric:
            # non-trainable for symmetric case
            self.register_buffer("clip_valn", 1.0 * temp_clipvals.clone())
        elif self.qscheme.single_sided:
            # non-trainable and always fixed at 0
            self.register_buffer("clip_valn", torch.tensor(0.0))
        else:
            self.register_parameter(
                "clip_valn", torch.nn.Parameter(1.0 * temp_clipvals.clone())
            )

        # make sure it's consistent with other quantizers
        # here we assume scales/zps are derived from clipvals,
        # i.e. no direct training on scales/zp (need to adj for LSQ+)
        self.register_buffer(
            "scales", torch.zeros_like(self.clip_val, requires_grad=False)
        )
        self.register_buffer(
            "zero_point",
            torch.zeros_like(self.clip_val, requires_grad=False, dtype=torch.int),
        )

    def set_quantizer(self):
        """
        Set a quantizer STE.  To be overriden in child class.

        Raises:
            NotImplementedError: Quantizer selection is not implemented for quantizer
        """
        raise NotImplementedError(
            f"Quantizer selection is not implemented for quantizer {self}"
        )

    def forward(self, input_tensor: torch.Tensor):
        """
        General forward() function for quantizer classes.

        NOTE: Only call this for non-learnable quantizers, to update scales/zp,
            If learnable quantizers, e.g. PACT or LSQ, pass clipvals or scales directly
            To STE functions without calling calc_qparams()

        Args:
            input_tensor (torch.Tensor): Tensor to be quantized.

        Raises:
            ValueError: Single-sided qscheme has tensor min < 0.0

        Returns:
            torch.Tensor: Dequantized or Quantized output tensor.
        """
        if self.qscheme.single_sided and input_tensor.min() < 0.0:
            raise ValueError(
                "FMS Model Optimizer: input to single_sided quantizer should be non-negative."
            )
        output = self.quantizer.apply(
            input_tensor,
            self.num_bits,
            self.clip_valn,
            self.clip_val,
            self.dequantize,
            self.qscheme.symmetric,
            self.qscheme.qlevel_lowering,
        )
        return output

    def calc_qparams(self):
        """
        Compute quantization parameters.  To be overridden in child class.

        Raises:
            NotImplementedError: Calculation is not implemented.
        """
        raise NotImplementedError(
            f"scale/zp calculation is not implemented for quantizer {self}"
        )

    def __repr__(self):
        """
        Represent a Quantizer class as a string.

        Returns:
            str: Quantizer string
        """
        perCh_str = f", perCh{self.perCh}" if self.perCh else ""
        perGrp_str = f", perGrp{self.perGrp}" if self.perGrp else ""
        return (
            f"{self.__class__.__name__}(num_bits={self.num_bits}, "
            f"quantizer={self.quantizer}{perCh_str}{perGrp_str})"
        )
