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
Fixtures for quantization testing
"""

# Third Party
import pytest
import torch

# Local
# Qscheme class for defining quantizers
from fms_mo.quant_refactor.base_quant import Qscheme
from fms_mo.quant_refactor.lsq_rc import LSQPlus_rc, LSQQuantization_rc
from fms_mo.quant_refactor.pact2_rc import PACT2_rc
from fms_mo.quant_refactor.pact2sym_rc import PACT2Sym_rc

# Refactored quantizers
from fms_mo.quant_refactor.pact_rc import PACT_rc
from fms_mo.quant_refactor.pactplussym_rc import PACTplusSym_rc
from fms_mo.quant_refactor.qmax_rc import Qmax_rc

# Legacy quantizers
from fms_mo.quant_refactor.quantizers_new import (
    PACT,
    PACT2,
    SAWB,
    LSQPlus,
    LSQQuantization,
    PACT2Sym,
    PACTplusSym,
    Qmax,
)
from fms_mo.quant_refactor.sawb_rc import SAWB_rc

# Reference PyTorch Quantization class
from fms_mo.quant_refactor.torch_quantizer import TorchQuantizer

###################
# Tensor Fixtures #
###################

# Tensors sizes for random distributions
tensor_sizes = [10, 100, 1000]
tensor_single_sided_sizes = [(10, 3), (100, 3), (1000, 3)]


@pytest.fixture(scope="session", params=tensor_sizes)
def tensor(request):
    """
    Create a random normal distribution (mean=0, var=1) KxK tensor

    Args:
        request (tuple): Tensor size

    Returns:
        torch.Tensor: Random normal tensor
    """
    torch.manual_seed(23)
    k = request.param
    return torch.randn(k, k, requires_grad=False)


@pytest.fixture(scope="session", params=tensor_single_sided_sizes)
def tensor_single_sided(request):
    """
    Create a random one-sided normal distribution [0,max) KxK tensor

    Args:
        request (tuple): Tensor size

    Returns:
        torch.Tensor: Random normal tensor in [0,max)
    """
    torch.manual_seed(23)
    k, single_sided_max = request.param
    return single_sided_max * torch.rand(k, k, requires_grad=False)


############################
# Quantizer Tuple Fixtures #
############################

# Symmetric
qschemes_symmetric_params = []
for qunit in ["perT"]:
    for symmetric in [True]:
        for Nch in [None]:
            for Ngrp in [None]:
                for single_sided in [False]:
                    # needs to be disabled for some special cases
                    for qlevel_lowering in [True]:
                        qschemes_symmetric_params.append(
                            Qscheme(
                                unit=qunit,
                                symmetric=symmetric,
                                single_sided=single_sided,
                                qlevel_lowering=qlevel_lowering,
                                Nch=Nch,
                                Ngrp=Ngrp,
                                axis=None,
                            )
                        )

quantizer_symmetric_params = []
for num_bits in torch.tensor([8, 4]):
    for clip_high in torch.tensor([2.49, 2.50, 2.51]):
        for scheme in qschemes_symmetric_params:
            quantizer_symmetric_params.append(
                {
                    "num_bits": num_bits,
                    "clip_low": -clip_high,
                    "clip_high": clip_high,
                    "scheme": scheme,
                }
            )


@pytest.fixture(scope="session", params=quantizer_symmetric_params)
def quantizer_symmetric(request):
    """
    Fixture tuple for symmetric quantizer

    Args:
        request (dict): Dict for quantizer args

    Returns:
        dict: Tuple for quantizer args
    """
    return request.param


# Per channel symmetric params
# clip_high, Nch will be computed at test level from tensor
qschemes_symmetric_perCh_params = []
for qunit in ["perCh"]:
    for symmetric in [True]:
        for Ngrp in [False]:
            for single_sided in [False]:
                # needs to be disabled for some special cases
                for qlevel_lowering in [True]:
                    for axis in [0]:
                        qschemes_symmetric_perCh_params.append(
                            Qscheme(
                                unit=qunit,
                                symmetric=symmetric,
                                single_sided=single_sided,
                                qlevel_lowering=qlevel_lowering,
                                Nch=1,  # temp value
                                axis=axis,
                            )
                        )

quantizer_symmetric_perCh_params = []
for num_bits in torch.tensor([8, 4]):
    for scheme in qschemes_symmetric_perCh_params:
        quantizer_symmetric_perCh_params.append(
            {
                "num_bits": num_bits,
                # "clip_low": -clip_high,
                # "clip_high": clip_high,
                "scheme": scheme,
            }
        )


@pytest.fixture(scope="session", params=quantizer_symmetric_perCh_params)
def quantizer_symmetric_perCh(request):
    """
    Fixture tuple for symmetric quantizer w/ per channel clips

    Args:
        request (dict): Dict for quantizer args

    Returns:
        dict: Tuple for quantizer args
    """
    return request.param


# Asymmetric
qschemes_asymmetric_params = []
for qunit in ["perT"]:  # ['perT','perCh','perGrp']:
    for symmetric in [False]:
        for Nch in [None]:
            for Ngrp in [None]:
                for single_sided in [False]:
                    for qlevel_lowering in [False]:
                        qschemes_asymmetric_params.append(
                            Qscheme(
                                unit=qunit,
                                symmetric=symmetric,
                                single_sided=single_sided,
                                qlevel_lowering=qlevel_lowering,
                                Nch=Nch,
                                Ngrp=Ngrp,
                                axis=None,
                            )
                        )

quantizer_asymmetric_params = []
for num_bits in torch.tensor([8, 4]):
    for clip_low in torch.tensor([-3.25, -2.50, -1.75]):
        for clip_high in torch.tensor([1.33, 2.44, 3.55]):
            for scheme in qschemes_asymmetric_params:
                quantizer_asymmetric_params.append(
                    {
                        "num_bits": num_bits,
                        "clip_low": clip_low,
                        "clip_high": clip_high,
                        "scheme": scheme,
                    }
                )


@pytest.fixture(scope="session", params=quantizer_asymmetric_params)
def quantizer_asymmetric(request):
    """
    Fixture tuple for asymmetric quantizer

    Args:
        request (dict): Dict for quantizer args

    Returns:
        dict: Tuple for quantizer args
    """
    return request.param


# Create random clip vals for Per Channel ; must be accompanied by the same tensor
qschemes_asymmetric_perCh_params = []
for qunit in ["perCh"]:
    for symmetric in [False]:
        for Ngrp in [False]:
            for single_sided in [False]:
                # needs to be disabled for some special cases
                for qlevel_lowering in [True]:
                    for axis in [0]:
                        qschemes_asymmetric_perCh_params.append(
                            Qscheme(
                                unit=qunit,
                                symmetric=symmetric,
                                single_sided=single_sided,
                                qlevel_lowering=qlevel_lowering,
                                Nch=1,  # temp value
                                axis=axis,
                            )
                        )

quantizer_asymmetric_perCh_params = []
for num_bits in torch.tensor([8, 4]):
    for scheme in qschemes_asymmetric_perCh_params:
        quantizer_asymmetric_perCh_params.append(
            {
                "num_bits": num_bits,
                # "clip_low": -clip_high,
                # "clip_high": clip_high,
                "scheme": scheme,
            }
        )


@pytest.fixture(scope="session", params=quantizer_asymmetric_perCh_params)
def quantizer_asymmetric_perCh(request):
    """
    Fixture tuple for asymmetric quantizer w/ per channel clips

    Args:
        request (dict): Dict for quantizer args

    Returns:
        dict: Tuple for quantizer args
    """
    return request.param


# Single-Sided
qschemes_single_sided_params = []
for qunit in ["perT"]:  # ['perT','perCh','perGrp']:
    for symmetric in [False]:
        for Nch in [None]:
            for Ngrp in [None]:
                for single_sided in [True]:
                    for qlevel_lowering in [False]:
                        qschemes_single_sided_params.append(
                            Qscheme(
                                unit=qunit,
                                symmetric=symmetric,
                                single_sided=single_sided,
                                qlevel_lowering=qlevel_lowering,
                                Nch=Nch,
                                Ngrp=Ngrp,
                                axis=None,
                            )
                        )

quantizer_single_sided_params = []
for num_bits in torch.tensor([8, 4]):
    for clip_high in torch.tensor([2.10, 2.33, 2.55]):
        for scheme in qschemes_single_sided_params:
            quantizer_single_sided_params.append(
                {
                    "num_bits": num_bits,
                    "clip_low": 0.0,
                    "clip_high": clip_high,
                    "scheme": scheme,
                }
            )


# Create tuple to send to quantizers for single-sided clipping
@pytest.fixture(scope="session", params=quantizer_single_sided_params)
def quantizer_single_sided(request):
    """
    Fixture tuple for single-sided quantizer

    Args:
        request (dict): Dict for quantizer args

    Returns:
        dict: Tuple for quantizer args
    """
    return request.param


##################################
# Base Quantizer Options Fixture #
##################################

base_quantizer_option_params = []
for nativePT in [True, False]:
    for dequantize in [True, False]:
        base_quantizer_option_params.append(
            {"nativePT": nativePT, "dequantize": dequantize}
        )


@pytest.fixture(scope="session", params=base_quantizer_option_params)
def base_options(request):
    """
    Fixture for quantizer base options

    Args:
        request (dict): Base option dict

    Returns:
        dict: Base option tuple
    """
    return request.param


######################
# Quantizer Fixtures #
######################


@pytest.fixture
def torch_quantizer_symmetric(quantizer_symmetric):
    """
    Torch Quantizer w/ symmetric settings

    Args:
        quantizer_symmetric (dict): Symmetric quantizer settings

    Returns:
        torch.nn.Module: TorchQuantizer
    """
    return TorchQuantizer(
        num_bits=quantizer_symmetric["num_bits"],
        clip_low=quantizer_symmetric["clip_low"],
        clip_high=quantizer_symmetric["clip_high"],
        qscheme=quantizer_symmetric["scheme"],
    )


@pytest.fixture
def torch_quantizer_symmetric_perCh(quantizer_symmetric_perCh):
    """
    Torch Quantizer w/ symmetric settings for perCh

    Args:
        quantizer_symmetric (dict): Symmetric quantizer settings

    Returns:
        torch.nn.Module: TorchQuantizer
    """
    return TorchQuantizer(
        num_bits=quantizer_symmetric_perCh["num_bits"],
        clip_low=quantizer_symmetric_perCh["clip_low"],
        clip_high=quantizer_symmetric_perCh["clip_high"],
        qscheme=quantizer_symmetric_perCh["scheme"],
    )


@pytest.fixture
def torch_quantizer_asymmetric(quantizer_asymmetric):
    """
    Torch Quantizer w/ asymmetric settings

    Args:
        quantizer_asymmetric (dict): Asymmetric quantizer settings

    Returns:
        torch.nn.Module: TorchQuantizer
    """
    return TorchQuantizer(
        num_bits=quantizer_asymmetric["num_bits"],
        clip_low=quantizer_asymmetric["clip_low"],
        clip_high=quantizer_asymmetric["clip_high"],
        qscheme=quantizer_asymmetric["scheme"],
    )


@pytest.fixture
def torch_quantizer_asymmetric_perCh(quantizer_asymmetric_perCh):
    """
    Torch Quantizer w/ asymmetric settings for perCh

    Args:
        quantizer_asymmetric (dict): Asymmetric quantizer settings

    Returns:
        torch.nn.Module: TorchQuantizer
    """
    return TorchQuantizer(
        num_bits=quantizer_asymmetric_perCh["num_bits"],
        clip_low=quantizer_asymmetric_perCh["clip_low"],
        clip_high=quantizer_asymmetric_perCh["clip_high"],
        qscheme=quantizer_asymmetric_perCh["scheme"],
    )


@pytest.fixture
def torch_quantizer_single_sided(quantizer_single_sided):
    """
    Torch Quantizer w/ single-sided settings

    Args:
        quantizer_single_sided (dict): Single-sided quantizer settings

    Returns:
        torch.nn.Module: TorchQuantizer
    """
    return TorchQuantizer(
        num_bits=quantizer_single_sided["num_bits"],
        clip_low=quantizer_single_sided["clip_low"],
        clip_high=quantizer_single_sided["clip_high"],
        qscheme=quantizer_single_sided["scheme"],
    )


@pytest.fixture
def pact_quantizer_single_sided(quantizer_single_sided):
    """
    PACT quantizer w/ single-sided settings

    Args:
        quantizer_single_sided (dict): Single-sided quantizer settings

    Returns:
        torch.autograd.Function: PACT
    """
    return PACT(
        num_bits=quantizer_single_sided["num_bits"],
        # init_clip_valn=quantizer_single_sided["clip_low"],
        init_clip_val=quantizer_single_sided["clip_high"],
        # qscheme=quantizer_single_sided["scheme"],
    )


@pytest.fixture
def pact_rc_quantizer_single_sided(quantizer_single_sided):
    """
    PACT_rc quantizer w/ single-sided settings

    Args:
        quantizer_single_sided (dict): Single-sided quantizer settings

    Returns:
        torch.autograd.Function: PACT_rc
    """
    return PACT_rc(
        num_bits=quantizer_single_sided["num_bits"],
        init_clip_valn=quantizer_single_sided["clip_low"],
        init_clip_val=quantizer_single_sided["clip_high"],
        qscheme=quantizer_single_sided["scheme"],
    )


@pytest.fixture
def pact2_quantizer_asymmetric(quantizer_asymmetric):
    """
    PACT2 quantizer w/ asymmetric settings

    Args:
        quantizer_asymmetric (dict): Asymmetric quantizer settings

    Returns:
        torch.autograd.Function: PACT2
    """
    return PACT2(
        num_bits=quantizer_asymmetric["num_bits"],
        init_clip_valn=quantizer_asymmetric["clip_low"],
        init_clip_val=quantizer_asymmetric["clip_high"],
        # qscheme=quantizer_asymmetric["scheme"],
    )


@pytest.fixture
def pact2_rc_quantizer_asymmetric(quantizer_asymmetric):
    """
    PACT2_rc quantizer w/ asymmetric settings

    Args:
        quantizer_asymmetric (dict): Asymmetric quantizer settings

    Returns:
        torch.autograd.Function: PACT2_rc
    """
    return PACT2_rc(
        num_bits=quantizer_asymmetric["num_bits"],
        init_clip_valn=quantizer_asymmetric["clip_low"],
        init_clip_val=quantizer_asymmetric["clip_high"],
        qscheme=quantizer_asymmetric["scheme"],
    )


@pytest.fixture
def pact2sym_quantizer_symmetric(quantizer_symmetric):
    """
    PACT2Sym quantizer w/ symmetric settings

    Args:
        quantizer_symmetric (dict): Symmetric quantizer settings

    Returns:
        torch.autograd.Function: PACT2Sym
    """
    return PACT2Sym(
        num_bits=quantizer_symmetric["num_bits"],
        # init_clip_valn=quantizer_asymmetric["clip_low"],
        init_clip_val=quantizer_symmetric["clip_high"],
        # qscheme=quantizer_asymmetric["scheme"],
    )


@pytest.fixture
def pact2sym_rc_quantizer_symmetric(quantizer_symmetric):
    """
    PACT2Sym_rc quantizer w/ symmetric settings

    Args:
        quantizer_symmetric (dict): Symmetric quantizer settings

    Returns:
        torch.autograd.Function: PACT2Sym_rc
    """
    return PACT2Sym_rc(
        num_bits=quantizer_symmetric["num_bits"],
        init_clip_valn=quantizer_symmetric["clip_low"],
        init_clip_val=quantizer_symmetric["clip_high"],
        qscheme=quantizer_symmetric["scheme"],
    )


@pytest.fixture
def pactplussym_quantizer_symmetric(quantizer_symmetric):
    """
    PACT+Sym quantizer w/ symmetric settings

    Args:
        quantizer_symmetric (dict): Symmetric quantizer settings

    Returns:
        torch.autograd.Function: PACT+Sym
    """
    return PACTplusSym(
        num_bits=quantizer_symmetric["num_bits"],
        # init_clip_valn=quantizer_symmetric["clip_low"],
        init_clip_val=quantizer_symmetric["clip_high"],
        # qscheme=quantizer_symmetric["scheme"],
    )


@pytest.fixture
def pactplussym_rc_quantizer_symmetric(quantizer_symmetric):
    """
    PACT+Sym_rc quantizer w/ symmetric settings

    Args:
        quantizer_symmetric (dict): Symmetric quantizer settings

    Returns:
        torch.autograd.Function: PACT+Sym_rc
    """
    return PACTplusSym_rc(
        num_bits=quantizer_symmetric["num_bits"],
        init_clip_valn=quantizer_symmetric["clip_low"],
        init_clip_val=quantizer_symmetric["clip_high"],
        qscheme=quantizer_symmetric["scheme"],
    )


@pytest.fixture
def sawb_quantizer_symmetric(quantizer_symmetric):
    """
    SAWB quantizer w/ symmetric settings

    Args:
        quantizer_symmetric (dict): Symmetric quantizer settings

    Returns:
        torch.autograd.Function: SAWB
    """
    return SAWB(
        num_bits=quantizer_symmetric["num_bits"],
        # init_clip_valn=quantizer_symmetric["clip_low"],
        # init_clip_val=quantizer_symmetric["clip_high"],
        # qscheme=quantizer_symmetric["scheme"],
    )


@pytest.fixture
def sawb_quantizer_symmetric_perCh(quantizer_symmetric_perCh):
    """
    SAWB quantizer w/ symmetric settings

    Args:
        quantizer_symmetric (dict): Symmetric quantizer settings

    Returns:
        torch.autograd.Function: SAWB
    """
    return SAWB(
        num_bits=quantizer_symmetric_perCh["num_bits"],
        # init_clip_valn=quantizer_symmetric["clip_low"],
        # init_clip_val=quantizer_symmetric["clip_high"],
        # qscheme=quantizer_symmetric["scheme"],
    )


@pytest.fixture
def sawb_rc_quantizer_symmetric(quantizer_symmetric):
    """
    SAWB_rc quantizer w/ symmetric settings

    Args:
        quantizer_symmetric (dict): Symmetric quantizer settings

    Returns:
        torch.autograd.Function: SAWB_rc
    """
    return SAWB_rc(
        num_bits=quantizer_symmetric["num_bits"],
        # init_clip_valn=quantizer_symmetric["clip_low"],
        # init_clip_val=quantizer_symmetric["clip_high"],
        qscheme=quantizer_symmetric["scheme"],
    )


@pytest.fixture
def qmax_quantizer_symmetric(quantizer_symmetric):
    """
    Qmax quantizer w/ symmetric settings

    Args:
        quantizer_symmetric (dict): Symmetric quantizer settings

    Returns:
        torch.autograd.Function: Qmax
    """
    return Qmax(
        num_bits=quantizer_symmetric["num_bits"],
        # init_clip_valn=quantizer_symmetric["clip_low"],
        # init_clip_val=quantizer_symmetric["clip_high"],
        # qscheme=quantizer_symmetric["scheme"],
    )


@pytest.fixture
def qmax_rc_quantizer_symmetric(quantizer_symmetric):
    """
    Qmax_rc quantizer w/ symmetric settings

    Args:
        quantizer_symmetric (dict): Symmetric quantizer settings

    Returns:
        torch.autograd.Function: Qmax_rc
    """
    return Qmax_rc(
        num_bits=quantizer_symmetric["num_bits"],
        # init_clip_valn=quantizer_symmetric["clip_low"],
        # init_clip_val=quantizer_symmetric["clip_high"],
        qscheme=quantizer_symmetric["scheme"],
    )


@pytest.fixture
def lsq_quantizer_single_sided(quantizer_single_sided):
    """
    LSQ quantizer w/ single-sided settings

    Args:
        quantizer_single_sided (dict): Single-sided quantizer settings

    Returns:
        torch.autograd.Function: LSQ
    """
    return LSQQuantization(
        num_bits=quantizer_single_sided["num_bits"],
        # init_clip_valn = quantizer_single_sided['clip_low'],
        init_clip_val=quantizer_single_sided["clip_high"],
        # scheme = quantizer_single_sided['scheme'],
    )


@pytest.fixture
def lsq_rc_quantizer_single_sided(quantizer_single_sided):
    """
    LSQ_rc quantizer w/ single-sided settings

    Args:
        quantizer_single_sided (dict): Single-sided quantizer settings

    Returns:
        torch.autograd.Function: LSQ_rc
    """
    return LSQQuantization_rc(
        num_bits=quantizer_single_sided["num_bits"],
        init_clip_valn=quantizer_single_sided["clip_low"],
        init_clip_val=quantizer_single_sided["clip_high"],
        scheme=quantizer_single_sided["scheme"],
    )


@pytest.fixture
def lsqplus_quantizer_symmetric(quantizer_symmetric):
    """
    LSQ+ quantizer w/ symmetric settings

    Args:
        quantizer_symmetric (dict): Symmetric quantizer settings

    Returns:
        torch.autograd.Function: LSQ+
    """
    return LSQPlus(
        num_bits=quantizer_symmetric["num_bits"],
        clip_valn=quantizer_symmetric["clip_low"],
        clip_val=quantizer_symmetric["clip_high"],
        # scheme = quantizer_symmetric['scheme'],
    )


@pytest.fixture
def lsqplus_quantizer_asymmetric(quantizer_asymmetric):
    """
    LSQ+ quantizer w/ asymmetric settings

    Args:
        quantizer_asymmetric (dict): Asymmetric quantizer settings

    Returns:
        torch.autograd.Function: LSQ+
    """
    return LSQPlus(
        num_bits=quantizer_asymmetric["num_bits"],
        clip_valn=quantizer_asymmetric["clip_low"],
        clip_val=quantizer_asymmetric["clip_high"],
        # scheme = quantizer_asymmetric['scheme'],
    )


@pytest.fixture
def lsqplus_rc_quantizer_symmetric(quantizer_symmetric):
    """
    LSQ+_rc quantizer w/ asymmetric settings

    Args:
        quantizer_symmetric (dict): Symmetric quantizer settings

    Returns:
        torch.autograd.Function: LSQ+_rc
    """
    return LSQPlus_rc(
        num_bits=quantizer_symmetric["num_bits"],
        clip_valn=quantizer_symmetric["clip_low"],
        clip_val=quantizer_symmetric["clip_high"],
        scheme=quantizer_symmetric["scheme"],
    )


@pytest.fixture
def lsqplus_rc_quantizer_asymmetric(quantizer_asymmetric):
    """
    LSQ+_rc quantizer w/ asymmetric settings

    Args:
        quantizer_asymmetric (dict): Asymmetric quantizer settings

    Returns:
        torch.autograd.Function: LSQ+_rc
    """
    return LSQPlus_rc(
        num_bits=quantizer_asymmetric["num_bits"],
        clip_valn=quantizer_asymmetric["clip_low"],
        clip_val=quantizer_asymmetric["clip_high"],
        scheme=quantizer_asymmetric["scheme"],
    )
