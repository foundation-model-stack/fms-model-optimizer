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
Test get_activation_quantizer

Currently under construction.
Only checks that the function signiture is callable without error.
"""

# Third Party
import pytest

# Local
from fms_mo.quant_refactor.get_quantizer_rc import get_activation_quantizer_rc

# Activation inputs
# qa_mode="PACT",
# nbits=32,
# clip_val=None,
# clip_valn=None,
# non_neg=False,
# align_zero=True,
# extend_act_range=False,
# use_PT_native_Qfunc=False,
# use_subnormal=False,

# Symmetric Activation Params
qa_mode_symmetric_params = [
    "pactsym",
    "pactsym+",
    "maxsym",
]
activation_symmetric_params = []
for qa_mode in qa_mode_symmetric_params:
    for num_bits in [8, 4]:
        for clip_val in [3.5]:
            for clip_valn in [-clip_val]:
                for non_neg in [False]:
                    for qlevel_lowering in [True, False]:
                        for extend_act_range in [True, False]:
                            for nativePT in [True, False]:
                                for use_subnormal in [True, False]:
                                    activation_symmetric_params.append(
                                        {
                                            "qa_mode": qa_mode,
                                            "num_bits": num_bits,
                                            "clip_val": clip_val,
                                            "clip_valn": clip_valn,
                                            "non_neg": non_neg,
                                            "qlevel_lowering": qlevel_lowering,
                                            "extend_act_range": extend_act_range,
                                            "nativePT": nativePT,
                                            "use_subnormal": use_subnormal,
                                        }
                                    )


@pytest.fixture(params=activation_symmetric_params)
def activation_symmetric(request):
    """
    Activation Symmetric params as dict[key,val]

    Returns:
        dict: One activation symmetric param dict
    """
    return request.param


# Asymmetric Activation Params
qa_mode_asymmetric_params = [
    "pact",
    "pact+",
    "max",
    "minmax",
    "lsq+",
]
activation_asymmetric_params = []
for qa_mode in qa_mode_asymmetric_params:
    for num_bits in [8, 4]:
        for clip_val in [3.5]:
            for clip_valn in [-2.4]:
                for non_neg in [False]:
                    for qlevel_lowering in [True, False]:
                        for extend_act_range in [True, False]:
                            for nativePT in [True, False]:
                                for use_subnormal in [True, False]:
                                    activation_asymmetric_params.append(
                                        {
                                            "qa_mode": qa_mode,
                                            "num_bits": num_bits,
                                            "clip_val": clip_val,
                                            "clip_valn": clip_valn,
                                            "non_neg": non_neg,
                                            "qlevel_lowering": qlevel_lowering,
                                            "extend_act_range": extend_act_range,
                                            "nativePT": nativePT,
                                            "use_subnormal": use_subnormal,
                                        }
                                    )


@pytest.fixture(params=activation_asymmetric_params)
def activation_asymmetric(request):
    """
    Activation Asymmetric params as dict[key,val]

    Returns:
        dict: One activation asymmetric param dict
    """
    return request.param


# Single-sided Activation Params
qa_mode_single_sided_params = [
    "pact",
    "pact+",
]
activation_single_sided_params = []
for qa_mode in qa_mode_single_sided_params:
    for num_bits in [8, 4]:
        for clip_val in [3.5]:
            for clip_valn in [0]:
                for non_neg in [True]:
                    for qlevel_lowering in [True, False]:
                        for extend_act_range in [True, False]:
                            for nativePT in [True, False]:
                                for use_subnormal in [True, False]:
                                    activation_single_sided_params.append(
                                        {
                                            "qa_mode": qa_mode,
                                            "num_bits": num_bits,
                                            "clip_val": clip_val,
                                            "clip_valn": clip_valn,
                                            "non_neg": non_neg,
                                            "qlevel_lowering": qlevel_lowering,
                                            "extend_act_range": extend_act_range,
                                            "nativePT": nativePT,
                                            "use_subnormal": use_subnormal,
                                        }
                                    )


@pytest.fixture(params=activation_single_sided_params)
def activation_single_sided(request):
    """
    Activation Single-sided params as dict[key,val]

    Returns:
        dict: One activation single-sided param dict
    """
    return request.param


def test_get_activation_symmetric_rc(activation_symmetric):
    """
    Test get_activation_quantizer is callable

    Args:
        activation_symmetric (dict): Function input
    """
    get_activation_quantizer_rc(
        qa_mode=activation_symmetric["qa_mode"],
        nbits=activation_symmetric["num_bits"],
        clip_val=activation_symmetric["clip_val"],
        clip_valn=activation_symmetric["clip_valn"],
        non_neg=activation_symmetric["non_neg"],
        align_zero=activation_symmetric["qlevel_lowering"],
        extend_act_range=activation_symmetric["extend_act_range"],
        use_PT_native_Qfunc=activation_symmetric["nativePT"],
        use_subnormal=activation_symmetric["use_subnormal"],
    )


def test_get_activation_asymmetric_new(activation_asymmetric):
    """
    Test get_activation_quantizer is callable

    Args:
        activation_asymmetric (dict): Function input
    """
    get_activation_quantizer_rc(
        qa_mode=activation_asymmetric["qa_mode"],
        nbits=activation_asymmetric["num_bits"],
        clip_val=activation_asymmetric["clip_val"],
        clip_valn=activation_asymmetric["clip_valn"],
        non_neg=activation_asymmetric["non_neg"],
        align_zero=activation_asymmetric["qlevel_lowering"],
        extend_act_range=activation_asymmetric["extend_act_range"],
        use_PT_native_Qfunc=activation_asymmetric["nativePT"],
        use_subnormal=activation_asymmetric["use_subnormal"],
    )


def test_get_activation_single_sided_rc(activation_single_sided):
    """
    Test get_activation_quantizer is callable

    Args:
        activation_single_sided (dict): Function input
    """
    get_activation_quantizer_rc(
        qa_mode=activation_single_sided["qa_mode"],
        nbits=activation_single_sided["num_bits"],
        clip_val=activation_single_sided["clip_val"],
        clip_valn=activation_single_sided["clip_valn"],
        non_neg=activation_single_sided["non_neg"],
        align_zero=activation_single_sided["qlevel_lowering"],
        extend_act_range=activation_single_sided["extend_act_range"],
        use_PT_native_Qfunc=activation_single_sided["nativePT"],
        use_subnormal=activation_single_sided["use_subnormal"],
    )
