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
Test get_weight_quantizer

Currently under construction.
Only checks that the function signiture is callable without error.
"""

# Third Party
import pytest
import torch

# Local
from fms_mo.quant_refactor.get_quantizer_rc import get_weight_quantizer_rc

# Weight inputs
# qw_mode="SAWB+",
# nbits=32,
# clip_val=None,
# clip_valn=None,
# align_zero=True,
# w_shape=None,
# recompute=False,
# perGp=None,
# use_PT_native_Qfunc=False,
# use_subnormal=False,

# Symmetric weight_ Params
# TODO (bmgroth) - enable perCh and perGrp tests
qw_mode_symmetric_params = [
    "sawb",
    "sawb16",
    # "sawbperCh",
    "sawb+",
    "sawbinterp",
    "max",
    # "maxperCh",
    # "maxperGp",
    "minmax",
    # "minmaxperCh",
    # "minmaxperGp",
    "pact",
    "pact+",
    "lsq+",
]
weight_symmetric_params = []
for qw_mode in qw_mode_symmetric_params:
    for num_bits in [8, 4]:
        for clip_val in [3.5]:
            for clip_valn in [-clip_val]:
                for qlevel_lowering in [True, False]:
                    for w_shape in [torch.randn(10, 10)]:  # random 10x10 tensor
                        for recompute in [True, False]:
                            for perGp in [None]:  # checked for is not None
                                for nativePT in [True, False]:
                                    for use_subnormal in [True, False]:
                                        weight_symmetric_params.append(
                                            {
                                                "qw_mode": qw_mode,
                                                "num_bits": num_bits,
                                                "clip_val": clip_val,
                                                "clip_valn": clip_valn,
                                                "qlevel_lowering": qlevel_lowering,
                                                "w_shape": w_shape,
                                                "recompute": recompute,
                                                "perGp": perGp,
                                                "nativePT": nativePT,
                                                "use_subnormal": use_subnormal,
                                            }
                                        )


@pytest.fixture(params=weight_symmetric_params)
def weight_symmetric(request):
    """
    Weight symmetric params as dict[key,val]

    Returns:
        dict: One weight symmetric param dict
    """
    return request.param


def test_get_weight_symmetric_rc(weight_symmetric):
    """
    Test get_weight_quantizer is callable

    Args:
        weight_symmetric (dict): Function input
    """
    get_weight_quantizer_rc(
        qw_mode=weight_symmetric["qw_mode"],
        nbits=weight_symmetric["num_bits"],
        clip_val=weight_symmetric["clip_val"],
        clip_valn=weight_symmetric["clip_valn"],
        align_zero=weight_symmetric["qlevel_lowering"],
        w_shape=weight_symmetric["w_shape"],
        recompute=weight_symmetric["recompute"],
        perGp=weight_symmetric["perGp"],
        use_PT_native_Qfunc=weight_symmetric["nativePT"],
        use_subnormal=weight_symmetric["use_subnormal"],
    )
