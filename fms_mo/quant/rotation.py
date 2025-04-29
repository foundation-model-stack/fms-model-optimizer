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

"""Util functions related to Hadamard rotation."""

# Third Party
import torch

# Local
from fms_mo.utils.hadamard_util import matmul_hadU_cuda


class RotQuantWrapper(torch.nn.Module):
    """Add a wrapper to fms-mo quantizers. Objects of this class could have two rotation tensors,
    and basic formula is:

        self.quantizer(self.rot_left @ input_tensor @ self.rot_right)

    NOTE rot_xxx could be optional, depending on whether it's for weights or activations.
    For example, in SpinQuant QKV Linears will looks like (pseudo-code, "self" are not refering
    to the same objects here):
        qx = self.quantize_feature(x)                       # no rotation, just a normal quantizer
        qw_q = self.quantize_weight(self.weight, R1_t)      # need left rotation only
        qw_k = self.quantize_weight(sefl.weight, R1_t)
        qw_v = self.quantize_weight(sefl.weight, R1_t, R2)  # need both left and right rotation

        return F.linear(qx, qw, bias)

    for MLP down_proj
        qx = self.quantize_feature(x, None, R4)             # for activation, should be x @ R
        qw = self.quantize_weight(sefl.weight, R4_t, R1)

        return F.linear(qx, qw, bias)

    Also need to make sure self.R is pointing to a nn.Parameter() if training on R is needed.
    """

    def __init__(self, quantizer, *args, **kwargs):
        self.online_full_had = kwargs.pop("online_full_had", None)
        self.f32_had = kwargs.pop("f32_had", None)
        super().__init__(*args, **kwargs)
        self.quantizer = quantizer
        self.R_left = None
        self.R_right = None
        self.K_left = None  # if K_xxx > 1, R_xxx is a special had matrix
        self.K_right = None

    def forward(self, input_tensor):
        org_dtype = input_tensor.dtype

        if self.online_full_had:
            # online hadamard => rotation for activation. should be input_tensor @ R_right
            # cannot be fused into W and no training, either.
            if self.fp32_had:
                input_tensor = input_tensor.float()
            input_tensor = matmul_hadU_cuda(
                input_tensor, self.R_right, self.K_right
            ).to(org_dtype)

            return input_tensor

        # not online => rotation for weights, could be fused into W later.
        if self.R_left:
            input_tensor = self.R_left @ inp_tensor
        if self.R_right:
            inp_tensor = inp_tensor @ self.R_right

        return inp_tensor
