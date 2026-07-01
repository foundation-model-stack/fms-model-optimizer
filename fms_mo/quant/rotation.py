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
from fms_mo.utils.hadamard_util import matmul_hadU, matmul_hadU_cuda


class RotQuantWrapper(torch.nn.Module):
    """Add a wrapper to fms-mo quantizers. Objects of this class could have two rotation tensors,
    and basic formula is:

        quantizer(Rot_left @ input_tensor @ Rot_right)

    But Rot_xxx could be optional, depending on whether it's for weights or activations.

    For weights, two possible use cases in SpinQuant are:
        (A^-1 W) and (A^-1 W B).
    Since linear.weight is already W^T and should stay as (rotated W)^T , these two cases will be
        (A^-1 W)^T = W^T (A^-1)^T = W^T A, as A is a Hadamard matrix
        (A^-1 W B)^T = B^T W^T A
    ** Furthermore, depending on R1 is A (v_proj) or B (o_ and down_proj), computation could be
        slightly different
            if R1 is A (R_left):
                calc W^T A first -> (W^T A)^T -> reshape -> *B -> .t() then ready for linear
            else R1 is B (R_right):
                calc B^T W^T first -> reshape -> *A -> ready for linear

    For activation (online rotation), it will always be (input_tensor @ R_right)

    then    return F.linear(qx, qw, bias)

    NOTE
    0. If online_full_had == False and self.R_left is None => do nothing, apply quantizer ONLY.
    1. Make sure self.R is pointing to a nn.Parameter() if training on R is needed.
    2. Because R is a ptr to a nn.Param tensor, it CANNOT store a "transposed" copy, hence the use
        of self.transpose flags if needed.
    """

    def __init__(self, quantizer=None, *args, **kwargs):
        self.online_full_had = kwargs.pop("online_full_had", None)
        self.compute_dtype = kwargs.pop("compute_dtype", torch.float64)
        super().__init__(*args, **kwargs)
        self.quantizer = quantizer
        self.R_left = None
        self.R_right = None
        self.K_left = None
        self.K_right = None
        self.R1_is_left = True  # see dosstring above
        self.transpose_right = False  # this flag is for online rotation only
        # if K_xxx == 1, use exact hadamard matrix. (R_xxx won't be needed). but if K > 1, R will
        # be one of the 12 special had matrix. (they are stored in a binary file)

    def forward(self, inp):
        org_dtype = inp.dtype

        if self.R_left is not None:
            # Case 1: Weight rotation
            #       as Activation rotation will only have R_right. If R_left exists for A =>
            #       should have absorbed R_left for A into prev layer's W.
            #       Hence, R_left is not None can only mean weight rotation, not online =>
            #       could be either 1) R_left only or 2) both R_left and R_right.

            in_feat, out_feat = inp.shape[-1], inp.shape[0]  # input is W^T (out, in)
            if self.R1_is_left:
                # for q, k, v, up, gate, calc W^T A first. see details in docstring
                inp = inp.to(self.compute_dtype) @ self.R_left.to(self.compute_dtype)

                if self.R_right is not None:
                    had_dim = self.R_right.shape[0]
                    inp = inp.t()  # (W^T A) ^T = A^T W, shape is (in, out)
                    inp = inp.reshape(-1, out_feat // had_dim, had_dim)
                    inp = inp.to(self.compute_dtype) @ self.R_right.to(
                        self.compute_dtype
                    )
                    inp = inp.reshape((in_feat, out_feat)).t()

            else:
                assert self.R_right is not None, "R1_is_right but R_right is None."

                # for o, down, calc B^T W^T first, where R1 is B
                inp = self.R_right.t().to(self.compute_dtype) @ inp.to(
                    self.compute_dtype
                )
                had_dim = self.R_left.shape[0]
                inp = inp.t()  # this will be W, not W^T, i.e. (in, out)
                w_shape = inp.shape
                inp = inp.reshape(-1, in_feat // had_dim, had_dim)
                inp = inp.to(self.compute_dtype) @ self.R_left.to(self.compute_dtype)
                inp = inp.reshape((out_feat, in_feat))

        elif self.R_right is not None or self.K_right == 1:
            # Case 2: rotation for activation. should always be (inp @ R_right)
            if self.online_full_had:
                # Case 2-1: online, no training to R. When R_right is None (K==1), use exact size
                if self.compute_dtype in [torch.float, torch.float64]:
                    # follow SpinQuant paper, use no higher than fp32 for online had
                    inp = inp.float()

                # matmul_hadU_cuda already include 1/sqrt(shape[-1])
                if self.transpose_right and self.R_right is not None:
                    inp = matmul_hadU_cuda(inp, self.R_right.t(), self.K_right)
                else:
                    inp = matmul_hadU_cuda(inp, self.R_right, self.K_right)
                    # inp = matmul_hadU(inp)
            else:
                # Case 2-2: offline (such as last R before lm_head)
                if self.transpose_right:
                    inp = inp.to(self.compute_dtype) @ self.R_right.t().to(
                        self.compute_dtype
                    )
                else:
                    inp = inp.to(self.compute_dtype) @ self.R_right.to(
                        self.compute_dtype
                    )

        # Case 3: both R_left and R_right are None and K!=1=> No Rotation, apply quantizer if exist.

        inp = inp.to(org_dtype)

        if self.quantizer:
            # with torch.no_grad():
            inp = self.quantizer(inp)

        return inp

    def __repr__(self):
        """Simplified repr for RotQuantizer. Shows name and nbits."""
        repr_str = "Only("
        if self.quantizer is not None:
            repr_str = f"{self.quantizer.__class__.__name__}("

        if self.R_left is not None or self.online_full_had:
            # will do W or A rotation
            repr_str = (
                "Rot"
                + repr_str
                + f"{'' if self.R_left is None else 'Rl'},{'' if self.R_right is None else 'Rr'})"
            )

        return repr_str


class EmbeddingRotWrapper(torch.nn.Module):
    """Simply add a Rotation after input embeddings. original code looks like

            input_embeds = self.embed_tokens(input_ids)

        This wrapper will be:

            input_embeds = self.embed_tokens(input_ids)
            dtype = input_embeds.dtype
            if self.R:
                input_embeds = input_embeds @ self.R).to(dtype)
            return input_embeds

    Also need to make sure self.R is pointing to a nn.Parameter() if training on R is needed.
    """

    def __init__(self, emb, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.emb = emb
        self.R = None
        self.compute_dtype = torch.float64

    def forward(self, inp_ids):
        inp_embeds = self.emb(inp_ids)
        org_dtype = inp_embeds.dtype
        if self.R is not None:
            inp_embeds = (
                inp_embeds.to(self.compute_dtype) @ self.R.to(self.compute_dtype)
            ).to(org_dtype)
        return inp_embeds

    def __repr__(self):
        """Simplified repr for RotEmb."""
        repr_str = f"Rot{str(self.emb)}"
        if self.R is not None:
            repr_str.replace(")", ", Rr)")
        return repr_str
