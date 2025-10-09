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

# This code is based on QuaRot(https://github.com/spcl/QuaRot/tree/main/quarot).
# Licensed under Apache License 2.0.
# Adapted from https://github.com/Cornell-RelaxML/quip-sharp/blob/main/lib/utils/matmul_had.py
# and https://github.com/facebookresearch/SpinQuant/blob/main/utils/hadamard_utils.py
"""
Change original "text tensor implementation" into binaries for better efficiency. Only has 12
sizes available in the safetensors file. [12, 20, 28, 36, 40, 44, 52, 60, 108, 140, 156, 172]
"""

# Standard
from pathlib import Path

# Third Party
from fast_hadamard_transform import hadamard_transform  # pylint: disable=import-error
from safetensors import safe_open
import torch

# TODO make sure it's a persistent cache so we don't need to load from file everytime
cwd = Path(__file__).parent
hadKs = {}
with safe_open(cwd / "hadk.safetensors", framework="pt", device="cuda") as f:
    for K_str in f.keys():  # K is a str
        hadKs[K_str] = f.get_tensor(K_str)


class HadamardTransform(torch.autograd.Function):
    """The unnormalized Hadamard transform (i.e. without dividing by sqrt(2))"""

    # TODO seems redundant, insdie hadamard_transform(), backward is already handled...?
    @staticmethod
    def forward(_ctx, u):
        return hadamard_transform(u)

    @staticmethod
    def backward(_ctx, grad):
        return hadamard_transform(grad)


def get_hadK(n, transpose=False):
    """Simplify the implementation and use binary tensors instead of text implementation."""
    hadK = None
    for K in [172, 156, 140, 108, 60, 52, 44, 40, 36, 28, 20, 12]:
        if n % K == 0 and is_pow2(n // K):
            hadK = hadKs[str(K)]
            if transpose:
                hadK = hadK.T
            break

    if hadK is None:
        if is_pow2(n):
            K = 1
        else:
            raise RuntimeError(
                f"{n} is not power of 2 or does not have a special size Hadamard available."
            )

    return hadK, K


def matmul_hadU(X, transpose=False):
    """Borrowed from SpinQuant."""
    n = X.shape[-1]
    hadK, K = get_hadK(n, transpose)
    input_ = X.clone().view(-1, n, 1)
    output = input_.clone()
    while input_.shape[1] > K:
        input_ = input_.view(input_.shape[0], input_.shape[1] // 2, 2, input_.shape[2])
        output = output.view(input_.shape)
        output[:, :, 0, :] = input_[:, :, 0, :] + input_[:, :, 1, :]
        output[:, :, 1, :] = input_[:, :, 0, :] - input_[:, :, 1, :]
        output = output.view(input_.shape[0], input_.shape[1], -1)
        (input_, output) = (output, input_)
    del output

    if K > 1:
        # Do not explicitly repeat - OOM
        # input_ = torch.bmm(
        #     hadK.repeat(len(input_), 1, 1).to(input_.device).to(input_.dtype), input_)
        # Use bcast instead
        input_ = hadK.view(1, K, K).to(input_) @ input_

    return input_.view(X.shape) / torch.tensor(n).sqrt()


def matmul_hadUt(X):
    """Borrowed from SpinQuant."""
    return matmul_hadU(X, transpose=True)


def random_hadamard_matrix(size, device):
    """Borrowed from SpinQuant."""
    # See https://cornell-relaxml.github.io/quip-sharp/
    # Section "Randomized Hadamard Transformation"
    Q = torch.randint(low=0, high=2, size=(size,)).to(torch.float64)
    Q = Q * 2 - 1
    Q = torch.diag(Q)
    return matmul_hadU(Q).to(device)


def hadamard_matrix(size, device):
    """Borrowed from SpinQuant."""
    Q = torch.eye(size)
    return matmul_hadU(Q).to(device)


def matmul_hadU_cuda(X, hadK, K):
    """Borrowed from SpinQuant."""
    n = X.shape[-1]
    if K == 1:
        return HadamardTransform.apply(X.contiguous()) / torch.tensor(n).sqrt()
    # if transpose:
    #     hadK = hadK.T.contiguous()
    input_ = X.view(-1, K, n // K)
    input_ = HadamardTransform.apply(input_.contiguous()) / torch.tensor(n).sqrt()
    input_ = hadK.to(input_.device).to(input_.dtype) @ input_
    return input_.reshape(X.shape)


# def matmul_hadUt_cuda(X, hadK, K):
#     """Borrowed from SpinQuant."""
#     return matmul_hadU_cuda(X, hadK, K, transpose=True)


def apply_exact_had_to_linear(module, had_dim=-1, output=False, R2=None):
    """Borrowed from SpinQuant."""
    assert isinstance(module, torch.nn.Linear)
    in_features, out_features = module.in_features, module.out_features

    if had_dim != -1:
        assert is_pow2(had_dim), "Hadamard dimension must be a power of 2!"

    W_ = module.weight.data
    dtype = W_.dtype
    dev = W_.device
    init_shape = W_.shape
    W_ = W_.float().cuda()

    if had_dim == -1:
        if output:
            had_K, K = get_hadK(out_features)
            W_ = matmul_hadU_cuda(W_.t(), had_K, K).t()
        if not output:
            had_K, K = get_hadK(in_features)
            W_ = matmul_hadU_cuda(W_, had_K, K)
    else:
        hadK = hadamard_matrix(had_dim, "cuda").to(torch.float64)
        if R2 is not None:
            hadK = R2.to(torch.float64)
        if output:
            W_ = W_.t()
            transposed_shape = W_.shape
            temp = W_.reshape(-1, transposed_shape[-1] // had_dim, had_dim)
            temp = temp.to(torch.float64) @ hadK
            W_ = temp.reshape(transposed_shape).t()
        else:
            init_shape = W_.shape
            temp = W_.reshape(-1, init_shape[-1] // had_dim, had_dim)
            temp = temp.to(torch.float64) @ hadK
            W_ = temp.reshape(init_shape)
    module.weight.data = W_.to(device=dev, dtype=dtype)


def is_pow2(n):
    """Borrowed from SpinQuant."""
    return (n & (n - 1) == 0) and (n > 0)


# hadamard matrices for had12, had36.pal2, had52,will,
# # had60.pal, had108.pal, had140.pal, had156.will, had172.will:
# http://www.neilsloane.com/hadamard/index.html
