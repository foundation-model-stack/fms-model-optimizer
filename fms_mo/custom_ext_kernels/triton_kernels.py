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

"""This file contains external kernels for FP and INT8 matmul written in triton."""

# Third Party
import torch

# Local
from fms_mo.utils.import_utils import available_packages

# Assume any calls to the file are requesting triton
if not available_packages["triton"]:
    raise ImportError(
        "triton python package is not avaialble, please check your installation."
    )

# Third Party
# pylint: disable=wrong-import-position
from triton.language.extra import libdevice
import triton
import triton.language as tl

DTYPE_I8 = [torch.int8]
DTYPE_F8 = [torch.float8_e4m3fn, torch.float8_e5m2]
DTYPE_8BIT = DTYPE_I8 + DTYPE_F8


def get_cuda_autotune_config(chunk_size=None):
    """Basic use of triton.Config() is like:
    triton.Config(
        {'BLOCK_SIZE_M': i,
        'BLOCK_SIZE_N': j,
        'BLOCK_SIZE_K': k,
        'GROUP_SIZE_M': l
        },
        num_stages=m,
        num_warps=n
    )
    User could override BLOCK_SIZE_K to a certain chunk_size (must >16).
    """
    test_combinations = [
        (128, 256, 64, 8, 3, 8),
        (64, 256, 32, 8, 4, 4),
        (128, 128, 32, 8, 4, 4),
        (128, 64, 32, 8, 4, 4),
        (64, 128, 32, 8, 4, 4),
        (128, 32, 32, 8, 4, 4),
        (64, 32, 32, 8, 5, 2),
        (32, 64, 32, 8, 5, 2),
        # Good config for fp8 inputs.
        (128, 256, 128, 8, 3, 8),
        (256, 128, 128, 8, 3, 8),
        (256, 64, 128, 8, 4, 4),
        (64, 256, 128, 8, 4, 4),
        (128, 128, 128, 8, 4, 4),
        (128, 64, 64, 8, 4, 4),
        (64, 128, 64, 8, 4, 4),
        (128, 32, 64, 8, 4, 4),
    ]
    return [
        triton.Config(
            {
                "BLOCK_SIZE_M": i,
                "BLOCK_SIZE_N": j,
                "BLOCK_SIZE_K": chunk_size if chunk_size else k,
                "GROUP_SIZE_M": l,
            },
            num_stages=m,
            num_warps=n,
        )
        for i, j, k, l, m, n in test_combinations
    ]


# `triton.jit`'ed functions can be auto-tuned by using the `triton.autotune` decorator,
# which consumes:
#   - A list of `triton.Config` objects that define different configurations of
#       meta-parameters (e.g., `BLOCK_SIZE_M`) and compilation options (e.g., `num_warps`) to try
#   - An auto-tuning *key* whose change in values will trigger evaluation of all the
#       provided configs
# => Need to avoid using auto-tune for real model inference! But for micro-benchmarking purpose, we
#       could enable the decorator below
# @triton.autotune(configs=get_cuda_autotune_config(), key=['M', 'N', 'K'])
@triton.jit
def matmul_kernel(
    # Pointers to matrices
    a_ptr,
    b_ptr,
    c_ptr,
    # Matrix dimensions
    M,
    N,
    K,
    # The stride variables represent how much to increase the ptr by when moving by 1
    # element in a particular dimension. E.g. `stride_am` is how much to increase `a_ptr`
    # by to get the element one row down (A has M rows).
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    chunk_trun_bits,
    max_acc_bits,  # pylint: disable=unused-argument
    clamp_acc_to_dl16,
    truncate_then_accumulate,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    ACTIVATION: tl.constexpr,
):
    """Kernel for computing the matmul C = A x B that include LSB truncation.
    A has shape (M, K), B has shape (K, N) and C has shape (M, N)
    Args:
        chunk_trun_bits (int): number of LSB to truncate/round. [0 to 23]
    """
    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    # See above `L2 Cache Optimizations` section for details.
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # ----------------------------------------------------------
    # Create pointers for the first blocks of A and B.
    # We will advance this pointer as we move in the K direction
    # and accumulate
    # `a_ptrs` is a block of [BLOCK_SIZE_M, BLOCK_SIZE_K] pointers
    # `b_ptrs` is a block of [BLOCK_SIZE_K, BLOCK_SIZE_N] pointers
    # See above `Pointer Arithmetic` section for details
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix.
    # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
    # of fp32 values for higher accuracy.
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    ## ------ prepare LSB rounding/truncation masks outside the for loop -------
    round_bit, trun_mask = round_and_trun_mask(chunk_trun_bits, clamp_acc_to_dl16)

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load the next block of A and B, generate a mask by checking the K dimension.
        # If it is out of bounds, set it to 0.
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        # We accumulate along the K dimension.
        if truncate_then_accumulate:
            accumulator_inner = tl.dot(a, b, input_precision="ieee")
        else:
            accumulator_inner = tl.dot(a, b, accumulator, input_precision="ieee")
        # tl.dot() default is using TF32 approximation, not good enough for LSB truncation exp

        ## ------ add chunky LSB rounding/masking --------
        if clamp_acc_to_dl16 or chunk_trun_bits > 0:
            accumulator_inner = round_and_trun(
                accumulator_inner, round_bit, trun_mask, clamp_acc_to_dl16
            )
        ## ---------------------------------------------------------
        if truncate_then_accumulate:
            accumulator += accumulator_inner
        else:
            accumulator = accumulator_inner

        # Advance the ptrs to the next K block.
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    # You can fuse arbitrary activation functions here
    # while the accumulator is still in FP32!
    if ACTIVATION == "leaky_relu":
        accumulator = leaky_relu(accumulator)

    c = accumulator  # do not cast to (tl.float16) just yet

    # -----------------------------------------------------------
    # Write back the block of the output matrix C with masks.
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


# Reminder: avoid auto-tune for real model inference! But for micro-benchmarking purpose, could
#           enable the decorator below
# @triton.autotune(configs=get_cuda_autotune_config(),key=['M', 'N', 'K'],)
@triton.jit
def imatmul_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    chunk_trun_bits,
    max_acc_bits,
    clamp_acc_to_dl16,  # pylint: disable=unused-argument
    truncate_then_accumulate,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    ACTIVATION: tl.constexpr,
):
    """Kernel for computing the INT matmul D = A x B + C that include LSB truncation and MSB
    clamping. A and B should be INT8, C/D should be INT32. (similar to the float version.)
    A has shape (M, K), B has shape (K, N) and C has shape (M, N)
    Args:
        chunk_trun_bits (int): number of LSBs to truncate/round.
    """
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)

    # accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.int32)
    accumulator = tl.load(c_ptrs, mask=c_mask, other=0.0)
    ## ------ prepare MSB/LSB rounding/truncation masks -------
    round_bit = 1 << (chunk_trun_bits - 1) if chunk_trun_bits > 0 else 0
    acc_min = -(1 << (max_acc_bits - 1))
    acc_max = -acc_min - 1
    ## ---------------------------------------------------------

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        if truncate_then_accumulate:
            accumulator_inner = tl.dot(a, b, input_precision="ieee")
        else:
            accumulator_inner = tl.dot(a, b, accumulator, input_precision="ieee")

        ## ------ INT MSB truncation is simulated by clamping,
        #           "special" INT LSB truncation by right and left shift --------
        if max_acc_bits < 32:
            accumulator_inner = tl.maximum(
                tl.minimum(accumulator_inner, acc_max), acc_min
            )
        if chunk_trun_bits != 0:
            accumulator_inner = (accumulator_inner + round_bit) >> chunk_trun_bits
            accumulator_inner = accumulator_inner << chunk_trun_bits
        ## ---------------------------------------------------------
        if truncate_then_accumulate:
            accumulator += accumulator_inner
        else:
            accumulator = accumulator_inner

        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    if ACTIVATION == "leaky_relu":
        accumulator = leaky_relu(accumulator)

    c = accumulator

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    tl.store(c_ptrs, c, mask=c_mask)


@triton.jit
def matmul_kernel_DABC(
    # Pointers to matrices
    a_ptr,
    b_ptr,
    c_ptr,
    # Matrix dimensions
    M,
    N,
    K,
    # The stride variables represent how much to increase the ptr by when moving by 1
    # element in a particular dimension. E.g. `stride_am` is how much to increase `a_ptr`
    # by to get the element one row down (A has M rows).
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    chunk_trun_bits,
    max_acc_bits,  # pylint: disable=unused-argument
    clamp_acc_to_dl16,
    truncate_then_accumulate,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    ACTIVATION: tl.constexpr,
):
    """Kernel for computing the matmul D = A x B + C that include LSB truncation.
    A has shape (M, K), B has shape (K, N) and C/D has shape (M, N).
    NOTE:
        C should be consistent with accumulator dtype, e.g. fp8xfp8 -> fp32.
        *D ptr is supposed to be the same as C ptr, no need to provide D as arg
        **we can be used C to verify unintended truncation by CUDA as well.
    Args:
        chunk_trun_bits (int): number of LSB to truncate/round. [0 to 23]
    """
    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    # See above `L2 Cache Optimizations` section for details.
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # ----------------------------------------------------------
    # Create pointers for the first blocks of A and B.
    # We will advance this pointer as we move in the K direction
    # and accumulate
    # `a_ptrs` is a block of [BLOCK_SIZE_M, BLOCK_SIZE_K] pointers
    # `b_ptrs` is a block of [BLOCK_SIZE_K, BLOCK_SIZE_N] pointers
    # See above `Pointer Arithmetic` section for details
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)

    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix.
    # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
    # of fp32 values for higher accuracy, i.e. C should have been cast to fp32 already
    accumulator = tl.load(c_ptrs, mask=c_mask, other=0.0)
    ## ------ prepare LSB rounding/truncation masks outside the for loop -------
    round_bit, trun_mask = round_and_trun_mask(chunk_trun_bits, clamp_acc_to_dl16)

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load the next block of A, B, and C, generate a mask by checking the K dimension.
        # If it is out of bounds, set it to 0.
        # D = truncation(A*B) + C
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        # We accumulate along the K dimension. but apply truncation on local A*B first
        if truncate_then_accumulate:
            accumulator_inner = tl.dot(a, b, input_precision="ieee")
        else:
            accumulator_inner = tl.dot(a, b, accumulator, input_precision="ieee")
        # tl.dot() default is using TF32 approximation, not good enough for LSB truncation exp
        # NOTE: tl.dot(a, b, c) should correspond to a CUDA mma instruction, typically "c = a*b+c".
        #       If this mma instruction uses "reduced-precision" under the hood, not only a*b will
        #       be accumulated in that precision, there's a chance c will be cast to that "lower"
        #       precision as well, hence, could lose some precision!

        ## ------ add chunky LSB rounding/masking --------
        if clamp_acc_to_dl16 or chunk_trun_bits > 0:
            accumulator_inner = round_and_trun(
                accumulator_inner, round_bit, trun_mask, clamp_acc_to_dl16
            )
        ## ---------------------------------------------------------
        if truncate_then_accumulate:
            accumulator += accumulator_inner
        else:
            accumulator = accumulator_inner

        # Advance the ptrs to the next K block.
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    # You can fuse arbitrary activation functions here
    # while the accumulator is still in FP32!
    if ACTIVATION == "leaky_relu":
        accumulator = leaky_relu(accumulator)

    d = accumulator  # do not cast to (tl.float16) just yet

    # -----------------------------------------------------------
    # Write back the block of the output to matrix "C" with masks.
    tl.store(c_ptrs, d, mask=c_mask)


@triton.jit
def leaky_relu(x):
    """Activation function that could be fused into matmul kernel"""
    return tl.where(x >= 0, x, 0.01 * x)


@triton.jit
def round_and_trun_mask(chunk_trun_bits, clamp_acc_to_dl16):
    """
    Rounding and LSB truncation masks only need to be generated once.
    These mask will be applied on "inner" accumulator, which is alway FP32 (e8m23). We may truncate
    up to 23b for mantissa. If DL16/DL8, pay attention to exponent bias.
    Examples: 20b -> trun_mask = 0xFFF00000, round_bit = 0x00080000
               8b -> trun_mask = 0xFFFFFF00, round_bit = 0x00000080
    """
    if clamp_acc_to_dl16:
        # DL16 is e6m9, hence, truncate 23 - 9 = 14 bits
        chunk_trun_bits = 14
    round_bit = 1 << (chunk_trun_bits - 1) if chunk_trun_bits > 0 else 0
    trun_mask = ~tl.cast((1 << chunk_trun_bits) - 1, tl.uint32)
    return round_bit, trun_mask


@triton.jit
def round_and_trun(x, round_bit, trun_mask, clamp_acc_to_dl16):
    """Round and truncate (usually for accumulator)."""
    x = libdevice.uint_as_float((libdevice.float_as_uint(x) + round_bit) & trun_mask)  # pylint: disable=assignment-from-no-return

    if clamp_acc_to_dl16:
        # clamp to DL16 min/max:
        #   max = 2^32 * 1.(1111 1111 0)_base2 = 2^32*(2 - 2^-9) = 8581545984.0
        #         greater than this will become +inf (or -inf)
        #   min = 2^-31 * 1.(0000 0000 1)_base2 = 2^-31*(1 + 2^-9)> = 4.665707820095122e-10
        #         smaller than this will become 0
        dl16_max = 8581545984.0
        dl16_min = 4.665707820095122e-10
        x = tl.where(x >= dl16_max, float("inf"), x)
        x = tl.where(x <= -dl16_max, float("-inf"), x)
        x = tl.where(tl.abs(x) < dl16_min, 0, x)
    return x


def tl_matmul_chunk_truncate(
    a,
    b,
    c=None,
    activation="",
    chunk_trun_bits=0,
    chunk_size=16,
    max_acc_bits=32,
    truncate_then_accumulate=True,
    cast_output_to_input_dtype=None,
    clamp_acc_to_dl16=False,
):
    """Triton matmul for HW behavior simulation. Supports float and int8.
    i. variable chunk size (i.e., BLOCK_SIZE_K)
    ii. LSB truncation, must <23 if using float.
    iii. assume D = A*B + C, where C is optional. If C exists, it will be updated inplace.

    Args:
        a, b: input tensors. FloatX, X in [32, 16, 8] or INT8.
        activation (str, optional): activation func to be fused, see relu example.
        chunk_trun_bits (int, optional): number of LSBs to be truncated/rounded.
        chunk_size (int, optional): BLOCK_SIZE_K, some HW has specific chunk size. must >= 16.
        max_acc_bits (int, optional): num of bits for the accumulator, e.g. if INT24 is used, will
                                        clamp each chunk of a*b to [-2**23-1, 2**23].
                                        (only used by INT)
        clamp_acc_to_dl16(bool): Only used by FP8, whether to clamp local accumulator (FP32) to DL16
        truncate_then_accumulate (bool, optional): if True, c = truncate(a*b) + c, otherwise
                                                    c = truncate(a*b+c)
        cast_output_to_input_dtype (bool, optional): accumulator has higher prec than input, usually
                                                    FP32 or INT32. by default we cast the final
                                                    output to the same dtype as input for non-8bits.

    Returns:
        _type_: _description_

    NOTE:
    use empirical way to determine BLOCK sizes, may not be optimal. But need to avoid autotune for
    real model inference. otherwise auto-tune may be triggered in every forward call.
    """

    # Check constraints.
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.is_contiguous(), "Matrix A must be contiguous"
    assert a.dtype == b.dtype, "Input dtypes inconsistent"

    if cast_output_to_input_dtype is None:
        cast_output_to_input_dtype = a.dtype not in DTYPE_8BIT
    allowed_dtypes = [torch.float, torch.bfloat16, torch.float16]
    cuda_cc = torch.cuda.get_device_capability()
    if cuda_cc[0] >= 8:
        allowed_dtypes += DTYPE_I8
    if cuda_cc[0] >= 9 or cuda_cc == (8, 9):
        allowed_dtypes += DTYPE_F8
    assert a.dtype in allowed_dtypes, "Input dtype is not supported"

    # Allocates output, always accumulate in FP32 (if floats) or INT32 then cast
    def isPowerofTwo(x):
        """triton-specific limitation: block size needs to be power of 2."""
        return (x & (x - 1)) == 0

    min_chunk_size = 32 if a.dtype in DTYPE_8BIT else 16

    # because min k (chunk size in this case) for fp16/bf16 is 16, if smaller is needed, we could
    # insert 0s in between elements, e.g. pad [m,k] -> [m,2k], [k,n]->[2k,n], out=[m,n] unchanged.
    if chunk_size == 8 and a.dtype in [
        torch.float8_e4m3fn,
        torch.int8,
        torch.float16,
        torch.bfloat16,
    ]:
        exp_ratio = min_chunk_size // chunk_size
        a_padded = torch.zeros(
            a.shape[0], a.shape[1] * exp_ratio, dtype=a.dtype, device=a.device
        )
        a_padded[:, ::exp_ratio] = a
        a = a_padded
        b_padded = torch.zeros(
            b.shape[0] * exp_ratio, b.shape[1], dtype=b.dtype, device=b.device
        )
        b_padded[::exp_ratio, :] = b
        b = b_padded
        chunk_size = min_chunk_size
    else:
        chunk_size = (
            max(chunk_size, min_chunk_size)
            if isPowerofTwo(chunk_size)
            else min_chunk_size
        )

    M, K = a.shape
    K, N = b.shape
    if a.dtype in DTYPE_I8:
        acc_dtype = torch.int32
        mm_kernel = imatmul_kernel
    else:
        acc_dtype = torch.float32
        mm_kernel = matmul_kernel if c is None else matmul_kernel_DABC
        assert chunk_trun_bits < 23, "FP32 accumulator only has 23 mantissa bits"

    if c is None:
        c_org_dtype = a.dtype
        c = torch.zeros((M, N), device=a.device, dtype=acc_dtype)
    else:
        # if C is in fp16, accumulate in fp32 no matter what, decide whether to cast back later
        c_org_dtype = c.dtype
        c = c.to(acc_dtype)
        assert c.shape[0] == M and c.shape[1] == N, "C shape is inconsistent with A B."

    # 1D launch kernel where each block gets its own program.
    def grid(META):
        return (
            triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
        )

    if M < 1024 or N < 1024:
        kernel_config = {
            "BLOCK_SIZE_M": 64,
            "BLOCK_SIZE_K": chunk_size,
            "BLOCK_SIZE_N": 32,
            "GROUP_SIZE_M": 8,
            "num_warps": 2,
            "num_stages": 5,
        }
    else:
        kernel_config = {
            "BLOCK_SIZE_M": 128,
            "BLOCK_SIZE_K": chunk_size,
            "BLOCK_SIZE_N": 64,
            "GROUP_SIZE_M": 8,
            "num_warps": 4,
            "num_stages": 4,
        }

    mm_kernel[grid](
        a,
        b,
        c,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
        chunk_trun_bits=chunk_trun_bits,
        max_acc_bits=max_acc_bits,
        clamp_acc_to_dl16=clamp_acc_to_dl16,
        truncate_then_accumulate=truncate_then_accumulate,
        ACTIVATION=activation,
        **kernel_config,  # if using auto-tune, comment this line out.
    )
    return c.to(c_org_dtype) if cast_output_to_input_dtype else c
