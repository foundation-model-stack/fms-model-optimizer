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


"""This file contains external kernel registrations, compilation, and packing functions.
Some functions may require additional packages, e.g. gptqmodel, cutlass (source clone)
"""

# pylint: disable=ungrouped-imports,unused-argument,c-extension-no-member
# disable unused args because of torch op registration signature

# Standard
from functools import partial
from pathlib import Path
import logging
import os
import time

# Third Party
from packaging.version import Version
from transformers.pytorch_utils import Conv1D
import numpy as np
import torch
import torch.utils.cpp_extension as cpp_ext

# Local
from fms_mo.utils.import_utils import available_packages
from fms_mo.utils.utils import default_device_selection

MIN_BLOCK_SIZE = 5
disable_torchtrt = True

logger = logging.getLogger(__name__)

DEFAULT_DEVICE = default_device_selection()

# PT version check and choose proper registration decorators/functions
# basically 2.1 doesn't have a functional op reg, only supports decorator
#   and 2.3 only has functional, no decorator, 2.4+ has both
pt_ver = Version(torch.__version__)
if pt_ver == Version("2.1"):
    # Third Party
    import torch._custom_ops as custom_ops

    reg_op = custom_ops.custom_op
    kernel_impl = custom_ops.impl
    reg_fake = custom_ops.impl_abstract
    reg_op_func = None

elif pt_ver < Version("2.4"):
    # Third Party
    import torch._custom_ops as custom_ops
    import torch.library as lib

    reg_op = (
        custom_ops.custom_op
    )  # NOTE 2.3 doesn't seem to have this in torch.library.
    reg_op_func = lib.define  # NOTE this is func, not decorator
    kernel_impl = lib.impl
    reg_fake = lib.impl_abstract

elif pt_ver >= Version("2.4"):
    # Third Party
    import torch.library as lib

    reg_op = partial(lib.custom_op, mutates_args=())
    reg_op_func = lib.define  # NOTE this is func, not decorator
    kernel_impl = lib.register_kernel
    reg_fake = lib.register_fake

else:
    raise RuntimeError("Custom Op registration only works for >PT2.1")


def cublas_ops_load_and_reg(run_unit_test=False):
    """Compile and register a few cuBlas GEMM functions under torch.ops.cublas_gemm namespace.
    We could use:
    1. torch.utils.cpp_extension.load to compile a cpp file, or
    2. torch.utils.cpp_extension.load_inline() and "string" source codes instead of a file
    Args:
        run_unit_test: bool. Run unit tests after Op registration. (if unit tests defined.)

    NOTE:
        i. can add "extra_cflags", such as "-fpermissive"
        ii. if is_python_module set to True, PYBIND11 is needed in the C++ source
        iii. PyTorch <2.1 does not support custom Op registration, and this registration signature
            evolved a lot from 2.1 to 2.4. TODO: Need to add 2.4 support.
        iv. In some cases, torch.compile seems to convert some of the (x, y) into int32 for
            QMatmulDebug. we need to force ".float().contiguous()" for debug purpose
        v. need to make sure CUDA_HOME is set (and correctly)

    """

    if hasattr(torch.ops.cublas_gemm, "gemm_nt_i8i32"):
        logger.info("Our custom cublas functions have been registered already!")
        need_registration = False

    else:
        if os.environ.get("CUDA_HOME", None) in [None, "", "None"]:
            os.environ["CUDA_HOME"] = "/usr/local/cuda"

        t0 = time.perf_counter()
        cpp_ext.load(
            name="cublas_gemm",
            sources=[
                f"{os.path.dirname(__file__)}/custom_ext_kernels/ext_kernel_cublas.cpp"
            ],
            extra_cflags=["-O3"],
            is_python_module=False,
            with_cuda=True,
            verbose=False,
        )
        compile_time = time.perf_counter() - t0
        logger.info(f"Inline compile for CUBLAS took {compile_time:.4f} sec.")
        need_registration = True

    if run_unit_test:
        m = 2
        n = 64
        k = 64

        X = torch.randint(-8, 7, (m, k), dtype=torch.int8, device="cuda")
        X_t = X.t().contiguous()
        W = torch.randint(-8, 7, (k, n), dtype=torch.int8, device="cuda")
        W_t = W.t().contiguous()

        # torch.matmul or "@" does not support int8
        out_base = X.type(torch.int).to("cpu") @ W.type(torch.int).to("cpu")
        out_base = out_base.type(torch.int).to("cuda")
        logger.info(f"out_base:\n{out_base}")

        logger.info("==f32==")
        out = torch.ops.cublas_gemm.gemm_nn_f32(X, W).type(torch.int)
        logger.info(f"out_nn:\n{out}")
        logger.info(f"close to out_base? {torch.allclose(out_base, out)}")

        out = torch.ops.cublas_gemm.gemm_tn_f32(X_t, W).type(torch.int)
        logger.info(f"out_tn:\n{out}")
        logger.info(f"close to out_base? {torch.allclose(out_base, out)}")

        out = torch.ops.cublas_gemm.gemm_nt_f32(X, W_t).type(torch.int)
        logger.info(f"out_nt:\n{out}")
        logger.info(f"close to out_base? {torch.allclose(out_base, out)}")

        out = torch.ops.cublas_gemm.gemm_tt_f32(X_t, W_t).type(torch.int)
        logger.info(f"out_tt:\n{out}")
        logger.info(f"close to out_base? {torch.allclose(out_base, out)}")

        logger.info("==i32==")
        out = torch.ops.cublas_gemm.gemm_nt_i32(X, W_t)
        logger.info(f"out_nt:\n{out}")
        logger.info(f"close to out_base? {torch.allclose(out_base, out)}")

        logger.info("==i8i32==")
        out = torch.ops.cublas_gemm.gemm_nt_i8i32(X, W_t)
        logger.info(f"out_nt:\n{out}")
        logger.info(f"close to out_base? {torch.allclose(out_base, out)}")

        logger.info("==i8i8==")
        out = torch.ops.cublas_gemm.gemm_nt_i8i8(X, W_t)
        logger.info(f"out_nt:\n{out}")
        logger.warning(
            "Output is not checked against base because output bitwidth is smaller"
        )

    # register ops if needed
    if need_registration is False:
        return

    namespace = "mygemm"
    if hasattr(torch.ops, "mygemm") and hasattr(torch.ops.mygemm, "gemm_nt_i8i32"):
        logger.info("gemm_nt_i8i32 has been registered already!")
        return

    @reg_op(f"{namespace}::gemm_nt_i8i32")
    def gemm_nt_i8i32(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Inputs have to be 2D -> [m,n] @ ([p,n]).t(), i.e. y is transposed"""
        return NotImplemented

    @kernel_impl(f"{namespace}::gemm_nt_i8i32")
    def gemm_nt_i8i32_impl(x, y):
        """Inputs have to be 2D -> [m,n] @ ([p,n]).t(), i.e. y is transposed"""
        return torch.ops.cublas_gemm.gemm_nt_i8i32(x, y)

    @reg_fake(f"{namespace}::gemm_nt_i8i32")
    def gemm_nt_i8i32_impl_abstract(x, y):
        """Inputs have to be 2D -> [m,n] @ ([p,n]).t(), i.e. y is transposed"""
        assert x.dtype == y.dtype
        assert x.shape[-1] == y.shape[-1]
        new_shape = list(x.shape)
        new_shape[-1] = y.shape[-2]
        dtype = x.dtype if x.dtype != torch.int8 else torch.float
        return torch.empty(
            tuple(new_shape), dtype=dtype, device=x.device, requires_grad=False
        )

    @reg_op(f"{namespace}::gemm_nt_f32")
    def gemm_nt_f32(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Inputs have to be 2D -> [m,n] @ ([p,n]).t(), i.e. y is transposed"""
        return NotImplemented

    @kernel_impl(f"{namespace}::gemm_nt_f32")
    def gemm_nt_f32_impl(x, y):
        """Inputs have to be 2D -> [m,n] @ ([p,n]).t(), i.e. y is transposed"""
        return torch.ops.cublas_gemm.gemm_nt_f32(
            x.float().contiguous(), y.float().contiguous()
        )

    @reg_fake(f"{namespace}::gemm_nt_f32")
    def gemm_nt_f32_impl_abstract(x, y):
        """Inputs have to be 2D -> [m,n] @ ([p,n]).t(), i.e. y is transposed"""
        assert x.dtype == y.dtype
        assert x.shape[-1] == y.shape[-1]
        new_shape = list(x.shape)
        new_shape[-1] = y.shape[-2]
        return torch.empty(
            tuple(new_shape), dtype=torch.float, device=x.device, requires_grad=False
        )

    @reg_op(f"{namespace}::gemm_nn_f32")
    def gemm_nn_f32(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Inputs has to be 2D -> [m,n] @ [n,p], i.e. y is NOT transposed"""
        return NotImplemented

    @kernel_impl(f"{namespace}::gemm_nn_f32")
    def gemm_nn_f32_impl(x, y):
        """Inputs has to be 2D -> [m,n] @ [n,p], i.e. y is NOT transposed"""
        return torch.ops.cublas_gemm.gemm_nn_f32(x, y)

    @reg_fake(f"{namespace}::gemm_nn_f32")
    def gemm_nn_f32_impl_abstract(x, y):
        """Inputs has to be 2D -> [m,n] @ [n,p], i.e. y is NOT transposed"""
        assert x.dtype == y.dtype
        assert x.shape[1] == y.shape[0]
        new_shape = list(x.shape)
        new_shape[1] = y.shape[1]
        return torch.empty(
            tuple(new_shape), dtype=torch.float, device=x.device, requires_grad=False
        )

    logger.info(
        f"A few CUBLAS gemm functions have been loaded and registered under torch.ops.{namespace}."
    )
    return


def cutlass_ops_load_and_reg(qcfg=None, run_unit_test=False):
    """Compile and register CUTLASS kernels
    The name of i8i32 is because the output tensor from this kernel is i32, cast if needed.
    Args:
        qcfg: dict. quant config
        run_unit_test: bool. Run unit tests after Op registration. (if unit tests defined.)

    The compile requires include/ folder from "cutlass" source, please git clone if needed.
    NOTE:
        1. package by "pip install nvidia-cutlass" does not come with include folder...
        2. this version can only do 2D tensor x 2D tensor, and N T layout only. batched tensor will
            result in wrong answers.
        3. Op registration signature changed drastically from PT 2.1 to 2.4. TODO: add 2.4 support

    Search priority is:
        1. respect user defined location thru os.environ["CUTLASS_PATH"]
        2. check user home dir, e.g., '/home/userId/cutlass'
        3. unlikely to work, but check site-package/cutlass

    CUTLASS Python API reference: https://github.com/NVIDIA/cutlass/tree/main/python
    """

    if qcfg is None:
        qcfg = {}
    else:
        qcfg["CUTLASS_AVAILABLE"] = False

    if hasattr(torch.ops, "cutlass_gemm") and hasattr(
        torch.ops.cutlass_gemm, "i8i32nt"
    ):
        logger.info("Our custom CUTLASS gemm functions have been loaded already!")
        qcfg["CUTLASS_AVAILABLE"] = True

    else:
        # compile and register the new Ops
        os.environ["PATH"] = os.environ["PATH"] + ":/usr/local/cuda/bin"
        os.environ["TORCH_CUDA_ARCH_LIST"] = "8.0"
        os.environ["CUDA_HOME"] = "/usr/local/cuda"

        fms_mo_root_dir = Path(__file__).parent.parent
        site_pkg_dir = Path(torch.__file__).parent.parent
        source_dir = fms_mo_root_dir / "custom_ext_kernels"
        name = "cutlass_i8i32gemm"
        cuda_file = source_dir / f"{name}.cu"
        cpp_file = os.path.join(source_dir.absolute(), name + ".cpp")

        if "CUTLASS_PATH" in os.environ and os.environ["CUTLASS_PATH"] is not None:
            CUTLASS_PATH = os.environ["CUTLASS_PATH"]
        elif (Path.home() / "cutlass").exists() and (
            Path.home() / "cutlass" / "include"
        ).exists():
            CUTLASS_PATH = Path.home() / "cutlass"
        elif (site_pkg_dir / "cutlass").exists() and (
            site_pkg_dir / "cutlass" / "include"
        ).exists():
            CUTLASS_PATH = site_pkg_dir / "cutlass"
        else:
            raise RuntimeError(
                "Cannot find CUTLASS_PATH! Please set Env Var or git clone the cutlass "
                "repo to user home directory.\n    We need cutlass/include/ for building kernels!"
            )
        logger.info(f"Found CUTLASS include path under {CUTLASS_PATH}.")

        extra_cuda_cflags = ["-std=c++17"]
        plan_i8i32_tensorop = cpp_ext.load(
            "cutlass_mm",
            [cpp_file, cuda_file],
            extra_cuda_cflags=extra_cuda_cflags,
            extra_include_paths=[
                os.path.join(CUTLASS_PATH, "include"),
                os.path.join(CUTLASS_PATH, "tools/util/include"),
            ],
            verbose=True,
        )

        # Register op
        namespace = "cutlass_gemm"

        @reg_op(f"{namespace}::i8i32nt")
        def i8i32nt(m1: torch.Tensor, m2: torch.Tensor) -> torch.Tensor:
            """Inputs should be [m,k] and [k,n] with NT layout"""
            return NotImplemented

        # Generic implementation
        @kernel_impl(
            f"{namespace}::i8i32nt", "cuda"
        )  # kw changed from "types" to "device_types"
        def i8i32nt_impl(m1, m2):
            """Inputs should be [m,k] and [k,n] with NT layout"""
            CDshape = (m1.shape[0], m2.shape[1])
            C = torch.empty(
                CDshape, dtype=torch.int32, device="cuda", requires_grad=False
            )
            assert m1.shape[1] == m2.shape[0]
            D = plan_i8i32_tensorop.run(m1, m2, C)
            return D

        # Abstract implementation
        @reg_fake(f"{namespace}::i8i32nt")
        def i8i32nt_impl_abstract(m1, m2):
            """Inputs should be [m,k] and [k,n] with NT layout"""
            assert m1.dtype == m2.dtype == torch.int8
            assert m1.is_cuda and m2.is_cuda
            out_shape = (m1.shape[0], m2.shape[1])
            return torch.empty(
                out_shape, dtype=torch.int32, device="cuda", requires_grad=False
            )

        logger.info(
            "A few CUTLASS gemm functions have been loaded and registered to "
            f"torch.ops.{namespace}."
        )

        # Register op
        namespace = "cutlass"

        @reg_op(f"{namespace}::conv2di8i32nt")
        def conv2di8i32nt(
            inputs: torch.Tensor,
            weight: torch.Tensor,
            stride: int,
            padding: int,
            dilation: int,
        ) -> torch.Tensor:
            return NotImplemented

        # Generic implementation
        @kernel_impl(f"{namespace}::conv2di8i32nt", "cuda")
        def conv2di8i32nt_impl(inputs, weight, stride, padding, dilation):
            """
            Supported param types does not include Tuple which is the default type in PT for stride,
            padding, dilation. Have to accept int and then expand to Tuple.
            """
            CDshape = (inputs.shape[0], weight.shape[1])
            stride = (stride, stride)
            padding = (padding, padding)
            dilation = (dilation, dilation)
            # [optional] allocate an empty tensor for output
            C = torch.empty(
                CDshape, dtype=torch.int32, device="cuda", requires_grad=False
            )

            conv_output = plan_i8i32_tensorop.run_conv2d(
                inputs,
                weight,
                C,
                stride,
                padding,
                dilation,
                alpha=1,
                beta=0,
                split_k_mode="serial",
                split_k_slices=1,
                output_nchw_chlast=True,
            )
            return conv_output

        # Abstract implementation
        @reg_fake(f"{namespace}::conv2di8i32nt")
        def conv2di8i32nt_impl_abstract(
            inputs, weight, stride, padding, dilation
        ):  # inputs are [m,k] and [k,n] with NT layout
            assert inputs.dtype == weight.dtype == torch.int8
            assert inputs.is_cuda and weight.is_cuda
            out_shape = (inputs.shape[0], weight.shape[1])
            return torch.empty(
                out_shape, dtype=torch.int32, device="cuda", requires_grad=False
            )

        logger.info(
            "A few CUTLASS conv2d functions have been loaded and registered to "
            f"torch.ops.{namespace}."
        )
        if qcfg:
            qcfg["CUTLASS_AVAILABLE"] = True

    if run_unit_test:

        def create_test_tensors(Nbatch, M, N, K, ele_type, accum_type):
            """Keep in mind that we need [m,k]@[k,n] with N T layout for TensorCores, meaning B
            tensor is contiguous at [n,k] then .t()
            """
            if Nbatch in [0, 1]:
                Ashape = (M, K)
                Bshape = (K, N)
                CDshape = (M, N)
            else:
                Ashape = (Nbatch, M, K)
                Bshape = (Nbatch, K, N)
                CDshape = (Nbatch, M, N)

            A = torch.randint(
                -8, 7, Ashape, dtype=ele_type, device="cuda", requires_grad=False
            )
            B = torch.randint(
                -8, 7, Bshape, dtype=ele_type, device="cuda", requires_grad=False
            )
            C = torch.zeros(
                CDshape, dtype=accum_type, device="cuda", requires_grad=False
            )
            D = torch.zeros(
                CDshape, dtype=accum_type, device="cuda", requires_grad=False
            )

            return A, B, C, D

        elem_type = torch.int8
        accum_type = torch.int32
        bs = 1
        m, n, k = 512, 512, 512

        # test 1, k = 4m
        A_i8, B_i8, C_i8, D_i8 = create_test_tensors(
            bs, m, n, 4 * m, elem_type, accum_type
        )
        BT_i8 = B_i8.permute(0, 2, 1) if len(B_i8.shape) == 3 else B_i8.t()
        D_i8 = plan_i8i32_tensorop.run(A_i8, BT_i8.contiguous().t(), C_i8)
        D_torch = torch.matmul(A_i8.float(), B_i8.float())
        logger.info(
            f"[CUTLASS i8i8 Test 1] [{m},{4*m}]@[{4*m},{n}] numerical error",
            torch.norm(D_i8.float() - D_torch),
        )

        A_i8, B_i8, C_i8, D_i8 = create_test_tensors(
            bs, 4 * m, 4 * n, k, elem_type, accum_type
        )
        BT_i8 = B_i8.permute(0, 2, 1) if len(B_i8.shape) == 3 else B_i8.t()
        D_i8 = plan_i8i32_tensorop.run(A_i8, BT_i8.contiguous().t(), C_i8)
        D_torch = torch.matmul(A_i8.float(), B_i8.float())
        logger.info(
            f"[CUTLASS i8i8 Test 2] [{4*m},{k}]@[{k},{4*n}] numerical error",
            torch.norm(D_i8.float() - D_torch),
        )


def exllama_ops_load_and_reg(qcfg=None, run_unit_test=False):
    """Register Exllama kernels borrowed from gptqmodel
    Args:
        qcfg: dict. quant config
        run_unit_test: bool. Run unit tests after Op registration. (if unit tests defined.)

    NOTE:
        1. need to install gptqmodel python package
        2. Op registration signature changed drastically from torch 2.1 - 2.4. TODO: add 2.4 support

    see https://github.com/ModelCloud/GPTQModel for installation instructions
    """
    if qcfg is None:
        qcfg = {}
    elif qcfg:
        qcfg["GPTQMODEL_AVAILABLE"] = False

    namespace = "gptqmodel_gemm"
    # check before compile
    if hasattr(torch.ops, namespace) and hasattr(
        torch.ops.gptqmodel_gemm, "exv1_i4f16"
    ):
        logger.info("Custom GPTQModel functions have been loaded already!")
        qcfg["GPTQMODEL_AVAILABLE"] = True
        need_registration = False
    else:
        need_registration = (
            available_packages["gptqmodel_exllama_kernels"]
            and available_packages["gptqmodel_exllamav2_kernels"]
        )

        if not need_registration:
            logger.warning(
                "Please check the installation of GPTQModel package."
                "External kernels cannot be used this time."
            )
            return

        # Third Party
        import gptqmodel_exllama_kernels
        import gptqmodel_exllamav2_kernels

        # Register op
        @reg_op(f"{namespace}::exv1_i4f16")
        def exv1_i4f16(x: torch.Tensor, q4: int, q4_width: int) -> torch.Tensor:
            """q4 is the handle, i.e. INT "address", to packed weight, not a tensor"""

        # Generic implementation
        @kernel_impl(f"{namespace}::exv1_i4f16", "cuda")
        def exv1_i4f16_impl(x, q4, q4_width):
            """q4 is the handle, i.e. INT "address", to packed weight, not a tensor"""
            outshape = x.shape[:-1] + (q4_width,)
            x = x.view(-1, x.shape[-1])
            output = torch.empty(
                (x.shape[0], q4_width), dtype=torch.float16, device=x.device
            )

            gptqmodel_exllama_kernels.q4_matmul(x, q4, output)
            return output.view(outshape)

        # Abstract implementation
        @reg_fake(f"{namespace}::exv1_i4f16")
        def exv1_i4f16_abstract(x, q4, q4_width):
            outshape = x.shape[:-1] + (q4_width,)
            return torch.empty(
                outshape, dtype=torch.float16, device=x.device, requires_grad=False
            )

        # Register op
        @reg_op(f"{namespace}::exv2_i4f16")
        def exv2_i4f16(
            x: torch.Tensor, q4: int, q4_width: int, force_cuda: bool
        ) -> torch.Tensor:
            """q4 is the handle, i.e. INT "address", to packed weight, not a tensor"""

        # Generic implementation
        @kernel_impl(f"{namespace}::exv2_i4f16", "cuda")
        def exv2_i4f16_impl(x, q_handle, q4_width, force_cuda):
            """q4 is the handle, i.e. INT "address", to packed weight, not a tensor"""
            outshape = x.shape[:-1] + (q4_width,)
            x = x.view(-1, x.shape[-1])
            output = torch.empty(
                (x.shape[0], q4_width), dtype=torch.float16, device=x.device
            )

            gptqmodel_exllamav2_kernels.gemm_half_q_half(
                x, q_handle, output, force_cuda
            )
            return output.view(outshape)

        # Abstract implementation
        @reg_fake(f"{namespace}::exv2_i4f16")
        def exv2_i4f16_abstract(x, q4, q4_width, force_cuda):
            outshape = x.shape[:-1] + (q4_width,)
            return torch.empty(
                outshape, dtype=torch.float16, device=x.device, requires_grad=False
            )

        # Wrappers for better graph representation, i.e. try to pass real tensors instead of just
        # one handle, even though the extra terms will not be used in the external Kernel...
        @reg_op(f"{namespace}::exv2_i4f16_fxinputs")
        def exv2_i4f16_fxinputs(
            x: torch.Tensor,
            qw: torch.Tensor,
            qzeros: torch.Tensor,
            scales: torch.Tensor,
            g_idx: torch.Tensor,
            q4: int,
            q4_width: int,
            force_cuda: bool,
        ) -> torch.Tensor: ...

        # Generic implementation
        @kernel_impl(f"{namespace}::exv2_i4f16_fxinputs", "cuda")
        def exv2_i4f16_fxinputs_impl(
            x, qw, qzeros, scales, g_idx, q_handle, q4_width, force_cuda
        ):
            outshape = x.shape[:-1] + (q4_width,)
            x = x.view(-1, x.shape[-1])
            output = torch.empty(
                (x.shape[0], q4_width), dtype=torch.float16, device=x.device
            )

            gptqmodel_exllamav2_kernels.gemm_half_q_half(
                x, q_handle, output, force_cuda
            )
            return output.view(outshape)

        # Abstract implementation
        @reg_fake(f"{namespace}::exv2_i4f16_fxinputs")
        def exv2_i4f16_fxinputs_abstract(
            x, qw, qzeros, scales, g_idx, q4, q4_width, force_cuda
        ):
            outshape = x.shape[:-1] + (q4_width,)
            return torch.empty(
                outshape, dtype=torch.float16, device=x.device, requires_grad=False
            )

        logger.info(
            f"New GPTQModel gemm functions have been loaded and registered to \
            torch.ops.{namespace}."
        )
        if qcfg:
            qcfg["GPTQMODEL_AVAILABLE"] = True

    if run_unit_test:
        return NotImplemented


def imatmul_ops_reg(
    useINTkernel="triton",
    mm_func=torch.matmul,
    AB_dtype=torch.float,
    D_dtype=torch.float,
):
    """This function will register a dummy Q_imatmul Op for better "graph representation".
    Args:
        useINTkernel: str|bool. ["cutlass", "triton", False]. choose to use a) real INT matmul, e.g.
                    cutlass or triton kernel or b) "simulated" imatmul using torch.matmul.
                    For b), could use D_dtype to select fp16 or fp32 accumulation
        mm_func: matmul func to be used when useINTkernel is True, should be a real callable kernel
                from cutlass, but for debug purpose, could use torch.matmul as well.
        AB_dtype: datatype for input tensors
        D_dtype: datatype for accumulation and output tensor

    NOTE:
        1. dQ and correction term needs to be handled in the QLinear module
        2. using PT2.2 syntax, won't be compatible with PT2.1
        3. Although we already registered torch.ops.cutlass_gemm.i8i32nt, which would call the same
            backend "mm_func", we DO NOT want to call i8i32nt here because it will create an extra
            layer of wrapper, i.e., fms_mo.imatmul -> cutlass_gemm.i8i32nt -> real imm func,
            instead, we want  fms_mo.imatmul ->                         real imm func,

    """
    if hasattr(torch.ops, "fms_mo") and hasattr(torch.ops.fms_mo, "imatmul"):
        logger.info("imatmul dummy op has already been registered.")
        return

    if pt_ver > Version("2.1"):
        # Functional form Op registration should work for > 2.1, even though decorator form is
        # recommended for 2.4+
        reg_op_func("fms_mo::imatmul", "(Tensor m1, Tensor m2) -> Tensor")
        reg_op_func(
            "fms_mo::iaddmm",
            "(Tensor bias, Tensor m1, Tensor m2, Tensor scale_i, Tensor scale_w) -> Tensor",
        )
        reg_op_func(
            "fms_mo::q_iaddmm_dq",
            "(Tensor bias, Tensor m1, Tensor m2, Tensor scale_i, "
            "Tensor zp_i, Tensor scale_w) -> Tensor",
        )
        reg_op_func("fms_mo::q_per_t_sym", "(Tensor x, Tensor s, Tensor zp) -> Tensor")
    elif pt_ver == Version("2.1"):
        raise NotImplementedError(
            "PyTorch 2.1 could support custom Ops, but we've only implemented for 2.2+ for this Op."
        )

    @kernel_impl("fms_mo::imatmul", "default")
    def imatmul(m1, m2):
        """Support 1) real INT matmul and 2) simulated "INT-like" FP matmul
        m1.shape = [m,k], m2.shape = [k,n] (i.e. already transposed back to "normal dim"), real
        memory layout should still be NT, check stride if needed.

        Case 1: assume m1 is INT8 (shifted from UINT8 already), m2 is INT8
        Case 2: m1, m2 are INT-like float for simulation. In some cases HF set accumulation to fp16
                , which overflowed easily. Need to create a fp32 output placeholder tensor to
                enforce fp32 accumulation

        NOTE: If m1 is 3D, i.e. [b,m,k], reshape to [b*m,k] for matmul then reshape back to [b,m,n]
                before return

        """

        re_shape = (-1, m1.shape[-1])
        tar_shape = tuple(m1.shape[:-1]) + (m2.shape[1],)
        m1 = m1.view(re_shape)

        if useINTkernel in ["triton", "cutlass"]:
            assert (
                m1.dtype == torch.int8 and m2.dtype == torch.int8
            ), "When using int matmul, inputs must be 2D and INT8."
            return mm_func(m1, m2).reshape(tar_shape)

        outf32_or_f16 = torch.empty(
            (m1.shape[0], m2.shape[1]), dtype=D_dtype, device=m1.device
        )
        mm_func(m1.to(AB_dtype), m2.to(AB_dtype), out=outf32_or_f16)
        return outf32_or_f16.to(torch.int32).reshape(tar_shape)

    @reg_fake("fms_mo::imatmul")
    def imatmul_abstract(m1, m2):
        tar_shape = tuple(m1.shape[:-1]) + (m2.shape[1],)
        return torch.empty(
            tar_shape, dtype=torch.int32, device=m1.device, requires_grad=False
        )

    @kernel_impl("fms_mo::iaddmm", "default")
    def iaddmm(bias, m1, m2, scale_i, scale_w):
        """This iaddmm should share the same signaure with torch.addmm, but, normal addmm is
                out = beta*bias + alpha*m1@m2
        here we use scale_input and scale_w as alpha, beta
        NOTE: m1 = x, m2 = W.t(), and x has been reshape to 2D already, i.e. m1.shape=[m,k] and
              m2.shape=[k,n], mem layout = NT
        """
        return scale_i.float() * torch.ops.fms_mo.imatmul(m1, m2) * scale_w + bias

    @reg_fake("fms_mo::iaddmm")
    def iaddmm_abstract(bias, m1, m2, scale_i, scale_w):
        return torch.empty(
            (m1.shape[0], m2.shape[1]),
            dtype=torch.float,
            device=m1.device,
            requires_grad=False,
        )

    @kernel_impl("fms_mo::q_iaddmm_dq", "default")
    def q_iaddmm_dq(bias, m1, m2, scale_i, zp_i, scale_w):
        """Take float tensors m1 and INT tensor m2, quantize m1 to INT8 -> imatmul -> dequant back
        to float then output.

        Args:
            m1 (fp16 or fp32): 2D tensor, activations. [m,k]
            m2 (int8): INT weights, already transposed to [k,n] and mem layout is column-major

        NOTE:
            1. When using torch.matmul as imatmul, be caureful about accu precision and overflow. In
                some cases HF set accumulation to fp16, which overflowed imatmul easily. To avoid
                that, create a fp32 out tensor to enforce fp32 accum.
            2. somehow native torch.quantize_per_tensor() is compatible with torch.compile in PT2.2,
                hence we have to add a wrapper for it here.

        """

        m1_dtype = m1.dtype  # should be fp16 or fp32

        # Q(m1) to INT8 (-128 to 127, not 0 to 255 UINT8)
        assert m2.dtype == torch.int8, f"weight tensor is of incorrect dtype {m2.dtype}"
        m1 = torch.clamp((m1 / scale_i + zp_i - 128).round(), -128, 127).to(torch.int8)

        if useINTkernel:
            mm_i32 = mm_func(m1, m2)
        else:
            outf32_or_f16 = torch.empty(
                (m1.shape[0], m2.shape[1]), dtype=D_dtype, device=m1.device
            )
            mm_func(m1.to(AB_dtype), m2.to(AB_dtype), out=outf32_or_f16)
            mm_i32 = outf32_or_f16.to(torch.int32)

        return (scale_i.float() * mm_i32 * scale_w + bias).to(m1_dtype)

    @reg_fake("fms_mo::q_iaddmm_dq")
    def q_iaddmm_dq_abstract(bias, m1, m2, scale_i, zp_i, scale_w):
        return torch.empty(
            (m1.shape[0], m2.shape[1]),
            dtype=m1.dtype,
            device=m1.device,
            requires_grad=False,
        )

    @kernel_impl("fms_mo::q_per_t_sym", "default")
    def q_per_t(x, s, zp):
        return torch.quantize_per_tensor(x, s, zp, torch.qint8).int_repr()

    @reg_fake("fms_mo::q_per_t_sym")
    def q_per_t_abstract(x, s, zp):
        return torch.empty(
            x.shape, dtype=torch.int8, device=x.device, requires_grad=False
        )

    return


def lower_qmodel_cutlass(
    mod: torch.nn.Module,
    exam_inp_eval,
    qcfg,
    use_inductor=False,
    model_dtype=torch.float16,
):
    """
    Examplar GPU lowering function using cutlass. Only swap Qlinears in transformers, nothing else.

    Args:
        mod: nn.Module. should be a fms_mo Qmodel, will do inplace layer swapping, no deepcopy
        exam_inp_eval: one "ready-to-run" data that can be fed to the model like model(**data). This
                        will be used for torch.compile
        use_inductor: bool or str, [False, True, 'default', "reduce-overhead", 'max-autotune'], will
                        be used as the "mode" in torch.compile().
        model_dtype: default to FP16, but sometimes may need BF16 or even FP32
    """
    # Third Party
    from torch.ao.quantization.utils import _parent_name

    # Local
    from fms_mo.modules.linear import QLinear, QLinearINT8Deploy

    currDev = getattr(mod, "device", next(mod.parameters()).device)
    mod.cpu().to(model_dtype)
    cutlass_ops_load_and_reg(qcfg)

    for name, m in mod.named_modules():
        if isinstance(m, QLinear):
            if m.num_bits_weight != 8 or m.num_bits_feature != 8:
                logger.warning(
                    f"Only support INT8 lowering for now, cannot lower {name}"
                )
                continue
            parent_name, module_name = _parent_name(name)
            parent_mod = mod.get_submodule(parent_name)
            target_qclass = QLinearINT8Deploy

            # NOTE: some kernels may require the module to be placed on CPU, eg. GPTQ
            m.to(currDev)

            new_module = target_qclass.from_fms_mo(
                m, target_device=currDev, use_inductor=use_inductor
            )
            setattr(parent_mod, module_name, new_module)
            m.to("cpu")

    # NOTE: In case some components in the old module are shared with the new module and have been
    #       moved to CPU during swapping, we move the entire model back to device one more time
    mod.to(currDev)

    if use_inductor is True:
        use_inductor = "default"
    if use_inductor:
        with torch.no_grad():
            mod = torch.compile(mod, mode=use_inductor)
            mod(**exam_inp_eval)

    logger.info(f"\nModel lowered {'and compiled' if use_inductor else ''}.\n{mod}")

    return mod


def lower_qmodel_triton(
    model: torch.nn.Module,
    use_dyn_max_act=False,
    max_acc_bits=32,
    clamp_acc_to_dl16=False,
    num_lsb_to_truncate=0,
    chunk_size=32,
    layer_to_exclude=None,
):
    """
    Examplar GPU lowering function using triton. Only swap Linear/Qlinear in transformers.
    Triton kernel can be used to:
    1. test INT8 or FP8 HW performance (kernel is not optimized)
    2. simulate MSB/LSB truncation effect or special dtype (DL16) accumulation

    Args:
        model: nn.Module. should be a fms_mo Qmodel, will do inplace layer swapping, no deepcopy
        use_dyn_max_act: bool or int, can be False or -1 for per-token, or -2 for perCh. will use
                        dynamic max quantizer for activation if not False.
        max_acc_bits: max bits for accumulator, typically FP32 for all FP matmuls and INT32 for all
                        INT matmuls. But some HW could use fewer bits to trade-off power
                        efficiency at the expense of higher chance of accumulation "overflow".
                        For example, an INT24 accumulator can only hold values ranged from -2^23 to
                        2^23 -1, as opposed to typical range -2^31 to -2^31 -1.
        clamp_acc_to_dl16: clamp local accumulator to DL16 (1-6-9) range. To simulate this special
                        dtype effect on accumulation.
        num_lsb_to_truncate: number of bits to truncate from LSB side. For example, given fp32 is
                        s1e8m23, if we choose to truncate 13 mantissa bits from right most side,
                        i.e. LSB, the resulting number will be s1e8m10, which is TF32.
        chunk_size: given a matmul of (m, k) @ (k, n), the inner product will be "accumulated" along
                    k-dim. Since the entire matrix will be partitioned into smaller tiles when being
                    computed, accumulator will only add a certain num of elements in one shot. This
                    "chunk size" in k-dim will affect the overflow/underflow of accumulator.
    """
    # Third Party
    from torch.ao.quantization.utils import _parent_name

    # Local
    from fms_mo.modules.linear import LinearFPxAcc, QLinear, QLinearINT8Deploy

    # Currently QLinearINT8 has more options in dynamic quantization than LinearFP. Here we resolve
    # the differences as a patch solution (will unify the codes in future release)
    linFP_dyn_code = (
        "per_token"
        if use_dyn_max_act in [-1, -2]
        else "per_tensor"
        if use_dyn_max_act
        else False
    )

    if layer_to_exclude is None:
        layer_to_exclude = []
    elif isinstance(layer_to_exclude, str):
        layer_to_exclude = [
            layer_to_exclude,
        ]
    elif not isinstance(layer_to_exclude, (list, tuple)):
        raise RuntimeError("layer_to_exclude has to be either str, list, or tuple.")

    for name, m in model.named_modules():
        if not isinstance(m, (QLinear, torch.nn.Linear)) or name in layer_to_exclude:
            continue
        parent_name, module_name = _parent_name(name)
        parent_mod = model.get_submodule(parent_name)

        # Only support simulations of 1) QLinear -> INT8, 2) nnLinear->FP8 for now
        if isinstance(m, QLinear):
            new_lin = QLinearINT8Deploy.from_fms_mo(
                m,
                use_int_kernel="triton",
                use_dynamic_max_act_Qfunc=use_dyn_max_act,
                max_acc_bits=max_acc_bits,
                truncate_lsb=num_lsb_to_truncate,
                chunk_size=chunk_size,
            )
        else:
            new_lin = LinearFPxAcc.from_nn(
                m,
                trun_bits=num_lsb_to_truncate,
                chunk_size=chunk_size,
                dynamic_fp8=linFP_dyn_code,
                clamp_acc_to_dl16=clamp_acc_to_dl16,
            )

        setattr(
            parent_mod,
            module_name,
            new_lin,
        )

    logger.info(f"\nModel lowering with triton kernel is done.\n{model}")


### -------------------------------------------------------------
#   GPTQ tensor packing functions for Exllama kernel
### -------------------------------------------------------------


def pack_old(linear, scales, zeros, g_idx, bits=4):
    """This pack function is borrowed from exllamav1
    It's super slow and kept only for debugging purpose
    NOTE:
        1. intweight needs .clamp to [0, 15] or it would have something out of the INT4 range
        2. zero -=1 will be added back in CUDA kernel
    """
    W = linear.weight.data.clone()
    if isinstance(linear, torch.nn.Conv2d):
        W = W.flatten(1)
    if isinstance(linear, Conv1D):
        W = W.t()

    assert g_idx is not None

    infeatures = W.shape[1]

    scales = scales.t().contiguous()
    zeros = zeros.t().contiguous()
    scale_zeros = zeros * scales
    scales = scales.clone().half()
    intweight = []
    for idx in range(infeatures):
        intweight.append(
            torch.round((W[:, idx] + scale_zeros[g_idx[idx]]) / scales[g_idx[idx]]).to(
                torch.int
            )[:, None]
        )

    intweight = torch.cat(intweight, dim=1)
    intweight = intweight.t().contiguous().clamp(0, 15)
    intweight = intweight.numpy().astype(np.uint32)
    i = 0
    row = 0
    qweight = np.zeros(
        (intweight.shape[0] // 32 * bits, intweight.shape[1]), dtype=np.uint32
    )
    while row < qweight.shape[0]:
        if bits in [4]:
            for j in range(i, i + (32 // bits)):
                qweight[row] |= intweight[j] << (bits * (j - i))
            i += 32 // bits
            row += 1
        else:
            raise NotImplementedError("Only 4 bits are supported.")

    qweight = qweight.astype(np.int32)
    qweight = torch.from_numpy(qweight)

    zeros -= 1
    zeros = zeros.numpy().astype(np.uint32)
    qzeros = np.zeros((zeros.shape[0], zeros.shape[1] // 32 * bits), dtype=np.uint32)
    i = 0
    col = 0
    while col < qzeros.shape[1]:
        if bits in [4]:
            for j in range(i, i + (32 // bits)):
                qzeros[:, col] |= zeros[:, j] << (bits * (j - i))
            i += 32 // bits
            col += 1
        else:
            raise NotImplementedError("Only 4 bits are supported.")

    qzeros = qzeros.astype(np.int32)
    qzeros = torch.from_numpy(qzeros)

    return qweight, qzeros, scales


def pack_vectorized(linear, scales, zeros, g_idx, bits=4, device="cuda:0"):
    """Vectorized pack function.
    Instead of iterating over all rows in the input matrix, select (rows/pack_size) rows with stride
    to do this in pack_size iterations. => All operations are vectorized

    NOTE:
        1. _quant_weight_trans expects WT with shape (infeatures, outfeatures), which facilitates
            vectorized math
        2. the datatype of the tensors are not half (not the same as pack_old)
        3. intweight needs .clamp to [0, 15] or it would have something out of the INT4 range
        4. zero -=1 will be added back in CUDA kernel
    """

    def _quant_weight_trans(WT, scales, zeros, g_idx, rows):
        """Use g_idx to select the rows from zeros and scales
        this creates a new array that is (outfeatures, infeatures)
        """
        zeros *= scales
        zeros = zeros[g_idx]
        scales = scales[g_idx]

        # vectorized quantization
        WT = (WT + zeros) / scales
        return WT

    def _pack_rows_nbits(x, nbits, trans=False):
        """Generalized packing function returns a packed array
        When trans==True, x is transposed before packing and
        packed qx is transposed again before returning
        """
        device = x.device
        pack_size = 32 // nbits

        if trans:
            x = x.t().contiguous()
        assert x.shape[0] % pack_size == 0

        qx = torch.zeros(
            (x.shape[0] // pack_size, x.shape[1]), dtype=torch.int, device=device
        )

        for j in range(pack_size):
            qx |= (x[j::pack_size] & 0xF) << (j * nbits)

        if trans:
            return qx.t().contiguous()

        return qx

    W = linear.weight.data.clone()
    if isinstance(linear, torch.nn.Conv2d):
        W = W.flatten(1)
    if isinstance(linear, Conv1D):
        W = W.t()

    assert g_idx is not None
    assert bits == 4, "Only 4 bits are supported."

    infeatures = W.shape[1]
    scales = scales.t().contiguous()
    zeros = zeros.t().contiguous()

    # Run quant on device
    WT = _quant_weight_trans(
        W.t().to(device),
        scales.to(device),
        zeros.to(device),
        g_idx.to(device),
        infeatures,
    )
    intweight = WT.round().to(torch.int)
    intweight = intweight.clamp(0, 15)

    # Pack weight in int32 tensor
    qweight = _pack_rows_nbits(intweight, bits)

    # Pack zeros in int32 tensor
    zeros -= 1
    qzeros = _pack_rows_nbits(zeros.to(torch.int), bits, trans=True)

    # TODO: Confirm what should the final device be for the tensors (for returns)?
    scales = scales.clone().half()

    return qweight, qzeros, scales


def get_gpu_memory_usage(device_id: int = 0, unit="MiB"):
    """calculate GPU memory usage"""
    if unit not in ["MiB", "GiB"]:
        raise ValueError("Unit must be either MiB or GiB")
    mem_free, mem_tot = torch.cuda.mem_get_info(device_id)
    n = 2 if unit == "MiB" else 3
    mem_used = (mem_tot - mem_free) // 1024**n
    mem_tot = mem_tot // 1024**n
    return mem_used, mem_tot


def pack_rows_nbits(x, nbits):
    """Standalone vectorized pack function.
    Instead of iterating over all rows in the input matrix
    Select (rows/pack_size) rows with stride to do this in pack_size iterations
    All operations are vectorized
    """
    device = x.device
    pack_size = 32 // nbits

    assert x.shape[0] % pack_size == 0

    qx = torch.zeros(
        (x.shape[0] // pack_size, x.shape[1]), dtype=torch.int, device=device
    )

    for j in range(pack_size):
        qx |= (x[j::pack_size] & 0xF) << (j * nbits)

    return qx


def unpack_rows_nbits(qw, nbits):
    """Derived from vectorized version of pack func
    NOTE: qW is [in_feat, out_feat], normal Linear.W = [out_feat, in_feat]
    """
    device = qw.device
    pack_size = 32 // nbits

    w = torch.zeros(
        (qw.shape[0] * pack_size, qw.shape[1]), dtype=torch.int, device=device
    )

    for j in range(pack_size):
        w[j::pack_size] = (qw >> (j * nbits)) & 0xF

    return w


def dq_gptq_w(qw, scales, g_idx, zp=8):
    """Assume zp is always 8, no need to unpack qzeros
    w_int dim is same as qW, [in_feat, out_feat]
    """
    w_int = unpack_rows_nbits(qw, 4)
    full_scales = torch.empty((g_idx.shape[0], scales.shape[1]), device=qw.device)
    for col in range(scales.shape[1]):
        full_scales[:, col] = scales[g_idx, col]
    return torch.mul((w_int - zp).to(torch.float16), full_scales)


def swap_nnlinear_to_quantlinear(model, qconfig, prefix=None, qlinear2use=None):
    """For HF LLaMA, prefix = model.layers or layers, but for FMS LLaMA could be
    decoder.model.layers or just layers.
    Args:
    qconfig: HF model.config.quantization_config
    qlinear2use: is a qclass to use, this class has to be subclassed from Exllama QuantLinear, but
                hard to verify without instantiate an obj, hence just check nn.Module
    return:
        inplace swapping, no need to return the model afterward, but could

    """
    if prefix is None:
        prefix = ["model.layers"]
    if isinstance(qconfig, dict):
        bits = qconfig["bits"]
        group_size = qconfig["group_size"]
        exVer = qconfig["exllama_config"]["version"]
    elif (
        not isinstance(qconfig, dict)
        and hasattr(qconfig, "to_dict")
        and callable(qconfig.to_dict)
    ):
        qconfig_dict = qconfig.to_dict()
        bits = qconfig_dict["bits"]
        group_size = qconfig_dict["group_size"]
        exVer = qconfig_dict["exllama_config"]["version"]
    else:
        raise RuntimeError("Unknown qconfig type. Please check")

    # allow user to provide custom quantlinear wrapper
    if qlinear2use is not None and issubclass(qlinear2use, torch.nn.Module):
        QuantLinear = qlinear2use
    elif exVer == 1:
        # Third Party
        from gptqmodel.nn_modules.qlinear.exllama import (
            ExllamaQuantLinear as QuantLinear,
        )
    else:
        # Third Party
        from gptqmodel.nn_modules.qlinear.exllamav2 import (
            ExllamaV2QuantLinear as QuantLinear,
        )

    num_swapped = 0
    for n, m in model.named_modules():
        if isinstance(m, torch.nn.Linear) and any(n.startswith(pre) for pre in prefix):
            currDev = m.weight.device
            if currDev == torch.device("meta"):
                logger.warning(f"{n} is on meta device, please check!")
                target_device = "cuda"
            else:
                target_device = currDev

            newQLinear = QuantLinear(
                bits,
                group_size,
                m.in_features,
                m.out_features,
                True,  # TODO: It's hard-coded as True in HF from_pretrained, we follow suit, but
                # could be (m.bias!=None), there is a chance the original module doesn't have bias
                # but GPTQ needs it, also probably safer to determine this flag based on the ckpt
                # to be loaded instead of the existing old module
                weight_dtype=m.weight.dtype,
            )
            parent_fullname = n[: n.rindex(".")]
            mod_name = n[n.rindex(".") + 1 :]
            mod_parent = model.get_submodule(parent_fullname)

            setattr(mod_parent, mod_name, newQLinear.to(target_device))
            num_swapped += 1

            if currDev != torch.device("meta"):
                m.cpu()

    logger.info(f"Total {num_swapped} nn.Linear layers were replaced by QuantLinear.")
    return model
