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

# Standard
# NOTE: do not use .side_effect, which will create a MagicMock obj and induce memory leak issues
# In the main loop, we can simply use "mock.patch("torch.bmm", new=mockbmm)" as our context manager
# Creating this "general purpose" patch_torch_bmm context manager in case we need to patch

"""
General utils for fms_mo
"""

# Standard
from contextlib import ExitStack, contextmanager
from functools import partial
from typing import Any, Callable, Dict, List, Tuple, Union
from unittest import mock
import logging
import sys

# Third Party
from transformers.tokenization_utils_base import BatchEncoding
import torch

logger = logging.getLogger(__name__)


def move_to(obj, device):
    """
    Moves the given object to the specified device.

    Args:
        obj: The object to be moved. Can be a PyTorch tensor, dictionary, list, or tuple.
        device: The device to move the object to.

    Returns:
        The moved object.

    Examples:
    >>> x = torch.tensor([1, 2, 3], device='cpu')
    >>> y = move_to(x, 'cuda')
    >>> y
    tensor([1, 2, 3], device='cuda')
    >>> z = {'a': x, 'b': x}
    >>> z = move_to(z, 'cuda')
    >>> z['a']
    tensor([1, 2, 3], device='cuda')
    >>> z['b']
    tensor([1, 2, 3], device='cuda')
    """
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    if isinstance(obj, (dict, BatchEncoding)):
        return {k: move_to(v, device) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        res = [move_to(v, device) for v in obj]
        if isinstance(obj, tuple):
            return tuple(res)
        return res
    logger.warning(f"Invalid type for move_to for {obj} type {type(obj)} to {device}")
    return obj


def mockbmm(
    mat1,
    mat2,
    default_to_torch=False,
    target_line_num=[0],
):
    """
    This function is used to mock the behavior of the bmm function in PyTorch.
    It is used to work around the fact that the bmm function in PyTorch is not
    implemented for all data types, and we need to use it in some cases.

    Args:
        mat1 (Tensor): The first input tensor.
        mat2 (Tensor): The second input tensor.

    Returns:
        Tensor: The result of the modified mock matrix multiplication.
    """
    cf = sys._getframe()
    qbmm_mod = None
    qbmm_lineno = cf.f_back.f_lineno
    if qbmm_lineno not in target_line_num:
        default_to_torch = True
    while (not default_to_torch) and cf.f_back and qbmm_mod is None:
        # First frame is QBmm's forward itself, can start searching from previous stack
        cf = cf.f_back
        if (
            "forward" in cf.f_code.co_name or "_attn" in cf.f_code.co_name
        ) and "self" in cf.f_locals:
            mod_calling_bmm_function = cf.f_locals["self"]
            # If not found -> default to torch.matmul
            qbmm_mod = getattr(mod_calling_bmm_function, f"QBmm{qbmm_lineno}", None)
    del cf
    if qbmm_mod is None and default_to_torch:
        qbmm_mod = torch.matmul
    return qbmm_mod(mat1, mat2)


def mockmatmul(
    mat1,
    mat2,
    default_to_torch=False,
    target_line_num=[0],
):
    """
    Patches torch.matmul() with QBmm( torch.bmm() )

    Args:
        mat1 (torch.Tensor): The first matrix to be multiplied.
        mat2 (torch.Tensor): The second matrix to be multiplied.
        target_bmm_lineno: Only patch matmul/bmm on the line number previously found by qmodel_prep.
                            i.e., matmuls/bmms other than self-attn will not be patched.
                            => need to make sure qmodel_prep only found 2.

    Returns:
        torch.Tensor: The result of the mock matrix multiplication.
    NOTE:
        1. First frame is mockmatmul itself. One frame back (cf.f_back) is where torch.matmul
            happened, whose line number is the one used for QBmm<xxx>
        2. QBmm module may not be attached to the immediate frame where torch.matmul happened. Need
            to trace back and find the frame with both "forward" in name and "self" in locals, i.e.
            a class (nn.module) has a function named "forward" something
        3. Keep default_to_torch=False unless really needed, otherwise if something went wrong with
            QBmm detection, it could go to default silently, which would be very difficult to debug.
    """
    cf = sys._getframe()
    qbmm_mod = None
    qbmm_lineno = cf.f_back.f_lineno
    if qbmm_lineno not in target_line_num:
        default_to_torch = True
    while (not default_to_torch) and cf.f_back and qbmm_mod is None:
        cf = cf.f_back
        if (
            "forward" in cf.f_code.co_name or "_attn" in cf.f_code.co_name
        ) and "self" in cf.f_locals:
            mod_calling_bmm_function = cf.f_locals["self"]
            # If not found -> default to torch.bmm
            qbmm_mod = getattr(mod_calling_bmm_function, f"QBmm{qbmm_lineno}", None)
    del cf

    # Didn't find the corresponding QBmm, default the call to torch.bmm, which only accepts 3D
    if qbmm_mod is None and default_to_torch:
        # Need to reshape if inputs are 2d or 4d
        if len(mat1.shape) == len(mat2.shape) and len(mat2.shape) in [2, 4]:
            tar_shape = [mat1.shape[-2], mat1.shape[-1]]
            if len(mat1.shape) == 4:
                tar_shape = mat1.shape[:2] + tar_shape
            mat1 = mat1.reshape([-1, mat1.shape[-2], mat1.shape[-1]])
            mat2 = mat2.reshape([-1, mat2.shape[-2], mat2.shape[-1]])
        output = torch.bmm(mat1, mat2)
        output = output.reshape(tar_shape)
        return output
    return qbmm_mod(mat1, mat2)


@contextmanager
def patch_torch_bmm(qcfg):
    """
    Patch the torch.bmm (function, not a module) with a quantized version (class).
    interface is the same, but this class acts as a multiplexer based on the caller stack info
    """

    if qcfg is not None:
        # could be 'torch.bmm', 'torch.matmul', or None
        ops_to_patch = qcfg.get("which2patch_contextmanager", None)
        tar_ln = list(qcfg["bmm_prep"]["layers_with_bmm"].values())[0]
        # if qcfg["bmm_prep"]["bmm_only_in_self_attn"] is False, may need to enable default_to_torch
        # in mock functions, e.g. partial(mockmatmul, default_to_torch=True)
        # This is in case a model uses extra matmuls, and QBmmXXX is not found or attached properly.
        new_target = (
            partial(mockbmm, target_line_num=tar_ln)
            if ops_to_patch == "torch.bmm"
            else partial(mockmatmul, target_line_num=tar_ln)
            if ops_to_patch == "torch.matmul"
            else None
        )
    else:
        new_target = None

    with ExitStack() as stack:
        if new_target is not None:
            # pylint: disable=unused-variable
            ctxs = [stack.enter_context(mock.patch(ops_to_patch, new=new_target))]
        # NOTE: In model_prep, we already create all the QBmm instances
        yield


def apply_masks(model, qcfg):
    """
    Applies masks to the model's parameters based on the given configuration.

    Args:
        model (torch.nn.Module): The model to which the masks will be applied.
        qcfg (dict): A dictionary containing the configuration parameters for pruning.
    """
    # logger = qcfg.get("logger", logger)
    p_layer = qcfg.get("p_per_layer")
    for n, module in model.named_modules():
        if hasattr(module, "get_mask"):
            if p_layer is not None:
                if p_layer in n or "pooler" in n:
                    module.get_mask()
                    logger.info(f"Now apply mask to {n}")
            else:
                if qcfg["p_skip"] in n:
                    module.mask = None
                    logger.info(f"skipping prune module {n} and set its mask to None")
                else:
                    module.get_mask()


def prepare_input(
    device, data: Union[torch.Tensor, Dict, Tuple, List]
) -> Union[torch.Tensor, Dict, Tuple, List]:
    """
    Prepares one `data` before feeding it to the model, be it a tensor
    or a nested list/dictionary of tensors.
    """
    if isinstance(data, dict):
        return type(data)({k: prepare_input(device, v) for k, v in data.items()})
    if isinstance(data, (tuple, list)):
        return type(data)(prepare_input(device, v) for v in data)
    if isinstance(data, torch.Tensor):
        kwargs = {"device": device}
        return data.to(**kwargs)

    logger.warning(
        "data input to prepare_input must be Dict, "
        "Tuple, List or torch.Tensor and currently is",
        type(data),
    )
    return data


def prepare_inputs(
    device: str, inputs: Union[Tuple, List, torch.Tensor, Dict[str, Any], BatchEncoding]
) -> Union[Tuple, List, torch.Tensor, Dict[str, Any]]:
    """
    Prepare `inputs` before feeding them to the model, converting them to tensors
    if they are not already and handling potential state.

    Arguments:
        inputs: can be torch.Tensor type because in certain cases .data is called on input.

    Return:
        inputs: allow returning Tuple, List, Tensor or Dict.
    """
    if isinstance(inputs, BatchEncoding):
        return inputs.to(device)
    return prepare_input(device, inputs)


def prepare_data_4_fwd(
    data_mb: Union[Tuple, List, torch.Tensor, Dict[str, Any]],
    qcfg,
    prefwdproc: Union[Callable[[Any], Tuple], str, None] = None,
    currDev: str = "cuda",
) -> Union[Tuple, List, torch.Tensor]:
    """
    Pre-processing function for minibatch (output from loader) which is used in tracing
    and qmodel_calib

    Arguments:
        data_mb: data/dummy input used for tracing or calibration.
        qcfg: fms_mo_config
        prefwdproc: user provided function.
        currDev: str, currently takes of value of cuda or cpu

    Returns:
        data_mb: data/dummy input used for tracing or calibration.
    """
    # FIXME: There is definitely a type confusion here. Check prepare_inputs,
    # Prepare_input and the requirements of this function.
    if prefwdproc:
        if callable(prefwdproc):
            # Pre-processing function, we pass the whole batch directly,
            # user needs to make sure the func "takes Any and returns a tuple"
            # FIXME: added isinstance check based on comment above:
            data_mb = prefwdproc(data_mb)
        elif prefwdproc == "toDevice":
            if qcfg["qwav2vec"]:
                if isinstance(data_mb, torch.Tensor):
                    data_mb = prepare_inputs(currDev, data_mb.data)  # Any
                else:
                    raise TypeError(".data can only be called on torch.Tensor")
            else:
                # Any
                data_mb = prepare_inputs(currDev, data_mb)
        else:
            raise AssertionError(
                "Undefined case for pre-processing function, please check your 'prefwdproc'."
            )
    else:
        if isinstance(data_mb, (tuple, list)):
            # Mini-batch in typical CNN, transformers' dataloader uses dict instead of tuple
            if len(data_mb) <= 2:
                # NOTE: Assumes model only needs the 1st element
                data_mb = (data_mb[0].to(currDev),)
                # NOTE: if DP is used, 1 batch from loader in DP mode is (bs_per_GPU * N_GPU),
                # if we feed one whole batch to model for tracing (1 GPU), it can cause OOM)
                if qcfg["wasDPmodel"]:
                    data_mb = (data_mb[0][:2],)  # slice partial batch for DP only
            elif len(data_mb) > 2:
                raise AssertionError(
                    "Non-conventional dataloader format\n"
                    + "         Please define prefwdproc for pre-processing \n"
                    + "         which allows model( prefwdproc(output_from_loader) )."
                )
        elif isinstance(data_mb, torch.Tensor):
            # Keep as tuple for torch.jit.trace()
            data_mb = (data_mb.to(currDev),)
        else:
            raise TypeError(
                "Data/dataloader provided by the user seems to have a type not in (tuple, list, or "
                "Tensor). \nIf the data fetched from dataloader needs extra processing before being"
                " fed to model. Please consider using a `prefwdproc` function such that \n"
                "       model( prefwdproc(data_fetched_from_loader) )"
            )

    # NOTE: Found type conflict. prepare_inputs returns type Dict[str, Union[torch.Tensor, Any]]
    if isinstance(data_mb, (tuple, list)):
        return data_mb
    if isinstance(data_mb, torch.Tensor):
        logger.warning("data_mb is being returned as a Tensor")
        return data_mb
    raise TypeError("Return type of data_mb should be tuple")


def default_device_selection():
    """
    Default device selection will set to CUDA if cuda is available otherwise it uses CPU
    """
    default_device = None
    if torch.cuda.is_available():
        default_device = "cuda"
    else:
        default_device = "cpu"
    return default_device


def checkpoint_summary(path_to_ckpt, print_to_file=False, show_details=False):
    """Open a checkpoint (safetensors format) and summarize data type vs total size."""
    # Standard
    from pathlib import Path
    import json

    # Third Party
    from safetensors import safe_open
    import pandas as pd

    ckpt_fp8_path = Path(path_to_ckpt)
    # 1. parse index -> which tensor in which file
    filename_keys = {sf.name: [] for sf in ckpt_fp8_path.glob("*.safetensors")}
    if len(filename_keys) > 1:  # only when model is > 5GB
        ckpt_fp8_idx = ckpt_fp8_path / "model.safetensors.index.json"
        with open(ckpt_fp8_idx, encoding="utf-8") as f:
            ckpt_fp8 = json.load(f)

        for k, fname in ckpt_fp8["weight_map"].items():
            filename_keys[fname].append(k)
    else:
        fname = list(filename_keys.keys())[0]
        with safe_open(ckpt_fp8_path / fname, framework="pt", device="cpu") as f:
            filename_keys[fname] = list(f.keys())

    # 2. summarize
    summary_fp8 = {"layer": [], "shape": [], "mem (MB)": [], "dtype": []}
    for fname, keys in filename_keys.items():
        with safe_open(ckpt_fp8_path / fname, framework="pt", device="cpu") as f:
            for key in keys:
                tmp = f.get_tensor(key)
                summary_fp8["layer"].append(key)
                summary_fp8["shape"].append(list(tmp.shape))
                summary_fp8["mem (MB)"].append(tmp.numel() * tmp.element_size() / 1e6)
                summary_fp8["dtype"].append(str(tmp.dtype))

    df_summary_fp8 = pd.DataFrame(summary_fp8)

    logger_or_print = logger.info if print_to_file else print
    logger_or_print(
        pd.pivot_table(
            df_summary_fp8,
            index="dtype",
            values=["layer", "mem (MB)"],
            aggfunc={"layer": "count", "mem (MB)": "sum"},
        )
    )
    if show_details:
        logger_or_print(df_summary_fp8.to_markdown())
