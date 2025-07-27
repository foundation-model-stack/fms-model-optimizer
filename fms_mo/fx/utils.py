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

"""Utils for FX graph parsing and external kernel lowering"""

# Standard
from typing import Any, Dict, Optional
import logging
import operator
import os

# Third Party
import pandas as pd
import torch

# Local
from fms_mo.modules.linear import QLinear, QLinearDebug, QLinearW4A32Debug
from fms_mo.utils.import_utils import available_packages
from fms_mo.utils.utils import default_device_selection

try:
    # Local
    from fms_mo.modules.linear import QLinearCutlassI8I32NT

    cutlass_available = True
except ImportError:
    cutlass_available = False

try:
    # Local
    from fms_mo.modules.linear import QLinearExv1WI4AF16, QLinearExv2WI4AF16

    gptqmodel_available = True
except ImportError:
    gptqmodel_available = False


MIN_BLOCK_SIZE = 5
disable_torchtrt = True


logger = logging.getLogger(__name__)

DEFAULT_DEVICE = default_device_selection()
CLASS_TO_FUNC = {
    torch.nn.Conv2d: torch.nn.functional.conv2d,
    torch.nn.Linear: torch.nn.functional.linear,
    torch.nn.BatchNorm2d: torch.nn.functional.batch_norm,
    torch.nn.ReLU: torch.nn.functional.relu,
    torch.nn.LayerNorm: torch.nn.functional.layer_norm,
}


def check_qclass_fallback_based_on_min_feat(
    ref_module, qclass_frontup, qclass_fallback=None, feat_thresh=16
):
    """Check if a Qclass can be used for this specific ref_module (the module to be converted)
    Some kernels has limitations on min feature, if target_qclass is not compatible, this fallback()
    will return a proper qclass to use.

    NOTE:
    1. FasterTransformer is not supported in fms_mo anymore
    2. don't have a QConv_fallback yet

    Args:
        ref_module (nn.Module): the module to be checked
        qclass_frontup (nn.Module): target Qclass for ref_module
        qclass_fallback (nn.Module, optional): class to use if qclass_frontup won't work.
        feat_thresh (int, optional): min requirement for feature size. Defaults to 16.

    Raises:
        RuntimeError: only support Linear and Conv2d

    Returns:
        _type_: the Qclass that can be used for lowering
    """

    qclass_has_constraints = [
        QLinearDebug,
    ]
    if cutlass_available:
        qclass_has_constraints += [QLinearCutlassI8I32NT]
    if gptqmodel_available:
        qclass_has_constraints += [QLinearExv1WI4AF16, QLinearExv2WI4AF16]

    qclass = type(ref_module)
    if issubclass(qclass, torch.nn.Linear):
        min_feat_or_ch = min(ref_module.in_features, ref_module.out_features)
        qclass_fallback = QLinearDebug if qclass_fallback is None else qclass_fallback
    elif issubclass(qclass, torch.nn.Conv2d):
        min_feat_or_ch = min(ref_module.in_channels, ref_module.out_channels)
        qclass_fallback = None
    else:
        raise RuntimeError(
            f"Reference module {ref_module} is neither Linear nor Conv2d."
        )

    if min_feat_or_ch < feat_thresh and issubclass(
        qclass_frontup, tuple(qclass_has_constraints)
    ):
        logger.info(
            f"{ref_module} has min feature size ({min_feat_or_ch}) <16. "
            "Fall back to PyTorch implementation"
        )
        return qclass_fallback
    return qclass_frontup


def lower_qmodel_to_ext_kernels(
    mod, exam_inp_eval, qcfg, useInductor=True, model_dtype=torch.float16
):
    """Prepare a quantized model with specialized kernels that can utilize the HW for acceleration
    e.g. modify a trained INT8 model using real GPU INT8 matrix multiplication
    Basic mechanism is just inplace layer swapping, for example, Linear layers will be replaced by
    QLinearINT8Deploy which will be calling CUTLASS kernel instead of torch.nn.functional.linear

    NOTE:
    1. user need to define a mapping thru    qcfg["ext_kernel_mapping_mod"]
    2. to make it simple, only swap user specified qclass, nothing else
    3. move the module to GPU before swapping to accelerate scale/zp calculations
    4. gptq_post_init() must be done at model level, or OOM and incorrect results easily

    Args:
        mod (torch.nn.Module): model to be 'lowered'
        exam_inp_eval (_type_): an example of input data structure to model.forward()
        qcfg (dict): quantization config dict
        useInductor (bool|str, optional): flag to invoke torch.compile after swapping. possible
                                        values are True, False, "reduce-overhead", "max-autotune"...
        model_dtype (_type_, optional): model data type. default to FP16, could be BF16 or even FP32

    Returns:
        model: ready to run

    TODO:
    1. Streamline this function in v0.2 in terms of class mapping.
    2. Assert acceptable "nbits" for given Qclasses, e.g. 4 for GPTQ 8 for cutlass.
    """

    # Third Party
    from torch.ao.quantization.utils import _parent_name

    currDev = getattr(mod, "device", next(mod.parameters()).device)
    mod.cpu().to(model_dtype)

    qclass_must_start_from_cpu = None
    using_gptq = False
    if (
        available_packages["gptqmodel"]
        and available_packages["exllama_kernels"]
        and available_packages["exllamav2_kernels"]
    ):
        # Local
        from fms_mo.modules.linear import QLinearExllamaV1, QLinearExllamaV2

        using_gptq = any(
            issubclass(c, (QLinearExllamaV1, QLinearExllamaV2))
            for c in qcfg["ext_kernel_mapping_mod"].values()
        )
        qclass_must_start_from_cpu = (
            QLinearExllamaV1,
            QLinearExllamaV2,
        )

    mod2swap = {
        n: m
        for n, m in mod.named_modules()
        if type(m) in qcfg["ext_kernel_mapping_mod"]
    }

    for name, module in mod2swap.items():
        parent_name, module_name = _parent_name(name)
        parent_mod = mod.get_submodule(parent_name)

        target_qclass = qcfg["ext_kernel_mapping_mod"][type(module)]

        if isinstance(module, QLinear):
            qclass = check_qclass_fallback_based_on_min_feat(
                module,
                qclass_frontup=target_qclass,
                qclass_fallback=(
                    QLinearW4A32Debug if module.num_bits_weight == 4 else QLinearDebug
                ),
            )

        if issubclass(qclass, qclass_must_start_from_cpu):
            module.to("cpu")
        else:
            module.to(currDev)  # see Note 3

        new_module = qclass.from_fms_mo(
            module, target_device=currDev, useInductor=useInductor
        )
        setattr(parent_mod, module_name, new_module)

        module.to("cpu")

    if using_gptq:
        # Third Party
        from gptqmodel.utils.model import hf_gptqmodel_post_init as gptq_post_init

        mod_tmp = gptq_post_init(mod_tmp, use_act_order=False)  # see Note 4

    mod.to(currDev)
    logger.info(mod)

    if useInductor:
        with torch.no_grad():
            mod = torch.compile(
                mod, mode="default" if useInductor is True else useInductor
            )

            mod(**exam_inp_eval)

    return mod


def cus_backend_plotFXAtenGM(
    gm_fx,
    sample_inp,
    filename_tag="",
    show_details=False,
):
    """This is a custom backend for plotting the GraphModule, both FX and Aten.
    Args:
        gm_fx: GraphModule in FX or Aten IR, will be passed in by Dynamo.
        sample_inp: Sample input tensor, will be passed in by Dynamo.
        filename_tag: str. User provided filename.
        show_details: bool. Whether to plot more details for each node.

    NOTE:
    1. Use it as a "compile backend", e.g. torch.compile(model, backend=cus_backend_plotFxAtenGM)
        default output filename is FX.svg and Aten.svg.
    2. Plotting more than ~1500 nodes will make the downstream plot engine "dot" very slow.
    3. torch.quant_per_t() cannot be traced well (as of PT2.2). either implement faketensor wrapper
        (see example in custom_ext_kernels/utils.py "fms_mo::q_per_t_sym") or avoid using it
    """

    # Third Party
    from torch._decomp import get_decompositions
    from torch._functorch.aot_autograd import aot_module_simplified, make_boxed_compiler

    MAX_NODE_TO_PLOT = 1000
    plot_graph_module(
        gm_fx,
        outputname=f"FX_{filename_tag}.svg",
        Nnode_to_plot=MAX_NODE_TO_PLOT,
        show_details=show_details,
    )

    def fake_fwd_comp(gm_aten, inp):  # pylint: disable=unused-argument
        plot_graph_module(
            gm_aten,
            outputname=f"Aten_{filename_tag}.svg",
            Nnode_to_plot=MAX_NODE_TO_PLOT,
            show_details=show_details,
        )

        return gm_aten.forward

    with torch.no_grad():
        gm_fwd = aot_module_simplified(
            gm_fx,
            sample_inp,
            fw_compiler=make_boxed_compiler(fake_fwd_comp),
            decompositions=get_decompositions([torch.ops.aten.rsub.Scalar]),
        )
    return gm_fwd


def lname_to_org_name(Lname):
    """Lname is from list(n.meta['nn_module_stack'].values())[-1][0], usually looks like
        "L__self___self_bert_layer" or "getattr_L__self___layer1___0___conv1"
    original name would be something like
        "L['self'].self.bert.layer" or
        "L['self']._modules['layer1']._modules['0']._modules['downsample']._modules['0']"
    where '[', ']', "'", and '.' are all replaced by _ already. It could also look like
        "getattr(L['self'].E1.encoder, '0')", or even
        "getattr(getattr(L['self'].layer1, '0').downsample, '0')", or
        "getattr( "body", "suffix"  )".remain" should become body+"."+suffix+".remain"
    """
    while Lname.startswith("getattr("):
        idx_right_paren = Lname.rindex(")")
        idx_right_comma = Lname.rindex(",")
        remain = (
            "" if idx_right_paren + 1 == len(Lname) else Lname[idx_right_paren + 1 :]
        )
        body = Lname[len("getattr(") : idx_right_comma]
        suffix = Lname[idx_right_comma + 1 : idx_right_paren].replace("'", "").lstrip()
        Lname = body + "." + suffix + remain

    org_mod_name = None
    if "." in Lname:
        Lname = Lname.replace("_modules['", "").replace("']", "")
        idx_first_dot = Lname.index(".")
        org_mod_name = Lname[idx_first_dot + 1 :].replace("[", ".").replace("]", "")
    elif Lname.startswith("L__self___"):
        Lname = Lname[len("L__self___") :]
        # make sure only single _ exists, no more __ and ___
        if "__" not in Lname and "___" not in Lname:
            org_mod_name = Lname.replace("_", ".")

    return org_mod_name


def get_org_mod_name_of_fx_node(
    node, gm=None, lut_fx2org: Optional[Dict[str, str]] = None
):
    """Given a FX node, could be call_module or call_fuction, find out the original module name,
    based on meta data

    NOTE:
    1. if graph breaks, it won't provide the full name corresponding to top level
    2. gm.meta.get("dynamo_flat_name_to_original_fqn") seems to be designed for parameters/buffers
        not modules, possibly still need to use our lname_to_org_name()
    3. by matching parameter tensors from original model and dynamo graphmodule, we could create a
        LUT for fx module name to original name (only Linear and Conv, or mod with parameters). BUT
        we also need self-attention module for torch.matmul ops => add a partial matching to infer,
        e.g. if we have 'L__self___model_layers_slice_None__32__None___0_self_attn_q_proj' in LUT
            'L__self___model_layers_slice_None__32__None___0_self_attn' can be infered easily

    Args:
        node (fx.node): fx node of interest
        gm (GraphModule, optional): FX graph containing the given fx node. could be useful when
                                    parsing the node name
        lut_fx2org (dict, optional): LUT from fx module name to original module name

    Returns:
        str: corresponding name on original graph
    """
    org_name = f"Unknown:{node.name}"
    if lut_fx2org is None:
        lut_fx2org = {}
    if "nn_module_stack" in node.meta:
        n_fx_mod_name = list(node.meta["nn_module_stack"].keys())[-1]
        n_fx_org_mod_name = list(node.meta["nn_module_stack"].values())[-1][0]
        if n_fx_mod_name in lut_fx2org:
            org_name = lut_fx2org[n_fx_mod_name]
        elif gm and isinstance(node.target, str):
            LUT = gm.meta.get("dynamo_flat_name_to_original_fqn", {})  # see Note 2
            org_name = LUT.get(node.target, None)
        else:
            for k, v in lut_fx2org.items():
                if k.startswith(n_fx_mod_name):
                    suffix = k[len(n_fx_mod_name) :]
                    suffix = "." + suffix[1:]  # replace leading "_" with "."
                    if v.endswith(suffix):
                        org_name = v[: -len(suffix)]
                    break

        if org_name.startswith("Unknown:"):
            org_name = lname_to_org_name(n_fx_org_mod_name)

    return org_name


def get_target_op_from_node(node: torch.fx.Node, gm: torch.fx.GraphModule):
    """Determine the real target Op of a given node depending on the type of node.op

    Args:
        node (torch.fx.Node): node of interest
        gm (torch.fx.GraphModule): graph containing the given node

    Returns:
        Any: could be a str or callable if it's a call_func/method, or nn.Module if 'call_module'...
    """

    if node.op in ["call_function", "placeholder", "call_method", "output"]:
        return node.target

    if node.op == "call_module":
        return gm.get_submodule(node.target)

    if node.op == "get_attr":
        return getattr(gm, node.target)


def get_target_op_from_mod_or_str(mod_or_str, verbose=False):
    """Similar to the above, get_target_op_from_node(), but for non-nodes

    NOTE:
    1. some Ops starts with "i", e.g. iadd is inplace add
    2. if callable, e.g. torch.matmul, torch.bmm, -> return itself, but if torch.nn.Linear, Conv2d->
        return both class and functional form

    Args:
        mod_or_str (_type_): module of op "name" in string
        verbose (bool, optional): whether to print more info. Defaults to False.

    Returns:
        _type_: _description_
    """
    if isinstance(mod_or_str, str):
        possible_op = [
            getattr(torch.nn.functional, mod_or_str, None),
            getattr(torch, mod_or_str, None),
            getattr(operator, mod_or_str, None),
            getattr(operator, "i" + mod_or_str, None),  # see Note 1
        ]
        clean_up = [op for op in possible_op if op is not None]
        if clean_up == []:
            if verbose:
                logger.info(f"Cannot identify the real Op for {mod_or_str}")
            clean_up = mod_or_str
        return clean_up

    if callable(mod_or_str):
        if mod_or_str in CLASS_TO_FUNC:
            return [mod_or_str, CLASS_TO_FUNC[mod_or_str]]
        return mod_or_str

    if verbose:
        logger.info(f"Cannot identify the real Op for {mod_or_str}")
    return mod_or_str


#############
## Helpers ##
#############


def model_size_Wb(mod, unit="MB", print_to_file=True, show_details=False):
    """Checks model size, only count weight and bias

    NOTE:
    1. Usually module.weight is a Tensor, but for torch's quantized module, weight is packed with
        bias using qint8, mod.weight will be a callable instead of a tensor.

    Args:
        mod (nn.Module): a model
        unit (str, optional): 'MB' or 'GB. Defaults to "MB".

    Returns:
        float: model size in desired unit
    """

    mem_use = 0
    if unit not in ["MB", "GB"]:
        logger.warning(
            f"Unrecognized unit for memory summary: {unit}. Will use MB instead."
        )
        unit = "MB"

    summary_weights = {"layer": [], "shape": [], f"mem ({unit})": [], "dtype": []}
    for n, m in mod.named_modules():
        w = getattr(m, "weight", None)
        w_dtype, w_shape = None, None
        if callable(w):  # see Note 1.
            w_mat, b_mat = w()[:2]
            mem_use = (
                w_mat.numel() * w_mat.element_size()
                + b_mat.numel() * b_mat.element_size()
            )
            w_dtype = str(w_mat.dtype)
            w_shape = w_mat.shape

        elif isinstance(w, torch.Tensor):
            mem_use = w.numel() * w.element_size()
            if hasattr(m, "bias") and m.bias is not None:
                mem_use += m.bias.numel() * m.bias.element_size()
            w_dtype = str(w.dtype)
            w_shape = w.shape

        if w_shape:
            mem_use = mem_use / 1e9 if unit == "GB" else mem_use / 1e6

            summary_weights["layer"].append(n)
            summary_weights["shape"].append(w_shape)
            summary_weights[f"mem ({unit})"].append(mem_use)
            summary_weights["dtype"].append(w_dtype)

    df_summary_weights = pd.DataFrame(summary_weights)
    logger_or_print = logger.info if print_to_file else print
    logger_or_print("[check model size] Summary of W/b tensors in this model:")
    logger_or_print(
        "\n%s",
        str(
            pd.pivot_table(
                df_summary_weights,
                index="dtype",
                values=["layer", f"mem ({unit})"],
                aggfunc={"layer": "count", f"mem ({unit})": "sum"},
            )
        ),
    )
    if show_details:
        logger_or_print(df_summary_weights.to_markdown())

    return df_summary_weights[f"mem ({unit})"].sum().item()


def plot_graph_module(
    gm,
    modules=None,
    outputname="test.svg",
    plot_type="dot",
    verbose=False,
    show_details=False,
    skip_nodes=None,
    Nnode_to_plot=None,
    additional_coloring_rules=None,
    lut_fx_mod_name_to_org: Optional[Dict[str, str]] = None,
):
    """Plots a GraphModule in .SVG format to visualize the compute graph. If graphviz/pygraphviz is
    not installed properly, this function will just print out a message and do nothing.

    Args:
        gm (GraphModule): GraphModule to be plotted
        modules (dict, optional): name->module mapping. Defaults to None.
        outputname (str, optional): output filename. Defaults to "test.svg".
        plot_type (str, optional): plot format. Defaults to "dot". options include [neato, dot,
                                    twopi, circo, fdp, nop, wc, acyclic, gvpr, gvcolor, ccomps,
                                    sccmap, tred, sfdp, unflatten]
        verbose (bool, optional): print debug messages. Defaults to False.
        show_details (bool, optional): include more information on the plot. usually call_mod and
                                        get_attr has very long node name, hard to read, therefore,
                                        Defaults to False.
        skip_nodes (list(nodes), optional): nodes we don't want to plot, could be node names or
                                            node_type. Defaults to None.
        Nnode_to_plot (int, optional): number of nodes to plot, usually <1500. Defaults to None.
        additional_coloring_rules (dict, optional): a dict for node name-> color. use pre-defined
                                                    colors if not exists. Defaults to None.
    NOTE:
    1. don't mix the use of node.name and node.target. they are different usually.
    2. node.op could be ['call_module', 'call_function', 'get_attr', 'placeholder', 'output']

    """
    if not available_packages["pygraphviz"]:
        logger.error("pygraphviz is not installed properly")
        return

    # Third Party
    from tqdm import tqdm
    import pygraphviz as pgv

    if skip_nodes is None:
        skip_nodes = []

    G = pgv.AGraph(strict=False, directed=True)
    nodes_of_intr = [
        n
        for n in gm.graph.nodes
        if (n.name not in skip_nodes and n.op not in skip_nodes)
    ]
    if Nnode_to_plot is None:
        Nnode_to_plot = len(nodes_of_intr)

    if verbose:
        logger.info(nodes_of_intr, flush=True)
    if modules is None:
        modules = dict(gm.named_modules())

    pbar = tqdm(nodes_of_intr, total=Nnode_to_plot, ncols=120)
    for i, node_ptr in zip(range(Nnode_to_plot), pbar):
        pbar.set_description(
            f"Exporting GraphModule to SVG file: {i+1}/{Nnode_to_plot}"
        )
        nname = node_ptr.name  # see Note 1
        ntype = node_ptr.op

        if "nn_module_stack" in node_ptr.meta:
            fx_mod_name = list(node_ptr.meta["nn_module_stack"].keys())[-1]
        else:
            fx_mod_name = ""

        mod = modules.get(str(node_ptr.target), "")

        n_tar = ntype
        if ntype in [
            "call_function",
        ]:
            n_tar += f": {str(node_ptr.target).replace('<','').replace('>','')}"
        elif ntype in ["call_module", "get_attr"]:
            org_mod_name = get_org_mod_name_of_fx_node(
                node_ptr, lut_fx2org=lut_fx_mod_name_to_org
            )
            n_tar += f": {org_mod_name}"
            if node_ptr.target.startswith(fx_mod_name + "_"):
                attr_name = node_ptr.target[len(fx_mod_name) + 1 :]
                attr_name = attr_name.split("self___")[-1]
                n_tar += f"|{attr_name}"
        elif ntype in ["call_method"]:
            n_tar += f": {node_ptr.target}"

        if show_details:
            labelStr = (
                f"{{{n_tar}}}"
                if ntype in ["call_module", "get_attr"]
                else f"{{%{nname}|{n_tar}}}"
            )
        else:
            labelStr = f"%{nname}"

        isQuantizer = (
            (
                ntype == "get_attr"
                and any(kw in nname for kw in ["_scale", "_zero_point", "_zp"])
            )
            or (ntype == "call_function" and "quantize_per" in nname)
            or (ntype == "call_method" and "dequant" in nname)
            or "q_per_t" in nname
        )

        isSubmodule = ntype == "call_module" and (
            isinstance(getattr(gm, node_ptr.target), torch.fx.GraphModule)
        )

        fcolor = (
            "#b2d3e4"
            if "Conv" in str(mod)
            or isinstance(mod, (torch.nn.Conv2d, torch.ao.nn.quantized.Conv2d))
            else "#FFF2CC"
            if isQuantizer or "post_process" in nname
            else "#DDDDDD"
            if (
                "mm" in nname
                or "linear" in nname
                or ("matmul" in nname or "matmul" in fx_mod_name)
                or isinstance(mod, (torch.nn.Linear, torch.ao.nn.quantized.Linear))
            )
            else "#C5E0B4"
            if "If" in nname or "FPout" in nname or isSubmodule
            else "white"
        )

        if additional_coloring_rules:
            fcolor = additional_coloring_rules.get(nname, fcolor)

        G.add_node(
            nname, fillcolor=fcolor, style="filled", label=labelStr, shape="record"
        )

        for j in node_ptr.users:
            G.add_edge(nname, j.name)

    if verbose and plot_type == "dot":
        G.write(outputname + ".dot")
        logger.info(
            "Output graph to dot file. !! It could be very slow for DenseNet or complicated nets !!"
        )
    G.layout(prog=plot_type)

    if os.path.exists(outputname):
        os.remove(outputname)
    G.draw(outputname)


def debug_backend(
    gm: torch.fx.GraphModule,
    exam_inp: Any,  # pylint: disable=unused-argument
    subgraph_id: list = None,
):
    """Prints graph/nodes in table format and plot in SVG format. Supports graph breaks as well, in
    which case will save multiple SVGs with a suffix serial number.

    Args:
        gm (torch.fx.GraphModule): graph to be printed/plotted
        exam_inp (Any): not used here, but required call signature by dynamo
        subgraph_id (list): graph serial number when running into graph breaks, will be the suffix
                            for file name. use list to make it a persistant "global" variable

    Returns:
        GraphModule: required by dynamo
    """
    if subgraph_id is None:
        subgraph_id = [0]
    logger.info(f"Graph ID {subgraph_id}: (total nodes = {len(list(gm.graph.nodes))})")
    gm.graph.print_tabular()
    plot_graph_module(
        gm,
        outputname=f"debug{subgraph_id[0]}.svg",
        Nnode_to_plot=1200,
        show_details=True,
    )
    subgraph_id[0] += 1
    return gm
