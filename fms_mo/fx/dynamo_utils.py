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
"""This file contains graph searching and analysis functions specifically designed for GraphModule
in FX IR, i.e. the GraphModule received in dynamo custom backend. Older version based on TorchScript
can be found in fms_mo/utils.
"""

# Standard
from typing import Dict, Optional
import logging

# Third Party
import torch

# Local
from fms_mo.fx.utils import (
    get_org_mod_name_of_fx_node,
    get_target_op_from_mod_or_str,
    get_target_op_from_node,
)
from fms_mo.utils.import_utils import available_packages

logger = logging.getLogger(__name__)

# From PyTorch 2.5+, graphModule received in dynamo custom backend will be Aten IR instead of FX IR,
# i.e. no "call_module" nodes, all parameter tensors become "placeholder" nodes, and etc...
# This following flag will make dynamo behaves like PyTorch 2.4. Only use it when model_analyzer()
# really stop working and hard to recover.
# Ref: https://pytorch.org/tutorials/recipes/regional_compilation.html

# torch._dynamo.config.inline_inbuilt_nn_modules = False


def run_fwd_once(model, sample_inp):
    """Convenient function to run model once using correct input unpack."""
    with torch.no_grad():
        if isinstance(sample_inp, dict) or all(
            hasattr(sample_inp, k) for k in ("keys", "values", "items")
        ):
            out = model(**sample_inp)
        elif isinstance(sample_inp, tuple):
            out = model(*sample_inp)
        elif isinstance(sample_inp, torch.Tensor):
            out = model(sample_inp)
        else:
            try:
                #   assume user provided input is ready-to-run...
                out = model(sample_inp)
            except RuntimeError:
                logger.info(
                    f"Unknown data structure for example_input.{type(sample_inp)} Please check."
                )
    return out


def dfs_gm(
    gm,
    targetOp=None,
    target_node=None,  # only match one, not one in many
    start_nodes=None,
    stop_nodes=None,
    stopOp=None,
    stop_depth=99999,
    show_results=False,
    find1stOnly=False,
    reverse=False,
    prescreenOp=None,
    hook=None,
    return_nodes=False,
    lut_fx_mod_name_to_org: Optional[Dict[str, str]] = None,
):
    """Depth-First Search at FX IR level, to replace our old TorchScript equivalent func
    Because FX IR is a higher level IR, should have much fewer
    distractions than Aten IR. In terms of graph analysis, it should be better than TorchScript's IR

    Args:
        gm (GraphModule): graph to be searched, in FX IR
        targetOp (nn.Module|str, optional): search target. could be nn.Module, like nn.Linear,
                                            or str like 'conv2d', 'linear'... Defaults to None.
        target_node (node, optional): similar to targetOp but it's a node instead. Defaults to None.
        start_nodes (node, optional): can specify which node the DFS will start from, will use
                                    "placeholder" or "output" nodes (if reverse==True) if None.
        stop_nodes (_type_, optional): search stop criteria. Defaults to None.
        stopOp (nn.Module|str, optional): similar to targetOp/stop_nodes. Defaults to None.
        stop_depth (int, optional): use depth as a stop criteria. Defaults to 99999.
        show_results (bool, optional): print search results, for debug purpose. Defaults to False.
        find1stOnly (bool, optional): stop after first match is found. Defaults to False.
        reverse (bool, optional): search in reverse direction, i.e. end to start. Defaults to False.
        prescreenOp (nn.Module|str, optional): Ops to ignore. Defaults to None.
        hook (callable, optional): a function to apply to the node if search criteria is satisfied.
        return_nodes (bool, optional): default return is a dict, could be simplified to nodes only
                                        if this flag is True. sometimes we may want to do further
                                        analysis on the nodes we found, like further DFS on each of
                                        the nodes. hence, convenient to return nodes instead of dict

    Returns:
        dict|str: a dict like {(node, line_num):depth, ...} or just the node names

    NOTE:
    1. For transformers, don't want to search the branch coming from 'attention_mask' or
        'token_type_id', hence we screen out placeholder nodes with those keywords.
    2. When parsing fx nodes back to original module name, if searching for func, like bmm/matmul,
        will return the module containing that function. We utilize "line_num" in the return to make
        sure node/module unique, as same op could be called twice in one module, but this "line_num"
        could change with model definition, raise version/compatibility concern.
    3. In order to sort return dict by depth, we need to use "key=" in sort func.
        Since dict.items() creates an iterable, like [(k0,v0), (k1,v1), (k2,v2),...]
        "key=..." in sorted() is a callable to apply to each item (tuple) from iterable to extract
        the "key" that will be used for sorting, i.e. "lambda item: item[1])" will give us "depth"
        https://stackoverflow.com/questions/613183/how-do-i-sort-a-dictionary-by-value
    """
    assert (
        targetOp is not None or target_node is not None
    ), "Please provide either targetOp or target_node. Cannot be None for both."
    # prep a few counter, hash tables, and lists
    if stop_nodes is None:
        stop_nodes = []
    if stopOp is None:
        stopOp = []
    if prescreenOp is None:
        prescreenOp = []

    node_found = {}
    node_traced = []
    visited = {n: False for n in gm.graph.nodes}
    inp_nodes = [
        n
        for n in gm.graph.nodes
        if n.op == "placeholder" and "mask" not in n.name and "token" not in n.name
    ]
    out_nodes = [n for n in gm.graph.nodes if n.op == "output"]
    if start_nodes is None:
        start_nodes = inp_nodes if not reverse else out_nodes

    if not isinstance(targetOp, (list, tuple)):
        targetOp = [targetOp]
    if not isinstance(start_nodes, (list, tuple)):
        start_nodes = [start_nodes]
    real_targetOps = []
    for op_i in targetOp:
        real_ops = get_target_op_from_mod_or_str(op_i)
        if isinstance(real_ops, list):
            real_targetOps.extend(real_ops)
        else:
            real_targetOps.append(real_ops)

    if stopOp:
        if not isinstance(stopOp, (list, tuple)):
            stopOp = [stopOp]
        stopOp = [get_target_op_from_mod_or_str(m) for m in stopOp]

    def _dfs(curr_node, depth):
        """_dfs is the recursive function that performs real DFS.

        NOTE:
        1. _dfs() can access all the kwargs from DFS() scope
        2. to check if targetOp or target_node:
            a. if curr_op is a module, (type(curr_op) in targetOp) will work
            b. if curr_op is a func, (curr_op in targetOp) will work
            c. may need to check each str(targetOp_i) in curr_node.name:
        3. some ops could be called twice in one module, like bmm
            need to know the physical line number of each call as an unique identifier
        4. node.meta['stack_trace'] records look like "filepath/name, line num, actual code"
        5. stop_nodes/Op will be included in the found, if they satisfy check 2). make sure the
            starting node (where the search begins) does not satisfy this criteria. if needed, added
            "filters" to stopOp in the outer loop
        6. When deciding next nodes, only care about args that are node, e.g. a func call like
            sum(node1, dim=[a constant]), dim will also be an input but no need for DFS
        7. only when the last node of _dfs is visited (branch merge), a br_end will returned


        Args:
            curr_node (node): current node
            depth (int): depth in the graph
        """

        if visited[curr_node]:
            return curr_node

        curr_op = get_target_op_from_node(curr_node, gm)
        visited[curr_node] = True

        # 1) before doing anything, filter by prescreenOp first!
        if any(Op_i == curr_op for Op_i in prescreenOp):
            return
        # 2) check if this node is targetOp or target_node, see Note 2
        if (
            curr_node == target_node
            or type(curr_op) in real_targetOps
            or curr_op in real_targetOps
        ):
            # physical line number of the call as an unique identifier, see Note 3
            calltrace = curr_node.meta["stack_trace"].split("  File ")[-1]
            line_num = int(calltrace.split(", ")[1].replace("line ", ""))
            node_found[(curr_node, line_num)] = depth

            if show_results:
                logger.info(f"{curr_node.name}")
            if find1stOnly:
                return  # TODO apply hooks first?
            if hook and callable(hook):
                hook(curr_node)
        # 3) check if this node satisfies stopOp/stop_node, see Note 5
        if curr_node in stop_nodes or curr_op in stopOp or depth > stop_depth:
            if show_results:
                logger.info("Ran into a stop criteria, either node, op, or depth ")
            return
        node_traced.append(curr_node)

        next_nodes = (
            curr_node.all_input_nodes if reverse else list(curr_node.users.keys())
        )  # see Note 6
        for i, next_i in enumerate(next_nodes):
            if i > 0:
                if show_results:
                    logger.info(
                        f"Start searching branch #{i}/{len(next_nodes)} of node {curr_node.name}"
                    )

            br_end = _dfs(next_i, depth + 1)  # see Note 7

            if br_end is not None:
                if show_results:
                    logger.info(f"Current branch merged into {br_end.name}")

        return

    for node_i in start_nodes:
        _dfs(node_i, depth=0)
        if show_results:
            logger.info(
                f"Total nodes traced={len(node_traced)}, found {len(node_found)} "
                f"that satisfy the criteria {targetOp}"
            )

    if not return_nodes:
        org_mod_names = {}
        for n_ln, d in node_found.items():
            n, line_num = n_ln  # unpack tuple
            org_mod_names[
                get_org_mod_name_of_fx_node(n, gm, lut_fx_mod_name_to_org), line_num
            ] = d  # see Note 2

        return dict(
            sorted(org_mod_names.items(), key=lambda item: item[1])
        )  # see Note 3

    return node_found


def find_conv_on_shortcut_gm(
    gm: torch.fx.GraphModule,
    lut_fx_mod_name_to_org: Optional[Dict[str, str]] = None,
    lut_name_to_mod=None,
):
    """Identify Conv on shortcut using FX GM DFS
    It's (almost) specific for ResNet-like CNNs, will return a list of module names (as used in the
    original model, not FX module names)

    NOTE:
    1. Not interested in embedding here, could check either the module type or input data type (Long
        or INT), but instead we simply check node.name and node.target
    2. shortcut Convs must have channel_out > channel_in, on the other hand, FPN must have
        channel_out <= channel_in
    3. find 'add' nodes, add is the end of a branch, return will be a dict like
        {(node0, line_num0):depth0,... }
    4. search upward to find where the branch begins, i.e. branch_begin should have more than 1
        child, here we set a safety and look for within 20 levels.
        2.1 make sure the branch_start nodes we found are the same node. possible reason for
            different "branch start" nodes are:
            a. run into search limit (20 levels) or
            b. branch-in-branch situation, eg a concat exists in one of the branches
            c. they are really from different branches
            but if one of the branch_start node is a parent of others, that should suffice

    5. count levels of each branch, decide which one is the shortcut
    """

    if lut_name_to_mod is None:
        lut_name_to_mod = {}

    # 1. Find "add" nodes, including inplace add as some may use "out+=shortcut"
    nodes_add = dfs_gm(gm, ["add"], return_nodes=True)

    qconv_candidate = []
    search_limit = 20
    for node_i, _ in nodes_add:
        if any(
            ("embed" in ni.name or "embed" in str(ni.target))
            for ni in node_i.all_input_nodes
        ):
            continue

        # 2. Find where this branch begins, ie common node when going up along parent nodes of Add
        levels_from_add = [1] * len(node_i.all_input_nodes)
        branch_start = [None] * len(node_i.all_input_nodes)
        for j, p_j in enumerate(node_i.all_input_nodes):
            while len(p_j.users) < 2 and levels_from_add[j] < search_limit:
                levels_from_add[j] += 1
                if p_j.all_input_nodes:
                    p_j = p_j.all_input_nodes[
                        0
                    ]  # TODO what if there are more than 1 inputs?
                else:
                    break
            branch_start[j] = p_j

        # 2.1 make sure all the branch_start node are the same node
        if any(br_i != branch_start[0] for br_i in branch_start[1:]):
            real_branch_start = branch_start[0]
            for br_i in branch_start[1:]:
                new_node_is_common_parent = (
                    len(
                        dfs_gm(
                            gm,
                            target_node=real_branch_start,
                            start_nodes=[br_i],
                            stop_nodes=[real_branch_start],
                        )
                    )
                    > 0
                )
                old_node_is_common_parent = (
                    len(
                        dfs_gm(
                            gm,
                            target_node=br_i,
                            start_nodes=[real_branch_start],
                            stop_nodes=[br_i],
                        )
                    )
                    > 0
                )
                if br_i != real_branch_start and new_node_is_common_parent:
                    real_branch_start = br_i
                elif br_i == real_branch_start or old_node_is_common_parent:
                    # safety check, confirm current branch_start is the real branch start
                    raise NotImplementedError
                else:
                    # rare case, the two inputs to Add are really from different branches,
                    # not a "short cut Add", continue
                    real_branch_start = None
                    break
        else:
            # most likely case, all the branch_start are the same node
            real_branch_start = branch_start[0]

        # 3. find the shortest path, i.e. shortcut
        idxSC = levels_from_add.index(min(levels_from_add))
        if real_branch_start:
            tempNodes = dfs_gm(
                gm,
                [torch.nn.Conv2d],
                start_nodes=[node_i.all_input_nodes[idxSC]],
                stop_nodes=[real_branch_start],
                find1stOnly=True,
                reverse=True,
                return_nodes=True,
            )
            if tempNodes:
                n_conv_i = list(tempNodes.keys())[0][0]
                if n_conv_i.op == "call_module":
                    conv_mod = gm.get_submodule(n_conv_i.target)
                else:
                    # in case aten IR is being used
                    conv_mod_name = get_org_mod_name_of_fx_node(
                        n_conv_i, lut_fx2org=lut_fx_mod_name_to_org
                    )
                    conv_mod = lut_name_to_mod.get(conv_mod_name, None)
                    if not isinstance(conv_mod, torch.nn.Conv2d):
                        continue
                if conv_mod.out_channels > conv_mod.in_channels:  # see Note 2
                    qconv_candidate.append(
                        get_org_mod_name_of_fx_node(
                            n_conv_i, gm, lut_fx_mod_name_to_org
                        )
                    )

    return qconv_candidate


def find_1st_last_gm(
    gm,
    firstOps=None,
    lastOps=None,
    return_1st_last_sep=False,
    lut_fx_mod_name_to_org: Optional[Dict[str, str]] = None,
):
    """Identify the first and last layer of interests
    Usually only interested in Conv and Linear, but could be others as well
    NOTE:
    1. even though we specify "find1stOnly=True", it is very common in transformers that
        shortcuts will make every linear look like "first" linear. Therefore, we will use "depth",
        i.e. .value() of the returned dict from dfs_gm(), to determine the real "first" nodes
    2. when checking min_depth, we add [999] in case candidates list is empty

    TODO in old version, we have "stopOp='prim::TupleUnpack'" as a hack for Llama, need to confirm
    """
    if firstOps is None:
        firstOps = [torch.nn.Conv2d, torch.nn.Linear]
    if lastOps is None:
        lastOps = [torch.nn.Conv2d, torch.nn.Linear]
    first_candidates = dfs_gm(
        gm,
        targetOp=firstOps,
        find1stOnly=True,
        lut_fx_mod_name_to_org=lut_fx_mod_name_to_org,
    )
    last_candidates = dfs_gm(
        gm,
        targetOp=lastOps,
        find1stOnly=True,
        reverse=True,
        lut_fx_mod_name_to_org=lut_fx_mod_name_to_org,
    )

    min_depth = min(list(first_candidates.values()) + [999])
    real_first = [
        mod_lineNo[0]
        for mod_lineNo, depth in first_candidates.items()
        if depth == min_depth
    ]

    min_depth = min(list(last_candidates.values()) + [999])
    real_last = [
        mod_lineNo[0]
        for mod_lineNo, depth in last_candidates.items()
        if depth == min_depth
    ]

    if return_1st_last_sep:
        return real_first, real_last

    return real_first + real_last


def find_single_sided_op_gm(
    gm,
    op_of_interest=None,
    return_LUTs=False,
    verbose=False,
    lut_fx_mod_name_to_org: Optional[Dict[str, str]] = None,
):
    """Try to determine the "polarity" of output of "every nodes" based on their inputs and the Op
    itself, then decide which Conv/Linear (or user-specified Op) will use single-sided quantizer

    Algo:
    1. make a table of "isActOutPositiveOnly" for every node. init as None, will be True/False later
    2. mark those known Ops, then derive the rest from the relationship with the known ones.
    3. check the op/node of interest using "isActOutPositiveOnly" table

    NOTE:
    a. "index_put" or "masked_fill" is filling part of the target tensor using an idx tensor and a
        given src tensor, may fill negative values and turn tensor 2-sided. It's a call_method node
        with args = (input tensor, bool idx, value to be filled)
    b. For those "call_module" nodes, only activations will show up in all_input_nodes,
        weight and bias are "attributes of that module", not nodes (or a get_attr node) on the graph
    c. Because there could be None (undertermined) inputs still, need to check before using the
        "inferred" properties
    d. to(torch.bool) will make tensor 0 or 1 -> bounded and positive only
    e. When checking branches, e.g. Add
        e1. if one of the inputs to Add is bi-dir, Add's output is bi-dir
        e2. if all inputs are determined already, can infer from inputs
    f. for all other Ops, eg, view, reshape, transpose, with "single (meaningful) parent", simply
        makes output polarity = input polarity
    g. if none of the inputs is bi-dir and undetermined (None) still exists, call it a "risky node"
        as it could be inaccurately determined.

    if return_LUTs -> return isActOutPositiveOnly and isActOutBounded
    """
    if op_of_interest is None:
        op_of_interest = [torch.nn.Conv2d, torch.nn.Linear]

    # Step 1. Prepare look-up tables
    isActOutPositiveOnly = {}
    isActOutBounded = {}
    for n in gm.graph.nodes:
        if n.op in ["placeholder", "output"]:
            isActOutPositiveOnly[n] = False
            isActOutBounded[n] = False
        else:
            isActOutPositiveOnly[n] = None
            isActOutBounded[n] = None

    # corresponds to ReLU, ReLU6, Sigmoid
    RectifierOp = [
        torch.nn.functional.relu,
        torch.nn.ReLU,
        torch.nn.functional.hardtanh,
        torch.nn.functional.sigmoid,
        torch.nn.functional.softmax,
        torch.nn.functional.cross_entropy,
        torch.nn.functional.binary_cross_entropy,
    ]
    RectifierKW = ["range"]
    BiDirOp = [
        torch.nn.Conv2d,
        torch.nn.Linear,
        torch.nn.LayerNorm,
        torch.nn.BatchNorm2d,
        torch.matmul,
        torch.bmm,
        torch.nn.functional.scaled_dot_product_attention,
        torch.nn.functional.layer_norm,
        torch.nn.functional.batch_norm,
    ]
    BiDirKW = ["emb", "layernorm", "batchnorm"]
    AddOp = get_target_op_from_mod_or_str("add")
    ConcatOp = [torch.stack, torch.cat]  # TODO check "list construct" on FX graph?
    # Bounded Ops: NOTE silu is only bounded on the neg side.
    BoundedOp = [
        torch.nn.functional.hardtanh,
        torch.nn.functional.sigmoid,
        torch.nn.functional.softmax,
    ]

    loop_counter = 0
    search_limit = 10
    Nundetermined = sum(v is None for v in isActOutPositiveOnly.values())
    while Nundetermined > 0 and loop_counter < search_limit:
        # Step 2: go through all the nodes ONCE and try our best to determine their 'polarity'
        for n in gm.graph.nodes:
            if n.op in ["placeholder", "output"]:
                continue
            currOp = get_target_op_from_node(n, gm)
            parentOps = [get_target_op_from_node(pi, gm) for pi in n.all_input_nodes]
            all_inputs_determined = all(
                isActOutPositiveOnly[pi] is not None for pi in n.all_input_nodes
            )  # see Note c
            inferred_output = all(isActOutPositiveOnly[pi] for pi in n.all_input_nodes)
            inferred_bounded = all(isActOutBounded[pi] for pi in n.all_input_nodes)
            # --- Deterministic cases
            if isinstance(currOp, torch.Tensor):
                # get_attr nodes could end up as tensors
                isActOutPositiveOnly[n] = torch.all(
                    currOp >= 0
                ).item()  # torch True->python True
                isActOutBounded[n] = False
            elif currOp == "to" and torch.bool in n.args:  # see Note d
                isActOutPositiveOnly[n] = True
                isActOutBounded[n] = True
            elif (
                currOp in RectifierOp
                or type(currOp) in RectifierOp
                or any(kw in str(n.target) for kw in RectifierKW)
            ):
                isActOutPositiveOnly[n] = True
                isActOutBounded[n] = currOp in BoundedOp
            elif (
                currOp in BiDirOp
                or type(currOp) in BiDirOp
                or any(kw in str(n.target) for kw in BiDirKW)
            ):
                isActOutPositiveOnly[n] = False
                isActOutBounded[n] = currOp in BoundedOp
            elif currOp == "masked_fill":  # see Note a
                if n.args[2] < 0:
                    isActOutPositiveOnly[n] = False
                elif all_inputs_determined:
                    isActOutPositiveOnly[n] = inferred_output
                isActOutBounded[n] = False
            # --- Dependent cases, can be "undertermined" if insufficient info
            elif currOp in AddOp or currOp in ConcatOp:  # see Note e
                if any((pOp in BiDirOp or type(pOp) in BiDirOp) for pOp in parentOps):
                    isActOutPositiveOnly[n] = False
                    isActOutBounded[n] = False
                elif all_inputs_determined:
                    isActOutPositiveOnly[n] = inferred_output
                    isActOutBounded[n] = inferred_bounded
            elif len(n.all_input_nodes) == 1 and all_inputs_determined:  # see Note f
                isActOutPositiveOnly[n] = inferred_output
                isActOutBounded[n] = inferred_bounded
            else:
                # NOTE unknown ops, do nothing for now, will likely resolved after a few passes
                continue
            # One round of polarity check completed

        loop_counter += 1
        Nundetermined = sum(v is None for v in isActOutPositiveOnly.values())

    if loop_counter == 10 and Nundetermined > 0 and verbose:
        logger.warning("Polarity check reaches loop limit (10) and still not finished.")

    if return_LUTs:
        return isActOutPositiveOnly, isActOutBounded

    node_of_interest = dfs_gm(
        gm,
        targetOp=op_of_interest,
        return_nodes=True,
        lut_fx_mod_name_to_org=lut_fx_mod_name_to_org,
    )

    SingleSidedOps = []
    risky_nodes = []
    for n, _ in node_of_interest:
        input_pos = [isActOutPositiveOnly[p] for p in n.all_input_nodes]
        if (False not in input_pos) and (None in input_pos):  # see Note g
            risky_nodes.append(n)
        if all(input_pos):
            SingleSidedOps.append(
                get_org_mod_name_of_fx_node(n, gm, lut_fx_mod_name_to_org)
            )

    if risky_nodes:
        logger.warning(
            "The polarity (single or double-sided) of some of the nodes cannot be \n"
            "accurately determined. It won't be an issue for INT8, but please double check the \n"
            "quantizer selection of critical layers. Use 'qspecial_layers' if needed"
        )

    return SingleSidedOps


def find_qkvsync_candidates_gm(
    gm, return_nodes=False, lut_fx_mod_name_to_org: Optional[Dict[str, str]] = None
):
    """Identify groups of Linears that share the same parent. It's a transformer-specific feature.

    NOTE:
    1. QanyNet5() will simply force the candidates to use the same nn.parameter for clipvals, i.e.
        QKV nodes will be using the same pointer to the same tensor object, then optimizer will take
        care of the rest.
    2. QanyNet is expecting LUT[an instance of nn.Linear] = another instance of nn.Linear
        e.g., LUT[mod1]=mod2, LUT[mod2]=mod2, LUT[mod3]=mod2, when mod1/2/3 shares the same parent
        BUT, because in latest pytorch, modules in GraphModule (named like l_bert_encoder_0_xxx )
          may not be identical the original ones, e.g. bert.encoder.0.xxx
          (seems like some wrappers might have been applied during Dynamo tracing)
          so we use original "module names" instead, and TODO need to adjust Qanynet/makeModules()

    """
    all_linears = dfs_gm(gm, targetOp=[torch.nn.Linear], return_nodes=True)

    # 1. sort/group by depth
    LUTdep2nodes = {}
    for n_ln, depth in all_linears.items():
        if depth in LUTdep2nodes:
            LUTdep2nodes[depth].append(n_ln[0])
        else:
            LUTdep2nodes[depth] = [n_ln[0]]

    # 2. throw away depths with single node
    LUTdep2nodes = {
        depth: nodes for depth, nodes in LUTdep2nodes.items() if len(nodes) > 1
    }
    if return_nodes:
        return LUTdep2nodes

    # 3. re-arranging the info into desired format for later use
    my_1st_sibling = {}
    Nlayers = 0
    Nshared_parents = 0
    for depth, nodes in LUTdep2nodes.items():
        parents = [ni.all_input_nodes[0] for ni in nodes]
        org_mod_names = [
            get_org_mod_name_of_fx_node(ni, gm, lut_fx_mod_name_to_org) for ni in nodes
        ]
        if all(p == parents[0] for p in parents[1:]):
            Nshared_parents += 1
            for org_name_i in org_mod_names:
                my_1st_sibling[org_name_i] = org_mod_names[0]
                Nlayers += 1

    return my_1st_sibling


def find_silu_gm(gm, lut_fx_mod_name_to_org: Optional[Dict[str, str]] = None):
    """Special handle for Conv following silu, specific for EffDet and etc
    LLM could use SiLU as well (llama?), but not relavent to this func
    """
    siluConv = {}
    for n, _ in dfs_gm(gm, targetOp=[torch.nn.Conv2d], return_nodes=True):
        p_node = n.all_input_nodes[0]
        gp_nodes = p_node.all_input_nodes

        pOp = get_target_op_from_node(p_node, gm)
        gpOp = get_target_op_from_node(gp_nodes[0], gm) if gp_nodes else None

        if torch.nn.functional.silu in [pOp, gpOp]:
            siluConv[get_org_mod_name_of_fx_node(n, gm, lut_fx_mod_name_to_org)] = {
                "qa_mode": "qsilu"
            }

    return siluConv


def find_rpn_fpn_gm(
    gm,
    verbose=False,
    Nsubgraph=0,
    lut_fx_mod_name_to_org: Optional[Dict[str, str]] = None,
):
    """For object detection CNN models, RPN (RegionProposalNetwork) and FPN (FeaturePyramidNetwork)
    are commonly used. prefer to skip them, but may be ok to quantize in some cases.

    NOTE:
    1. We try to identify the architecture, not the class from torchvision, such as
        from torchvision.ops import FeaturePyramidNetwork
        from torchvision.models.detection.rpn import RegionProposalNetwork
        FPN signature is interp/add/interp/add/interp/add
    2. seems like Dynamo cannot trace obj det models WITHOUT a graph break, e.g. FasterRCNN in
        torchvision.models.detection. So here we assume we need to identify from a subgraph
    3. FPN needs to start from the 1st interp, "add" also has 2 users. "fpn end" will add or concat
        at least 4 layers of feature maps
    4. start and end nodes have to be list. DFS output is {(org_mod_name, line_num):depth, ...},
        only need the org_mod_names
    5. There are also 4 point-wise Conv/BN, referred to as inner_block, connected to the backbone.
        first PW Conv should be right above the interp and BN of the fpn_start, the other 3 Conv/BN
        are one of the inputs to FPN Adds
    6. RPN searching is not implemented yet.

    """

    add_op = get_target_op_from_mod_or_str("add")
    fpn_convs = []
    fpn_inner_blocks = []

    n_interp = [
        n
        for n in gm.graph.nodes
        if n.op == "call_function" and n.target is torch.nn.functional.interpolate
    ]
    # 1. search for FPN, see Note 1
    for n in n_interp:
        child = list(n.users.keys())[0]  # interpolation has only 1 child
        Grandchild = [
            gc
            for gc in child.users.keys()
            if gc.target is torch.nn.functional.interpolate
        ]
        chkChild = child.target in add_op
        chkGC = (
            len(Grandchild) == 1
            and Grandchild[0].target is torch.nn.functional.interpolate
        )
        if chkGC:
            GreatGrandchild = list(Grandchild[0].users.keys())[0]
            chkGGC = GreatGrandchild.target in add_op
        else:
            chkGGC = False

        if chkChild and chkGC and chkGGC:
            # found FPN, see Note 3
            fpn_st_nodes = n.all_input_nodes
            while len(fpn_st_nodes[0].users) != 2:
                fpn_st_nodes = fpn_st_nodes[0].all_input_nodes
            fpn_end_node = GreatGrandchild
            while len(fpn_end_node.all_input_nodes) < 4:
                fpn_end_node = list(fpn_end_node.users.keys())[0]
            fpn_convs = dfs_gm(
                gm,
                targetOp=[torch.nn.Conv2d],
                start_nodes=fpn_st_nodes,
                stop_nodes=[fpn_end_node],
                lut_fx_mod_name_to_org=lut_fx_mod_name_to_org,
            )
            fpn_convs = [mod_name for mod_name, ln in fpn_convs.keys()]  # see Note 4
            fpn_adds = dfs_gm(
                gm,
                targetOp=add_op,
                start_nodes=fpn_st_nodes,
                stop_nodes=[fpn_end_node],
                return_nodes=True,
            )
            # will check for Conv/BN as parent and grandparent
            nodes_to_chk4ConvBN = [na for na, ln in fpn_adds.keys()] + [n]

            for nj in nodes_to_chk4ConvBN:
                for p in nj.all_input_nodes:
                    gp = p.all_input_nodes[0]
                    tar_op_p = get_target_op_from_node(p, gm)
                    tar_op_gp = get_target_op_from_node(gp, gm)
                    if (
                        isinstance(tar_op_p, torch.nn.BatchNorm2d)
                        or tar_op_p is torch.nn.functional.batch_norm
                    ) and (
                        isinstance(tar_op_gp, torch.nn.Conv2d)
                        and tar_op_gp.kernel_size in [1, (1, 1)]
                    ):
                        fpn_inner_blocks.append(
                            get_org_mod_name_of_fx_node(
                                gp, lut_fx2org=lut_fx_mod_name_to_org
                            )
                        )
            fpn_convs += fpn_inner_blocks

            if verbose:
                logger.info(f"In subgraph {Nsubgraph} found FPN Convs {fpn_convs}")
            break

    # TODO 2. search for RPN not implemented yet

    return fpn_convs


def find_and_prep_bmm_gm(gm, lut_fx_mod_name_to_org: Optional[Dict[str, str]] = None):
    """Previously with TorchScript, we use this func to perform 2 tasks:
        a) create QBmms, and then attach them to the model,
        b) set up qcfg["which2patch_contextmanager"] so that patch_torch_bmm() context
            manager can work properly

    NOTE:
    1. we want to keep both inputs to bmm2 in 8bit, and the one handles the output from softmax will
        use Qmax quantizer, i.e. m1 in bmm2(m1, m2)
    2. because now it's processed under FX GM, attaching new modules to GM is not useful TODO will
        put QBmm creation info in the return dict and let QanyNet() handle it.
    3. if sdpa is enabled, we will not find any bmm/matmul
    4. dfs_gm returns {(node, line_num):depth, ...}, rearrange to
        {"module name":[(node1, line_num1, depth1), (node2, line_num2, depth2)],...}
        keep depth to determine the call sequence of nodes, ie. same as line num sequence
    5. n_ln_depth is a list like [(n0,l0,d0), (n1,l1,d1), (n2,l2,d2),...]
        "key="" in sorted() is a callable to be applied to each item from iterable,
        i.e. (n,l,d) tuple in this case, to extract the "key" that will be used for sorting
    6. use line number to determine which is bmm1, which is bmm2
    """

    all_bmms = dfs_gm(gm, targetOp=[torch.bmm], return_nodes=True)
    all_matmuls = dfs_gm(gm, targetOp=[torch.matmul], return_nodes=True)
    all_sdpas = dfs_gm(
        gm,
        targetOp=[torch.nn.functional.scaled_dot_product_attention],
        return_nodes=True,
    )

    return_dict = {"which2patch_contextmanager": None, "layers_with_bmm": {}}

    # check which2patch, either bmm or matmul, can't be both
    Nbmm_found = len(all_bmms)
    Nmatmul_found = len(all_matmuls)
    if Nbmm_found > 0 and Nmatmul_found == 0:
        return_dict["which2patch_contextmanager"] = "torch.bmm"
        LUT2sort = all_bmms
    elif Nbmm_found == 0 and Nmatmul_found > 0:
        return_dict["which2patch_contextmanager"] = "torch.matmul"
        LUT2sort = all_matmuls
    else:
        if Nbmm_found > 0 and Nmatmul_found > 0:
            raise RuntimeError(
                "Both bmm and matmul are found. Not sure which to patch."
            )
        if Nbmm_found == 0 and Nmatmul_found == 0 and len(all_sdpas) > 0:
            logger.warning(
                "No bmm and matmul are found. Likely SDPA is enabled. "
                "Will patch nothing!"
            )

        return return_dict

    LUTmodname2linenum = {}  # see Note 4
    for node_line_num, depth in LUT2sort.items():
        node, line_num = node_line_num
        org_mod_name = get_org_mod_name_of_fx_node(node, gm, lut_fx_mod_name_to_org)
        if org_mod_name in LUTmodname2linenum:
            LUTmodname2linenum[org_mod_name] += [(node, line_num, depth)]
        else:
            LUTmodname2linenum[org_mod_name] = [(node, line_num, depth)]

    # sort the items in each module "by depth" to determine which is bmm1 and bmm2, see Note 5
    for mod_name, n_ln_depth in LUTmodname2linenum.items():
        sorted_n_ln_depth = sorted(n_ln_depth, key=lambda item: item[2])

        n_bmm2, ln, depth = sorted_n_ln_depth[1]
        find_softmax_from_bmm2_m1 = dfs_gm(
            gm,
            targetOp=["softmax"],
            start_nodes=n_bmm2.all_input_nodes[0],
            stop_depth=10,
            reverse=True,
            return_nodes=True,
        )
        find_softmax_from_bmm2_m2 = dfs_gm(
            gm,
            targetOp=["softmax"],
            start_nodes=n_bmm2.all_input_nodes[1],
            stop_depth=10,
            reverse=True,
            return_nodes=True,
        )
        assert (
            len(find_softmax_from_bmm2_m1) == 1 and len(find_softmax_from_bmm2_m2) == 0
        ), "cannot find softmax in bmm2's first input. please double check"

        return_dict["layers_with_bmm"][mod_name] = [
            ln for n, ln, depth in sorted_n_ln_depth
        ]  # see Note 6

    logger.info(f"Found {Nbmm_found} torch.bmm OPs and {Nmatmul_found} torch.matmul")
    logger.info(
        f"context manager is set to intercept {return_dict['which2patch_contextmanager']}"
    )
    return return_dict


def add_prefix_to_list_or_dict(list_or_dict, prefix, update_both_k_and_v=False):
    """For graph breaks, the names we identify could be just partial name
    need to add a prefix in some cases
    """
    if prefix in [None, ""] or not isinstance(prefix, str):
        # safety check in case no prefix or none-str is passed in
        return list_or_dict

    if isinstance(list_or_dict, list):
        new_list_or_dict = [f"{prefix}.{it}" for it in list_or_dict]
    elif isinstance(list_or_dict, dict):
        if update_both_k_and_v:
            new_list_or_dict = {
                f"{prefix}.{k}": f"{prefix}.{v}" for k, v in list_or_dict.items()
            }
        else:
            new_list_or_dict = {f"{prefix}.{k}": v for k, v in list_or_dict.items()}
    else:
        logger.warning("Inputs is not list or dict, prefix was not added")
        new_list_or_dict = list_or_dict

    return new_list_or_dict


def model_analyzer(
    model,
    sample_inp,
    qcfg,
    plotsvg=None,
):
    """Main entry point for model analysis, basically try to determine which layers to quantize.
    Two options:
    1: User provides layer name patterns "TO QUANTIZE" (NOT layers "TO SKIP")
        a. this option will not perform any tracing but only name matching
            !!! Preferred option for LLM, as LLM tracing will be slow and prone to fail
        b. this option does not support BMMs, since BMM are not layers.
        c. User can still specify additional layers (exact match) in ['qskip_layer_name']
    2. Use Dynamo to replace TorchScript tracing in old qmodel_prep(),

    NOTE:
    1. Will use lut_weight2modname to find the prefix for subgraphs, should graph break. As module
        seems to have extra layer of wrapper from Dynamo, matching module, i.e. id(module), may lead
        to incorrect results, matching weights (tensor) should be consistent.
    2. For subgraph, we might be getting a partial "original name", such as layer.0.xxx instead of
        "encoder".layer.0.xxx => we can use param pointers to identify the real original name and
        "subtract" the partial name out and get the prefix
    3. may need to rewrite DFS to utilize "custom stack" for really large LLM, e.g. there is an
        upper limit don't use recursive directly)
    4. qcfg.get("QBmm") is set by qmodel_prep() based on qbmm_nbits.
    5. bmm2 m1 (input was from softmax) needs special treatment

    """

    qcfg["N_backend_called"] = 0
    lut_weight2modname = {
        mod.weight: n
        for n, mod in model.named_modules()
        if isinstance(mod, (torch.nn.Linear, torch.nn.Conv2d))
    }  # see Note 1

    def cus_backend_model_analyzer(
        gm_fx,
        exam_inp,  # pylint: disable=unused-argument
        is_transformers=True,
        plotsvg=None,
    ):
        """Need to collect the following info:
        1. conv on shortcut (ResNet specific)
        2. first and last layer
        3. single-/double- sided nodes, eg, following ReLU
        4. FPN (also see TODO in RPN)
        5. identify SiLU
        6. bmm/matmul, create and attach QBmm if needed, setup context manager
        7. QKV sync, only sync activation clipvals

        for transformers
        a. try not to trace from "attention_mask" or non-"input_ids" inputs
        b. do not skip the first. but can skip the "last" as it's usually the prediction head

        NOTE:
        1. basic tests are like below, it should be covered by unittest in the future
                mod_lin_all = dfs_gm(gm_fx, torch.nn.Linear)
            mod_lin_1st = dfs_gm(gm_fx, torch.nn.Linear, find1stOnly=True)
            mod_lin_last = dfs_gm(gm_fx, torch.nn.Linear, reverse=True, find1stOnly=True)
            bmm_all = dfs_gm(gm_fx, [torch.matmul, torch.bmm])
            sdpa_all = dfs_gm(gm_fx, [torch.nn.functional.scaled_dot_product_attention])
            add_all = dfs_gm(gm_fx, ['add'])
        2.  for subgraphs, the "first" layers are usually not really starting from the beginning
            of the full graph/model. also, it's difficult to know whether it ends at where the
            full graph ends, hence 'last' could be incorrect, either.
            => cannot determine first/last in this case. simply adding nothing to skip_candidates
        3. if transformer with graph_break, 'last' may be incorrect
        4. if no graph breaks or the first of the subgraphs, should be the top level with no prefix
        5. find_and_prep_bmm_gm() return dict looks like {"which2patch_contextmanager": None,
            "layers_with_bmm": {} }
        6. find_qkvsync_candidates_gm() return dict is {'layer.name.0':'layer.name.1',...}, need to
            add prefix to both k and v


        """
        qcfg["N_backend_called"] += 1
        lut_fx_mod_name_to_org = {
            n.replace(".weight", ""): lut_weight2modname[p]
            for n, p in gm_fx.named_parameters()
            if p in lut_weight2modname
        }
        prefix = None
        if qcfg["N_backend_called"] > 1:  # subgraph found, see Note 2
            # TODO this approach only works for FX IR (call_module nodes are not functionalized)
            #       need an update for Aten IR cases
            for n in gm_fx.graph.nodes:
                if n.op == "call_module":
                    mod = gm_fx.get_submodule(n.target)
                    if isinstance(mod, (torch.nn.Linear, torch.nn.Conv2d)):
                        real_org_modname = lut_weight2modname[mod.weight]
                        part_org_modname = get_org_mod_name_of_fx_node(
                            n, gm_fx, lut_fx_mod_name_to_org
                        )
                        idx = real_org_modname.rindex(part_org_modname)
                        if idx > 1:
                            prefix = real_org_modname[: idx - 1]  # remove trailing '.'
                            break

        if plotsvg:
            # Local
            from fms_mo.fx.utils import plot_graph_module

            if not isinstance(plotsvg, str):
                plotsvg = f"debug{qcfg['N_backend_called']}.svg"
            plot_graph_module(
                gm_fx,
                outputname=plotsvg,
                show_details=True,
                Nnode_to_plot=1000,
                lut_fx_mod_name_to_org=lut_fx_mod_name_to_org,
            )

        # Graph checks begin. Use append, add prefix if needed
        skip_candidates = []
        # Check 1. Conv on shortcut
        all_conv = [
            m
            for _, m in gm_fx.named_modules()
            if isinstance(m, torch.nn.Conv2d) or issubclass(type(m), torch.nn.Conv2d)
        ]
        # if gm is using aten IR, only ops can be seen, no modules.
        conv_ops = dfs_gm(
            gm_fx,
            targetOp=[torch.nn.Conv2d, torch.nn.functional.conv2d],
            return_nodes=True,
        )
        lut_name_to_mod = {n: m for m, n in qcfg["LUTmodule_name"].items()}
        if len(all_conv) > 0 or len(conv_ops) > 0:
            skip_candidates += find_conv_on_shortcut_gm(
                gm_fx, lut_fx_mod_name_to_org, lut_name_to_mod
            )

        # Check 2. first/last, see Note 2 and 3, NOTE that transformers are handled differently
        if qcfg["N_backend_called"] > 1:
            skip_candidates += []
        elif not is_transformers:
            # see Note 4
            skip_candidates += find_1st_last_gm(
                gm_fx, lut_fx_mod_name_to_org=lut_fx_mod_name_to_org
            )
        qcfg["qskip_layer_name"] += add_prefix_to_list_or_dict(skip_candidates, prefix)

        # Check 3: single/double sided
        qcfg["qsinglesided_name"] += add_prefix_to_list_or_dict(
            find_single_sided_op_gm(
                gm_fx, lut_fx_mod_name_to_org=lut_fx_mod_name_to_org
            ),
            prefix,
        )

        # Check 4: identify RPN/FPN
        # temporarily disable this check, TODO test find_rpn_fpn_gm() again

        # NOTE: The following 3 funcs return dict instead of list. Use update() instead of append().
        # Check 5: Conv+SiLU
        qcfg["qspecial_layers"].update(
            add_prefix_to_list_or_dict(
                find_silu_gm(gm_fx, lut_fx_mod_name_to_org), prefix
            )
        )

        # Check 6: BMM
        temp_dict = find_and_prep_bmm_gm(gm_fx, lut_fx_mod_name_to_org)  # see Note 5
        if len(temp_dict["layers_with_bmm"]) > 0:
            temp_dict["layers_with_bmm"] = add_prefix_to_list_or_dict(
                temp_dict["layers_with_bmm"], prefix
            )
            qcfg["bmm_prep"]["which2patch_contextmanager"] = temp_dict[
                "which2patch_contextmanager"
            ]
            qcfg["bmm_prep"]["layers_with_bmm"].update(temp_dict["layers_with_bmm"])
            # make sure there are ONLY 2 bmm per layer (self_attention). some models may use
            # additional bmm/matmuls. Raise warning if that's the case.
            num_layers = len(temp_dict["layers_with_bmm"])
            num_bmms = 0
            seen_line_num = []
            for line_nums in temp_dict["layers_with_bmm"].values():
                num_bmms += len(line_nums)
                for line_num in line_nums:
                    if line_num not in seen_line_num:
                        seen_line_num.append(line_num)
            qcfg["bmm_prep"]["bmm_only_in_self_attn"] = True
            if num_bmms != num_layers * 2 or len(seen_line_num) != 2:
                qcfg["bmm_prep"]["bmm_only_in_self_attn"] = False
                logger.warning(
                    "This model uses additional matmul/bmm other than those in self-attention. "
                    "If you plan to quantize self-attention, please note that the additional bmms "
                    "may also be quantized!"
                    f"{temp_dict['layers_with_bmm']}\n"
                )

        # Check 7: QKV
        temp_dict = find_qkvsync_candidates_gm(
            gm_fx, lut_fx_mod_name_to_org=lut_fx_mod_name_to_org
        )  # see Note 6
        temp_dict = add_prefix_to_list_or_dict(
            temp_dict, prefix, update_both_k_and_v=True
        )
        qcfg["qkvsync_my_1st_sibling"].update(temp_dict)
        # NOTE we put these info into qcfg because some processings need
        #       TO BE CONTINUED after Dynamo completed. see below

        return gm_fx

    # --- perform tracing with Dynamo
    # Standard
    from functools import partial

    # Third Party
    from transformers import PreTrainedModel

    if issubclass(type(model), torch.nn.Module):
        model_param_size = (
            sum(p.numel() for p in model.parameters()) / 1e9
        )  # in billions
        model_to_be_traced = model.forward
    else:
        model_to_be_traced = model
        model_param_size = 999

    transformer_model_classes = (PreTrainedModel,)

    if available_packages["torchvision"]:
        # Third Party
        # pylint: disable = import-error
        from torchvision.models import VisionTransformer

        transformer_model_classes += (VisionTransformer,)

    is_transformers = issubclass(type(model), transformer_model_classes)
    if model_param_size > 1:
        # Standard
        import sys

        org_recur_lim = sys.getrecursionlimit()
        sys.setrecursionlimit(4000)  # default is 1000

    cus_bknd = partial(
        cus_backend_model_analyzer,
        is_transformers=is_transformers,
        plotsvg=plotsvg,
    )

    for it in ["qskip_layer_name", "qsinglesided_name"]:
        if it not in qcfg:
            qcfg[it] = []
    for it in ["qkvsync_my_1st_sibling", "qspecial_layers"]:
        if it not in qcfg:
            qcfg[it] = {}
    if "bmm_prep" not in qcfg:
        qcfg["bmm_prep"] = {"which2patch_contextmanager": None, "layers_with_bmm": {}}

    if is_transformers:
        # NOTE simplified method to determine 1st/last modules for transformers.
        # will not work if model has multiple parallel heads at the end, e.g. obj det
        def call_seq_hook(mod, *_args, **kwargs):
            mod_name = kwargs.get("mod_name", lut_weight2modname.get(mod.weight, None))
            if mod_name is None:
                raise RuntimeError("cannot determine module name, plz check model.")

            qcfg["mod_call_seq"].append(mod_name)

        h_hooks = []
        qcfg["mod_call_seq"] = []
        for n, m in model.named_modules():
            if isinstance(m, (torch.nn.Linear, torch.nn.Conv2d)):
                h_hooks.append(
                    m.register_forward_hook(partial(call_seq_hook, mod_name=n))
                )

        with torch.no_grad():
            run_fwd_once(model, sample_inp)

        for h in h_hooks:
            h.remove()

        # only add last layer
        qcfg["qskip_layer_name"] += [qcfg["mod_call_seq"][-1]]

        if available_packages["torchvision"]:
            # unless it's a ViT, skip first Conv as well
            if issubclass(type(model), VisionTransformer) and isinstance(
                model.get_submodule(qcfg["mod_call_seq"][0]), torch.nn.Conv2d
            ):
                qcfg["qskip_layer_name"] += [qcfg["mod_call_seq"][0]]

    with torch.no_grad():
        model_opt = torch.compile(
            model_to_be_traced,
            backend=cus_bknd,
        )
        run_fwd_once(model_opt, sample_inp)

        del model_opt

    if model_param_size > 1:
        sys.setrecursionlimit(org_recur_lim)

    # ------ model analysis is finished, but there are a few remaining things to be done

    # a) qkvsync dict update from "module names" to "module instances"
    #   NOTE when graph break happened, qkvsync() may only find partial QKV names. For example,
    # as opposed to ["model.layers.0.self_attn.q_proj", ..., "model.layers.1.self_attn.q_proj", ...]
    # it may report ["self_attn.q_proj", "self_attn.k_proj", ...]
    # Therefore, length of qcfg["qkvsync_my_1st_sibling"] will be much shorter and keys of this dict
    # won't exist in full list (like all_linears below).
    all_linears = set(
        n for n, m in model.named_modules() if isinstance(m, torch.nn.Linear)
    )

    if any(k not in all_linears for k in qcfg["qkvsync_my_1st_sibling"]):
        # qcfg["qkvsync_my_1st_sibling"] dict is like {q:q, k:q, v:q,...}, here we need a simpler
        # dict like {q:[q,k,v], gate:[up, gate]}
        lut_all_siblings = {}
        for me, sib_1st in qcfg["qkvsync_my_1st_sibling"].items():
            if sib_1st not in lut_all_siblings:
                lut_all_siblings[sib_1st] = [sib_1st]
            elif me not in lut_all_siblings[sib_1st]:
                lut_all_siblings[sib_1st].append(me)

        full_sib_list = {}
        for me, all_sibs in lut_all_siblings.items():
            partial_matches = [lin for lin in all_linears if me in lin]
            # here lin is full_name, me and all_sibs are partial
            for lin in partial_matches:
                prefix = lin[: lin.index(me)]
                for sib in all_sibs:
                    full_sib_list[prefix + sib] = prefix + me
                    all_linears.remove(prefix + sib)
        # all_linears will still have down_proj, out_proj, lm_head, and maybe others
        qcfg["qkvsync_my_1st_sibling"] = full_sib_list

    updated_dict = {
        model.get_submodule(mod): model.get_submodule(sib)
        for mod, sib in qcfg["qkvsync_my_1st_sibling"].items()
    }
    qcfg["qkvsync_my_1st_sibling"] = updated_dict

    # b) qbmm creation and attaching to model
    if qcfg.get("QBmm"):  # see Note 4
        # Local
        QBmm = qcfg["mapping"]["matmul_or_bmm"]

        qcfg["which2patch_contextmanager"] = qcfg["bmm_prep"][
            "which2patch_contextmanager"
        ]
        isbmm = qcfg["which2patch_contextmanager"] == "torch.bmm"
        for mod_name, line_nums in qcfg["bmm_prep"]["layers_with_bmm"].items():
            mod_bmm_happened = model.get_submodule(mod_name)
            for whichQBmm, ln in enumerate(line_nums, start=1):
                nbits = qcfg[f"nbits_bmm{whichQBmm}"]
                newQBmm = QBmm(
                    num_bits_m1=max(nbits, 8) if whichQBmm == 2 else nbits,
                    num_bits_m2=nbits,
                    qm1_mode=qcfg[f"bmm{whichQBmm}_qm1_mode"],
                    qm2_mode=qcfg[f"bmm{whichQBmm}_qm2_mode"],
                    m1_unidirectional=(whichQBmm == 2),
                    m1_bounded=(whichQBmm == 2),  # see Note 5
                    m2_unidirectional=False,
                    m2_bounded=False,
                    replaceBmm=isbmm,
                    qcfg=qcfg,
                )
                setattr(mod_bmm_happened, f"QBmm{ln}", newQBmm)

        # add auto QBmm check to last layer if any QBmms in model (only for transformers)
        def qbmm_auto_check(_mod, *_args, **_kwargs):
            """Automatic QBmm check. This hook will be attached to the last module and check once
            only at the end of first forward() call. Throw a "warning" if a model has QBmm attached
            but not called (as it could be intentional.)
            """
            num_called_qbmms = []
            for lay, line_nums in qcfg["bmm_prep"]["layers_with_bmm"].items():
                for ln in line_nums:
                    qbmm_i = model.get_submodule(f"{lay}.QBmm{ln}")
                    num_called_qbmms.append(qbmm_i.num_module_called == 1)

            if not all(num_called_qbmms):
                err_msg = (
                    "QBmms were attached but not called during forward()."
                    "Possibly patch_torch_bmm() context manager is missing."
                )
                if qcfg["force_stop_if_qbmm_auto_check_failed"]:
                    raise RuntimeError(err_msg)
                logger.warning(err_msg)

            qcfg["hook_qbmm_auto_check"].remove()

        last_mod = model.get_submodule(qcfg["mod_call_seq"][-1])
        qcfg["hook_qbmm_auto_check"] = last_mod.register_forward_hook(qbmm_auto_check)

    # c) identify RPN/FPN
    # TODO this hack only works for torchvision models. will use find_rpn_fpn_gm()

    if available_packages["torchvision"]:
        # Third Party
        # pylint: disable = import-error
        from torchvision.models.detection.rpn import RegionProposalNetwork
        from torchvision.ops import FeaturePyramidNetwork

        rpnfpn_prefix = []
        rpnfpn_convs = []
        for n, m in model.named_modules():
            if isinstance(m, (FeaturePyramidNetwork, RegionProposalNetwork)):
                rpnfpn_prefix.append(n)
            if isinstance(m, torch.nn.Conv2d) and any(
                n.startswith(p) for p in rpnfpn_prefix
            ):
                rpnfpn_convs.append(n)
                if n not in qcfg["qskip_layer_name"]:
                    qcfg["qskip_layer_name"].append(n)

    if qcfg["N_backend_called"] > 1:
        logger.warning(
            f"Found {qcfg['N_backend_called']} graph breaks during Dynamo tracing!! \n"
            f"First/Last layer, which usually needs to stay unquantized, may not be identified"
            f" correctly now. Please double-check layers being skipped:\n"
            f"{qcfg['qskip_layer_name']}\n NOTE: Users can control layer selection by adding layer"
            f"names to:\n"
            f"1. qcfg['qskip_layer_name'], qcfg['qspecial_layers'] (need to be exact names, will be"
            f"added on top of what is found automatically.), or \n"
            f"2. qcfg['qlayer_name_pattern'] (partial names, will bypass automatic search."
        )
