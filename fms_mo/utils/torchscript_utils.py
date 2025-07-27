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
This file contains utils related to torchscript
"""
# pylint: disable=c-extension-no-member

# Standard
from copy import deepcopy
from typing import List, Tuple
import logging
import sys

# Third Party
from transformers.tokenization_utils_base import BatchEncoding
import torch

# Local
from fms_mo.quant.quantizers import transformers_prepare_input
from fms_mo.utils.import_utils import available_packages
from fms_mo.utils.utils import move_to, patch_torch_bmm, prepare_data_4_fwd

logger = logging.getLogger(__name__)


def parse_operation(op_str: str):
    """
    Takes a string searches for the last '(' and ')' and separates it into the operator and operands

    Args:
        op_str (str):
            e.g. "^LearnedClippedLinearQuantizeSTE_rev1(4, True, False, None, None)(%input.5, %80)"

    Returns:
        tuple: A tuple containing operator (str), and operands List[str]:
            e.g. "^LearnedClippedLinearQuantizeSTE_rev1(4, True, False, None, None)"
            e.g. "[%input.5, %80]"
    """
    last_open_parenthesis_index = op_str.rfind("(")
    last_close_parenthesis_index = op_str.rfind(")")
    operator = op_str[:last_open_parenthesis_index]
    operands = op_str[
        last_open_parenthesis_index + 1 : last_close_parenthesis_index
    ].split(",")
    operands = [operand.strip() for operand in operands] if operands != [""] else None
    return operator, operands


class Node:
    r"""
    A class representing a node in a PyTorch model.

    Starting example:
    %input.28 : Float(1:238144, 64:3721, 61:61, 61:1, requires_grad=1, device=cpu)
    = aten::_convolution(%input.27, %184, %60, %185, %186, %187, %57, %188, %59, %57, %57,
    %56, %56), scope: __module.layer1/__module.layer1.2/__module.layer1.2.conv2
    # directory_path \ file.py:419:0

    Step 1: Remove everything following # such that example becomes:

    %input.28 : Float(1:238144, 64:3721, 61:61, 61:1, requires_grad=1, device=cpu)
    = aten::_convolution(%input.27, %184, %60, %185, %186, %187, %57, %188, %59, %57, %57,
    %56, %56), scope: __module.layer1/__module.layer1.2/__module.layer1.2.conv2

    Step 2: Simplify "scope" string such the example becomes:

    %input.28 : Float(1:238144, 64:3721, 61:61, 61:1, requires_grad=1, device=cpu)
    = aten::_convolution(%input.27, %184, %60, %185, %186, %187, %57, %188, %59, %57, %57,
    %56, %56), scope: __module.layer1.2.conv2

    Step 3: Further parse into a dict

    Example:
    '%77 : Float(16:1632000, 3:544000, 80:6800, 80:85, 85:1, requires_grad=1,
    device=cuda:0), %78 : Float(16:408000, 3:136000, 40:3400, 40:85, 85:1, requires_grad=1,
    device=cuda:0), %79 : Float(16:102000, 3:34000, 20:1700, 20:85, 85:1, requires_grad=1,
    device=cuda:0) = prim::TupleUnpack(%1248)'

    Attributes:
        name (str): The name of the node.
        obj (str): The object associated with the node.
        Op (str): The operation performed by the node.
        operator (str): The operator type of the node.
        operands (list): A list of operands associated with the node.
        parents (list): A list of parent nodes.
        children (list): A list of child nodes.
        scope (str): The scope of the node.
        modname (str): The module name of the node.
        lineno (int): The line number of the node.
        unpackIdx (int): The index of the unpack operation.
        ch_in (list): The input channels of the node.
        ch_out (list): The output channels of the node.
        TSparents (list): The native PyTorch script parents of the node.
        TSoutputs (list): The native PyTorch script outputs of the node.

    Methods:
        __init__(node_input, dictionary_of_nodes): Initializes the Node object.
        __repr__(): Returns a string representation of the Node object.
    """

    def __init__(self, node_input, dictionary_of_nodes: dict):
        """
        Initialize a Node object.

        Args:
            node_input (str or torch._C.Node): The input to the Node object.
                If it's a string, it represents the node definition as a string.
                If it's a torch._C.Node, it represents a native TorchScript node.
            dictionary_of_nodes (dict): A dictionary that keeps track of all the nodes in the graph.
        """
        if isinstance(node_input, torch._C.Node):
            node_input_repr = node_input.__repr__().replace("\n", "")
            native_torchscript_node = node_input
            native_torchscript_parents = [
                "%" + n.__repr__().split(" defined in")[0]
                for n in native_torchscript_node.inputs()
            ]
            native_torchscript_outputs = [
                "%" + n.__repr__().split(" defined in")[0]
                for n in native_torchscript_node.outputs()
            ]
        elif isinstance(node_input, str):
            node_input_repr = node_input
            native_torchscript_node = None
            native_torchscript_parents = None
            native_torchscript_outputs = None
        else:
            logger.warning(
                "Input to class Node is neither a string nor a torchscript node"
            )
            return

        if "# " in node_input_repr:
            line_number = node_input_repr.split("#")[1].split(":")[-2]
            node_input_repr = node_input_repr.split("#")[0]
        else:
            line_number = None

        if "scope:" in node_input_repr:
            temp_str = node_input_repr.split("scope:")
            scope_repr = temp_str[1].split("/")[-1]
            node_input_repr = temp_str[0]
        else:
            scope_repr = None
        module_name = (
            scope_repr.replace("__module.", "") if scope_repr is not None else ""
        )
        module_name = module_name.replace(
            "model.", ""
        )  # Remove model. for shorter names

        start_index = 0
        if " = " in node_input_repr:
            if node_input_repr.count(" = ") == 1:
                node_def, op_str = node_input_repr.split(" = ")
            else:
                # e.g.,  %2206 : Float(1, 3, 1, 1, 2, strides=[6, 2, 2, 2, 1], requires_grad=0,
                # device=cpu) = prim::Constant[value=(1,1,1,.,.) =
                # 1.2500  1.6250  (1,2,1,.,.) =    2.0000  3.7500  (1,3,1,.,.) =
                # 4.1250  2.8750 [ CPUFloatType{1,3,1,1,2} ]](), scope: __module.module_list.106
                idx1steq = node_input_repr.find(" = ")
                node_def, op_str = (
                    node_input_repr[:idx1steq],
                    node_input_repr[idx1steq + 3 :],
                )
            operator, operands = parse_operation(op_str)
            if "aten::_conv" in op_str:
                self.ch_in = list(native_torchscript_node.inputs())[0].type().sizes()
                # NOTE: Needed for finding shortcut convolutions later
                self.ch_out = list(native_torchscript_node.outputs())[0].type().sizes()
        else:
            node_def = node_input_repr
            op_str, operator, operands = None, None, None
        node_def_in_one_line = node_def.count(" : ")
        # when unpacking OPs, will create N instances of node, i.e. not pointing to the same "self"
        node_instances = [self] + [
            deepcopy(self) for _ in range(node_def_in_one_line - 1)
        ]

        for node_index, node_instance in enumerate(node_instances):
            if node_index == node_def_in_one_line - 1:
                end_index = len(node_def)
            else:
                current_colon_index = node_def.find(" : ", start_index)
                next_colon_index = node_def.find(" : ", current_colon_index + 1)
                end_index = node_def.rfind("%", start_index, next_colon_index) - 2
            working_str = node_input_repr[start_index:end_index]
            start_index = end_index + 2

            node_instance.name, node_instance.obj = working_str.split(" : ")
            node_instance.name = node_instance.name.strip()
            if native_torchscript_outputs:
                if node_instance.name not in native_torchscript_outputs:
                    logger.error(
                        f"Node def {node_instance.name} not in nativeTSoutputs "
                        f"{native_torchscript_outputs}"
                    )
            node_instance.Op = op_str
            if node_def_in_one_line > 1:
                node_instance.unpackIdx = node_index
            if line_number:
                node_instance.lineno = line_number
            node_instance.operator = operator
            # This is the name of parents, not the pointer to the parent nodes
            node_instance.parents = operands
            node_instance.parents_ptr = []
            node_instance.scope = scope_repr
            node_instance.modname = module_name
            node_instance.children = []
            node_instance.children_ptr = []
            node_instance.TSparents = native_torchscript_parents
            node_instance.TSoutputs = native_torchscript_outputs
            # graph.dictionary_of_nodes will keep a record of all the nodes
            dictionary_of_nodes[node_instance.name] = node_instance

    def __repr__(self):
        return f"{self.name} "


class Graph:
    """
    Class for Graph

    Attributes:
        dictionary_of_nodes (dict): A dictionary that maps node names to Node objects.
        inputs (list): A list of Node objects representing the input nodes of the graph.
        disable_plots (bool): A flag indicating whether to disable plotting functionality.
        model_node (Node): A Node object representing the model node of the graph.
        return_node (list): A list of Node objects representing the return nodes of the graph.

    """

    def __init__(self, graph):
        """
        Initializes the Graph object by parsing the given torch._C.Graph object.

        Args:
            graph (torch._C.Graph): The torch._C.Graph object to be parsed.
        """
        self.dictionary_of_nodes = {}
        self.inputs = []
        self.disable_plots = not available_packages["pygraphviz"]

        list_str = None
        if isinstance(graph, torch._C.Graph):
            graph_repr = graph.__repr__()
            list_str = graph_repr.split("\n")

        # Parse header, i.e. first few lines, def of graph inputs, first arg must be model itself
        # example: graph(%self.1 : __torch__.torchvision.models.resnet.ResNet,
        #      %input.1 : Float(1:178608, 3:59536, 244:244, 244:1, requires_grad=0, device=cpu)):
        curr_line = 0
        header = ""
        is_header = True
        left_parenthesis, right_parenthesis = 0, 0
        while is_header and curr_line < 10:
            line_str_i = list_str[curr_line]
            header += line_str_i
            left_parenthesis = header.count("(")
            right_parenthesis = header.count(")")
            # Unbalanced parenthesis means still in header
            is_header = left_parenthesis != right_parenthesis

            if line_str_i.endswith(","):
                line_str_i = line_str_i[:-1]
            elif line_str_i.endswith("):"):
                line_str_i = line_str_i[:-2]

            if curr_line == 0:
                self.model_node = Node(
                    line_str_i.replace("graph(", ""), self.dictionary_of_nodes
                )
            else:
                self.inputs.append(Node(line_str_i, self.dictionary_of_nodes))

            curr_line += 1

        # Parse body, i.e. def of nodes. we now can utilize TS native nodes
        for graph_node in graph.nodes():
            n_ptr = Node(graph_node, self.dictionary_of_nodes)
            if "prim::If" in graph_node.kind():
                n_ptr.scope = "Either"
                for bi in graph_node.blocks():
                    for nj in bi.nodes():
                        nj_ptr = Node(nj, self.dictionary_of_nodes)
                    n_ptr.scope += f" {nj_ptr.name} or"
                n_ptr.scope = n_ptr.scope[:-3]

        # Parse last line, return, usually looks like "return (%xxx)"
        temp_str = graph.return_node().__repr__()
        idx_s = temp_str.index("(")
        idx_e = temp_str.rindex(")")
        self.return_node = [
            self.dictionary_of_nodes[ri.strip()]
            for ri in temp_str[idx_s + 1 : idx_e].split(",")
        ]

        # Second pass add children info and update pointer lists
        for key_node, value_node in self.dictionary_of_nodes.items():
            if value_node.parents is not None:
                value_node.parents_ptr = [
                    self.dictionary_of_nodes[pi] for pi in value_node.parents
                ]
                for pi in (
                    value_node.parents_ptr
                ):  # Here we can use pointer to parents directly
                    pi.children.append(key_node)
                    pi.children_ptr.append(value_node)
                # Verify consistency between our data structure vs native torchscript's
                if set(value_node.parents) != set(value_node.TSparents):
                    logger.info(
                        f"{key_node} inconsistent parents {value_node.parents} "
                        f"{value_node.TSparents}"
                    )

        logger.info("torchscript inlined_graph parsed successfully!")

    def DFS(
        self,
        target_op,
        verbose: bool = False,
        insert_annotation: bool = False,
        find_first_only: bool = False,
        reverse: bool = False,
        node_st=None,
        stop_op=None,
        ignore_op=None,
        hook=None,
    ):
        """
        Performs depth first search on the graph.

        Args:
            target_op (str or list of str): The target operation(s) to search for.
            verbose (bool): Whether to print search results for debugging purposes.
                        Defaults to False.
            insert_annotation (bool): Whether to insert annotations in the returned node list
                        for specific cases. Defaults to False.
            find_first_only (bool): Whether to stop after finding the first node that matches
                        the target operation(s). Defaults to False.
            reverse (bool): Whether to search in reverse direction, from the end to the beginning
                        of the graph. Defaults to False.
            node_st (Node or str): The starting node for the search. If not specified, it defaults
                        to the input nodes of the graph.
            stop_op (str or list of str): The operations to stop the search at.
                        If not specified, it defaults to None.
            ignore_op (str or list of str): The operations to ignore during the search.
                        If not specified, it defaults to None.
            hook (callable, optional): A function to apply to the node if it matches the
                        target operation(s). Defaults to None.

        Returns:
            list: A list of Node objects that match the target operation(s).
        """
        # Make a table to record visit history, to avoid redundant search
        self.visited = {key: False for key in self.dictionary_of_nodes}
        if isinstance(target_op, str):
            # 1. If target_op is not a list of strings, make it a list for easier looping.
            # 2. a list of criteria means more than one search criteria is allowed
            target_op = [target_op]

        if isinstance(stop_op, str):
            stop_op = [stop_op]
        if isinstance(node_st, Node):
            node_st = [
                node_st
            ]  # Make it a single element list, so we can use for loop easier.
        if ignore_op is None:
            ignore_op = ["aten::size"]  # Default prescreen for now
        elif isinstance(ignore_op, str):
            ignore_op = [ignore_op] + ["aten::size"]
        else:  # a list
            ignore_op += ["aten::size"]

        # Search from a given starting node, the beginning (input nodes) or the end (return node)
        starting_nodes = (
            node_st
            if node_st is not None
            else self.inputs
            if not reverse
            else self.return_node
        )

        self.node_found = []
        self.node_traced = 0
        for node_i in starting_nodes:
            self.br_begin = None
            self._dfs(
                node_i,
                target_op,
                verbose,
                insert_annotation,
                find_first_only,
                reverse,
                node_st,
                stop_op,
                ignore_op,
                hook,
            )
            if verbose:
                logger.info(
                    f"Nodes traced={self.node_traced}/{len(self.dictionary_of_nodes.keys())}, "
                    f"found {len(self.node_found)} that satisfy the criteria {target_op}"
                )
        return self.node_found

    def _dfs(
        self,
        curr_node,
        target_op,
        verbose,
        insert_annotation,
        find_first_only,
        reverse,
        node_st,
        stop_op,
        ignore_op,
        hook,
    ):
        # Make sure curr_node is a pointer, not a string
        if isinstance(curr_node, str):
            curr_node = self.dictionary_of_nodes[curr_node]

        if self.visited[curr_node.name]:  # Avoid redundant search
            return curr_node  # This is the end of a branch
        self.visited[curr_node.name] = True  # Mark this node visited first

        if curr_node.Op:  # e.g, input nodes' .Op=[]
            # Checking sequence ignore_op -> target_op -> stop_op
            # 1) bBfore checking if curr_node satisfies target_op, filter by ignore_op first
            if any(Op_i in curr_node.Op for Op_i in ignore_op):
                return
            # 2) Then check if satisfies target_op
            if any(Op_i in curr_node.Op for Op_i in target_op):
                self.node_found.append(curr_node)
                if verbose:
                    logger.info(
                        f"{curr_node.name} {curr_node.operator} {curr_node.scope}"
                    )
                if find_first_only:
                    return
                # Find_first_only means "stop searching after 1st node with target_op is found"
                # (if branches before found, 1st node for each branch)
                if hook:
                    hook(
                        curr_node
                    )  # Can apply a hook function if search criteria satisfied
            # 3) Then decide if it satisfies stop_op
            if stop_op:
                if any(Op_i in curr_node.Op for Op_i in stop_op):
                    return
                # we can also stop the search when run into any of the provided stop_op, but
                # make sure the 1st node where the search begins does not satisfy that criteria
                # for example, we could search from a Conv and hope to stop at the next Conv,
                # call by DFS(...,node_st=Conv.parent_ptr[i], stop_op='conv')
                # remember we have added our known "filters" to stop_op already in the outer loop
        self.node_traced += 1

        next_nodes = curr_node.children_ptr if not reverse else curr_node.parents_ptr
        end_nodes = self.return_node if not reverse else self.inputs
        if next_nodes:
            num_next_nodes: int = len(next_nodes)
            if num_next_nodes > 1 and insert_annotation:
                for next_node_elem in next_nodes[1:]:
                    self.node_found.append(
                        f"{curr_node.name}->{next_node_elem.name}, A branch begins here"
                    )

            for next_node_index, next_node_elem in enumerate(next_nodes):
                if next_node_index > 0:
                    self.br_begin = next_node_elem
                    next_node_elem.isOnBranch = True
                    if verbose:
                        logger.info(
                            f"Start searching branch #{next_node_index}/{num_next_nodes} of "
                            f"node {curr_node.name}"
                        )

                br_end = self._dfs(
                    next_node_elem,
                    target_op,
                    verbose,
                    insert_annotation,
                    find_first_only,
                    reverse,
                    node_st,
                    stop_op,
                    ignore_op,
                    hook,
                )

                if br_end is not None:
                    if verbose:
                        logger.info(f"current branch merged into {br_end.name}")
                    if insert_annotation:
                        self.node_found.append(
                            f"{self.br_begin.name}->{br_end.name}, A branch merges into "
                        )
                    next_node_elem.isBranchMerge = True
        elif curr_node in end_nodes:  # return node also has no child
            if verbose:
                logger.info(
                    "Run into a return-node or an input-node (if reverse search) "
                )
            if insert_annotation:
                self.node_found.append(f"{curr_node.name}, End of main branch")

    def brute_force_search(self, target_op):
        """
        Searches the nodes_dictionary directly for nodes with a matching operator.

        Args:
            target_op (str): The operator to search for.

        Returns:
            list: A list of nodes that match the target operator.
        """

        self.node_found = []
        for curr_node in self.dictionary_of_nodes.values():
            if curr_node.Op:
                if target_op in curr_node.Op:
                    self.node_found.append(curr_node)
                    logger.info(
                        f"{curr_node.name} {curr_node.operator} {curr_node.scope}"
                    )

        return self.node_found

    def plot_full(self, output_name="test.svg"):
        """
        Plots the full graph

        Args:
            output_name (str, optional): The name of the file to save the plot to.
                    Defaults to "test.svg".
        """
        if self.disable_plots:
            return
        # Third Party
        import pygraphviz as pgv

        G = pgv.AGraph(strict=False, directed=True)
        for node_name, node_ptr in self.dictionary_of_nodes.items():
            G.add_node(node_name.replace("%", ""), label=node_name, shape="record")
            for childen_node in node_ptr.children:
                G.add_edge(node_name.replace("%", ""), childen_node.replace("%", ""))
        G.layout(prog="dot")
        G.draw(output_name)

    def plot_short(
        self,
        kw,
        output_name="test.svg",
        plot_in_notebook: bool = False,
        verbose: bool = False,
        fields=1,
        showQW=True,
    ):
        """
        Plot a computation graph for a given keyword.

        Args:
            kw (str): The keyword to search for in the computation graph.
            output_name (str, optional): The name of the output file. Defaults to "test.svg".
            plot_in_notebook (bool, optional): Whether to plot the graph in the Jupyter notebook.
                        Defaults to False.
            verbose (bool, optional): Whether to logger.info verbose output. Defaults to False.
            fields (int, optional): The number of fields to include in the node label.
                        Defaults to 1.
            showQW (bool, optional): Whether to show the weight quantizers in the graph.
                        Defaults to True.
        """
        if self.disable_plots:
            return
        # Third Party
        import pygraphviz as pgv

        G = pgv.AGraph(strict=False, directed=True)
        nodes_of_intr = self.DFS(kw)
        if verbose:
            logger.info(nodes_of_intr)

        for node_ptr in nodes_of_intr:
            nname = node_ptr.name.replace("%", "")
            modname = (
                node_ptr.scope.replace("__module.", "")
                if node_ptr.scope is not None
                else ""
            )
            modname = modname.replace(
                "model.", ""
            )  # Further simplify the name for better visualization on SVG
            fieldsStr = ["%" + nname, node_ptr.operator, modname]
            labelStr = "|".join(
                fieldsStr[: min(fields, len(fieldsStr))]
            )  # Can plot with fewer fields for each node as needed
            fcolor = (
                "#b2d3e4"
                if "conv" in node_ptr.operator
                else (
                    "#FFF2CC"
                    if "^" in node_ptr.operator
                    else (
                        "#DDDDDD"
                        if (
                            "addmm" in node_ptr.operator
                            or "bmm" in node_ptr.operator
                            or "linear" in node_ptr.operator
                        )
                        else "#C5E0B4"
                        if "If" in node_ptr.operator
                        else "white"
                    )
                )
            )
            if "conv" in node_ptr.operator:
                ch_info = (
                    f"{node_ptr.ch_in}|{node_ptr.ch_out}"
                    if hasattr(node_ptr, "ch_in") and hasattr(node_ptr, "ch_out")
                    else ""
                )
                labelStr = (
                    f"{fieldsStr[0]}: \n| {{ {fieldsStr[1]}|{fieldsStr[2]} }}| "
                    f"{{ input:|output: }} |{{ {ch_info} }}"
                )
                # Also include weight quantizers
                nodeW = node_ptr.parents_ptr[1]
                if "^" in nodeW.Op and showQW:  # A quantizer is being used.
                    nWname = nodeW.name.replace("%", "")
                    modWname = (
                        nodeW.scope.replace("__module.", "")
                        if nodeW.scope is not None
                        else ""
                    )
                    modWname = modWname.replace("model.", "")
                    fieldsWStr = ["%" + nWname, nodeW.operator, modWname]
                    labelWStr = "|".join(fieldsWStr[: min(fields, len(fieldsWStr))])
                    G.add_node(
                        nWname,
                        fillcolor="#FFF2CC",
                        style="filled",
                        label=labelWStr,
                        shape="record",
                    )
                    G.add_edge(nWname, nname)
            G.add_node(
                nname, fillcolor=fcolor, style="filled", label=labelStr, shape="record"
            )

            for j in node_ptr.children:
                G.add_edge(nname, j.replace("%", ""))

        if verbose:
            G.write(output_name + ".dot")
            logger.info(
                "Output graph to .dot. It could be very slow for DenseNet or complicated nets."
            )
        G.layout(prog="dot")

        G.draw(output_name)
        if plot_in_notebook and available_packages["matplotlib"]:
            # Third Party
            import matplotlib.image as mpimg
            import matplotlib.pyplot as plt

            plt.figure(figsize=(16, 50))
            plt.imshow(mpimg.imread(output_name))
            plt.axis("off")

    def is_child(self, child_node, parent_node):
        """
        Checks if a node is a child node of a parent node

        Args:
            child_node (Node): The child node to check
            parent_node (Node): The parent node to check against

        Returns:
            bool: True if the child node is a child of the parent node, False otherwise
        """
        potential_match = self.DFS(child_node.Op, node_st=parent_node)
        if not isinstance(potential_match, list):
            potential_match = [potential_match]
        # Matching node.Op may be insufficient, need to check .obj as well
        is_child_node = any(node.obj == child_node.obj for node in potential_match)
        return is_child_node

    def id_rpn_from_last(self, last_candidates):
        """
        Identify the RPN candidates from the last candidates.

        Args:
            last_candidates (list): A list of candidate nodes.

        Returns:
            rpn_candidates (list): A list of identified RPN candidates.
        """
        grouped, rpn_candidates = [], []
        for n in last_candidates:
            if n.children_ptr:  # Sometimes last nodes have no children anymore
                child0 = n.children_ptr[0]
                if len(n.children) == 1 and (
                    "TupleConstruct" in child0.Op or "ListConstruct" in child0.Op
                ):
                    if child0 not in grouped:
                        grouped.append(
                            child0
                        )  # NOTE: We add the tuple construct, not the Op itself, here
        if len(grouped) > 1:
            num_member_per_group = [len(n.parents) for n in grouped]
            if max(num_member_per_group) % min(num_member_per_group) == 0:
                idx_max = num_member_per_group.index(max(num_member_per_group))
                rpn_candidates = grouped[idx_max].parents_ptr
        return rpn_candidates

    def find_fpn_convs(self):
        """
        Identify the FPN to be quantized

        Returns:
            List[Node]: A list of Node objects representing the FPN convolutions
        """
        fpn_begin = []
        fpn_end = []
        fpn_candidates = []
        temp_node = self.DFS("aten::upsample")
        children = (
            [n1 for n in temp_node for n1 in n.children_ptr if "aten::add" in n1.Op]
            if temp_node
            else []
        )
        grandchildren = (
            [n1 for n in children for n1 in n.children_ptr if "aten::upsample" in n1.Op]
            if children
            else []
        )
        ggchildren = (
            [n1 for n in grandchildren for n1 in n.children_ptr if "aten::add" in n1.Op]
            if grandchildren
            else []
        )
        if ggchildren:
            for n in ggchildren:
                fpn_begin += self.DFS(
                    "TupleConstruct", find_first_only=True, reverse=True, node_st=n
                )
                fpn_end += self.DFS(
                    "TupleConstruct", find_first_only=True, reverse=False, node_st=n
                )
            # Remove duplicates
            temp_node, fpn_begin = fpn_begin, []
            for n in temp_node:
                if n not in fpn_begin:
                    fpn_begin.append(n)
            temp_node, fpn_end = fpn_end, []
            for n in temp_node:
                if n not in fpn_end:
                    fpn_end.append(n)
            if len(fpn_begin) > 1 or len(fpn_end) > 1 or len(fpn_begin) != len(fpn_end):
                logger.warning(f"FPN detection is inconsistent. {fpn_begin} {fpn_end}")
            else:
                fpn_candidates = self.DFS(
                    "aten::_conv", node_st=fpn_begin[0], stop_op=fpn_end[0].Op
                )
        return fpn_candidates

    def __repr__(self):
        return (
            f" model node={self.model_node}\n inputs={self.inputs}\n "
            f"{self.dictionary_of_nodes}\nreturn={self.return_node}"
        )


def find_shortcut_conv_v2(graph, verbose=False):
    """
    Revised algorithm for finding convolutional modules on shortcut path
    TODO: make it a member functon of class Graph

    Args:
        graph (Graph): The input graph.
        verbose (bool, optional): Whether to print detailed information. Defaults to False.

    Returns:
        list: List of potential convolutional modules on shortcut path
    """
    # 1. Find the Add_ nodes
    assert isinstance(graph, Graph), "input needs to be an object of our custom graph"
    NodesAdd = graph.DFS(["aten::add(", "aten::add_("])
    irrNodes = []
    # Depending on user preference, could be out+=shortcut, out=out+shortcut or out=shortcut+out
    if verbose:
        logger.info(NodesAdd)

    qconv_candidate = []
    for node_i in NodesAdd:
        if all("Float" in n.obj for n in node_i.parents_ptr[:2]):
            # Make sure this Add Op is adding 2 float tensors (No interest in Long or Int Adds here)
            # 2. Find where branch begins, i.e. common node along 1st and 2nd parent node of Add,
            # record how many levels to the current Add
            node_i.parents_ptr[:2]
            levels_from_add = [1, 1]
            branch_nodes = [None, None]
            for j, p_j in enumerate(node_i.parents_ptr[:2]):
                while len(p_j.children_ptr) < 2 and levels_from_add[j] < 20:
                    levels_from_add[j] += 1
                    p_j = p_j.parents_ptr[0]
                branch_nodes[j] = p_j

            # 2.1 Make sure the branch node is the same (there might be cases where a secondary
            # branch, like a concat, exists in one of the branches)
            if branch_nodes[0] != branch_nodes[1]:
                # possible cause, run into search limit (20 levels) or branch-in-branch situation,
                # theoretically we should keep searching to confirm
                # but if we can confirm one of the two is a common parent, should suffice
                if graph.is_child(
                    child_node=branch_nodes[0], parent_node=branch_nodes[1]
                ):
                    real_branch_begin = branch_nodes[1]
                elif graph.is_child(
                    child_node=branch_nodes[1], parent_node=branch_nodes[0]
                ):
                    real_branch_begin = branch_nodes[0]
                else:
                    real_branch_begin = None
                    logger.warning(f"{node_i}'s branch analysis could be incorrect!")
            else:
                # Most likely case, the 2 searches return the same node
                real_branch_begin = branch_nodes[0]
            shorter_path = 0 if levels_from_add[0] < levels_from_add[1] else 1
            if real_branch_begin:
                temp_nodes = graph.DFS(
                    "conv",
                    find_first_only=True,
                    reverse=True,
                    node_st=node_i.parents_ptr[shorter_path],
                    stop_op=real_branch_begin.Op,
                )
                # shortcut Convs must have channel_out > channel_in, on the other hand,
                # FPN must have channel_out <= channel_in
                qconv_candidate += [n for n in temp_nodes if n.ch_out > n.ch_in]
        else:
            irrNodes.append(node_i)

    return qconv_candidate


def find_all_conv_sorted(graph, verbose=False):
    """
    Find all the convolutional modules in the graph, and insert those on shortcut path at the
    branch point (as a list, in case there are more than 1)

    Args:
        graph (Graph): The graph to be searched.
        verbose (bool, optional): Whether to print out information about the search process.
                Default is False.

    Returns:
        list: A list of sorted convolutional layers.
    """
    NodesConv = graph.DFS(["conv", "addmm"], verbose=False, insert_annotation=True)

    if verbose:
        for node_name in NodesConv:
            if isinstance(node_name, str):
                logger.info(f"[Annotation] {node_name}")
            else:
                logger.info(node_name.name)

    end_of_main_branch_index = 0
    branch_begin_index = []
    branch_merge_index = []
    dict_branch_first_node = {}
    for node_index, node_name in enumerate(NodesConv):
        if isinstance(node_name, str):
            if "begins" in node_name:
                # e.g. '%input.237->%input.240, A branch begins here'
                branch_begin_index.append(node_index)
                branch_first_node_name = node_name[
                    node_name.find("->") + 2 : node_name.find(",")
                ]
                if branch_first_node_name in dict_branch_first_node:
                    # this node already in dictBr1stNode, e.g. DenseNet
                    curr_val = dict_branch_first_node[branch_first_node_name]
                    if isinstance(curr_val, int):
                        dict_branch_first_node[branch_first_node_name] = [
                            curr_val,
                            node_index,
                        ]
                    else:
                        dict_branch_first_node[branch_first_node_name].append(
                            node_index
                        )
                else:
                    dict_branch_first_node[branch_first_node_name] = node_index
            elif "merges" in node_name:
                branch_merge_index.append(node_index)
            elif "End of main" in node_name:
                end_of_main_branch_index = node_index
            else:
                logger.error("Undefined annotations found")

    if len(branch_begin_index) != len(branch_merge_index):
        logger.error(
            f"Total number of branch_begins {len(branch_begin_index)} and branch_merges "
            f"{len(branch_merge_index)} mismatch"
        )
    else:
        revBrMerge = branch_merge_index[::-1] + [end_of_main_branch_index]
        for idx_st, idx_end in zip(revBrMerge[1:], revBrMerge[:-1]):
            if idx_end - idx_st > 1:
                strAnno = NodesConv[idx_end]
                br1stnode = strAnno[
                    : strAnno.find("->")
                ]  # e.g. '%input.248->%1730, A branch merges into'
                # look up dict['%input.248'] to get which line has '%xxxx->%input.248, ....',
                # insert this section of nodes into that line
                index_to_insert = dict_branch_first_node[br1stnode]
                if isinstance(index_to_insert, list):
                    index_to_insert = dict_branch_first_node[br1stnode].pop()
                NodesConv[index_to_insert] = NodesConv[idx_st + 1 : idx_end]

    new_nodes_conv = [
        node
        for node in NodesConv[:end_of_main_branch_index]
        if not isinstance(node, str)
    ]
    if verbose:
        logger.info("\nAfter insertion of Convs on branches into the list\n")
        for node_name in new_nodes_conv:
            if isinstance(node_name, list):
                logger.info(node_name)
            else:
                logger.info(node_name)
    return new_nodes_conv


def check_activation_dir(curr_node):
    """
    Check activation direction of a node in a PyTorch module. The function identifies the
    `isActOutUnidir` and `isActOutBounded` attributes of the current node.

    Args:
        curr_node (Node): The current node in the PyTorch module.
    """
    # Corresponds to ReLU, ReLU6, Sigmoid
    rectifier_ops: List[str] = [
        "aten::relu",
        "aten::hardtanh",
        "aten::sigmoid",
        "aten::softmax",
    ]
    #
    bidirectional_ops: List[str] = [
        "aten::_conv",
        "aten::addmm",
        "aten::linear",
        "aten::index_put_",  # Fills target tensor with a given src tensor based on an idx tensor
        "aten::matmul",
    ]
    add_ops: List[str] = ["aten::add(", "aten::add_("]
    concat_ops: List[str] = ["prim::ListConstruct"]
    # Bounded Ops, note that silu is only bounded on the neg side.
    bounded_ops: List[str] = [
        "aten::hardtanh",
        "aten::sigmoid",
        "aten::softmax",
    ]
    # Deterministic cases
    if any(rectifier_op in curr_node.Op for rectifier_op in rectifier_ops):
        curr_node.isActOutUnidir = True
        curr_node.isActOutBounded = bool(
            any(Op_i in curr_node.Op for Op_i in bounded_ops)
        )
    elif any(bidrectional_op in curr_node.Op for bidrectional_op in bidirectional_ops):
        curr_node.isActOutUnidir = False
        curr_node.isActOutBounded = bool(
            any(Op_i in curr_node.Op for Op_i in bounded_ops)
        )
    # Dependent cases, can be "undertermined" if insufficient info
    elif any(add_op in curr_node.Op for add_op in add_ops):
        # When checking branches, need to make sure all the parents has isActOutUnidir
        # If adding a tensor with bias (as in nn.linear), it's bi-dir
        if any("bias" in pi.Op for pi in curr_node.parents_ptr[:2]):
            curr_node.isActOutUnidir = False
            curr_node.isActOutBounded = False
        elif all(hasattr(pi, "isActOutUnidir") for pi in curr_node.parents_ptr[:2]):
            curr_node.isActOutUnidir = all(
                pi.isActOutUnidir is True for pi in curr_node.parents_ptr[:2]
            )
            curr_node.isActOutBounded = all(
                pi.isActOutBounded is True for pi in curr_node.parents_ptr[:2]
            )
        # If any of the parents is undetermined -> curr_node will be undertermined
    elif any(concat_op in curr_node.Op for concat_op in concat_ops):
        if all(hasattr(pi, "isActOutUnidir") for pi in curr_node.parents_ptr):
            curr_node.isActOutUnidir = all(
                pi.isActOutUnidir is True for pi in curr_node.parents_ptr
            )
        if all(hasattr(pi, "isActOutBounded") for pi in curr_node.parents_ptr):
            curr_node.isActOutBounded = all(
                pi.isActOutBounded is True for pi in curr_node.parents_ptr
            )
    elif curr_node.parents_ptr[0]:
        # For all other operations, simply make output.dir = input.dir.
        if hasattr(curr_node.parents_ptr[0], "isActOutUnidir"):
            curr_node.isActOutUnidir = curr_node.parents_ptr[0].isActOutUnidir
        if hasattr(curr_node.parents_ptr[0], "isActOutBounded"):
            curr_node.isActOutBounded = curr_node.parents_ptr[0].isActOutBounded
    else:
        logger.error(f"Exceptions found at {curr_node}")


def clear_act_dir_flag(current_node):
    """
    Clears the flag "isActOutUnidir" and "isActOutBounded" from the given node.

    Args:
        current_node (object): The node object to clear the flags from.
    """
    if hasattr(current_node, "isActOutUnidir"):
        delattr(current_node, "isActOutUnidir")
    if hasattr(current_node, "isActOutBounded"):
        delattr(current_node, "isActOutBounded")


def find_single_sided_conv_v2(graph, Op="conv"):
    """
    Try to determine the output directionality of "every nodes", then decide which Conv to
    use single-sided quantizer
    basically we add a flag ".isActOutUnidir" to each node if it can be determined
    (either True or False).

    Args:
        graph (Graph): The graph to search
        Op (str, optional): The operation type to search for. Defaults to "conv".

    Returns:
        List[]: a list of single sided convolution nodes
    """

    graph.DFS(":", hook=clear_act_dir_flag)

    for ni in graph.inputs:
        # inputNodes.isActOutUnidir won't affect the results, without this flag, DFS will not stop.
        ni.isActOutUnidir = False
    NodesConv = graph.DFS(Op)
    # 1st parent of Conv is the one that is responsible for activations
    NodesConvParent0 = [c.parents_ptr[0] for c in NodesConv]

    num_searched = 0
    num_conv_parents = len(NodesConvParent0)
    num_undetermined = num_conv_parents
    while num_undetermined > 0:
        graph.DFS(":", hook=check_activation_dir)
        num_searched += 1
        undetermined = [p for p in NodesConvParent0 if not hasattr(p, "isActOutUnidir")]
        num_undetermined = len(undetermined)
        logger.info(
            f"Searching single-sided {Op} Loop# {num_searched}, {num_undetermined}/ "
            f"{num_conv_parents} nodes are still undetermined.{undetermined}"
        )
        if num_searched > 10:
            logger.error(
                f"This search should've finished within {num_searched - 1} loops."
            )
            break

    single_sided_convolution = [c for c in NodesConv if c.parents_ptr[0].isActOutUnidir]

    # NOTE the flag, isActOutUnidir" will remain in the graph until next time calling this function

    return single_sided_convolution


def find_single_sided_bmm(graph, Op=("aten::bmm", "aten::matmul")):
    """
    Try to determine the output directionality of "every nodes",
    then decide which bmm (check both inputs) to use single-sided quantizer
    basically we add a flag ".isActOutUnidir" to each node if it can be determined
    (either True or False).

    Args:
        graph (Graph): The input graph to be searched.
        Op (tuple, optional): A tuple of operator names to search for.
                Defaults to ("aten::bmm", "aten::matmul").

    Returns:
        list: A list of nodes representing the single-sided BMM operations found in the graph.
    """

    graph.DFS(":", hook=clear_act_dir_flag)

    # inputNodes.isActOutUnidir won't affect the results, but without this flag, DFS will not stop.
    for ni in graph.inputs:
        ni.isActOutUnidir = False
    NodesBMM = graph.DFS(Op)
    NodesBMM = [
        b
        for b in NodesBMM
        if not ("aten::matmul" in b.Op and "aten::add_" in b.children_ptr[0].Op)
    ]
    # make sure if it's a matmul, it is not followed by an immediate 'add' (which is most
    # like translated from a nn.Linear)

    NodesBMMParents = [
        (b.parents_ptr[0], b.parents_ptr[1]) for b in NodesBMM
    ]  # check both parents

    num_searched = 0
    num_parents = len(NodesBMMParents)
    num_undetermined = num_parents
    while num_undetermined > 0:
        graph.DFS(":", hook=check_activation_dir)
        num_searched += 1
        undetermined_list = [
            (p0, p1)
            for (p0, p1) in NodesBMMParents
            if not (hasattr(p0, "isActOutUnidir") and hasattr(p1, "isActOutUnidir"))
        ]
        num_undetermined = len(undetermined_list)
        logger.info(
            f"Searching single-sided {Op} Loop# {num_searched}, {num_undetermined}/{num_parents} "
            f"nodes are still undetermined.{undetermined_list}"
        )
        if num_searched > 10:
            logger.error("This search should've finished within 10 loops.")
            break

    single_sided_bmm_list = [
        b
        for b in NodesBMM
        if b.parents_ptr[0].isActOutUnidir or b.parents_ptr[1].isActOutUnidir
    ]

    return single_sided_bmm_list


def get_module_from_node(net, node):
    """
    Get the module from the net that the node is pointing to (based on node.scope)

    Args:
        net (nn.Module): The neural network module.
        node (Node): The node in the network

    Returns:
        nn.Module: returns the module that the node is pointing to
    """
    mod_names = node.scope.replace("__module.", "").strip().split(".")
    if mod_names[0] == "model":
        mod_names.pop(0)
    curr_mod = net
    for mod_name in mod_names:
        curr_mod = getattr(curr_mod, mod_name, None)
    return curr_mod


def trace_and_reconstruct(
    model,
    dloader,
    quant_config,
    prefwdproc="toDevice",
    save_fname="temp_model.pt",
    dev=None,
    inplace=False,
):
    """
    This is the main function that handles tracing.

    Currently broken up into different sections to handle different models:
    1. Detectron2
    2. RNN/LSTM
    3. Transformers/BERT
    4. Others: Computer Vision/Object Detection

    Args:
        model (torch.nn.Module): The PyTorch model to be traced.
        dloader (torch.utils.data.DataLoader or torch.Tensor or BatchEncoding or list or dict):
            The data loader or data tensor to be used for tracing.
        quant_config (dict): A dictionary containing configuration parameters for tracing and
            quantization.
        prefwdproc (str, optional):
            The preprocessing function to be applied to the data before tracing.
            Defaults to "toDevice".
        save_fname (str, optional): The filename to save the traced model.
            Defaults to "temp_model.pt".
        dev (torch.device, optional): The device to use for tracing. Defaults to None.
        inplace (bool, optional): Whether to perform tracing and reconstruction in place on the
            original model. Defaults to False.

    Returns:
        Graph: The reconstructed graph of the traced model.
    """

    DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    # NOTE: operate on copy to avoid any modification on the original model, including forward()
    temp_model = deepcopy(model) if not inplace else model
    # If GPU OOM, use smaller (partial) batch size for tracing and calibration
    currDev = next(model.parameters()).device if dev is None else dev
    # For tracing, use as small batch size as possible, if qcfg['wasDPmodel'] else None because
    # we will perform trace and calib on single GPU for DP now, make sure batchsize is proper
    # do not want to trigger calibration for trace/plot graph, set tempmodel's calib_counter to 0.
    # (This only affects post-swapping tracing.)
    for name, b in temp_model.named_buffers():
        if name.endswith("calib_counter"):
            b.fill_(0)
        if name.endswith("ptq_calibration_counter"):
            b.fill_(0)

    # Try to get one batch of data and send to device, e.g. from dataloader, provided by user,
    # RNN loader is a different
    if (
        isinstance(dloader, torch.utils.data.dataloader.DataLoader)
        or "detectron2.modeling" in sys.modules
    ):
        data_mb = next(iter(dloader))
    elif isinstance(dloader, (torch.Tensor, BatchEncoding)):
        # Or user can provide one tensor in a format that model can accept as is
        data_mb = dloader.to(currDev)
        prefwdproc = None  # To avoid unintended slicing
    elif isinstance(dloader, list):  # assume user provides a list of ready-to-go data
        data_mb = dloader[0]
    elif isinstance(dloader, dict):
        # Most likely from transformers, just move to device one by one
        data_mb = {k: v.to(currDev) for k, v in dloader.items()}
    elif callable(dloader):
        # Some codes use fake loader (a callable func), this is just one of the many possible cases
        data_mb, _ = dloader()
    else:
        logger.warning(
            " Unknown case for extracting one batch of data. Will leave it as is. "
            "Please make sure it's ready for feeding to model."
        )
    data_mb = move_to(data_mb, dev if dev is not None else DEFAULT_DEVICE)

    # Special handling for model calling detectron2
    if "detectron2.modeling" in sys.modules:
        # https://detectron2.readthedocs.io/en/latest/tutorials/deployment.html
        # https://github.com/facebookresearch/detectron2/blob/
        #   a406e69f6687b9e3923db31bfda89c339e0a81c4/detectron2/export/flatten.py#L185
        raise RuntimeError("Detectron2 is not supported for the moment")
    # Special handling for RNN/LSTM
    if quant_config["isRNNmodel"]:
        hid = (
            torch.zeros(
                quant_config["nlayers"] * (quant_config["bidirectional"] + 1),
                quant_config["batch_size"],
                quant_config["nhid"],
            ).to(currDev),
            torch.zeros(
                quant_config["nlayers"] * (quant_config["bidirectional"] + 1),
                quant_config["batch_size"],
                quant_config["nhid"],
            ).to(currDev),
        )

        if "data_mb" in quant_config:
            data_mb = tuple(
                item_i.to(currDev) if isinstance(item_i, torch.Tensor) else item_i
                for item_i in data_mb
            )
        elif prefwdproc:
            if callable(prefwdproc):
                data_mb = prefwdproc(
                    data_mb
                )  # NOTE: make sure callable input and output are tuple
            elif prefwdproc == "toDevice":
                # 99% of the time, dataloader will return a tuple
                if isinstance(data_mb, tuple) or (
                    isinstance(data_mb, list) and len(data_mb) > 1
                ):
                    # pylint: disable=consider-using-generator
                    data_mb = tuple(
                        [
                            (
                                item_i.to(currDev)
                                if isinstance(item_i, torch.Tensor)
                                else item_i
                            )
                            for item_i in data_mb
                        ]
                    )
                    # Only pass the first N items to model, e.g. batch has (imgs, labels)
                    # and model only needs (imgs). user needs to specify N (=1 most of the time)
                    data_mb = data_mb[: quant_config["numparamsfromloadertomodel"]]
                    data_mb = (
                        (*data_mb,)
                        if quant_config.get("RNNmodel_wo_hidden", False)
                        else (*data_mb, hid)
                    )  # NOTE: Unpack the tuple/list and make it a tuple for torch.jit.trace
                elif isinstance(data_mb, torch.Tensor):
                    logger.warning(
                        "Data minibatch is a single tensor, please make sure this "
                        "is the correct data format to send to model."
                    )
                    data_mb = data_mb.to(currDev)
                    data_mb = (
                        (data_mb,)
                        if quant_config.get("RNNmodel_wo_hidden", False)
                        else (data_mb, hid)
                    )  # NOTE: Unpack the tuple/list and make it a tuple for torch.jit.trace
                else:
                    logger.error(
                        "RNN model dataloader returns abnormal mini batch, please check!"
                    )

        with torch.no_grad():
            traced_cell = torch.jit.trace(temp_model, data_mb)
    # Special handling for transformers/BERT
    elif (
        quant_config["QBmm"]
        or len(
            [
                n
                for n, m in temp_model.named_modules()
                if "attn" in n or "attention" in n
            ]
        )
        > 0
        and (not quant_config["qvit"])  # "transformers" in sys.modules
    ):
        # Using pre-forward function is not good enough, because model input needs to be dictionary,
        # but we have 2 extra slicing in generic code section
        data_mb = transformers_prepare_input(data_mb, dev=currDev)
        Nbatch_small_trace = max(quant_config["world_size"], 4)
        # Use small batch for tracing to avoid Out of Memory
        if isinstance(data_mb, (torch.Tensor, tuple)):
            # Assume user already slice/prep the tensor properly, use as is
            with torch.no_grad(), patch_torch_bmm(quant_config):
                traced_cell = torch.jit.trace(
                    temp_model, example_inputs=data_mb, check_trace=False, strict=False
                )
        elif isinstance(data_mb, (dict, BatchEncoding)):
            try:
                # may cause out of memory error without slicing
                with torch.no_grad(), patch_torch_bmm(quant_config):
                    traced_cell = torch.jit.trace(
                        temp_model,
                        example_kwarg_inputs=data_mb,
                        check_trace=False,
                        strict=False,
                    )
            except MemoryError:
                data_mb = (data_mb["input_ids"][:Nbatch_small_trace],)
                traced_cell = torch.jit.trace(
                    temp_model, example_inputs=data_mb, check_trace=False, strict=False
                )
        else:
            raise RuntimeError(
                "Incorrect input type for torchscript tracing, only tensor, "
                f"tuple or dict are allowed. got { type(data_mb)}"
            )
    # All other models, such as vision or object detection
    else:
        data_mb: Tuple = prepare_data_4_fwd(data_mb, quant_config, prefwdproc, currDev)

        with torch.no_grad():
            if quant_config["qvit"]:
                with patch_torch_bmm(quant_config):
                    traced_cell = torch.jit.trace(
                        temp_model, data_mb, check_trace=False, strict=False
                    )
            else:
                traced_cell = torch.jit.trace(
                    temp_model, data_mb, check_trace=False, strict=False
                )
        # NOTE: DP model would have been unwrapped at this point

    # PART 1.1: create a computation graph using our own data structure
    reconstructed_graph = Graph(traced_cell.inlined_graph)
    if quant_config["plotsvg"]:
        reconstructed_graph.plot_short(
            [":", "^"], output_name=save_fname + ".svg", fields=3
        )

    del traced_cell
    if not inplace:
        temp_model.to(torch.device("cpu"))
        del temp_model

    return reconstructed_graph


# Simplification old torchscript codes in qmodel_prep
def model_analyzer_ts(
    model, dloader, quant_config: dict, prefwdproc: callable, save_fname: str, dev: str
):
    """
       Option 2: trace the model and find candidates to quantize
    important operate on copy to avoid any potential modification on the original model,
       including calling forward()...

        Args:
            model (nn.Module): model
            dloader (torch.utils.data.DataLoader): user provided data, could be as simple as a batch
                of "ready-to-run" data, a list of data, or a dataloader.
            quant_config (dict): quantization configuration file
            prefwdproc (callable, optional): if the data fetched from dataloader needs further
                process before feeding to model, user can define this "pre-forward-process" func,
                which will be called in this way:
                    model( prefwdproc(data) ).
           save_fname (str): file name for saving
           dev (str): device type

       Returns:
           qskip_layer_name: A list of the names of the layers to skip during quantization.
           QsinglesidedConvs: A list of the names of the layers with single-sided convolutions.
    """

    reconstructed_graph = trace_and_reconstruct(
        model,
        dloader,
        quant_config,
        prefwdproc=prefwdproc,
        save_fname=save_fname + "_beforeQ",
        dev=dev,
    )

    # PART 1.2: Identify layers to skip quantization and single-sided convs,
    # e.g. following relu. (NOTE: BKM is shortcut, 1st/last, and DW. DW will be handled in QanyNet4)
    shortcut_candidates = (
        find_shortcut_conv_v2(reconstructed_graph)
        if not quant_config["qshortcutconv"]
        else []
    )
    if quant_config["isRNNmodel"]:
        first_candidates = (
            last_candidates
        ) = []  # last is usually Linear, which most of the time needs to be quantized
    else:
        first_candidates = (
            reconstructed_graph.DFS(
                ["aten::_conv", "aten::addmm", "aten::linear", "aten::embedding"],
                find_first_only=True,
                reverse=False,
                verbose=True,
                ignore_op="attention_mask",
            )
            if not quant_config["q1stlastconv"]
            else []
        )
        last_candidates = (
            reconstructed_graph.DFS(
                ["aten::_conv", "aten::addmm", "aten::linear"],
                find_first_only=True,
                stop_op="prim::TupleUnpack",  # NOTE: Hack for LLMs
                reverse=True,
            )
            if not quant_config["q1stlastconv"]
            else []
        )
        # nn.Linear will decompose into matmul+add_ in BERT (older PT), but "first" is everywhere
        potential_candidates = (
            reconstructed_graph.DFS(
                ["aten::matmul"], find_first_only=True, reverse=True
            )
            if not quant_config["q1stlastconv"]
            else []
        )  # Find last
        potential_last = [
            p for p in potential_candidates if "aten::add_" in p.children_ptr[0].Op
        ]
        if last_candidates:
            # Make sure newly found potential_lasts is not upstream (a parent) of any of
            # the existing "last candidates"
            potential_candidates = []
            for n_i in potential_last:
                if not any(
                    reconstructed_graph.ischild(l_j, n_i) for l_j in last_candidates
                ):
                    potential_candidates.append(n_i)

        last_candidates += potential_candidates
        # Special case for BERT when using "output_attentions=True", which will create extra
        # TupleConstr and search path for reverse DFS
        # As a result, last_candidates will include almost all the Linears
        if (
            quant_config.get("output_attentions", False)
            or quant_config.get("decoder_arch", False)
        ) and len(last_candidates) > 5:
            last_candidates = last_candidates[:1]

    # 1.2.1 try to identify RPN, which by default will be skipped
    rpn_candidates = reconstructed_graph.id_rpn_from_last(last_candidates)
    # NOTE: only if user turned on Qrpn flag and we found rpn in this net
    if quant_config.get("Qrpn", False) and rpn_candidates:
        last_candidates = [n for n in last_candidates if n not in rpn_candidates]

    q_skip_candidates = shortcut_candidates + last_candidates + first_candidates

    # 1.2.2 try to identify FPN, which by default will be quantized
    fpn_candidates = reconstructed_graph.find_fpn_convs()
    logger.info(
        f"Skip candidates: {q_skip_candidates}\n fpn candidates: {fpn_candidates}"
    )
    if quant_config.get("qskipfpn", False) and fpn_candidates:
        for fpn_candidate in fpn_candidates:
            if fpn_candidate not in q_skip_candidates:
                q_skip_candidates.append(fpn_candidate)
    # Backward compatibility, if Qfpn is True, make sure fpn_candidates are not on q_skip list
    elif quant_config.get("Qfpn", True) and fpn_candidates:
        q_skip_candidates = [n for n in q_skip_candidates if n not in fpn_candidates]

    # 1.2.3 Determine single or double sided Conv
    singlesidedConvs = find_single_sided_conv_v2(reconstructed_graph)

    # 1.2.4 Special handle for silu, for EffDet and etc
    if quant_config.get("specialhandleSiLU"):
        siluConv = []
        for c in reconstructed_graph.DFS(["aten::_conv"]):
            chk_pOp, chk_gpOp = False, False
            pOp = c.parents_ptr[0].Op
            if pOp:
                chk_pOp = "silu" in pOp
            if c.parents_ptr[0].parents_ptr:  # not []
                gpOp = c.parents_ptr[0].parents_ptr[0].Op
                if gpOp:
                    chk_gpOp = "silu" in gpOp
            if chk_pOp or chk_gpOp:
                siluConv += [c]
        quant_config["qspecial_layers"] = {
            sc.scope.replace("__module.", "").strip(): {"qa_mode": "qsilu"}
            for sc in siluConv
        }

    # 1.2.5 Find torch.bmm function calls, add QBmm modules to where bmm is called.
    # NOTE: BMM is used as a function, not module
    if quant_config["QBmm"] and quant_config["which2patch_contextmanager"] != "off":
        # ['QBmm'] is determined by nbits_bmm[1,2], if using QBertSelfAttn instead of func swapping,
        #  it could still be True
        # ['which2patch'] == 'off' will forcefully turn off this searching and QBmm attaching
        QBmm = quant_config["mapping"]["matmul_or_bmm"]

        find_single_sided_bmm(reconstructed_graph)
        # After search, flag "isActOutUnidir" and "isActOutBounded" will be available
        # in all nodes, need those flags to setup QBmm()

        num_qbmm_attached = 0
        num_bmm_found = 0
        for b in reconstructed_graph.DFS(["aten::bmm", "aten::matmul"]):
            # Make sure if it's a matmul, it is not followed by an immediate 'add'
            # (which is most like translated from a nn.Linear)
            if "aten::matmul" in b.Op and "aten::add_" in b.children_ptr[0].Op:
                continue
            isbmm = "bmm" in b.Op
            mod_names = (
                b.scope.replace("__module.", "").strip().split(".")
            )  # ['model','backbone',....] or ['backbone',...]
            # FIXME: Not sure why model need to be exclude here, but will cause opt model to fail

            curr_mod = model
            for m in mod_names:
                curr_mod = getattr(curr_mod, m)

            # NOTE: (add more codes here to filter out those we don't want to quantize)

            # because each module could call BMM more than once, need add the line number
            # to make the name unique
            # (i.e. "line number" is the real physical line number in xxx.py, which was
            # extracted by torchscript)
            # If we can find another QBmm -> which=2,
            whichQBmm = 1 + any(attr for attr in dir(curr_mod) if "QBmm" in attr)
            # TODO: Generalize to multi QBmm (use sum instead of any)
            newQBmm = QBmm(
                num_bits_m1=quant_config[f"nbits_bmm{whichQBmm}"],
                num_bits_m2=quant_config[f"nbits_bmm{whichQBmm}"],
                qm1_mode=quant_config[f"bmm{whichQBmm}_qm1_mode"],
                qm2_mode=quant_config[f"bmm{whichQBmm}_qm2_mode"],
                m1_unidirectional=b.parents_ptr[0].isActOutUnidir,
                m1_bounded=b.parents_ptr[0].isActOutBounded,
                m2_unidirectional=b.parents_ptr[1].isActOutUnidir,
                m2_bounded=b.parents_ptr[1].isActOutBounded,
                replaceBmm=isbmm,
                qcfg=quant_config,
            )
            setattr(curr_mod, f"QBmm{b.lineno}", newQBmm)
            quant_config["LUTmodule_name"][newQBmm] = (
                ".".join(mod_names) + f".QBmm{b.lineno}"
            )

            num_bmm_found += isbmm
            num_qbmm_attached += 1

        if num_bmm_found > 0 and num_qbmm_attached == num_bmm_found:
            quant_config["which2patch_contextmanager"] = "torch.bmm"
        elif num_bmm_found == 0 and num_qbmm_attached > 0:
            quant_config["which2patch_contextmanager"] = "torch.matmul"
        else:
            quant_config["which2patch_contextmanager"] = None
        logger.info(
            f"Found {num_bmm_found} torch.bmm and {num_qbmm_attached-num_bmm_found} torch.matmul"
        )
        logger.info(
            f"Context manager is set to intercept {quant_config['which2patch_contextmanager']}"
        )

    # 1.2.6 sync QKV's clipval (or any linears that share parent)
    # here we identify the groups that share parent (at this point, they are still nn.Linear)
    # QanyNet4() will just simply force them to use the same nn.parameter, then optimizer will
    # handle the rest
    if quant_config["qkvsync"]:
        linears = list(reconstructed_graph.DFS(["aten::linear"])) + [
            n
            for n in reconstructed_graph.DFS(["aten::matmul"])
            if "aten::add_" in n.children_ptr[0].Op
        ]
        # If it's a matmul, make sure it is followed by an immediate 'add'
        # (which is translated from a nn.Linear)
        linear_parents = [n.parents_ptr[0] for n in linears]
        quant_config["qkvsync_my_1st_sibling"] = {}
        # this LUT will help to find the 1st sibling of a given nn.Linear, if exists
        # the keys of this LUT are nn.Linear, and values are nn.Linear here, too. But values
        # will be updated by QanyNet to Qlinear later.
        for i, ni in enumerate(linears):
            pi = linear_parents[i]
            mod_i = get_module_from_node(model, ni)
            if isinstance(pi, Node):  # To avoid redundant search
                if mod_i not in quant_config["qkvsync_my_1st_sibling"]:
                    quant_config["qkvsync_my_1st_sibling"][mod_i] = mod_i
                # No records of siblings yet -> 1st of a group -> add mod_i itself in the LUT ->
                # all members of a group should be in the LUT, so that seq of search won't matter
                # e.g., LUT[mod1]=mod2, LUT[mod2]=mod2, LUT[mod3]=mod2
                for j in range(i + 1, len(linears)):
                    nj = linears[j]
                    pj = linear_parents[j]
                    if pj == pi:
                        mod_j = get_module_from_node(model, nj)
                        quant_config["qkvsync_my_1st_sibling"][mod_j] = mod_i
                        # In QanyNet we will try to replace all values of mod1 in LUT to Qmod1
                        linear_parents[j] = i  # To avoid redundant search later
        logger.info(
            f"Found {len(quant_config['qkvsync_my_1st_sibling'])} linear layers in "
            f"{len(set(quant_config['qkvsync_my_1st_sibling'].values()))} group "
            "that have shared parents."
        )

    qskip_layer_name, QsinglesidedConvs = [], []
    tempqskip_layer_name = [
        qsc.scope.replace("__module.", "").strip()
        for qsc in q_skip_candidates
        if qsc.scope
    ]
    tempQsinglesidedConvs = [
        ssc.scope.replace("__module.", "").strip() for ssc in singlesidedConvs
    ]
    # Sometimes, e.g detectron2, layer names have a prefix of 'model.', add both names with
    # and without 'model.' to the skip list just in case
    for fpn_candidate in tempqskip_layer_name:
        if fpn_candidate in quant_config["qskip_layer_name"]:
            continue  # If loading from json, Qskip_layer won't be empty to begin with

        qskip_layer_name.append(fpn_candidate)
        if fpn_candidate.startswith("model."):
            qskip_layer_name.append(fpn_candidate[6:])

    for fpn_candidate in tempQsinglesidedConvs:
        if fpn_candidate in quant_config["qsinglesided_name"]:
            continue  # If loading from json, Qsinglesided won't be empty to begin with

        QsinglesidedConvs.append(fpn_candidate)
        if fpn_candidate.startswith("model."):
            QsinglesidedConvs.append(fpn_candidate[6:])

    del reconstructed_graph

    return qskip_layer_name, QsinglesidedConvs
