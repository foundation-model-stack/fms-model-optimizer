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

"""Quantization LSTM modules"""
# pylint: disable=arguments-renamed

# Standard
from typing import Optional, Tuple

# Third Party
from packaging.version import Version
from torch import _VF, Tensor, nn
import torch
import torch.nn.functional as F

# Local
from fms_mo.quant.quantizers import get_activation_quantizer, get_weight_quantizer


class QLSTM(nn.LSTM):
    """
    A class that implements a quantized version of the LSTM cell.

    Attributes:
        input_size (int): The number of expected features in the input x.
        hidden_size (int): The number of features in the hidden state h.
        num_layers (int, optional): Number of recurrent layers. Default: 1.
        bias (bool, optional): If False, then the layer does not use bias.
                                    Default: True.
        batch_first (bool, optional): If True, then the input and output tensors are provided as
                                    (batch, seq, feature). Default: False.
        dropout (float, optional): If non-zero, introduces a Dropout layer on the outputs of each
                                    LSTM layer except the last layer, with dropout probability equal
                                    to dropout. Default: 0.0.
        bidirectional (bool, optional): If True, becomes a bidirectional LSTM. Default: False.
        num_bits_input (int, optional): The number of bits used to represent the input activations.
                                    Default: 32.
        num_bits_hidden (int, optional): The number of bits used to represent the hidden state
                                    activations. Default: 32.
        num_bits_weight (int, optional): The number of bits used to represent the weights.
                                    Default: 32.
        qi_mode (str, optional): The quantization mode for the input activations. Default: None.
        qh_mode (str, optional): The quantization mode for the hidden state activations.
                                    Default: None.
        qw_mode (str, optional): The quantization mode for the weights. Default: None.
        i_clip_val_init (float, optional): The initial value for the input clipping threshold.
                                    Default: None.
        i_clip_valn_init (float, optional): The initial value for the negative input clipping
                                    threshold. Default: None.
        h_clip_val_init (float, optional): The initial value for the hidden state clipping
                                    threshold. Default: None.
        h_clip_valn_init (float, optional): The initial value for the hidden state negative clipping
                                    threshold. Default: None.
        w_clip_val_init (float, optional): The initial value for the weight clipping threshold.
                                    Default: None.
        w_clip_valn_init (float, optional): The initial value for the negative weight clipping
                                    threshold. Default: None.
        align_zero (bool, optional): If True, aligns the zero point of the quantizers to zero.
                                    Default: True.
        qcfg (dict, optional): A dictionary containing the configuration parameters for the
                                    quantizers. Default: None.
    """

    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers=1,
        bias=True,
        batch_first=False,
        dropout=0.0,
        bidirectional=False,
        num_bits_input=32,
        num_bits_hidden=32,
        num_bits_weight=32,
        qi_mode=None,
        qh_mode=None,
        qw_mode=None,
        i_clip_val_init=None,
        i_clip_valn_init=None,
        h_clip_val_init=None,
        h_clip_valn_init=None,
        w_clip_val_init=None,
        w_clip_valn_init=None,
        align_zero=True,
        **kwargs,
    ):
        """
        Initializes the quantized LSTM cell.

        Args:
            input_size (int): The number of expected features in the input x.
            hidden_size (int): The number of features in the hidden state h.
            num_layers (int, optional): Number of recurrent layers. Default: 1.
            bias (bool, optional): If False, then the layer does not use bias.
                                        Default: True.
            batch_first (bool, optional): If True, then the input and output tensors are provided as
                                        (batch, seq, feature). Default: False.
            dropout (float, optional): If non-zero, introduces a Dropout layer on the outputs of
                                        each LSTM layer except the last layer, with dropout
                                        probability equal to dropout. Default: 0.0.
            bidirectional (bool, optional): If True, becomes a bidirectional LSTM. Default: False.
            num_bits_input (int, optional): The number of bits used to represent the input
                                        activations. Default: 32.
            num_bits_hidden (int, optional): The number of bits used to represent the hidden state
                                        activations. Default: 32.
            num_bits_weight (int, optional): The number of bits used to represent the weights.
                                        Default: 32.
            qi_mode (str, optional): The quantization mode for the input activations. Default: None.
            qh_mode (str, optional): The quantization mode for the hidden state activations.
                                        Default: None.
            qw_mode (str, optional): The quantization mode for the weights. Default: None.
            i_clip_val_init (float, optional): The initial value for the input clipping threshold.
                                        Default: None.
            i_clip_valn_init (float, optional): The initial value for the negative input clipping
                                        threshold. Default: None.
            h_clip_val_init (float, optional): The initial value for the hidden state clipping
                                        threshold. Default: None.
            h_clip_valn_init (float, optional): The initial value for the hidden state negative
                                        clipping threshold. Default: None.
            w_clip_val_init (float, optional): The initial value for the weight clipping threshold.
                                        Default: None.
            w_clip_valn_init (float, optional): The initial value for the negative weight clipping
                                        threshold. Default: None.
            align_zero (bool, optional): If True, aligns the zero point of the quantizers to zero.
                                        Default: True.
            **kwargs (dict): Additional keyword arguments.

        Returns:
            torch.nn.Module: The initialized PACT-based quantized LSTM cell.
        """
        super().__init__(
            input_size,
            hidden_size,
            num_layers=num_layers,
            bias=bias,
            batch_first=batch_first,
            dropout=dropout,
            bidirectional=bidirectional,
        )

        qcfg = kwargs.pop("qcfg")
        self.num_bits_input = num_bits_input
        self.num_bits_hidden = num_bits_hidden
        self.num_bits_weight = num_bits_weight
        self.qi_mode = qi_mode
        self.qh_mode = qh_mode
        self.qw_mode = qw_mode
        self.i_clip_val_init = (
            i_clip_val_init if i_clip_val_init else qcfg.get("i_clip_val_init", 1.0)
        )
        self.i_clip_valn_init = (
            i_clip_valn_init if i_clip_valn_init else qcfg.get("i_clip_valn_init", -1.0)
        )
        self.h_clip_val_init = (
            h_clip_val_init if h_clip_val_init else qcfg.get("h_clip_val_init", 1.0)
        )
        self.h_clip_valn_init = (
            h_clip_valn_init if h_clip_valn_init else qcfg.get("h_clip_valn_init", -1.0)
        )
        self.w_clip_val_init = (
            w_clip_val_init if w_clip_val_init else qcfg.get("w_clip_val_init", 1.0)
        )
        self.w_clip_valn_init = (
            w_clip_valn_init if w_clip_valn_init else qcfg.get("w_clip_valn_init", -1.0)
        )
        self.align_zero = align_zero
        self.num_directions = 2 if bidirectional else 1
        if not hasattr(self, "proj_size"):
            self.proj_size = 0  # A new param added since PyTorch > 1.8, compatiblity
        self.pact_plus = qi_mode.endswith("+")

        for layer in range(self.num_layers):
            # Layer 0 in one direction (inputs to layer 0 are the same in the two directions,
            # only need to quantize them once)
            if num_bits_input < 32 and layer == 0:
                setattr(
                    self,
                    f"quantize_input_layer{layer}",
                    get_activation_quantizer(
                        self.qi_mode,
                        nbits=self.num_bits_input,
                        clip_val=self.i_clip_val_init,
                        clip_valn=self.i_clip_valn_init,
                        non_neg=False,
                        align_zero=self.align_zero,
                        extend_act_range=bool(self.extend_act_range),
                    ),
                )

            for direction in range(self.num_directions):
                suffix = "_reverse" if direction else ""

                if num_bits_hidden < 32:
                    setattr(
                        self,
                        f"quantize_hidden_layer{layer}{suffix}",
                        get_activation_quantizer(
                            self.qh_mode,
                            nbits=self.num_bits_hidden,
                            clip_val=self.h_clip_val_init,
                            clip_valn=self.h_clip_valn_init,
                            non_neg=False,
                            align_zero=self.align_zero,
                            extend_act_range=bool(self.extend_act_range),
                        ),
                    )
                if num_bits_weight < 32:
                    for w_type in ["hh", "ih"]:
                        for gate in ["ig", "fg", "cg", "og"]:
                            setattr(
                                self,
                                f"quantize_weight_{w_type}_layer{layer}{suffix}_{gate}",
                                get_weight_quantizer(
                                    self.qw_mode,
                                    nbits=self.num_bits_weight,
                                    clip_val=self.w_clip_val_init,
                                    clip_valn=self.w_clip_valn_init,
                                    align_zero=self.align_zero,
                                ),
                            )

    def forward(
        self, x: Tensor, hx: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        self.quantize_rnn_weights()
        output, next_hidden = self.lstm_layers_and_time_steps(x, hx)

        return output, next_hidden

    def quantize_rnn_weights(self):
        """
        Quantizes the weights of an LSTM network.
        """
        self.all_qweights = []
        for layer in range(self.num_layers):
            for direction in range(self.num_directions):
                sublayer = (
                    layer * self.num_directions + direction
                )  # if unidirectional, sublayer == layer
                suffix = "_reverse" if direction else ""

                self.all_qweights.append([])
                for name in self._all_weights[
                    sublayer
                ]:  # Nested list of full precision weight *names (from nn.LSTM)
                    if "bias" in name or self.num_bits_weight == 32:
                        self.all_qweights[sublayer].append(self._parameters[name])
                        continue
                    # Quantize weights using separate quantizer to each gate (input, forget,
                    # cell, output)
                    w_type = "ih" if "ih" in name else "hh"
                    qweights_ig = getattr(
                        self, f"quantize_weight_{w_type}_layer{layer}{suffix}_ig"
                    )(self._parameters[name][: self.hidden_size, :])
                    qweights_fg = getattr(
                        self, f"quantize_weight_{w_type}_layer{layer}{suffix}_fg"
                    )(
                        self._parameters[name][
                            self.hidden_size : 2 * self.hidden_size, :
                        ]
                    )
                    qweights_cg = getattr(
                        self, f"quantize_weight_{w_type}_layer{layer}{suffix}_cg"
                    )(
                        self._parameters[name][
                            2 * self.hidden_size : 3 * self.hidden_size, :
                        ]
                    )
                    qweights_og = getattr(
                        self, f"quantize_weight_{w_type}_layer{layer}{suffix}_og"
                    )(self._parameters[name][3 * self.hidden_size :, :])
                    self.all_qweights[sublayer].append(
                        torch.cat(
                            (qweights_ig, qweights_fg, qweights_cg, qweights_og), dim=0
                        )
                    )

    def lstm_layers_and_time_steps(self, x, hidden):
        """
        Function for LSTM layers and time steps.

        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size).
            hidden: Tuple of initial hidden states and cell states, each of shape
                    (num_layers * num_directions, batch_size, hidden_size).

        Returns:
            A tuple containing the output tensor of shape (batch_size, sequence_length, hidden_size)
                and the updated hidden states and cell states.
        """
        if hidden is None:
            hidden = (
                x.new_zeros(2 * self.num_layers, x.size(0), self.hidden_size),
                x.new_zeros(2 * self.num_layers, x.size(0), self.hidden_size),
            )

        next_hidden = []

        if self.batch_first:
            x = x.transpose(
                0, 1
            )  # swap first 2 dimensions [bsz, seqN, inN] -> [seqN, bsz, inN]

        # Hidden is 2-element tuple, each of size (nlayers, bsz, hidN)
        # changed here to LIST of nlayers tuples: ([bsz, hidN],  [bsz, hidN]) x nlayers
        hidden = list(zip(*hidden))

        for layer in range(self.num_layers):
            # For bidirectional LSTM, store outputs of both directions (otherwise stays empty)
            qoutput_all_ts_bidir = []

            if (
                layer == 0
            ):  # inputs are quantized once (if bidirectional, same inputs go to two direction)
                if self.num_bits_input < 32:
                    qinput = getattr(self, f"quantize_input_layer{layer}")(x)
                else:
                    qinput = x

            for direction in range(self.num_directions):
                qoutput_all_ts = []
                suffix = "_reverse" if direction else ""
                sublayer = (
                    self.num_directions * layer + direction
                )  # if unidir, sublayer == layer

                # if reverse layer (direction == 1) => reverse step order
                steps = range(x.size(0) - 1, -1, -1) if direction else range(x.size(0))

                weight_layer = self.all_qweights[
                    sublayer
                ]  # NOTE: weight of each layer are unpacked into w_ih, w_hh, b_ih, b_iw

                qhidden_layer = []
                for k in steps:
                    # Quantize hidden at t=0 only, for t>0 they come already quantized from
                    # previous time step
                    if (k == 0 and direction == 0) or (
                        k == x.size(0) - 1 and direction == 1
                    ):
                        if hidden[sublayer][0].sum() and self.num_bits_hidden < 32:
                            qhidden = getattr(
                                self, f"quantize_hidden_layer{layer}{suffix}"
                            )(hidden[sublayer][0])
                        else:
                            qhidden = hidden[sublayer][0]
                        qhidden_layer = (
                            qhidden,
                            hidden[sublayer][1],
                        )  # Cell state (hidden[layer][1]) is not quantized

                    cell_output = _VF.lstm_cell(qinput[k], qhidden_layer, *weight_layer)

                    if (
                        self.num_bits_hidden < 32
                    ):  # Quantize cell output (will be fed to both next time step and next layer)
                        qoutput = getattr(
                            self, f"quantize_hidden_layer{layer}{suffix}"
                        )(cell_output[0])
                    else:
                        qoutput = cell_output[0]

                    if layer < self.num_layers - 1:
                        qoutput_all_ts.append(
                            qoutput
                        )  # Store hy for next layer (every time step)
                    else:
                        # Last layer (in either direction): save non-quantized hy as output
                        # (every time step)
                        qoutput_all_ts.append(cell_output[0])

                    if (k < x.size(0) - 1 and direction == 0) or (
                        k > 0 and direction == 1
                    ):
                        qhidden_layer = (
                            qoutput,
                            cell_output[1],
                        )  # (hy, cy) are passed back to next time step
                    else:
                        # NOTE: Do not quantize the hidden states of last time step
                        # (or first time step of reverse direction)
                        qhidden_layer = (
                            cell_output[0],
                            cell_output[1],
                        )
                # Flip outputs of reverse direction (above they are stored in descending k order)
                if direction:
                    qoutput_all_ts.reverse()

                next_hidden.append(qhidden_layer)

                qoutput_all_ts = torch.cat(qoutput_all_ts, 0).view(
                    x.size(0), *qoutput.size()
                )  # hy at every time step changed from list of tensors to tensor)

                if self.num_directions == 2:
                    qoutput_all_ts_bidir.append(qoutput_all_ts)

            if self.num_directions == 1:
                qinput = qoutput_all_ts
            else:
                # Concatenate directionality on hidden dimension [NOTE: qinput stores
                # hidden state (all time steps) of last processed layer]
                qinput = torch.cat(qoutput_all_ts_bidir, x.dim() - 1)

            # Dropout is applied in between layers (not before the first and not after last)
            if self.dropout != 0 and layer < self.num_layers - 1:
                qinput = F.dropout(
                    qinput, p=self.dropout, training=self.training, inplace=False
                )

        next_h, next_c = zip(*next_hidden)  # Separate hidden state from cell state
        next_hidden = (
            torch.cat(next_h, 0).view(
                self.num_layers * self.num_directions, *next_h[0].size()
            ),
            torch.cat(next_c, 0).view(
                self.num_layers * self.num_directions, *next_c[0].size()
            ),
        )

        if self.batch_first:
            qinput = qinput.transpose(
                0, 1
            )  # swap back first 2 dimensions [seqN, bsz, inN] -> [bsz, seqN, inN]

        return qinput, next_hidden

    def flatten_qweights(self):
        """
        This function flattens the weights of a LSTM layer in PyTorch.
        It takes in the layer's parameters and returns a flattened version of the weights.
        The function checks if the weights are on the GPU and if they are compatible with cuDNN,
        and then flattens them using the appropriate method depending on the PyTorch version.
        """
        _flat_qweights = [p for layerparams in self.all_qweights for p in layerparams]
        first_fw = _flat_qweights[0]
        dtype = first_fw.dtype
        for fw in _flat_qweights:
            if (
                not isinstance(fw.data, Tensor)
                or not (fw.data.dtype == dtype)
                or not fw.data.is_cuda
                or not torch.backends.cudnn.is_acceptable(fw.data)
            ):
                return
        unique_data_ptrs = set(p.data_ptr() for p in _flat_qweights)
        if len(unique_data_ptrs) != len(_flat_qweights):
            return
        with torch.cuda.device_of(first_fw):
            # Third Party
            from torch.backends.cudnn import rnn

            with torch.no_grad():
                if Version(torch.__version__) < Version("1.8"):
                    torch._cudnn_rnn_flatten_weight(
                        self._flat_weights,
                        4,
                        self.input_size,
                        rnn.get_cudnn_mode("LSTM"),
                        self.hidden_size,
                        self.num_layers,
                        self.batch_first,
                        self.bidirectional,
                    )
                else:
                    torch._cudnn_rnn_flatten_weight(
                        self._flat_weights,
                        4,
                        self.input_size,
                        rnn.get_cudnn_mode("LSTM"),
                        self.hidden_size,
                        self.proj_size,
                        self.num_layers,
                        self.batch_first,
                        self.bidirectional,
                    )
                # NOTE: self.batch_first and self.bidirectin need position argument

    def __repr__(self):
        """
        Return a string representation of the quantized LSTM layer.
        """
        expr = (
            f"{self.__class__.__name__}({self.input_size}, {self.hidden_size}, "
            f"{self.num_layers}),\n"
            f"\tinput_bits={self.num_bits_input}, hidden_bits={self.num_bits_hidden}, "
            f"w_bits={self.num_bits_weight}"
        )
        if self.num_bits_input < 32:
            expr += f", input_quant_fn={self.quantize_input_layer0}"
        if self.num_bits_hidden < 32:
            expr += f", hidden_quant_fn={self.quantize_hidden_layer0}"
        if self.num_bits_weight < 32:
            expr += f", wei_quant_fn={self.quantize_weight_ih_layer0_ig}"
        return expr


# KEEP THIS AT END OF FILE - classes must be declared
QLSTM_modules = (QLSTM,)


def isinstance_qlstm(module):
    """
    Checks if the given module is one of the available quantized LSTM classes.

    Args:
        module (nn.Module): The module to check.

    Returns:
        bool: True if the module is a quantized LSTM class, False otherwise.
    """
    return isinstance(module, QLSTM_modules)
