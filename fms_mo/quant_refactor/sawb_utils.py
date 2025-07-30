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
SAWB utility functions for computing clip values.

Raises:
    ValueError: SAWB not implemented for various num_bits
    ValueError: SAWB code is not implemented
"""

# Standard
from typing import Tuple

# Third Party
import torch


def sawb_params(
    input_tensor: torch.Tensor,
    num_bits: torch.Tensor,
    qlevel_lowering: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute SAWB symmetric clip value and # of quantized levels.

    Args:
        input_tensor (torch.Tensor): Tensor to be quantized.
        num_bits (torch.Tensor): Number of bit for quantization.
        qlevel_lowering (bool, optional): Specify lowering of quantized levels.
                Defaults to False.

    Raises:
        ValueError: SAWB not implemented for various num_bits

    Returns:
        [torch.Tensor, torch.Tensor]: SAWB quantiation parameters
    """
    with torch.no_grad():
        x = input_tensor.flatten()
        mu = x.abs().mean()
        std = x.mul(x).mean().sqrt()

        # dic_coeff = {2:(3.212, -2.178), 3:(7.509, -6.892), 4:(12.68, -12.80), 5:(17.74, -18.64)}
        dic_coeff = {
            2: (3.12, -2.064),
            3: (7.509, -6.892),
            4: (12.68, -12.80),
            5: (17.74, -18.64),
            8: (31.76, -35.04),
        }
        if num_bits > 8:
            raise ValueError(f"SAWB not implemented for num_bits={num_bits}")
        num_bits_int = (
            num_bits.item() if isinstance(num_bits, torch.Tensor) else num_bits
        )
        coeff = dic_coeff[num_bits_int]
        clip_val = coeff[1] * mu + coeff[0] * std

        # Overwrite negative clip_vals with abs.max
        if clip_val < 0.0:
            clip_val = x.abs().max()

        n_levels = 2**num_bits - 2 if qlevel_lowering else 2**num_bits - 1

        return n_levels, clip_val


def sawb_params_code(
    input_tensor: torch.Tensor,
    num_bits: torch.Tensor,
    code: int,
    perCh: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute SAWB symmetric clip value and # of quantized levels

    Args:
        input_tensor (torch.Tensor): Tensor to be quantized.
        num_bits (torch.Tensor): Number of bit for quantization.
        code (int): Pre-built SAWB constants.
        perCh (bool, optional): Use per channel quantization. Defaults to False.

    Raises:
        ValueError: Code is not implemented
        ValueError: Improper code provided

    Returns:
        [torch.Tensor, torch.Tensor]: SAWB quantiation parameters
    """
    with torch.no_grad():
        coeff_dict = {
            102: (3.12, -2.064),  # [-a, -a/3, a/3, a] equivalent to 2 bits
            103: (2.6, -1.71),  # [-a, 0, a]
            403: (12.035, -12.03),  # [-a, -6/7a, ..., 0, ..., 6/7a, a]
            703: (28.24, -30.81),
            803: (31.76, -35.04),
        }

        if not coeff_dict.get(code) is None:
            coeff = coeff_dict[code]
        else:
            raise ValueError(f"SAWB not implemented for code={code}")

        if perCh:
            reduce_dim = list(range(1, len(input_tensor.shape)))
            # conv W=[ch_o, ch_i, ki, ij], linear W=[ch_o, ch_i], reduce all dim but ch_out
            mu = torch.mean(input_tensor.abs(), dim=reduce_dim)
            std = torch.mean(input_tensor**2, dim=reduce_dim).sqrt()
            clip_val_vec = coeff[1] * mu + coeff[0] * std

            # Overwrite negative clip_vals with abs.max
            neg_clip_idx = clip_val_vec < 0.0
            if torch.any(neg_clip_idx):
                clip_val_max = torch.max(input_tensor.abs(), dim=1).values
                clip_val_vec[neg_clip_idx] = clip_val_max[neg_clip_idx]

            return None, clip_val_vec

        # per-tensor
        x = input_tensor.flatten()
        mu = x.abs().mean()
        std = x.mul(x).mean().sqrt()
        clip_val = coeff[1] * mu + coeff[0] * std

        # Overwrite negative clip_vals with abs.max
        if clip_val < 0.0:
            clip_val = x.abs().max()

        if code in [102]:
            n_levels = 2**num_bits - 1
        elif code in [103, 203, 403, 703, 803]:
            n_levels = 2**num_bits - 2
        else:
            raise ValueError(f"SAWB not implemented for code={code}")

        return n_levels, clip_val
