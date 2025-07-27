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
"""Simple example using a toy model to demo how to trigger mx in fms-mo."""

# Third Party
import numpy as np
import torch
import torch.nn.functional as F


class ResidualMLP(torch.nn.Module):
    def __init__(self, hidden_size, device="cuda"):
        super(ResidualMLP, self).__init__()

        self.layernorm = torch.nn.LayerNorm(hidden_size, device=device)
        self.dense_4h = torch.nn.Linear(hidden_size, 4 * hidden_size, device=device)
        self.dense_h = torch.nn.Linear(4 * hidden_size, hidden_size, device=device)
        self.dummy = torch.nn.Linear(hidden_size, hidden_size, device=device)
        # add a dummy layer because by default we skip 1st/last, if there are only 2 layers, all will be skipped

    def forward(self, inputs):
        norm_outputs = self.layernorm(inputs)

        # MLP
        proj_outputs = self.dense_4h(norm_outputs)
        proj_outputs = F.gelu(proj_outputs)
        mlp_outputs = self.dense_h(proj_outputs)
        mlp_outputs = self.dummy(mlp_outputs)

        # Residual Connection
        outputs = inputs + mlp_outputs

        return outputs


if __name__ == "__main__":
    # Third Party
    from tabulate import tabulate

    # Local
    from fms_mo import qconfig_init, qmodel_prep

    HIDDEN_DIM = 128
    x = np.random.randn(16, HIDDEN_DIM)
    x = torch.tensor(x, dtype=torch.float32, device="cuda")
    results = {
        "dtype": [],
        "output[0, 0]": [],
        "output[0, 1]": [],
        "output[0, 2]": [],
        "||ref - out_dtype||_2": [],
    }

    # --- Test 0. Run MLP as is
    model = ResidualMLP(HIDDEN_DIM)
    with torch.no_grad():
        out = model(x)
        results["dtype"].append("fp32")
        results["output[0, 0]"].append(out[0, 0])
        results["output[0, 1]"].append(out[0, 1])
        results["output[0, 2]"].append(out[0, 2])
        results["||ref - out_dtype||_2"].append(0)
    print(model)

    # --- Test 1. fms-mo qmodel_prep, replace Linear with our QLinear
    qcfg = qconfig_init()
    qcfg["nbits_a"] = 8
    qcfg["nbits_w"] = 8
    qmodel_prep(model, x, qcfg)
    with torch.no_grad():
        out_dtype = model(x)
        results["dtype"].append("fmsmo_int8")
        results["output[0, 0]"].append(out_dtype[0, 0])
        results["output[0, 1]"].append(out_dtype[0, 1])
        results["output[0, 2]"].append(out_dtype[0, 2])
        results["||ref - out_dtype||_2"].append(torch.norm(out - out_dtype).item())
    print(model)

    qcfg["nbits_a"] = 4
    qcfg["nbits_w"] = 4
    model = ResidualMLP(HIDDEN_DIM)
    qmodel_prep(model, x, qcfg)
    with torch.no_grad():
        out_dtype = model(x)
        results["dtype"].append("fmsmo_int4")
        results["output[0, 0]"].append(out_dtype[0, 0])
        results["output[0, 1]"].append(out_dtype[0, 1])
        results["output[0, 2]"].append(out_dtype[0, 2])
        results["||ref - out_dtype||_2"].append(torch.norm(out - out_dtype).item())
    print(model)

    # --- Test 2. now change mapping to MX
    # NOTE simply use qa_mode or qw_mode to trigger the use of mx, e.g. use "mx_" prefixed mode,
    #       qcfg["mapping"] and other qcfg["mx_specs"] content will be updated automatically

    for dtype_to_test in ["int8", "int4", "fp8_e4m3", "fp8_e5m2", "fp4_e2m1"]:
        qcfg["qw_mode"] = f"mx_{dtype_to_test}"
        qcfg["qa_mode"] = f"mx_{dtype_to_test}"
        model = ResidualMLP(HIDDEN_DIM)  # fresh model
        qmodel_prep(model, x, qcfg)
        with torch.no_grad():
            out_dtype = model(x)
            results["dtype"].append(f"mx{dtype_to_test}")
            results["output[0, 0]"].append(out_dtype[0, 0])
            results["output[0, 1]"].append(out_dtype[0, 1])
            results["output[0, 2]"].append(out_dtype[0, 2])
            results["||ref - out_dtype||_2"].append(torch.norm(out - out_dtype).item())
    print(model)

    print(tabulate(results, headers="keys", tablefmt="pipe", floatfmt=".4f"))

    print("DONE!")
