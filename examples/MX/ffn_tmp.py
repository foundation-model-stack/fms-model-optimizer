# Third Party

# from mx import Linear as Linear_mx  # Need to amend mx's Linear class
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
    # Add config arguments
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--hidden_size", default=128)
    # parser.add_argument("--device", default='cuda')
    # args = parser.parse_args()
    # Standard
    from functools import partial

    # Third Party
    from mx import MxSpecs
    from tabulate import tabulate

    # Local
    from fms_mo import qconfig_init, qmodel_prep

    x = np.random.randn(16, 128)
    x = torch.tensor(x, dtype=torch.float32, device="cuda")
    results = {"dtype": [], "output[0, :5]": [], "||ref - out_dtype||_2": []}

    # --- Test 0. Run MLP as is
    mlp = ResidualMLP(128)
    # mlp.to("cuda")
    with torch.no_grad():
        out = mlp(x)
        results["dtype"].append("fp32")
        results["output[0, :5]"].append(out[0, :5].tolist())
        results["||ref - out_dtype||_2"].append("-")
    print(mlp)

    # --- Test 1. fms-mo qmodel_prep, replace Linear with our QLinear
    qcfg = qconfig_init()
    qcfg["nbits_a"] = 8
    qcfg["nbits_w"] = 8
    model = qmodel_prep(mlp, x, qcfg)
    with torch.no_grad():
        out_dtype = model(x)
        results["dtype"].append("fms_int8")
        results["output[0, :5]"].append(out_dtype[0, :5].tolist())
        results["||ref - out_dtype||_2"].append(torch.norm(out - out_dtype).item())
    # print(model)

    qcfg["nbits_a"] = 4
    qcfg["nbits_w"] = 4
    mlp = ResidualMLP(128)
    model = qmodel_prep(mlp, x, qcfg)
    with torch.no_grad():
        out_dtype = model(x)
        results["dtype"].append("fms_int4")
        results["output[0, :5]"].append(out_dtype[0, :5].tolist())
        results["||ref - out_dtype||_2"].append(torch.norm(out - out_dtype).item())
    print(model)

    # --- Test 2. now change mapping to MX
    # NOTE simply use qa_mode or qw_mode to trigger the use of mx, e.g. use "mx_" prefixed mode,
    #       qcfg["mapping"] and other qcfg["mx_specs"] content will be updated automatically

    for dtype_to_test in ["int8", "int4", "fp8_e4m3", "fp8_e5m2", "fp4_e2m1"]:
        qcfg["qw_mode"] = f"mx_{dtype_to_test}"
        qcfg["qa_mode"] = f"mx_{dtype_to_test}"
        mlp = ResidualMLP(128)  # fresh model
        model = qmodel_prep(mlp, x, qcfg)
        with torch.no_grad():
            out_dtype = model(x)
            results["dtype"].append(f"mx{dtype_to_test}")
            results["output[0, :5]"].append(out_dtype[0, :5].tolist())
            results["||ref - out_dtype||_2"].append(torch.norm(out - out_dtype).item())
    print(model)

    print(tabulate(results, headers="keys"))

    print("DONE!")
