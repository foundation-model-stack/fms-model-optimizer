# Third Party
import pytest
import torch

# Local
from fms_mo import qmodel_prep
from fms_mo.utils.import_utils import available_packages
from fms_mo.utils.qconfig_utils import check_config, set_mx_specs
from tests.models.test_model_utils import delete_file, qmodule_error

if available_packages["mx"]:
    # Local
    # pylint: disable=ungrouped-imports
    from fms_mo.modules.bmm import QBmmMX
    from fms_mo.modules.linear import QLinearMX

    mx_qmodules = [
        QLinearMX,
        QBmmMX,
    ]


@pytest.fixture(autouse=True)
def delete_files():
    """
    Delete any known files lingering before starting test
    """
    delete_file("qcfg.json")


@pytest.mark.skipif(
    not available_packages["mx"],
    reason="Skipping mx_specs error test; No package found",
)
def test_config_mx_specs_error(
    model_residualMLP: torch.nn.Module,
    config_fp32_mx_specs: dict,
    bad_mx_specs_settings: list,
):
    """
    Check that mx_specs throw ValueError when presented with bad key,value pair

    Args:
        model_residualMLP (torch.nn.Module): Single fp32 model
        config_fp32_mx_specs (dict): Config for fp32 quantization w/ mx_specs
        bad_mx_specs_settings (list):
            List of invalid values for mx_specs
    """
    model_dtype = next(model_residualMLP.parameters()).dtype

    assert "mx_specs" in config_fp32_mx_specs
    mx_specs_temp = config_fp32_mx_specs.get("mx_specs")

    for key, bad_val in bad_mx_specs_settings:
        # Every time we change the value, we must reset mx_specs
        config_fp32_mx_specs["mx_specs"][key] = bad_val
        set_mx_specs(config_fp32_mx_specs)

        with pytest.raises(ValueError):
            check_config(config_fp32_mx_specs, model_dtype)

        # Reset to saved value
        config_fp32_mx_specs["mx_specs"] = mx_specs_temp


@pytest.mark.skipif(
    not available_packages["mx"],
    reason="Skipping mx_specs error test; No package found",
)
def test_config_mx_error(
    model_residualMLP: torch.nn.Module,
    config_fp32_mx: dict,
    bad_mx_config_settings: list,
):
    """
    Check that mx_specs throw ValueError when presented with bad key,value pair

    Args:
        model_residualMLP (torch.nn.Module): Single fp32 model
        config_fp32_mx (dict): Config for fp32 quantization w/ mx_specs
        bad_mx_specs_settings (list):
            List of invalid values for mx_specs
    """
    model_dtype = next(model_residualMLP.parameters()).dtype

    assert "mx_specs" not in config_fp32_mx

    for (
        config_key,
        mx_specs_key,
        config_bad_val,
        mx_specs_bad_val,
    ) in bad_mx_config_settings:
        # Second check config w/ "mx_" prefix
        mx_temp = config_fp32_mx[config_key]

        # Need to reset qcfg["mx_specs"] w/ bad val
        config_fp32_mx[config_key] = config_bad_val

        set_mx_specs(config_fp32_mx)
        assert "mx_specs" in config_fp32_mx
        assert config_fp32_mx["mx_specs"][mx_specs_key] == mx_specs_bad_val

        with pytest.raises(ValueError):
            check_config(config_fp32_mx, model_dtype)

        # Reset value and delete mx_specs
        config_fp32_mx[config_key] = mx_temp
        del config_fp32_mx["mx_specs"]


@pytest.mark.skipif(
    not torch.cuda.is_available() or not available_packages["mx"],
    reason="Skipped because CUDA or MX library was not available",
)
def test_residualMLP(
    model_residualMLP: torch.nn.Module,
    input_residualMLP: torch.FloatTensor,
    config_fp32_mx_specs: dict,
    mx_format: str,
):
    """
    Test residualMLP for qmodel_prep

    Args:
        model_residualMLP (torch.nn.Module): Single fp32 model.
        input_residualMLP (torch.FloatTensor): Random 16x128 tensor.
        config_fp32_mx_specs (dict): Config for fp32 quantization w/ mx_specs.
        mx_format (str): MX format for quantization.
    """
    # Remove any saved qcfg.json
    delete_file()

    config_fp32_mx_specs["mx_specs"]["w_elem_format"] = mx_format
    config_fp32_mx_specs["mx_specs"]["a_elem_format"] = mx_format
    set_mx_specs(config_fp32_mx_specs)

    qmodel_prep(
        model_residualMLP, input_residualMLP, config_fp32_mx_specs, use_dynamo=True
    )
    qmodule_error(model_residualMLP, 2, 1)

    # One layer should be QLinearMX
    found_qmodule_mx = False
    for _, module in model_residualMLP.named_modules():
        if any(isinstance(module, qmodule_mx) for qmodule_mx in mx_qmodules):
            found_qmodule_mx = True
            # Check that the desired mx format was propagated to class
            assert module.mx_specs["w_elem_format"] == mx_format
            assert module.mx_specs["a_elem_format"] == mx_format

    assert found_qmodule_mx


@pytest.mark.skipif(
    not available_packages["mx"],
    reason="Skipping mx_specs error test; No package found",
)
def test_mx_specs_after_qconfig_init(
    model_residualMLP: torch.nn.Module,
    input_residualMLP: torch.FloatTensor,
    config_fp32: dict,
):
    """
    Test if a default config w/ MX qmodes trigger setting mx_specs inside qmodel_prep

    Args:
        model_residualMLP (torch.nn.Module): Single fp32 model.
        input_residualMLP (torch.FloatTensor): Random 16x128 tensor.
        config_fp32 (dict): Config w/ fp32 settings.
    """
    config_fp32["qa_mode"] = "mx_fp8_e5m2"
    config_fp32["qw_mode"] = "mx_fp8_e5m2"

    assert "mx_specs" not in config_fp32

    qmodel_prep(model_residualMLP, input_residualMLP, config_fp32, use_dynamo=True)

    assert "mx_specs" in config_fp32
