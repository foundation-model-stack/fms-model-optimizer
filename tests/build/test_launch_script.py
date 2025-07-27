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

"""Unit Tests for accelerate_launch script."""

# Standard
import os
import tempfile
import glob

# Third Party
import pytest
import torch

# First Party
from build.accelerate_launch import main
from build.utils import serialize_args
from tests.artifacts.testdata import WIKITEXT_TOKENIZED_DATA_JSON, MODEL_NAME
from fms_mo.utils.error_logging import (
    USER_ERROR_EXIT_CODE,
    INTERNAL_ERROR_EXIT_CODE,
)
from fms_mo.utils.import_utils import available_packages


SCRIPT = os.path.join(os.path.dirname(__file__), "../..", "fms_mo/run_quant.py")
BASE_KWARGS = {
    "accelerate_launch_args": {"num_processes": 1},
    "model_name_or_path": MODEL_NAME,
}
BASE_GPTQ_KWARGS = {
    **BASE_KWARGS,
    **{
        "quant_method": "gptq",
        "bits": 4,
        "group_size": 64,
        "training_data_path": WIKITEXT_TOKENIZED_DATA_JSON,
        "device": "cuda",
    },
}
BASE_FP8_KWARGS = {
    **BASE_KWARGS,
    **{
        "quant_method": "fp8",
    },
}
BASE_DQ_KWARGS = {
    **BASE_KWARGS,
    **{
        "quant_method": "dq",
        "nbits_w": 8,
        "nbits_a": 8,
        "nbits_kvcache": 32,
        "qa_mode": "fp8_e4m3_scale",
        "qw_mode": "fp8_e4m3_scale",
        "qmodel_calibration_new": 0,
        "training_data_path": WIKITEXT_TOKENIZED_DATA_JSON,
    },
}


def setup_env(tempdir):
    """Setting up env var"""
    os.environ["OPTIMIZER_SCRIPT"] = SCRIPT
    os.environ["PYTHONPATH"] = "./:$PYTHONPATH"
    os.environ["TERMINATION_LOG_FILE"] = tempdir + "/termination-log"
    os.environ["SET_NUM_PROCESSES_TO_NUM_GPUS"] = "False"


def cleanup_env():
    """Unsetting env var that were previously set for each test"""
    os.environ.pop("OPTIMIZER_SCRIPT", None)
    os.environ.pop("PYTHONPATH", None)
    os.environ.pop("TERMINATION_LOG_FILE", None)


@pytest.mark.skipif(
    not available_packages["gptqmodel"],
    reason="Only runs if gptqmodel package is installed",
)
def test_successful_gptq():
    """Check if we can gptq models"""
    with tempfile.TemporaryDirectory() as tempdir:
        setup_env(tempdir)
        GPTQ_KWARGS = {**BASE_GPTQ_KWARGS, **{"output_dir": tempdir}}
        serialized_args = serialize_args(GPTQ_KWARGS)
        os.environ["FMS_MO_CONFIG_JSON_ENV_VAR"] = serialized_args

        assert main() == 0

        _validate_termination_files_when_quantization_succeeds(tempdir)
        _validate_quantization_output(tempdir, "gptq")


@pytest.mark.skipif(
    not available_packages["llmcompressor"],
    reason="Only runs if llm-compressor package is installed",
)
def test_successful_fp8():
    """Check if we can fp8 quantize models"""
    with tempfile.TemporaryDirectory() as tempdir:
        setup_env(tempdir)
        FP8_KWARGS = {**BASE_FP8_KWARGS, **{"output_dir": tempdir}}
        serialized_args = serialize_args(FP8_KWARGS)
        os.environ["FMS_MO_CONFIG_JSON_ENV_VAR"] = serialized_args

        assert main() == 0

        _validate_termination_files_when_quantization_succeeds(tempdir)
        _validate_quantization_output(tempdir, "fp8")


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Only runs if GPUs are available"
)
def test_successful_dq():
    """Check if we can dq models"""
    with tempfile.TemporaryDirectory() as tempdir:
        setup_env(tempdir)
        DQ_KWARGS = {**BASE_DQ_KWARGS, **{"output_dir": tempdir}}
        serialized_args = serialize_args(DQ_KWARGS)
        os.environ["FMS_MO_CONFIG_JSON_ENV_VAR"] = serialized_args

        assert main() == 0

        _validate_termination_files_when_quantization_succeeds(tempdir)
        _validate_quantization_output(tempdir, "dq")


def test_bad_script_path():
    """Check for appropriate error for an invalid optimization script location"""
    with tempfile.TemporaryDirectory() as tempdir:
        setup_env(tempdir)
        QUANT_KWARGS = {**BASE_KWARGS, **{"output_dir": tempdir}}
        serialized_args = serialize_args(QUANT_KWARGS)
        os.environ["FMS_MO_CONFIG_JSON_ENV_VAR"] = serialized_args
        os.environ["OPTIMIZER_SCRIPT"] = "/not/here"

        with pytest.raises(SystemExit) as pytest_wrapped_e:
            main()
        assert pytest_wrapped_e.type is SystemExit
        assert pytest_wrapped_e.value.code == INTERNAL_ERROR_EXIT_CODE
        assert os.stat(tempdir + "/termination-log").st_size > 0


def test_blank_config_json_env_var():
    """Check for appropriate error when the json job env var is empty"""
    with tempfile.TemporaryDirectory() as tempdir:
        setup_env(tempdir)
        os.environ["FMS_MO_CONFIG_JSON_ENV_VAR"] = ""
        with pytest.raises(SystemExit) as pytest_wrapped_e:
            main()
        assert pytest_wrapped_e.type is SystemExit
        assert pytest_wrapped_e.value.code == USER_ERROR_EXIT_CODE
        assert os.stat(tempdir + "/termination-log").st_size > 0


def test_blank_config_json_path():
    """Check for appropriate error when the json job config file is empty"""
    with tempfile.TemporaryDirectory() as tempdir:
        setup_env(tempdir)
        os.environ["FMS_MO_CONFIG_JSON_PATH"] = ""
        with pytest.raises(SystemExit) as pytest_wrapped_e:
            main()
        assert pytest_wrapped_e.type is SystemExit
        assert pytest_wrapped_e.value.code == USER_ERROR_EXIT_CODE
        assert os.stat(tempdir + "/termination-log").st_size > 0


def test_faulty_file_path():
    """Check for appropriate error when invalid training data path is provided"""
    with tempfile.TemporaryDirectory() as tempdir:
        setup_env(tempdir)
        faulty_path = os.path.join(tempdir, "non_existent_file.pkl")
        QUANT_KWARGS = {
            **BASE_KWARGS,
            **{"training_data_path": faulty_path, "output_dir": tempdir},
        }
        serialized_args = serialize_args(QUANT_KWARGS)
        os.environ["FMS_MO_CONFIG_JSON_ENV_VAR"] = serialized_args
        with pytest.raises(SystemExit) as pytest_wrapped_e:
            main()
        assert pytest_wrapped_e.type is SystemExit
        assert pytest_wrapped_e.value.code == USER_ERROR_EXIT_CODE
        assert os.stat(tempdir + "/termination-log").st_size > 0


def test_bad_base_model_path():
    """Check for appropriate error when invalid model name/path is provided"""
    with tempfile.TemporaryDirectory() as tempdir:
        setup_env(tempdir)
        DQ_KWARGS = {
            **BASE_DQ_KWARGS,
            **{"model_name_or_path": "/wrong/path", "output_dir": tempdir},
        }
        serialized_args = serialize_args(DQ_KWARGS)
        os.environ["FMS_MO_CONFIG_JSON_ENV_VAR"] = serialized_args
        with pytest.raises(SystemExit) as pytest_wrapped_e:
            main()
        assert pytest_wrapped_e.type is SystemExit
        assert pytest_wrapped_e.value.code == USER_ERROR_EXIT_CODE
        assert os.stat(tempdir + "/termination-log").st_size > 0


def test_config_parsing_error():
    """Check for appropriate error when the json job config cannot be parsed successfully"""
    with tempfile.TemporaryDirectory() as tempdir:
        setup_env(tempdir)
        DQ_KWARGS = {
            **BASE_DQ_KWARGS,
            **{"nbits_w": "eight", "output_dir": tempdir},
        }  # Intentional type error
        serialized_args = serialize_args(DQ_KWARGS)
        os.environ["FMS_MO_CONFIG_JSON_ENV_VAR"] = serialized_args
        with pytest.raises(SystemExit) as pytest_wrapped_e:
            main()
        assert pytest_wrapped_e.type is SystemExit
        assert pytest_wrapped_e.value.code == USER_ERROR_EXIT_CODE
        assert os.stat(tempdir + "/termination-log").st_size > 0

        with open(tempdir + "/termination-log", "r", encoding="utf-8") as f:
            contents = f.read()
        assert (
            contents
            == "Exception raised during optimization. This may be a problem with your input: The field `nbits_w` was assigned by `<class 'str'>` instead of `<class 'int'>`"  # pylint: disable=line-too-long
        )


def _validate_termination_files_when_quantization_succeeds(base_dir):
    """Check whether the termination log and .complete files exists"""
    assert os.path.exists(os.path.join(base_dir, "/termination-log")) is False
    assert os.path.exists(os.path.join(base_dir, ".complete")) is True
    # assert os.path.exists(os.path.join(base_dir, training_logs_filename)) is True


def _validate_quantization_output(base_dir, quant_method):
    """Check whether the tokenizer and quantized model artifacts exists"""
    # Check tokenizer files exist
    assert os.path.exists(os.path.join(base_dir, "tokenizer.json")) is True
    assert os.path.exists(os.path.join(base_dir, "special_tokens_map.json")) is True
    assert os.path.exists(os.path.join(base_dir, "tokenizer_config.json")) is True
    # assert os.path.exists(os.path.join(base_dir, "tokenizer.model")) is True

    # Check quantized model files exist
    if quant_method == "gptq":
        assert len(glob.glob(os.path.join(base_dir, "model*.safetensors"))) > 0
        assert os.path.exists(os.path.join(base_dir, "quantize_config.json")) is True
        assert os.path.exists(os.path.join(base_dir, "config.json")) is True

    elif quant_method == "fp8":
        assert len(glob.glob(os.path.join(base_dir, "model*.safetensors"))) > 0
        assert os.path.exists(os.path.join(base_dir, "generation_config.json")) is True
        assert os.path.exists(os.path.join(base_dir, "config.json")) is True
        assert os.path.exists(os.path.join(base_dir, "recipe.yaml")) is True

    elif quant_method == "dq":
        assert len(glob.glob(os.path.join(base_dir, "model*.safetensors"))) > 0
        assert os.path.exists(os.path.join(base_dir, "generation_config.json")) is True
        assert os.path.exists(os.path.join(base_dir, "config.json")) is True


def test_cleanup():
    """Runs to unset env variables that could disrupt other tests"""
    cleanup_env()
    assert True
