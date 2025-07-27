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

# This file is modified based on https://huggingface.co/TheBloke/Llama-2-70B-GPTQ/discussions/11
# and https://gist.github.com/TheBloke/b47c50a70dd4fe653f64a12928286682#file-quant_autogptq-py
# usage:
#   python -m fms_mo.run_quant --model_name_or_path Maykeye/TinyLLama-v0 \
#       --output_dir TinyLLama-v0-gptq --quant_method gptq --training_data_path data
#
#   python -m fms_mo.run_quant --model_name_or_path Maykeye/TinyLLama-v0 \
#       --output_dir TinyLLama-v0-fp8 --quant_method fp8 --training_data_path open_platypus

"""
Main entry point for quantize API for GPTQ, FP8 and DQ quantization techniques
"""

# Standard
import logging
import os
import sys
import time
import traceback

# Third Party
from datasets import load_from_disk
from torch.cuda import OutOfMemoryError
from transformers import (
    AutoModelForMaskedLM,
    AutoModelForQuestionAnswering,
    AutoTokenizer,
)
import torch
import transformers

# Local
from fms_mo.dq import run_dq
from fms_mo.training_args import (
    DataArguments,
    FMSMOArguments,
    FP8Arguments,
    GPTQArguments,
    ModelArguments,
    OptArguments,
)
from fms_mo.utils.config_utils import get_json_config
from fms_mo.utils.error_logging import (
    INTERNAL_ERROR_EXIT_CODE,
    USER_ERROR_EXIT_CODE,
    write_termination_log,
)
from fms_mo.utils.import_utils import available_packages
from fms_mo.utils.logging_utils import set_log_level


def quantize(
    model_args: ModelArguments,
    data_args: DataArguments,
    opt_args: OptArguments,
    fms_mo_args: FMSMOArguments = None,
    gptq_args: GPTQArguments = None,
    fp8_args: FP8Arguments = None,
):
    """Main entry point to quantize a given model with a set of specified hyperparameters

    Args:
        model_args (fms_mo.training_args.ModelArguments): Model arguments to be used when loading
            the model
        data_args (fms_mo.training_args.DataArguments): Data arguments to be used when loading the
            tokenized dataset
        opt_args (fms_mo.training_args.OptArguments): Generic optimization related arguments
        fms_mo_args (fms_mo.training_args.FMSMOArguments): Parameters to use for PTQ quantization
        gptq_args (fms_mo.training_args.GPTQArguments): Parameters to use for GPTQ quantization
        fp8_args (fms_mo.training_args.FP8Arguments): Parameters to use for FP8 quantization
    """

    logger = set_log_level(opt_args.log_level, "fms_mo.quantize")

    logger.info(f"{fms_mo_args}\n{opt_args.quant_method}\n")

    if opt_args.quant_method in ["gptq", "gptqv2"]:
        if not available_packages["gptqmodel"]:
            raise ImportError(
                "Quantization method has been selected as gptq but unable to use external library, "
                "gptqmodel module not found. For more instructions on installing the appropriate "
                "package, see https://github.com/ModelCloud/GPTQModel"
            )
        gptq_args.use_triton = gptq_args.use_triton and available_packages["triton"]
        run_gptq(model_args, data_args, opt_args, gptq_args)
    elif opt_args.quant_method == "fp8":
        if not available_packages["llmcompressor"]:
            raise ImportError(
                "Quantization method has been selected as fp8 but unable to use external library, "
                "llmcompressor module not found. \n"
                "For more instructions on installing the appropriate package, see "
                "https://github.com/vllm-project/llm-compressor"
                "main?tab=readme-ov-file#installation"
            )
        run_fp8(model_args, data_args, opt_args, fp8_args)
    elif opt_args.quant_method == "dq":
        run_dq(model_args, data_args, opt_args, fms_mo_args)
    else:
        raise ValueError(
            f"{opt_args.quant_method} is not a valid quantization technique option. \
            Please choose from: gptq, fp8, dq"
        )


def run_gptq(model_args, data_args, opt_args, gptq_args):
    """GPTQ quantizes a given model with a set of specified hyperparameters

    Args:
        model_args (fms_mo.training_args.ModelArguments): Model arguments to be used when loading
            the model
        data_args (fms_mo.training_args.DataArguments): Data arguments to be used when loading the
            tokenized dataset
        opt_args (fms_mo.training_args.OptArguments): Generic optimization related arguments
        gptq_args (fms_mo.training_args.GPTQArguments): Parameters to use for GPTQ quantization
    """

    # Third Party
    from gptqmodel import GPTQModel, QuantizeConfig
    from gptqmodel.models.auto import MODEL_MAP, SUPPORTED_MODELS
    from gptqmodel.utils.backend import BACKEND

    # Local
    from fms_mo.utils.custom_gptq_models import custom_gptq_classes

    logger = set_log_level(opt_args.log_level, "fms_mo.run_gptq")

    if opt_args.quant_method == "gptq":
        quantize_config = QuantizeConfig(
            bits=gptq_args.bits,
            group_size=gptq_args.group_size,
            desc_act=gptq_args.desc_act,
            damp_percent=gptq_args.damp_percent,
        )
    else:
        quantize_config = QuantizeConfig(
            bits=gptq_args.bits,
            group_size=gptq_args.group_size,
            desc_act=gptq_args.desc_act,
            damp_percent=gptq_args.damp_percent,
            v2=True,
            v2_memory_device="cpu",
        )

    # Add custom model_type mapping to gptqmodel LUT so GPTQModel can recognize them.
    for mtype, cls in custom_gptq_classes.items():
        if mtype in MODEL_MAP:
            logger.info(
                f"Custom GPTQ model type {mtype} already exists in default MODEL_MAP.\n"
                "The mapping for this type is overwritten by user defined type now."
                "Please make sure this is the behavior you'd like to have."
            )
        SUPPORTED_MODELS.append(mtype)
        MODEL_MAP[mtype] = cls

    model = GPTQModel.from_pretrained(
        model_args.model_name_or_path,
        quantize_config=quantize_config,
        torch_dtype=model_args.torch_dtype,
    )
    if model_args.device:
        model = model.to(model_args.device)

    logger.info(f"Loading data from {data_args.training_data_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path, use_fast=True
    )
    data = load_from_disk(data_args.training_data_path)
    data = data.with_format("torch")

    logger.info("Starting GPTQ quantization")
    start_time = time.time()
    model.quantize(
        data,
        backend=BACKEND.TRITON if gptq_args.use_triton else BACKEND.AUTO,
        batch_size=gptq_args.batch_size,
        calibration_enable_gpu_cache=gptq_args.cache_examples_on_gpu,
    )

    logger.info(
        f"Time to quantize model at {opt_args.output_dir}: {time.time() - start_time}"
    )

    logger.info(f"Saving quantized model and tokenizer to {opt_args.output_dir}")
    model.save_quantized(opt_args.output_dir)
    tokenizer.save_pretrained(opt_args.output_dir)


def run_fp8(model_args, data_args, opt_args, fp8_args):
    """FP8 quantizes a given model with a set of specified hyperparameters

    Args:
        model_args (fms_mo.training_args.ModelArguments): Model arguments to be used when loading
            the model
        data_args (fms_mo.training_args.DataArguments): Data arguments to be used when loading the
            tokenized dataset
        opt_args (fms_mo.training_args.OptArguments): Generic optimization related arguments
        fp8_args (fms_mo.training_args.FP8Arguments): Parameters to use for FP8 quantization
    """

    # Third Party
    from llmcompressor import oneshot
    from llmcompressor.modifiers.quantization import QuantizationModifier
    from llmcompressor.transformers import SparseAutoModelForCausalLM

    logger = set_log_level(opt_args.log_level, "fms_mo.run_fp8")

    if model_args.task_type == "lm":
        model = SparseAutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            torch_dtype=model_args.torch_dtype,
        )
    elif model_args.task_type == "qa":
        model = AutoModelForQuestionAnswering.from_pretrained(
            model_args.model_name_or_path,
            torch_dtype=model_args.torch_dtype,
        )
    elif model_args.task_type == "mlm":
        model = AutoModelForMaskedLM.from_pretrained(
            model_args.model_name_or_path,
            torch_dtype=model_args.torch_dtype,
        )
    else:
        raise ValueError(f"Unsupported task: {model_args.task_type}")
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)

    recipe = QuantizationModifier(
        targets=fp8_args.targets, scheme=fp8_args.scheme, ignore=fp8_args.ignore
    )

    logger.info("Starting FP8 quantization")
    start_time = time.time()
    oneshot(
        model=model,
        recipe=recipe,
        max_seq_length=data_args.max_seq_length,
        num_calibration_samples=data_args.num_calibration_samples,
    )
    logger.info(
        f"Time to quantize model at {opt_args.output_dir}: {time.time() - start_time}"
    )

    logger.info(f"Saving quantized model and tokenizer to {opt_args.output_dir}")
    model.save_pretrained(opt_args.output_dir)
    tokenizer.save_pretrained(opt_args.output_dir)


def get_parser():
    """Get the command-line argument parser."""
    parser = transformers.HfArgumentParser(
        dataclass_types=(
            ModelArguments,
            DataArguments,
            OptArguments,
            FMSMOArguments,
            GPTQArguments,
            FP8Arguments,
        )
    )
    return parser


def parse_arguments(parser, json_config=None):
    """Parses arguments provided either via command-line or JSON config.

    Args:
        parser: argparse.ArgumentParser
            Command-line argument parser.
        json_config: dict[str, Any]
            Dict of arguments to use with tuning.

    Returns:
        ModelArguments
            Arguments pertaining to which model we are going to quantize.
        DataArguments
            Arguments pertaining to what data we are going to use for optimization and evaluation.
        OptArguments
            Arguments generic to optimization.
        FMSMOArguments
            Configuration for PTQ quantization.
        GPTQArguments
            Configuration for GPTQ quantization.
        FP8Arguments
            Configuration for FP8 quantization.
    """
    if json_config:
        (
            model_args,
            data_args,
            opt_args,
            fms_mo_args,
            gptq_args,
            fp8_args,
        ) = parser.parse_dict(json_config, allow_extra_keys=True)
    else:
        (
            model_args,
            data_args,
            opt_args,
            fms_mo_args,
            gptq_args,
            fp8_args,
            _,
        ) = parser.parse_args_into_dataclasses(return_remaining_strings=True)

    model_args.torch_dtype = getattr(
        torch, model_args.torch_dtype.replace("torch.", ""), torch.bfloat16
    )

    return (
        model_args,
        data_args,
        opt_args,
        fms_mo_args,
        gptq_args,
        fp8_args,
    )


def main():
    """Main entry point for quantize API for GPTQ, FP8 and DQ quantization techniques"""

    parser = get_parser()
    logger = logging.getLogger()
    job_config = get_json_config()
    # accept arguments via command-line or JSON
    try:
        (
            model_args,
            data_args,
            opt_args,
            fms_mo_args,
            gptq_args,
            fp8_args,
        ) = parse_arguments(parser, job_config)

        logger = set_log_level(opt_args.log_level, __name__)

        logger.debug(
            f"Input args parsed: \nmodel_args {model_args}, data_args {data_args}, "
            f"opt_args {opt_args}, fms_mo_args {fms_mo_args}, gptq_args {gptq_args}, "
            f"fp8_args {fp8_args}"
        )
    except Exception as e:  # pylint: disable=broad-except
        logger.error(traceback.format_exc())
        write_termination_log(
            f"Exception raised during optimization. This may be a problem with your input: {e}"
        )
        sys.exit(USER_ERROR_EXIT_CODE)

    if opt_args.output_dir:
        os.makedirs(opt_args.output_dir, exist_ok=True)
        logger.info("Using the output directory at %s", opt_args.output_dir)
    try:
        quantize(
            model_args=model_args,
            data_args=data_args,
            opt_args=opt_args,
            fms_mo_args=fms_mo_args,
            gptq_args=gptq_args,
            fp8_args=fp8_args,
        )
    except (MemoryError, OutOfMemoryError) as e:
        logger.error(traceback.format_exc())
        write_termination_log(f"OOM error during optimization. {e}")
        sys.exit(INTERNAL_ERROR_EXIT_CODE)
    except FileNotFoundError as e:
        logger.error(traceback.format_exc())
        write_termination_log(f"Unable to load file: {e}")
        sys.exit(USER_ERROR_EXIT_CODE)
    except (TypeError, ValueError, EnvironmentError) as e:
        logger.error(traceback.format_exc())
        write_termination_log(
            f"Exception raised during optimization. This may be a problem with your input: {e}"
        )
        sys.exit(USER_ERROR_EXIT_CODE)
    except Exception as e:  # pylint: disable=broad-except
        logger.error(traceback.format_exc())
        write_termination_log(f"Unhandled exception during optimization: {e}")
        sys.exit(INTERNAL_ERROR_EXIT_CODE)


if __name__ == "__main__":
    main()
