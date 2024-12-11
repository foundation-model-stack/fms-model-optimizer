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
import time

# Third Party
import transformers
from datasets import load_from_disk
from transformers import AutoTokenizer

# First Party
from fms_mo.dq import run_dq
from fms_mo.training_args import (DataArguments, FMSMOArguments, FP8Args,
                                  GPTQArgs, ModelArguments)
from fms_mo.utils.import_utils import available_packages

logger = logging.Logger("fms_mo.main")


def quantize(
    model_args: ModelArguments,
    data_args: DataArguments,
    fms_mo_args: FMSMOArguments,
    gptq_args: GPTQArgs,
    fp8_args: FP8Args,
    quant_method: str,
    output_dir: str,
):
    """Main entry point to quantize a given model with a set of specified hyperparameters

    Args:
        model_args (fms_mo.training_args.ModelArguments): Model arguments to be used when loading
            the model
        data_args (fms_mo.training_args.DataArguments): Data arguments to be used when loading the
            tokenized dataset
        fms_mo_args (fms_mo.training_args.FMSMOArguments): Parameters to use for PTQ quantization
        gptq_args (fms_mo.training_args.GPTQArgs): Parameters to use for GPTQ quantization
        fp8_args (fms_mo.training_args.FP8Args): Parameters to use for FP8 quantization
        quant_method (str): Quantization technique, options are gptq and fp8
        output_dir (str) Output directory to write to
    """

    logging.info(f"{fms_mo_args}\n{quant_method}\n")
    if quant_method == "gptq":
        if not available_packages["auto_gptq"]:
            raise ImportError(
                "Quantization method has been selected as gptq but unable to use external library, "
                "auto_gptq module not found. For more instructions on installing the appropriate "
                "package, see https://github.com/AutoGPTQ/AutoGPTQ?tab=readme-ov-file#installation"
            )
        run_gptq(model_args, data_args, gptq_args, output_dir)
    elif quant_method == "fp8":
        if not available_packages["llmcompressor"]:
            raise ImportError(
                "Quantization method has been selected as fp8 but unable to use external library, "
                "llmcompressor module not found. \n"
                "For more instructions on installing the appropriate package, see "
                "https://github.com/vllm-project/llm-compressor/tree/"
                "main?tab=readme-ov-file#installation"
            )
        run_fp8(model_args, data_args, fp8_args, output_dir)
    elif quant_method == "dq":
        run_dq(model_args, data_args, fms_mo_args, output_dir)
    else:
        raise ValueError(
            "Not a valid quantization technique option. Please choose from: gptq, fp8, dq"
        )


def run_gptq(model_args, data_args, gptq_args, output_dir):
    """GPTQ quantizes a given model with a set of specified hyperparameters

    Args:
        model_args (fms_mo.training_args.ModelArguments): Model arguments to be used when loading
            the model
        data_args (fms_mo.training_args.DataArguments): Data arguments to be used when loading the
            tokenized dataset
        gptq_args (fms_mo.training_args.GPTQArgs): Parameters to use for GPTQ quantization
        output_dir (str) Output directory to write to
    """

    # Third Party
    from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
    from auto_gptq.modeling._const import SUPPORTED_MODELS
    from auto_gptq.modeling.auto import GPTQ_CAUSAL_LM_MODEL_MAP

    # First Party
    from fms_mo.utils.custom_gptq_models import custom_gptq_classes

    quantize_config = BaseQuantizeConfig(
        bits=gptq_args.bits,
        group_size=gptq_args.group_size,
        desc_act=gptq_args.desc_act,
        damp_percent=gptq_args.damp_percent,
    )

    # Add custom model_type mapping to auto_gptq LUT so AutoGPTQForCausalLM can recognize them.
    for mtype, cls in custom_gptq_classes.items():
        SUPPORTED_MODELS.append(mtype)
        GPTQ_CAUSAL_LM_MODEL_MAP[mtype] = cls

    model = AutoGPTQForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        quantize_config=quantize_config,
        torch_dtype=model_args.torch_dtype,
    )

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
        use_triton=gptq_args.use_triton,
        batch_size=gptq_args.batch_size,
        cache_examples_on_gpu=gptq_args.cache_examples_on_gpu,
    )

    logger.info(f"Time to quantize model at {output_dir}: {time.time() - start_time}")

    logger.info(f"Saving quantized model and tokenizer to {output_dir}")
    model.save_quantized(output_dir, use_safetensors=True)
    tokenizer.save_pretrained(output_dir)


def run_fp8(model_args, data_args, fp8_args, output_dir):
    """FP8 quantizes a given model with a set of specified hyperparameters

    Args:
        model_args (fms_mo.training_args.ModelArguments): Model arguments to be used when loading
            the model
        data_args (fms_mo.training_args.DataArguments): Data arguments to be used when loading the
            tokenized dataset
        fp8_args (fms_mo.training_args.FP8Args): Parameters to use for FP8 quantization
        output_dir (str) Output directory to write to
    """

    # Third Party
    from llmcompressor.modifiers.quantization import QuantizationModifier
    from llmcompressor.transformers import SparseAutoModelForCausalLM, oneshot

    model = SparseAutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path, torch_dtype=model_args.torch_dtype
    )
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
    logger.info(f"Time to quantize model at {output_dir}: {time.time() - start_time}")

    logger.info(f"Saving quantized model and tokenizer to {output_dir}")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


def main():
    """Main entry point for quantize API for GPTQ, FP8 and DQ quantization techniques"""

    parser = transformers.HfArgumentParser(
        dataclass_types=(
            ModelArguments,
            DataArguments,
            FMSMOArguments,
            GPTQArgs,
            FP8Args,
        )
    )

    parser.add_argument(
        "--quant_method",
        type=str.lower,
        choices=["gptq", "fp8", None, "none", "dq"],
        default="none",
    )

    parser.add_argument("--output_dir", type=str)

    (
        model_args,
        data_args,
        fms_mo_args,
        gptq_args,
        fp8_args,
        additional,
        _,
    ) = parser.parse_args_into_dataclasses(return_remaining_strings=True)
    quant_method = additional.quant_method
    output_dir = additional.output_dir

    logger.debug(
        "Input args parsed: \nmodel_args %s, data_args %s, fms_mo_args %s, "
        "gptq_args %s, fp8_args %s, quant_method %s, output_dir %s",
        model_args,
        data_args,
        fms_mo_args,
        gptq_args,
        fp8_args,
        quant_method,
        output_dir,
    )

    quantize(
        model_args=model_args,
        data_args=data_args,
        fms_mo_args=fms_mo_args,
        gptq_args=gptq_args,
        fp8_args=fp8_args,
        quant_method=quant_method,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    main()
