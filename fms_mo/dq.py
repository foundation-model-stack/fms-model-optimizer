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

# pylint: disable=arguments-renamed

"""
Script for direct quantization
"""

# Standard
from pathlib import Path
import logging
import os

# Third Party
from datasets import load_from_disk
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    default_data_collator,
)
import torch

# Local
from fms_mo import qconfig_init, qmodel_prep
from fms_mo.fx.utils import model_size_Wb
from fms_mo.quant.ptq import (
    calibration_llm_1GPU,
    dq_llm,
    get_act_scales,
    get_act_scales_1gpu,
)
from fms_mo.utils.dq_utils import config_quantize_smooth_layers
from fms_mo.utils.eval_utils import Evaluator, eval_llm_1GPU
from fms_mo.utils.utils import patch_torch_bmm, prepare_input

logger = logging.getLogger(__name__)


def run_dq(model_args, data_args, fms_mo_args, output_dir):
    """
    For direct quantization LLMs without optimization:
    Models are directly quantized into INT8 or FP8 precisions using
    static or dynamic quantization, type casting, and SmoothQuant techniques.
    Supporting quantizing both linear layers and bmm operations in attention, such as KV-Cache.

    Args:
        model_args (fms_mo.training_args.ModelArguments): Model arguments to be used when loading
            the model
        data_args (fms_mo.training_args.DataArguments): Data arguments to be used when loading the
            tokenized dataset
        fms_mo_args (fms_mo.training_args.FMSMOArguments): Parameters to use for DQ quantization
        output_dir (str) Output directory to write to
    NOTE:
        use dynamo tracing instead of torchscript by default. if torchscript is needed, change
        1) config_kwarks and 2) use_dynamo in qmodel_prep()
    """
    # for attention or kv-cache quantization, need to use eager attention
    attn_bits = [
        fms_mo_args.nbits_bmm1,
        fms_mo_args.nbits_bmm2,
        fms_mo_args.nbits_kvcache,
    ]
    if any(x != 32 for x in attn_bits):
        attn_implementation = "eager"
    else:
        attn_implementation = None
    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
        "torchscript": False,
        "attn_implementation": attn_implementation,
    }
    config = AutoConfig.from_pretrained(model_args.model_name_or_path, **config_kwargs)
    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,
        "use_fast": model_args.use_fast_tokenizer,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path, **tokenizer_kwargs
    )
    block_size = min(fms_mo_args.block_size, tokenizer.model_max_length)
    torch_dtype = (
        model_args.torch_dtype
        if model_args.torch_dtype in ["auto", None]
        else getattr(torch, model_args.torch_dtype)
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision="main",
        use_auth_token=True if model_args.use_auth_token else None,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=False,
    )

    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))

    logger.info(f"Initialized model is: \n {model}")
    logger.info(f"Model is at {model.device} after intialization")
    logger.info(f"Tokenizer is {tokenizer}, block size is {block_size}")
    qcfg = qconfig_init(recipe="dq", args=fms_mo_args)
    # for models that cannot fit in 1 GPU, keep it in CPU and use block-wise calibration.
    total_gpu_memory = 0.0
    if torch.cuda.is_available():
        total_gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
    model_size = model_size_Wb(model, unit="GB")
    gpu_mem_util_per = model_size / total_gpu_memory

    known_large_models = [
        "Llama-2-70b",
        "Mixtral-8x7B",
        "Llama-3-70B",
        "405B-Instruct",
        "Mistral-Large",
        "Nemotron",
    ]
    qcfg["large_model"] = any(
        name in model_args.model_name_or_path for name in known_large_models
    ) or (gpu_mem_util_per > 0.7)
    dev = "cpu" if qcfg["large_model"] else "cuda:0"

    if hasattr(model.config, "model_type"):
        qcfg["model_type"] = model.config.model_type

    qcfg["model"] = model_args.model_name_or_path
    # config layers to skip, smooth scale
    config_quantize_smooth_layers(qcfg)

    if any(x != 32 for x in attn_bits):
        logger.info("Quantize attention bmms or kvcache, use dynamo for prep")
        use_layer_name_pattern_matching = False
        qcfg["qlayer_name_pattern"] = []
        assert (
            qcfg["qlayer_name_pattern"] == []
        ), "ensure nothing in qlayer_name_pattern when use dynamo"
        use_dynamo = True
    else:
        logger.info("Do not quantize attention bmms")
        use_layer_name_pattern_matching = True
        use_dynamo = False

    qcfg["seq_len"] = block_size
    qcfg["model"] = model_args.model_name_or_path
    qcfg["smoothq"] = True
    qcfg["plotsvg"] = False

    calibration_dataset = load_from_disk(data_args.training_data_path)
    calibration_dataset = calibration_dataset.with_format("torch")

    dq_dataloader = DataLoader(
        calibration_dataset,
        shuffle=True,
        collate_fn=default_data_collator,
        batch_size=1,
    )

    # For loading or creating smoothquant scale.
    act_scale_directory = "./act_scales"
    if not os.path.exists(act_scale_directory):
        os.makedirs(act_scale_directory)

    if qcfg["act_scale_path"] is not None:
        act_scales = torch.load(qcfg["act_scale_path"], map_location="cpu")
    else:
        logger.info("Generate activation scales")
        if qcfg["large_model"]:
            act_scales = get_act_scales_1gpu(model, dq_dataloader, qcfg)
        else:
            if gpu_mem_util_per < 0.7:
                model.to(dev)

            act_scales = get_act_scales(model, dq_dataloader, qcfg)
        scale_file = f"{act_scale_directory}/{qcfg['model'].replace('/', '-')}" + ".pt"
        torch.save(act_scales, scale_file)

    qmodel_prep(
        model,
        dq_dataloader,
        qcfg,
        use_layer_name_pattern_matching=use_layer_name_pattern_matching,
        use_dynamo=use_dynamo,
        dev=dev,
        save_fname="dq",
    )
    logger.info(f"Quantized model {model}")
    logger.info("Starting to apply smooth scale")
    dq_llm(model, act_scales, qcfg)
    logger.info("Finished applying smooth scale")
    logger.info("==" * 20)
    if qcfg["qmodel_calibration_new"] > 0:
        logger.info("Starting to calibrate activation clip_val")
        if qcfg["large_model"]:
            calibration_llm_1GPU(qcfg, model, dq_dataloader)
        else:
            model.to("cuda:0")
            pbar = tqdm(
                dq_dataloader,
                desc=" calibration after applying smoothq scale and before inference",
                total=qcfg["qmodel_calibration_new"],
            )
            for data_mb, _ in zip(pbar, range(qcfg["qmodel_calibration_new"])):
                data_mb = prepare_input(model.device, data_mb)
                with patch_torch_bmm(qcfg):
                    model(**data_mb)

    logger.info(f"Saving quantized model and tokenizer to {output_dir}")
    model.save_pretrained(output_dir, use_safetensors=True)
    tokenizer.save_pretrained(output_dir)

    if fms_mo_args.eval_ppl:
        path_test = Path(data_args.test_data_path)
        arrow_files = list(path_test.glob("*.arrow"))
        pt_files = list(path_test.glob("*.pt"))
        if len(arrow_files) > 0:
            test_dataset = load_from_disk(data_args.test_data_path)
            test_dataset = test_dataset.with_format("torch")
        elif len(pt_files) > 0:
            test_dataset = torch.load(pt_files[0])

        logger.info(f"Model for evaluation: {model}")
        if qcfg["large_model"]:
            eval_llm_1GPU(qcfg, model, test_dataset)
        else:
            model.to(torch.device("cuda:0"))
            n_samples = int(test_dataset.input_ids.shape[1] / block_size)
            evaluator = Evaluator(test_dataset, "cuda", n_samples=n_samples)
            ppl = evaluator.evaluate(model, block_size=block_size)
            logger.info(f"Model perplexity: {ppl}")
        logger.info("-" * 50)
        logger.info("Finished evaluation")
