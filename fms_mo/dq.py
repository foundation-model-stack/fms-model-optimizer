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
from fms_mo.custom_ext_kernels.utils import (
    lower_qmodel_triton,  # pylint: disable=unused-import
)
from fms_mo.fx.utils import model_size_Wb
from fms_mo.quant.ptq import (
    calibration_llm_1GPU_v2,
    dq_llm,
    get_act_scales,
    get_act_scales_1gpu,
)
from fms_mo.utils.aiu_utils import save_for_aiu
from fms_mo.utils.dq_utils import config_quantize_smooth_layers
from fms_mo.utils.eval_utils import Evaluator, eval_llm_1GPU
from fms_mo.utils.utils import patch_torch_bmm, prepare_input

logger = logging.getLogger(__name__)


def run_dq(model_args, data_args, opt_args, fms_mo_args):
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
        opt_args (fms_mo.training_args.OptArguments): Generic optimization arguments to be used
            during DQ
        fms_mo_args (fms_mo.training_args.FMSMOArguments): Parameters to use for DQ quantization

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
        "torchscript": True,
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
        or not isinstance(model_args.torch_dtype, str)
        else getattr(torch, model_args.torch_dtype)
    )
    # NOTE for models that cannot fit in 1 GPU, keep it on CPU and use block-wise calibration.
    # or leverage HF's device_map="auto", BUT tracing will not work properly with "auto"
    total_gpu_memory = 1e-5
    if torch.cuda.is_available():
        total_gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9

    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision="main",
        use_auth_token=True if model_args.use_auth_token else None,
        torch_dtype=torch_dtype,
        device_map=model_args.device_map,
        low_cpu_mem_usage=bool(model_args.device_map),
    )

    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))

    logger.info(f"Initialized model is: \n {model}")
    logger.info(f"Model is at {model.device} after intialization")
    logger.info(f"Tokenizer is {tokenizer}, block size is {block_size}")
    qcfg = qconfig_init(recipe="dq", args=fms_mo_args)

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
    dev = "cpu" if qcfg["large_model"] else "cuda"
    if model_args.device_map is None:
        model.to(dev)

    if hasattr(model.config, "model_type"):
        qcfg["model_type"] = model.config.model_type

    qcfg["model"] = model_args.model_name_or_path
    # config layers to skip, smooth scale
    config_quantize_smooth_layers(qcfg)

    use_dynamo = True
    # use dynamo as default unless really needed, False -> fallback to TorchScript tracing
    if any(x != 32 for x in attn_bits):
        logger.info("Quantize attention bmms or kvcache, will use dynamo for prep")
        use_layer_name_pattern_matching = False
        qcfg["qlayer_name_pattern"] = []
        assert (
            qcfg["qlayer_name_pattern"] == []
        ), "ensure nothing in qlayer_name_pattern when use dynamo"
    else:
        logger.info("Attention bmms will not be quantized.")
        use_layer_name_pattern_matching = True

    qcfg["seq_len"] = block_size
    qcfg["model"] = model_args.model_name_or_path
    qcfg["smoothq"] = qcfg.get("smoothq_alpha", -1) >= 0 and "mx_specs" not in qcfg
    qcfg["plotsvg"] = False

    calibration_dataset = load_from_disk(data_args.training_data_path)
    calibration_dataset = calibration_dataset.with_format("torch")

    dq_dataloader = DataLoader(
        calibration_dataset,
        shuffle=True,
        collate_fn=default_data_collator,
        batch_size=1,
    )

    # For loading or creating smoothquant scale. Sometimes we may include scales in ckpt as well.
    if qcfg["smoothq"]:
        scale_file = Path(f"./act_scales/{qcfg['model'].replace('/', '-')}.pt")
        if qcfg.get("act_scale_path", None):
            # user provided a scale file (or a dir)
            scale_file_or_dir = Path(qcfg["act_scale_path"])
            if scale_file_or_dir.is_dir():
                scale_file = scale_file_or_dir / f"{qcfg['model'].replace('/', '-')}.pt"
            elif scale_file_or_dir.is_file():
                scale_file = scale_file_or_dir

        if not scale_file.parent.exists():
            scale_file.parent.mkdir(exist_ok=False)

        if scale_file.exists():
            act_scales = torch.load(
                scale_file, map_location=getattr(model, "device", dev)
            )

        else:
            logger.info("Generate activation scales")
            if qcfg["large_model"]:
                act_scales = get_act_scales_1gpu(model, dq_dataloader, qcfg)
            else:
                act_scales = get_act_scales(model, dq_dataloader, qcfg)
            torch.save(act_scales, scale_file)

    if fms_mo_args.aiu_sim_triton != "fp8":
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
        logger.info("==" * 20)

    if qcfg["smoothq"]:
        logger.info("Starting to apply smooth scale")
        dq_llm(model, act_scales, qcfg)
        logger.info("Finished applying smooth scale")

    if qcfg["qmodel_calibration_new"] > 0:
        logger.info("Starting to calibrate activation clip_val")
        if qcfg["large_model"]:
            calibration_llm_1GPU_v2(qcfg, model, dq_dataloader)
        else:
            model.to("cuda")
            pbar = tqdm(
                dq_dataloader,
                desc=" calibration after applying smoothq scale and before inference",
                total=qcfg["qmodel_calibration_new"],
            )
            for data_mb, _ in zip(pbar, range(qcfg["qmodel_calibration_new"])):
                data_mb = prepare_input(model.device, data_mb)
                with patch_torch_bmm(qcfg):
                    model(**data_mb)

    if opt_args.save_ckpt_for_aiu:
        logger.info(
            f"Saving model processed for AIU and tokenizer to {opt_args.output_dir}"
        )
        save_for_aiu(model, qcfg, output_dir=opt_args.output_dir, verbose=True)
    elif opt_args.save_ckpt:
        logger.info(f"Saving quantized model and tokenizer to {opt_args.output_dir}")
        model.save_pretrained(opt_args.output_dir, use_safetensors=True)
        tokenizer.save_pretrained(opt_args.output_dir)

    if fms_mo_args.aiu_sim_triton:
        # NOTE plz apply correct HW settings here, defaults are not real HW params
        lower_qmodel_triton(
            model,
            use_dyn_max_act=-1 if qcfg["qa_mode"] == "pertokenmax" else False,
            max_acc_bits=qcfg.get("max_acc_bits", 32),
            num_lsb_to_truncate=qcfg.get("lsb_trun_bits", 0),
            chunk_size=qcfg.get("chunk_size", 32),  # 1024
            clamp_acc_to_dl16=fms_mo_args.aiu_sim_triton == "fp8",
            # layer_to_exclude=["lm_head",]
        )

    if fms_mo_args.eval_ppl:
        path_test = Path(data_args.test_data_path)
        arrow_files = list(path_test.glob("*.arrow"))
        pt_files = list(path_test.glob("*.pt"))
        if len(arrow_files) > 0:
            test_dataset = load_from_disk(data_args.test_data_path)
            test_dataset = test_dataset.with_format("torch")
        elif len(pt_files) > 0:
            test_dataset = torch.load(pt_files[0], weights_only=False)

        logger.info(f"Model for evaluation: {model}")
        if qcfg["large_model"]:
            eval_llm_1GPU(qcfg, model, test_dataset)
        else:
            model.to(torch.device("cuda:0"))
            n_samples = int(test_dataset.input_ids.shape[1] / block_size)
            evaluator = Evaluator(test_dataset, "cuda", n_samples=n_samples)
            with patch_torch_bmm(qcfg):
                ppl = evaluator.evaluate(model, block_size=block_size)
            logger.info(f"Model perplexity: {ppl}")
        logger.info("-" * 50)
        logger.info("Finished evaluation")
