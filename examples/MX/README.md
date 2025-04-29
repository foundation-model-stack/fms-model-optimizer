# Direct Quantization (DQ) Using `microscaling`
This is the same example as in the [DQ](../DQ_SQ/README.md) folder, except using [microscaling](https://arxiv.org/abs/2310.10537) format. 

Here, we provide an example of direct quantization. In this case, we demonstrate DQ of `llama3-8b` model into MXINT8, MXFP8, MXFP6, MXFP4 for weights, activations, and/or KV-cache. Note that `MXFP8` is a different format compared to typical PyTorch FP8s (e4m3 or e5m2), see our other [FP8 example](../FP8_QUANT/README.md).  Mainly all the `mx` format are not natively supported by Hopper yet (some will be supported by Blackwell), which means the quantization configurations and corresponding behavior are simulated, no "speed up" should be expected.

## Requirements
- [FMS Model Optimizer requirements](../../README.md#requirements)
- Microsoft `microxcaling` python package, download [here](https://github.com/microsoft/microxcaling.git).
> [!TIP]
> After git clone BEFORE installation, first update `pyproject.toml` to remove the version constraints in [dependencies]. (simply comment those three lines out.) Then use `pip install -e .` as usual.

## QuickStart

**1. Prepare Data** for calibration process by converting into its tokenized form. An example of tokenization using `LLAMA-3-8B`'s tokenizer is below.

```python
from transformers import AutoTokenizer
from fms_mo.utils.calib_data import get_tokenized_data

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B", use_fast=True)
num_samples = 128
seq_len = 2048
get_tokenized_data("wiki", num_samples, seq_len, tokenizer, path_to_save='data')
```
> [!NOTE]
> - Users should provide a tokenized data file based on their need. This is just one example to demonstrate what data format `fms_mo` is expecting.
> - Tokenized data will be saved in `<path_to_save>_train` and `<path_to_save>_test`
> - If you have trouble downloading Llama family of models from Hugging Face ([LLama models require access](https://www.llama.com/docs/getting-the-models/hugging-face/)), you can use `ibm-granite/granite-8b-code` instead

**2. Apply DQ** by providing specific hyper-parameters such as `quant_method`, weight quantizers (`qw_mode`) and activation quantizers (`qa_mode`) etc. An example using `Meta-Llama-3-8B` and the tokenized training and test data is provided below.
```bash
python  -m fms_mo.run_quant \
        --model_name_or_path "meta-llama/Meta-Llama-3-8B" \
        --training_data_path data_train \
        --test_data_path data_test \
        --torch_dtype "float16" \
        --quant_method dq \
        --nbits_w 8 \
        --nbits_a 8 \
        --nbits_kvcache 32 \
        --qa_mode "mx_fp8_e4m3"\
        --qw_mode "mx_fp8_e4m3" \
        --output_dir "dq_test" \
        --eval_ppl
```
> [!TIP]
> To use MX format, simply assign `qa_mode` and `qw_mode` argument with a `mx_<dtype supported by mx package>`, e.g. `mx_fp8_e4m3` as in the above example. Corresponding `QLinearMX` wrappers will be used in place of `QLinear` as in other examples.

**3. Compare the Perplexity score** For user convenience, the code will print out perplexity (controlled by `eval_ppl` flag) at the end of the run, so no additional steps needed (if the logging level is set to `INFO` in terminal). You can check output in the logging file. `./fms_mo.log`.

# *TO BE UPDATED BELOW THIS LINE*


## Example Test Results
The perplexity of the INT8 and FP8 quantized models on the `wikitext` dataset is shown below:

| Model     |Type |QA            |QW            |DQ  |SQ  |Perplexity|
|:---------:|:---:|:------------:|:------------:|:--:|:--:|:--------:|
|`Llama3-8b`|INT8 |maxpertoken   |maxperCh      |yes |yes |6.21      |
|           |FP8  |fp8_e4m3_scale|fp8_e4m3_scale|yes |yes |6.19      |

## Code Walk-through

**1. KV caching**

In large language models (LLMs), key/value pairs are frequently cached during token generation, a process known as KV caching, to prevent redundant computations due to the autoregressive nature of token generation. However, the size of the KV cache increases with both batch size and context length, which can slow down model inference due to the need to access a large amount of data in memory. Quantizing the KV cache effectively reduces this memory bandwidth limitation, improving inference speed. To study the quantization behavior of KV cache, we can simply set the `nbits_kvcache` argument to 8-bit, then the KV cache will be quantized together with weights and activations. In addition, the `bmm1_qm1_mode`, `bmm1_qm2_mode`, and `bmm2_qm2_mode` [arguments](../../fms_mo/training_args.py) must be set to the same quantizer mode as `qa_mode`. **NOTE**: `bmm2_qm1_mode` should be kept as `minmax`.

The effect of setting the `nbits_kvcache` to 8 and its relevant code sections are:

- Enables eager attention for the quantization of attention operations, including KV cache.
    ```python
    # For attention or kv-cache quantization, need to use eager attention
    attn_bits = [fms_mo_args.nbits_bmm1, fms_mo_args.nbits_bmm2, fms_mo_args.nbits_kvcache]
    if any(attn_bits) != 32:
        attn_implementation = "eager"
    else:
        attn_implementation = None
    ```
-  Enables Dynamo for quantized model preparation. We use PyTorch's Dynamo tracer to identify the bmm and KV cache inside the attention block.
    ```python
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
    ```

**2. Define quantization config** including quantizers and hyperparameters. Here we simply use the default [dq recipe](../../fms_mo/recipies/dq.json).

```python
qcfg = qconfig_init(recipe="dq",args=fms_mo_args)
```

**3. Obtain activation scales for SmoothQuant (SQ)**

``` python
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
        act_scales = get_act_scales(model, dq_dataloader, qcfg)
    scale_file = f"{act_scale_directory}/{qcfg['model'].replace('/', '-')}" + ".pt"
    torch.save(act_scales, scale_file)
```

**4. Prepare the quantized model and attach activation scales** to quantized modules

```python
qmodel_prep(
    model,
    dq_dataloader,
    qcfg,
    use_layer_name_pattern_matching=use_layer_name_pattern_matching,
    use_dynamo=use_dynamo,
    dev=dev,
    save_fname='test'
)

dq_llm(model, act_scales, qcfg)
```

**5. Perform direct quantization** by calibrating quantizers (clip_vals)

``` python
if qcfg["qmodel_calibration_new"] > 0:
    logger.info("Starting to calibrate activation clip_val")
    if qcfg["large_model"]:
        calibration_llm_1GPU(qcfg, model, calibration_dataset)
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
```

**6. Check perplexity** (simple method to evaluate the model quality)

``` python
if fms_mo_args.eval_ppl:
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
```
