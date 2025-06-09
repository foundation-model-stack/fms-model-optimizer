# Generative Pre-Trained Transformer Quantization (GPTQ) of LLAMA-3-8B Model


For generative LLMs, very often the bottleneck of inference is no longer the computation itself but the data transfer. In such case, all we need is an efficient compression method to reduce the model size in memory, together with an efficient GPU kernel that can bring in the compressed data and only decompress it at GPU cache-level right before performing an FP16 computation. This approach is very powerful because it could reduce the number of GPUs for serving the model by 4X without sacrificing inference speed (some constraints may apply, such as batch size cannot exceed a certain number.) FMS Model Optimizer supports this "weight-only compression", or sometimes referred to as W4A16 or [GPTQ](https://arxiv.org/pdf/2210.17323) by leveraging `gptqmodel`, a third party library, to perform quantization.

## Requirements

- [FMS Model Optimizer requirements](../../README.md#requirements)
- `gptqmodel` is needed for this example. Use `pip install gptqmodel` or [install from source](https://github.com/ModelCloud/GPTQModel/tree/main?tab=readme-ov-file)
- Optionally for the evaluation section below, install [lm-eval](https://github.com/EleutherAI/lm-evaluation-harness)
    ```
    pip install lm-eval
    ```


## QuickStart
This end-to-end example utilizes the common set of interfaces provided by `fms_mo` for easily applying multiple quantization algorithms with GPTQ being the focus of this example. The steps involved are:

1. **Convert the dataset into its tokenized form.** An example of tokenization using `LLAMA-3-8B`'s tokenizer is below.

    ```python
    from transformers import AutoTokenizer
    from fms_mo.utils.calib_data import get_tokenized_data

    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B", use_fast=True)
    num_samples = 128
    seq_len = 2048
    get_tokenized_data("wiki", num_samples, seq_len, tokenizer, gptq_style=True, path_to_save='data')
    ```
> [!NOTE]
> - Users should provide a tokenized data file based on their need. This is just one example to demonstrate what data format `fms_mo` is expecting.
> - Tokenized data will be saved in `<path_to_save>_train` and `<path_to_save>_test`
> - If you have trouble downloading Llama family of models from Hugging Face ([LLama models require access](https://www.llama.com/docs/getting-the-models/hugging-face/)), you can use `ibm-granite/granite-8b-code` instead

2. **Quantize the model** using the data generated above, the following command will kick off the quantization job (by invoking `gptqmodel` under the hood.) Additional acceptable arguments can be found here in [GPTQArguments](../../fms_mo/training_args.py#L127).

    ```bash
    python -m fms_mo.run_quant \
        --model_name_or_path meta-llama/Meta-Llama-3-8B  \
        --training_data_path data_train \
        --quant_method gptq \
        --output_dir Meta-Llama-3-8B-GPTQ \
        --bits 4 \
        --group_size 128
    ```
    The model that can be found in the specified output directory (`Meta-Llama-3-8B-GPTQ` in our case) can be deployed and inferenced via `vLLM`.

> [!NOTE]
> - In GPTQ, `group_size` is a trade-off between accuracy and speed, but there is an additional constraint that `in_features` of the Linear layer to be quantized needs to be an **integer multiple** of `group_size`, i.e. some models may have to use smaller `group_size` than default.

> [!TIP]
> 1. If you see error messages regarding `exllama_kernels` or `undefined symbol`, try installing `gptqmodel` from [source](https://github.com/ModelCloud/GPTQModel/tree/main?tab=readme-ov-file).
> 2. If you need to work on a custom model that is not supported by GPTQModel, please add your class wrapper [here](../../fms_mo/utils/custom_gptq_models.py). Additional information [here](https://github.com/ModelCloud/GPTQModel/tree/main?tab=readme-ov-file#how-to-add-support-for-a-new-model).

3. **Inspect the GPTQ checkpoint**
    ```python
    from fms_mo.utils.utils import checkpoint_summary
    checkpoint_summary("Meta-Llama-3-8B-GPTQ")
    ```

    We can see that most of the tensors are saved in INT32 format (INT4 are not natively supported by PyTorch, hence, packed into INT32 instead.). If you further print out the summary DataFrame (by adding `show_details=True` flag), you will find out the layers remained in `float16` or `float32` are `scales` and `layernorm`. 

    ```
                    layer     mem (MB)
    dtype
    torch.bfloat16     67  2101.878784
    torch.float16     224   109.051904
    torch.int32       672  3521.904640
    ```

4. **Evaluate the quantized model**'s performance on a selected task using `lm-eval` library, the command below will run evaluation on [`lambada_openai`](https://huggingface.co/datasets/EleutherAI/lambada_openai) task and show the perplexity/accuracy at the end.

    ```bash
    lm_eval --model hf \
            --model_args pretrained="Meta-Llama-3-8B-GPTQ,dtype=float16,gptqmodel=True,enforce_eager=True" \
            --tasks lambada_openai \
            --num_fewshot 5 \
            --device cuda:0 \
            --batch_size auto
    ```

## Example Test Results

- Unquantized Model
- 
|Model       |    Tasks     |Version|Filter|n-shot|  Metric  |   |Value |   |Stderr|
|------------|--------------|------:|------|-----:|----------|---|-----:|---|-----:|
| LLAMA3-8B  |lambada_openai|      1|none  |     5|acc       |↑  |0.7103|±  |0.0063|
|            |              |       |none  |     5|perplexity|↓  |3.7915|±  |0.0727|

- Quantized model with the settings showed above (`desc_act` default to False.)
- 
|Model       |    Tasks     |Version|Filter|n-shot|  Metric  |   |Value  |   |Stderr|
|------------|--------------|------:|------|-----:|----------|---|------:|---|-----:|
| LLAMA3-8B  |lambada_openai|      1|none  |     5|acc       |↑  |0.6365 |±  |0.0067|
|            |              |       |none  |     5|perplexity|↓  |5.9307 |±  |0.1830|

- Quantized model with `desc_act` set to `True` (could improve the model quality, but at the cost of inference speed.)
- 
|Model       |    Tasks     |Version|Filter|n-shot|  Metric  |   |Value  |   |Stderr|
|------------|--------------|------:|------|-----:|----------|---|------:|---|-----:|
| LLAMA3-8B  |lambada_openai|      1|none  |     5|acc       |↑  |0.6193 |±  |0.0068|
|            |              |       |none  |     5|perplexity|↓  |5.8879 |±  |0.1546|

> [!NOTE]
> There is some randomness in generating the model and data, the resulting accuracy may vary ~$\pm$ 0.05.


## Code Walk-through

1.  Command line arguments will be used to create a GPTQ quantization config. Information about the required arguments and their default values can be found [here](../../fms_mo/training_args.py)

    ```python
    from gptqmodel import GPTQModel, QuantizeConfig

    quantize_config = QuantizeConfig(
        bits=gptq_args.bits,
        group_size=gptq_args.group_size,
        desc_act=gptq_args.desc_act,
        damp_percent=gptq_args.damp_percent,
    )

    ```

2. Load the pre_trained model with `gptqmodel` class/wrapper. Tokenizer is optional because we already tokenized the data in a previous step.

    ```python
    model = GPTQModel.from_pretrained(
        model_args.model_name_or_path,
        quantize_config=quantize_config,
        torch_dtype=model_args.torch_dtype,
    )
    ```

3. Load the tokenized dataset from disk.

    ```python
    data = load_from_disk(data_args.training_data_path)
    data = data.with_format("torch")
    ```

4. Quantize the model.

    ```python
    model.quantize(
        data,
        backend=BACKEND.TRITON if gptq_args.use_triton else BACKEND.AUTO,
        batch_size=gptq_args.batch_size,
        calibration_enable_gpu_cache=gptq_args.cache_examples_on_gpu,
    )
    ```

5. Save the logs and the resulting quantized model.

    ```python
    logger.info(f"Saving quantized model and tokenizer to {output_dir}")
    model.save_quantized(output_dir, use_safetensors=True)
    tokenizer.save_pretrained(output_dir) # optional
    ```
> [!NOTE]
> 1. GPTQ of a 70B model usually takes ~4-10 hours on A100.
