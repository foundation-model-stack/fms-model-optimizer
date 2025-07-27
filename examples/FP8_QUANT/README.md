# 8-bit Floating Point (FP8) Quantization of LLaMA-3-8B Model

There are two types of FP8 support in FMS Model Optimizer:

1. **mature FP8** which can generate a model that is ready for serving by `vllm`, and
2. **experimental FP8** that is simulation only but has more advanced quantization options.

This is an example of mature FP8, which under the hood leverages some functionalities in [llm-compressor](https://github.com/vllm-project/llm-compressor), a third-party library, to perform FP8 quantization. An example for the experimental FP8 can be found [here](../DQ_SQ/README.md)

## Requirements

- [FMS Model Optimizer requirements](../../README.md#requirements)
- Nvidia A100 family or higher
- The [llm-compressor](https://github.com/vllm-project/llm-compressor) library can be installed using pip:

    ```bash
    pip install llmcompressor
    ```
- To evaluate the FP8 quantized model, [lm-eval](https://github.com/EleutherAI/lm-evaluation-harness) and [vllm](https://github.com/vllm-project/vllm) libraries are also required.
    ```bash
    pip install vllm lm_eval
    ```

> [!CAUTION]
> `vllm` may require a specific PyTorch version that is different from what is installed in your current environment and it may force install without asking. Make sure it's compatible with your settings or create a new environment if needed.

## QuickStart
This end-to-end example utilizes the common set of interfaces provided by `fms_mo` for easily applying multiple quantization algorithms with FP8 being the focus of this example. The steps involved are:

1. **FP8 quantization through CLI**. Other arguments could be found here [FP8Arguments](../../fms_mo/training_args.py#L84).

    ```bash
    python -m fms_mo.run_quant \
        --model_name_or_path "meta-llama/Meta-Llama-3-8B" \
        --quant_method fp8 \
        --torch_dtype bfloat16 \
        --output_dir "Meta-Llama-3-8B-FP8"
    ```

> [!NOTE]
> - The quantized model and tokenizer will be saved to `output_dir`, but some additional temporary storage space may be needed.
> - Runtime ~ 1 min on A100. (model download time not included)
> - If you have trouble downloading Llama family of models from Hugging Face ([LLama models require access](https://www.llama.com/docs/getting-the-models/hugging-face/)), you can use `ibm-granite/granite-3.0-8b-instruct` instead

2. **Inspect the FP8 checkpoint**

    ```python
    from fms_mo.utils.utils import checkpoint_summary
    checkpoint_summary("Meta-Llama-3-8B-FP8")
    ```

    We can see that most of the tensors are saved in FP8 format (`torch.float8_e4m3fn`). If you further print out the summary DataFrame (by adding `show_details=True` flag), you will find out the layers remained in `bfloat16` are `embeddings` and `lm_head`. 

    ```
                            mem (MB)
    dtype                           
    torch.bfloat16       2104.631296
    torch.float8_e4m3fn  6979.321856
    ```
> [!NOTE]
> FP16 model file size on storage is ~16.07 GB while FP8 is ~8.6 GB.

3. **Evaluate the quantized model**'s performance on a selected task using `lm-eval` library, the command below will run evaluation on [`lambada_openai`](https://huggingface.co/datasets/EleutherAI/lambada_openai) task and show the perplexity/accuracy at the end.

    ```bash
    lm_eval --model vllm \
        --model_args pretrained="Meta-Llama-3-8B-FP8,add_bos_token=True,dtype=float16,enforce_eager=True" \
        --tasks lambada_openai \
        --device cuda:0 \
        --batch_size 1 \
        --num_fewshot 5
    ```

## Example Test Results
- BF16 (not quantized) LLAMA3-8B model.

|    Tasks     |Version|Filter|n-shot|  Metric  |   |Value |   |Stderr|
|--------------|------:|------|-----:|----------|---|-----:|---|-----:|
|lambada_openai|      1|none  |     5|acc       |↑  |0.7120|±  |0.0287|
|              |       |none  |     5|perplexity|↓  |3.8683|±  |0.3716|

- FP8 quantized LLAMA3-8B model.

|    Tasks     |Version|Filter|n-shot|  Metric  |   |Value |   |Stderr|
|--------------|------:|------|-----:|----------|---|-----:|---|-----:|
|lambada_openai|      1|none  |     5|acc       |↑  |0.7160|±  |0.0286|
|              |       |none  |     5|perplexity|↓  |3.8915|±  |0.3727|

## Code Walk-through

1. The non-quantized pre-trained model is loaded using model wrapper from `llm-compressor`. The corresponding tokenizer is constructed as well.

    ```python
    from llmcompressor.modifiers.quantization import QuantizationModifier
    from llmcompressor.transformers import SparseAutoModelForCausalLM
    from llmcompressor import oneshot

    model = SparseAutoModelForCausalLM.from_pretrained(model_args.model_name_or_path, torch_dtype=model_args.torch_dtype)
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    ```

2. Quantization setting is provided using `QuantizationModifier`, additional settings can be found in [FP8Arguments](../../fms_mo/training_args.py#L84).

    ```python
    recipe = QuantizationModifier(
        targets=fp8_args.targets,
        scheme=fp8_args.scheme,
        ignore=fp8_args.ignore,
    )
    ```

3. FP8 quantization is performed by calling the `oneshot` function.
    ```python
    oneshot(
        model=model,
        recipe=recipe,
        max_seq_length=data_args.max_seq_length,
        num_calibration_samples=data_args.num_calibration_samples,
    )
    ```

4. The quantized model and the tokenizer are then saved in `output_dir`.

    ```python
    logger.info("Saving quantized model and tokenizer to {}".format(output_dir))
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    ```

5. Model evaluation is done using `lm_eval` through `vllm`. Accuracy and perplexity on the `lambada_openai` task will be reported.

    ```bash
    lm_eval --model vllm \
        --model_args pretrained="Meta-Llama-3-8B-FP8", add_bos_token=True, dtype="float16", enforce_eager=True \
        --tasks lambada_openai \
        --device cuda:0 \
        --batch_size auto \
        --num_fewshot 5
    ```

> [!NOTE]
> Even though A100 does not support FP8 computation, `vllm` can still utilize the compressed FP8 model and use FP16 computation to perform efficient inference.
