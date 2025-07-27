# Train and prepare INT8 checkpoint for the AIU using Direct Quantization
This example builds on the [Direct Quantization (DQ) example](../DQ_SQ/README.md). We assume the user is already familiar with the DQ quantization process and would like to generate an INT8-quantized checkpoint that is made compliant with the requirements of the AIU/Spire accelerator.

Once created, this checkpoint can be run on the AIU by using an inference script from [aiu-fms-testing-utils](https://github.com/foundation-model-stack/aiu-fms-testing-utils).

For more information on the AIU/Spyre accelerator, see the following blogs:
- [Introducing the IBM Spyre AI Accelerator chip](https://research.ibm.com/blog/spyre-for-z)
- [IBM Power modernizes infrastructure and accelerates innovation with AI in the year ahead](https://newsroom.ibm.com/blog-ibm-power-modernizes-infrastructure-and-accelerates-innovation-with-ai-in-the-year-ahead)

## Requirements
- [FMS Model Optimizer requirements](../../README.md#requirements)

## QuickStart

**1. Prepare Data** as per DQ quantization process ([link](../DQ_SQ/README.md)). In this example, we assume the user wants to quantized RoBERTa-base model and has thus prepared the DQ data for it, stored under the folder `data_train` and `data_test`, by adapting the DQ example accordingly.

**2. Apply DQ with conversion** by providing the desired quantization parameters, as well as the flags `--save_ckpt_for_aiu` and `--recompute_narrow_weights`.

```bash
python  -m fms_mo.run_quant \
        --model_name_or_path "roberta-base" \
        --training_data_path data_train \
        --test_data_path data_test \
        --torch_dtype "float16" \
        --quant_method dq \
        --nbits_w 8 \
        --nbits_a 8 \
        --nbits_kvcache 32 \
        --qa_mode "pertokenmax"\
        --qw_mode "maxperCh" \
        --qmodel_calibration_new 1 \
        --output_dir "dq_test" \
        --save_ckpt_for_aiu \
        --recompute_narrow_weights
```
> [!TIP]
> - In this example, we are not evaluating the perplexity of the quantized model, but, if so desired, the user can add the `--eval_ppl` flag.
> - We set a single calibration example because the quantizers in use do not need calibration: weights remain static during DQ, so a single example will initialize the quantizer correctly, and the activation quantizer `pertokenmax` will dynamically recompute the quantization range at inference time, when running on the AIU.

**3. Reload checkpoint for testing** and validate its content (optional).

```python
sd = torch.load("dq_test/qmodel_for_aiu.pt", weights_only=True)
```

Check that all quantized layers have been converted to `torch.int8`, while the rest are `torch.float16`.

```python
# select quantized layers by name
roberta_qlayers = ["attention.self.query", "attention.self.key", "attention.self.value", "attention.output.dense", "intermediate.dense", "output.dense"]
# assert all quantized weights are int8
assert all(v.dtype == torch.int8 for k,v in sd.items() if any(n in k for n in roberta_qlayers) and k.endswith(".weight"))
# assert all other parameters are fp16
assert all(v.dtype == torch.float16 for k,v in sd.items() if all(n not in k for n in roberta_qlayers) or not k.endswith(".weight"))
```

> [!TIP]
> - We have trained the model with symmetric quantizer for activations (`qa_mode`). If an asymmetric quantizer is used, then the checkpoint will also carry a `zero_shift` parameters which is torch.float32, so this validation step should be modified accordingly.

Because we have used the `narrow_weight_recomputation` option along with a `maxperCh` (max per-channel) quantizer for weights, the INT weight matrices distributions have been widened. Most values of standard deviation (per channel) should surpass the empirical threshold of 20.

```python
[f"{v.to(torch.float32).std(dim=-1).mean():.4f}" for k,v in sd.items() if k.endswith(".weight") and any(n in k for n in roberta_qlayers)]
```

> [!TIP]
> - We cast the torch.int8 weights to torch.float32 to be able to apply the torch.std function.
> - For per-channel weights, the recomputation is applied per-channel. Here we print a mean across channels for help of visualization.
> - It is not a guarantee that the recomputed weights will exceed the empirical threshold after recomputation, but it is the case for several common models of BERT, RoBERTa, Llama, and Granite families.
