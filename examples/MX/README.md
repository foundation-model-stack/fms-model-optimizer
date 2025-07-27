# `microscaling` Examples Using a Toy Model and Direct Quantization (DQ)
Microscaling, or "MX", format, such as `MXFP8`, is a different numeric format compared to commonly used FP8 formats. For example, PyTorch provides two FP8 formats, which are 1 sign bit, 4 exponent bits, and 3 mantissa bits (denoted as `e4m3`) or 1 sign bit, 5 exponent bits, and 2 mantissa bits (`e5m2`), see our other [FP8 example](../FP8_QUANT/README.md) for more details.  On the other hand, all the `mx` formats are group-based data structure where each member of the group is using the specified format, e.g. FP8 for MXFP8, while each group has a shared (usually 8-bit) "scale".  Group size could be as small as 32 or 16, depending on hardware design.  One may consider each MXFP8 number actually requires 8.25 bits (when group size is 32) instead of 8 bits.  More details about microscaling can be found in [this OCP document](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf).

Here, we provide two simple examples of using MX format in `fms-mo`. 

> [!NOTE]
It is important to keep in mind that `mx` is not natively supported by Hopper GPUs yet (some will be supported by Blackwell), which means the quantization configurations and corresponding behavior are simulated. Hence, no real "speed up" should be expected.


## Requirements
- [FMS Model Optimizer requirements](../../README.md#requirements)
- Microsoft `microxcaling` python package, download [here](https://github.com/microsoft/microxcaling.git).
> [!TIP]
> `FMS-Model-Optimizer` and `microxcaling` have clashing dependency requirements for `PyTorch` packages.  We have created a patching solution to resolve this, run the following in command line:
``` bash
python3 ../install_patches.py
```
This patching file will either download the repo for you, or look for an already installed version in `$HOME` or the current working directory, then install the patch.
For more information, see `patches/README.md`.

## QuickStart

### Example 1
First example is based on a toy model with only a few Linear layers, in which only one Linear layer will be quantized with MX version of `int8`, `int4`, `fp8`, and `fp4`.  The example can simply be run as follow

```bash
>>> python simple_mx_example.py
```

Comparison between different formats, including the first 3 elements from output tensors and the norm compared to FP32 reference, is shown below.

| dtype      |   output[0, 0] |   output[0, 1] |   output[0, 2] |   \|\|ref - out_dtype\|\|<sub>2</sub> |
|:-----------|---------------:|---------------:|---------------:|------------------------:|
| fp32       |        -1.0491 |         0.5312 |        -1.6387 |                  0.0000 |
| fmsmo_int8 |        -1.0577 |         0.5346 |        -1.6508 |                  0.4937 |
| fmsmo_int4 |        -0.5885 |         0.5831 |        -1.7976 |                  8.2927 |
| mxint8     |        -0.6444 |         0.6828 |        -1.8626 |                  8.3305 |
| mxint4     |        -0.9089 |         0.6141 |        -1.7630 |                  8.0692 |
| mxfp8_e4m3 |        -0.8031 |         0.7262 |        -1.9581 |                  7.8554 |
| mxfp8_e5m2 |        -0.8471 |         0.7319 |        -1.7458 |                  8.1838 |
| mxfp4_e2m1 |        -0.7506 |         0.6123 |        -1.9311 |                  7.9936 |


### Example 2
The second example is the same as the [DQ example](../DQ_SQ/README.md), except using [microxcaling](https://arxiv.org/abs/2310.10537) format.  We only demonstrate `mxfp8` and `mxfp4` here, but MXINT8, MXFP8, MXFP6, MXFP4 are also available for weights, activations, and/or KV-cache. 

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
> [!NOTE]
> To use MX format, simply assign `qa_mode` and `qw_mode` argument with a `mx_<dtype supported by mx package>`, e.g. `mx_fp8_e4m3` as in the above example. Corresponding `QLinearMX` wrappers will be used in place of `QLinear` as in other examples.

**3. Compare the Perplexity score** For user convenience, the code will print out perplexity (controlled by `eval_ppl` flag) at the end of the run, so no additional steps needed (if the logging level is set to `INFO` in terminal). You can check output in the logging file. `./fms_mo.log`.


## Example Test Results
The perplexity of the INT8 and FP8 quantized models on the `wikitext` dataset is shown below:

| Model     |Type |QA            |QW            |DQ  |SQ  |Perplexity|
|:---------:|:---:|:------------:|:------------:|:--:|:--:|:--------:|
|`Llama3-8b`|INT8 |maxpertoken   |maxperCh      |yes |yes |6.22      |
|           |FP8  |fp8_e4m3_scale|fp8_e4m3_scale|yes |yes |6.19      |
|           |**MX**|mx_fp8_e4m3  |mx_fp8_e4m3   |yes |**no** |6.23   |
|           |**MX**|mx_fp4_e2m1  |mx_fp4_e2m1   |yes |**no** |8.22   |


> [!NOTE]
> SmoothQuant is disabled when `mx` is being used. See `dq.py` for more details.

