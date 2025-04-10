# Model Optimization Using Quantization-Aware Training (QAT)

FMS Model Optimizer supports [quantization](https://www.ibm.com/think/topics/quantization) of models which will enable the utilization of reduced-precision numerical format and specialized hardware to accelerate inference performance (i.e., make "calling a model" faster).

Generally speaking, matrix multiplication (matmul) is the main operation in a neural network. The goal of quantization is to convert a floating-point (FP) matmul into an integer (INT) matmul, which runs much faster and requires lower energy consumption. A simplified example would be:

$$X@W \approx \lfloor \frac{X}{s_x} \rceil @ \lfloor \frac{W}{s_w} \rceil*s_xs_w$$

- where $X$, $W$ are FP tensors whose elements are all within a certain range, e.g. $[-5.0, 5.0]$, $@$ is matmul operation, $\lfloor  \rceil$ is rounding operation, scaling factor $s_x, s_w$ in this case is simply $5/127$.
- On the right hand side, after scaling and rounding the tensors will only contain integers in the range of $[-127, 127]$, which can be stored as a 8-bit integer.
- We may now use an INT8 matmul instead of a FP32 matmul to perform the task then multiply the scaling factors afterward.
- **Important** The benefit from INT matmul should outweigh the overhead from scaling, rounding, and descaling. But rounding will inevitably introduce approximation errors. Luckily, we can mitigate the errors by taking these quantization related operations into account during the training process, hence the Quantization-aware training ([QAT](https://arxiv.org/pdf/1712.05877))!

In the following example, we will first create a fine-tuned FP16 model, and then quantize this model from FP16 to INT8 using QAT. Once the model is tuned and QAT'ed, you can observe the accuracy and the acceleration at inference time of the model.


## Requirements

- [FMS Model Optimizer requirements](../../README.md#requirements)
- The inferencing step requires Nvidia GPUs with compute capability > 8.0 (A100 family or higher)
- NVIDIA cutlass package (Need to clone the source, not pip install). Preferably place in user's home directory: `cd ~ && git clone https://github.com/NVIDIA/cutlass.git`
- [Ninja](https://ninja-build.org/)
- `PyTorch 2.3.1` (as newer version will cause issue for the custom CUDA kernel)


## QuickStart

> [!NOTE]
> This example is based on the HuggingFace [Transformers Question answering example](https://github.com/huggingface/transformers/tree/main/examples/pytorch/question-answering).

There are **three main steps** to try out the example as follows:

#### **1.  Fine-tune a model** with 16-bit floating point (FP16) precision:

```shell
export CUDA_VISIBLE_DEVICES=0

python run_qa_no_trainer_qat.py \
  --model_name_or_path google-bert/bert-base-uncased \
  --dataset_name squad \
  --per_device_train_batch_size 12 \
  --learning_rate 3e-5 \
  --num_train_epochs 2 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --output_dir ./fp16_ft_squad/ \
  --with_tracking \
  --report_to tensorboard \
  --attn_impl eager
```

> [!TIP]
> The script can take up to 40 mins to run (on a single A100). By default, it is configured for detailed logging. You can disable the logging by removing the `with_tracking` and `report_to` flags in the script. This can reduce the runtime by around 20 mins.

#### **2.  Apply QAT** on the fine-tuned model, which converts the precision data to 8-bit integer (INT8):

```shell
python run_qa_no_trainer_qat.py \
  --model_name_or_path ./fp16_ft_squad/ \
  --dataset_name squad \
  --per_device_train_batch_size 12 \
  --learning_rate 3e-5 \
  --num_train_epochs 2 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --output_dir ./qat_on_fp16ft \
  --with_tracking \
  --report_to tensorboard \
  --attn_impl eager \
  --do_qat \
  --pact_a_lr 1e-3
```

> [!TIP]
> The script can take up to 1.5 hours to run (on a single A100). Remove `with_tracking` and `report_to` flags can reduce the runtime to about 40 mins.

#### **3. Compare the accuracy and inference speed** of 16-bit floating point (FP16) and 8-bit integer (INT8) precision models:

```shell
export TOKENIZERS_PARALLELISM=false

python run_qa_no_trainer_qat.py \
  --model_name_or_path ./qat_on_fp16ft/ \
  --dataset_name squad \
  --per_device_train_batch_size 128 \
  --per_device_eval_batch_size 128 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --attn_impl eager \
  --do_lowering <cutlass or triton>
```

This script uses an "external kernel" instead of the `torch.matmul` kernel to perform real `INT8` matmuls. We have two options for INT kernel, one is written using Nvidia's CUDA/CUTLASS library and one is in Triton. Both will be compiled once just ahead of the run (i.e., just-in-time, JIT, compilation).  The compiled artifacts are usually stored in `~/.cache/torch_extensions/`. Remove this folder if a fresh recompile of the kernel is needed.

Checkout [Example Test Results](#example-test-results) to compare against your results.

## Example Test Results

For comparison purposes, here are some of the results from an A100. CUTLASS results were obtained with `PyTorch 2.3.1` while Triton results were obtained using `PyTorch 2.4.1`:

> [!NOTE]
> Accuracy could vary ~ +-0.2 from run to run.

|model|batch size|torch.compile|accuracy(F1)|inference speed (msec)|
|----|--:|---------:|----:|------------:|
|fp16|128|eager     |88.21 (as fine-tuned) |126.38|
|    |128|Inductor  |     |71.59|
|    |128|CUDAGRAPH |     |71.13|
|INT8 CUTLASS|128|eager     |88.33|329.45 <sup>1</sup>|
|    |128|Inductor  |88.42|67.87 <sup>2</sup>|
|    |128|CUDAGRAPH |--   |-- <sup>3</sup>|
|INT8 triton|128|eager     |88.10|358.51|
|    |128|Inductor  |88.13|99.91 <sup>4</sup>|
|    |128|CUDAGRAPH |88.13|100.21 <sup>4</sup>|

<sup>1</sup> `INT8` matmuls are ~2x faster than `FP16` matmuls. However, `INT8` models will have additional overhead compared to `FP16` models. For example, converting FP tensors to INT before INT matmul.

<sup>2</sup> Each of these additional quantization operations is relatively 'cheap', but the overhead of launching each job is not negligible. Using `torch.compile` can fuse the Ops and reduce the total number of jobs being launched.

<sup>3</sup> `CUDAGRAPH` is the most effective way to minimize job launching overheads and can achieve ~2X end-to-end speed-up in this case. However, there seem to be bugs associated with this option at the moment. Further investigation is still on-going.

<sup>4</sup> Unlike our CUTLASS `INT8` kernel, which is ~2x faster than `FP16` matmul, our Triton `INT8` is not as optimized and performs only comparable with `FP16` on mid-to-large tensor sizes. 

## Code Walk-through

In this section, we will deep dive into what happens during the example steps.

There are three parts to the example:

**1. Fine-tune a model with 16-bit floating point (FP16) precision**

Fine-tunes a BERT model on the question answering dataset, SQuAD. This step is based on the HuggingFace [Transformers Question answering example](https://github.com/huggingface/transformers/tree/main/examples/pytorch/question-answering). It was modified to collect additional training information in case we would like to tweak the hyper-parameters later.

**2. Apply Quantization using QAT**

For `INT8` quantization, we can achieve comparable accuracy with `FP16` by using [quantization-aware training (QAT)](https://arxiv.org/abs/2407.11062) or post-training quantization (PTQ) techniques. In this example we use QAT.

In a nutshell, QAT simply quantizes the weight and activation tensors before matrix multiplications (matmul) so that quantization errors will be taken into account during the training/loss optimization process. The code below is an example of preparing a model for QAT quantization prior to fine tuning:

```python
from fms_mo import qmodel_prep, qconfig_init

# Create a config dict using a default recipe and CLI args
# If same item exists in both, args take precedence over recipe.
qcfg = qconfig_init(recipe = 'qat_int8', args=args)

# Prepare a list of "ready-to-run" data for calibration
exam_inp = [next(iter(train_dataloader)) for _ in range(qcfg['qmodel_calibration']) ]

logger.info(f"--- Accuracy of {args.model_name_or_path} before QAT/PTQ")
squad_eval(model) # This is a fn modified from original script that checks accuracy

qmodel_prep(model, exam_inp, qcfg, optimizer, use_dynamo = True)

# (Continue to original fine-tuning script)
...

```

The resulting model is saved in the `qat_on_fp16ft` folder. Be aware that the weights are now different from the original `FP16` checkpoint in Step 1, but not yet converted to real `INT8`!

**3. Evaluate Inference Accuracy and Speed**

> [!NOTE]
> This step will compile an external kernel for INT matmul, which currently only works with `PyTorch 2.3.1`.

Here is an example code snippet used for evaluation:

```python
from fms_mo.modules.linear import QLinear, QLinearINT8Deploy
# ...

# Only need 1 batch (not a list) this time, will be used by `torch.compile` as well.
exam_inp = next(iter(train_dataloader))

qcfg = qconfig_init(recipe = 'qat_int8', args=args)
qcfg['qmodel_calibration'] = 0 # <----------- NOTE 1
qmodel_prep(model, exam_inp, qcfg, optimizer, use_dynamo = True,
            ckpt_reload=args.model_name_or_path) # <----------- NOTE 2

# ----------- NOTE 3
mod2swap = [n for n,m in model.named_modules() if isinstance(m, QLinear)]
for name in mod2swap:
    parent_name, module_name = _parent_name(name)
    parent_mod = model.get_submodule(parent_name)
    qmod = getattr(parent_mod, module_name)
    setattr(parent_mod, module_name, QLinearINT8Deploy.from_fms_mo(qmod))

# ...

with torch.no_grad():
    model = torch.compile(model) #, mode='reduce-overhead') # <----- NOTE 4
    model(**exam_inp)

# ...

return # Stop the run here, no further training loop
```

In this example:

- By default, QAT will run `calibration` to initialize the quantization related parameters (with a small number of training data). At the end of QAT, these parameters are saved with the checkpoint, as we DO NOT want to run calibration at deployment stage. Hence, `qcfg['qmodel_calibration'] = 0`.
- Quantization related parameters will not be automatically loaded by the HuggingFace method, as those are not part of the original BERT model. Hence calling `qmodel_prep(..., ckpt_reload=[path to qat ckpt])`.
- By replacing `QLinear` layers with `QLinearINT8Deploy`, it will call the external kernel instead of `torch.matmul`.
- `torch.compile` with `reduce-overhead` option will use CUDAGRAPH and achieve the most ideal speed-up. However, some models may not be fully compatible with this option.
