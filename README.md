# FMS Model Optimizer

![Lint](https://github.com/foundation-model-stack/fms-model-optimizer/actions/workflows/lint.yml/badge.svg?branch=main)
![Tests](https://github.com/foundation-model-stack/fms-model-optimizer/actions/workflows/test.yml/badge.svg?branch=main)
![Build](https://github.com/foundation-model-stack/fms-model-optimizer/actions/workflows/pypi.yml/badge.svg?branch=main)
[![Minimum Python Version](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
![Release](https://img.shields.io/github/v/release/foundation-model-stack/fms-model-optimizer)
![License](https://img.shields.io/github/license/foundation-model-stack/fms-model-optimizer)


## Introduction

FMS Model Optimizer is a framework for developing reduced precision neural network models. [Quantization](https://www.ibm.com/think/topics/quantization) techniques, such as [quantization-aware-training (QAT)](https://arxiv.org/abs/2407.11062), [post-training quantization (PTQ)](https://arxiv.org/abs/2102.05426), and several other optimization techniques on popular deep learning workloads are supported.

## Highlights

- **Python API to enable model quantization:** With the addition of a few lines of codes, module-level and/or function-level operations replacement will be performed.
- **Robust:** Verified for INT 8/4-bit quantization on important vision/speech/NLP/object detection/LLMs.
- **Flexible:** Options to analyze the network using PyTorch Dynamo, apply best practices, such as clip_val initialization, layer-level precision setting, optimizer param group setting, etc. during quantization.
- **State-of-the-art INT and FP quantization techniques** for weights and activations, such as SmoothQuant, SAWB+ and PACT+.
- **Supports key compute-intensive operations** like Conv2d, Linear, LSTM, MM and BMM

## Supported Models

| | GPTQ | FP8 | PTQ | QAT |
|---|------|-----|-----|-----|
| Granite      |:white_check_mark:|:white_check_mark:|:white_check_mark:|:black_square_button:|
| Llama        |:white_check_mark:|:white_check_mark:|:white_check_mark:|:black_square_button:|
| Mixtral      |:white_check_mark:|:white_check_mark:|:white_check_mark:|:black_square_button:|
| BERT/Roberta |:white_check_mark:|:white_check_mark:|:white_check_mark:|:white_check_mark:   |

**Note**: Direct QAT on LLMs is not recommended

## Getting Started

### Requirements

1. **🐧 Linux system with Nvidia GPU (V100/A100/H100)**
2. Python 3.10 to Python 3.12
3. CUDA >=12

*Optional packages based on optimization functionality required:*

- **GPTQ** is a popular compression method for LLMs: 
    - [gptqmodel](https://pypi.org/project/gptqmodel/) or build from [source](https://github.com/ModelCloud/GPTQModel)
- If you want to experiment with **INT8** deployment in [QAT](./examples/QAT_INT8/) and [PTQ](./examples/PTQ_INT8/) examples:
    - Nvidia GPU with compute capability > 8.0 (A100 family or higher)
    - Option 1:
        - [Ninja](https://ninja-build.org/)
        - Clone the [CUTLASS](https://github.com/NVIDIA/cutlass) repository
        - `PyTorch 2.3.1` (as newer version will cause issue for the custom CUDA kernel used in these examples)
    - Option 2:
        - use triton kernel included. But this kernel is currently not faster than FP16.
- **FP8** is a reduced precision format like **INT8**:
    - Nvidia A100 family or higher
    - [llm-compressor](https://github.com/vllm-project/llm-compressor)
- To enable compute graph plotting function (mostly for troubleshooting purpose):
    - [matplotlib](https://matplotlib.org/)
    - [graphviz](https://graphviz.org/)
    - [pygraphviz](https://pygraphviz.github.io/)

> [!NOTE]
> PyTorch version should be < 2.4 if you would like to experiment deployment with external INT8 kernel.

### Installation

We recommend using a Python virtual environment with Python 3.9+. Here is how to setup a virtual environment using [Python venv](https://docs.python.org/3/library/venv.html):

```
python3 -m venv fms_mo_venv
source fms_mo_venv/bin/activate
```

> [!TIP]
> If you use [pyenv](https://github.com/pyenv/pyenv), [Conda Miniforge](https://github.com/conda-forge/miniforge) or other such tools for Python version management, create the virtual environment with that tool instead of venv. Otherwise, you may have issues with installed packages not being found as they are linked to your Python version management tool and not `venv`.

There are 2 ways to install the FMS Model Optimizer as follows:

#### From Release

To install from release ([PyPi package](https://pypi.org/project/fms-model-optimizer/)):

```shell
python3 -m venv fms_mo_venv
source fms_mo_venv/bin/activate
pip install fms-model-optimizer
```

#### From Source

To install from source(GitHub Repository):

```shell
python3 -m venv fms_mo_venv
source fms_mo_venv/bin/activate
git clone https://github.com/foundation-model-stack/fms-model-optimizer
cd fms-model-optimizer
pip install -e .
```

#### Optional Dependencies
The following optional dependencies are available:
- `fp8`: `llmcompressor` package for fp8 quantization
- `gptq`: `GPTQModel` package for W4A16 quantization
- `mx`: `microxcaling` package for MX quantization
- `opt`: Shortcut for `fp8`, `gptq`, and `mx` installs
- `aiu`: `ibm-fms` package for AIU model deployment
- `torchvision`: `torch` package for image recognition training and inference
- `triton`: `triton` package for matrix multiplication kernels
- `examples`: Dependencies needed for examples
- `visualize`: Dependencies for visualizing models and performance data
- `test`: Dependencies needed for unit testing
- `dev`: Dependencies needed for development

To install an optional dependency, modify the `pip install` commands above with a list of these names enclosed in brackets.  The example below installs `llm-compressor` and `torchvision` with FMS Model Optimizer:

```shell
pip install fms-model-optimizer[fp8,torchvision]

pip install -e .[fp8,torchvision]
```
If you have already installed FMS Model Optimizer, then only the optional packages will be installed.

### Try It Out!

To help you get up and running as quickly as possible with the FMS Model Optimizer framework, check out the following resources which demonstrate how to use the framework with different quantization techniques:

 - Jupyter notebook tutorials (It is recommended to begin here):
    - [Quantization tutorial](tutorials/quantization_tutorial.ipynb):
        - Visualizes a random Gaussian tensor step-by-step along the quantization process
        - Build a quantizer and quantized convolution module based on this process
- [Python script examples](./examples/)

## Docs

Dive into the [design document](./docs/fms_mo_design.md) to get a better understanding of the
framework motivation and concepts.

## Contributing

Check out our [contributing guide](CONTRIBUTING.md) to learn how to contribute.
