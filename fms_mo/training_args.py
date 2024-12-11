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

"""
Arguments used for quantization
"""

# Standard
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ModelArguments:
    """Dataclass for model related arguments."""

    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    torch_dtype: Optional[str] = field(
        default=None,
        metadata={"help": ["bfloat16", "float16", "float", "auto"]},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={
            "help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) \
            or not."
        },
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Where do you want to store the pretrained models downloaded from \
            huggingface.com"
        },
    )
    model_revision: str = field(
        default="main",
        metadata={
            "help": "The specific model version to use (can be a branch name, tag name or \
            commit id)."
        },
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `huggingface-cli login` (necessary to \
                use this script with private models)."
            )
        },
    )


@dataclass
class DataArguments:
    """Dataclass for data related arguments."""

    training_data_path: str = field(
        default=None,
        metadata={"help": "Path to the training data in JSON/JSONL format"},
    )
    training_data_config: str = field(default=None)
    test_data_path: str = field(
        default=None,
        metadata={"help": "Path to the test data in JSON/JSONL format"},
    )
    max_seq_length: Optional[int] = field(default=2048)
    num_calibration_samples: Optional[int] = field(default=512)


@dataclass
class FMSMOArguments:
    """Dataclass arguments used by fms_mo native quantization functions."""

    nbits_w: int = field(default=32, metadata={"help": ("weight precision")})
    nbits_a: int = field(default=32, metadata={"help": ("activation precision")})
    nbits_bmm1: int = field(default=32, metadata={"help": ("attention bmm1 precision")})
    nbits_bmm2: int = field(default=32, metadata={"help": ("attention bmm2 precision")})
    nbits_kvcache: int = field(default=32, metadata={"help": ("kv-cache precision")})
    qw_mode: str = field(default="sawb+", metadata={"help": ("weight quantizer")})
    qa_mode: str = field(default="pact+", metadata={"help": ("activation quantizer")})
    bmm1_qm1_mode: str = field(default="pact", metadata={"help": ("bmm1.m1 quanitzer")})
    bmm1_qm2_mode: str = field(default="pact", metadata={"help": ("bmm1.m2 quanitzer")})
    bmm2_qm1_mode: str = field(default="pact", metadata={"help": ("bmm2.m1 quanitzer")})
    bmm2_qm2_mode: str = field(default="pact", metadata={"help": ("bmm2.m1 quanitzer")})
    smoothq_alpha: float = field(default=0.65, metadata={"help": "smooth quant alpha"})
    qmodel_calibration: int = field(
        default=0,
        metadata={"help": "Num of batches for Qmodel calibration, using model copy."},
    )
    qmodel_calibration_new: int = field(
        default=0,
        metadata={
            "help": (
                "Num of batches for Qmodel calibration. "
                "NOTE! First num of iterations will be used for calibration."
            )
        },
    )
    block_size: Optional[int] = field(
        default=2048, metadata={"help": "input sequence length after tokenization"}
    )
    eval_ppl: bool = field(default=False)


@dataclass
class GPTQArgs:
    """Dataclass for GPTQ related arguments that will be used by auto-gptq."""

    bits: int = field(default=4, metadata={"choices": [2, 3, 4, 8]})
    group_size: int = field(default=-1)
    damp_percent: float = field(default=0.01)
    desc_act: bool = field(default=False)
    static_groups: bool = field(default=False)
    sym: bool = field(default=True)
    true_sequential: bool = field(default=True)
    batch_size: int = 1
    use_triton: bool = False
    use_cuda_fp16: bool = True
    autotune_warmup_after_quantized: bool = False
    cache_examples_on_gpu: bool = True


@dataclass
class FP8Args:
    """Dataclass for FP8 related arguments that will be used by llm-compressor."""

    targets: str = field(default="Linear")
    scheme: str = field(default="FP8_DYNAMIC")
    ignore: List[str] = field(default_factory=lambda: ["lm_head"])
