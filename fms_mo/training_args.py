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
from typing import List, Optional, Union, get_args, get_origin


@dataclass
class TypeChecker:
    """Parent dataclass used by other args dataclasses to support input type validation."""

    def __post_init__(self):
        for name, field_type in self.__annotations__.items():
            val = self.__dict__[name]
            invalid_val = False
            if get_origin(field_type) is Union:
                if not type(val) in get_args(field_type):
                    invalid_val = True
            elif not get_origin(field_type) is list:
                if not isinstance(val, field_type):
                    invalid_val = True
            else:
                if not (
                    get_origin(val) is list
                    or type(val) is list  # pylint: disable=unidiomatic-typecheck
                    or all(isinstance(item, int) for item in val)
                ):
                    invalid_val = True

            if invalid_val:
                current_type = type(val)
                raise TypeError(
                    f"The field `{name}` was assigned by `{current_type}` instead of `{field_type}`"
                )


@dataclass
class ModelArguments(TypeChecker):
    """Dataclass for model related arguments."""

    model_name_or_path: str = field(default="facebook/opt-125m")
    task_type: str = field(
        default="lm",
        metadata={
            "choices": ["lm", "qa", "mlm"],
            "help": (
                "Instantiate model for selected task: 'lm' (language modeling), 'qa' "
                "(question answering, for encoders), 'mlm' (masked language modeling, "
                "for encoders)."
            ),
        },
    )
    torch_dtype: str = field(default="bfloat16")
    device_map: Optional[str] = field(
        default=None,
        metadata={
            "help": "can be 'auto', 'balanced', 'balanced_low_0', 'sequential' or something like"
            " {'encoder':'cuda:1', 'decoder': 'cuda:2'}.\n"
            "HF will try to move modules between cpu and cuda automatically during inference."
        },
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
    device: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "`torch.device`: The device on which the module is (assuming that all the module \
                parameters are on the same device)."
            )
        },
    )


@dataclass
class DataArguments(TypeChecker):
    """Dataclass for data related arguments."""

    training_data_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the training data in JSON/JSONL format"},
    )
    training_data_config: Optional[str] = field(default=None)
    test_data_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the test data in JSON/JSONL format"},
    )
    max_seq_length: int = field(default=2048)
    num_calibration_samples: int = field(default=512)


@dataclass
class OptArguments(TypeChecker):
    """Dataclass for optimization related arguments."""

    quant_method: str = field(
        metadata={
            "choices": ["gptq", "gptqv2", "fp8", "dq"],
            "help": "Quantization technique",
        }
    )
    output_dir: str = field(
        metadata={
            "help": "Output directory to write quantized model artifacts and log files to"
        }
    )
    log_level: str = field(
        default="INFO",
        metadata={"help": "The log level to adopt during optimization."},
    )
    save_ckpt: bool = field(
        default=True,
        metadata={"help": "Save quantized checkpoint."},
    )
    save_ckpt_for_aiu: bool = field(
        default=False,
        metadata={"help": "Prepare and save AIU-compliant checkpoint."},
    )


@dataclass
class FMSMOArguments(TypeChecker):
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
    aiu_sim_triton: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "AIU simulation with triton kernel. ['int8', 'fp8', None]\n"
                "'int8' mode will trigger qmodel_prep() and swap QLinears"
                "'fp8' mode will directly replace existing nn.Linears"
            )
        },
    )
    recompute_narrow_weights: bool = field(
        default=False,
        metadata={"help": "Apply recomputation during checkpoint saving for AIU."},
    )


@dataclass
class GPTQArguments(TypeChecker):
    """Dataclass for GPTQ related arguments that will be used by gptqmodel."""

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
class FP8Arguments(TypeChecker):
    """Dataclass for FP8 related arguments that will be used by llm-compressor."""

    targets: str = field(default="Linear")
    scheme: str = field(default="FP8_DYNAMIC")
    ignore: List[str] = field(default_factory=lambda: ["lm_head"])
