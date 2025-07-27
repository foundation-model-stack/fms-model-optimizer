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

"""Allow users to add new GPTQ classes for their custom models easily."""

# Local
from fms_mo.utils.import_utils import available_packages

if available_packages["gptqmodel"]:
    # Third Party
    from gptqmodel.models.base import BaseGPTQModel

    class GraniteGPTQForCausalLM(BaseGPTQModel):
        """Enable Granite for GPTQ."""

        layer_type = "GraniteDecoderLayer"
        layers_node = "model.layers"
        base_modules = ["model.embed_tokens", "model.norm"]
        layer_modules = [
            ["self_attn.k_proj", "self_attn.v_proj", "self_attn.q_proj"],
            ["self_attn.o_proj"],
            ["mlp.up_proj", "mlp.gate_proj"],
            ["mlp.down_proj"],
        ]

    class GraniteMoeGPTQForCausalLM(BaseGPTQModel):
        """Enable Granite MOE for GPTQ."""

        layer_type = "GraniteMoeDecoderLayer"
        layers_node = "model.layers"
        base_modules = ["model.embed_tokens", "model.norm"]
        layer_modules = [
            ["self_attn.k_proj", "self_attn.v_proj", "self_attn.q_proj"],
            ["self_attn.o_proj"],
            ["block_sparse_moe.input_linear", "block_sparse_moe.output_linear"],
        ]

    # NOTE: Keys in this table are huggingface config."model_type" (see the corresponding field in
    #       config.json). Make sure you cover the ones in the model family you want to use,
    #       as they may not be under the same model_type. See Granite as an example.
    custom_gptq_classes = {
        # "granite": GraniteGPTQForCausalLM,
        "granitemoe": GraniteMoeGPTQForCausalLM,
    }
