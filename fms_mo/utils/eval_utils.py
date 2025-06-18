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
Evaluation utils for DQ
"""

# Standard
import logging

# Third Party
from torch import nn
from tqdm import tqdm
import torch

# Local
from fms_mo.quant.ptq import cache_block0_inputs, get_blocks
from fms_mo.utils.utils import move_to, patch_torch_bmm

logger = logging.getLogger(__name__)


@torch.no_grad()
def eval_llm_1GPU(qcfg, model, test_dataset, pre_cache_func=None, **kwargs):  # pylint: disable=unused-argument
    """
    Evaluate causal LLM with 1GPU, return perplexity
    Note:
    1. currently taking test_dataset as dict (instead of dataloader)
    2. Used for models that cannot fit into a 1 GPU. Will need to move modules back and forth.
    3. Keep hid_state on device to reduce uncessary data transfer.
    """
    model.eval()
    dev = "cuda"
    qcfg["batch_size"] = 1  # for dataloading, always use batch_size of 1
    qcfg["dtype"] = next(iter(model.parameters())).dtype
    seq_len = qcfg["seq_len"]
    qcfg["n_samples"] = int(test_dataset.input_ids.shape[1] / seq_len)
    # --- Phase 0 cache the inputs of the block0---
    use_cache = model.config.use_cache
    model.config.use_cache = False
    blocks, emb, emb_pos, emb_ln, ln_f, lm_head = get_blocks(model, qcfg["model_type"])
    cache_block0_inputs(
        model,
        test_dataset,
        qcfg,
        blocks,
        emb=emb,
        emb_pos=emb_pos,
        emb_ln=emb_ln,
        dev="cpu",
    )
    logger.info("Done, caching inputs to block0 for evaluation")

    # Phase 1: compute blocks and last linear layer
    pbar = tqdm(blocks, desc="evaluation: compute blocks")

    qcfg["cached_input"] = [
        inp.clone().detach().to(dev) for inp in qcfg["cached_block0_input"]
    ]
    kw_to_use = {
        kw_org: kw_new
        for kw_org, kw_new in qcfg["kw_to_cache"].items()
        if len(qcfg[kw_new]) == len(qcfg["cached_input"])
    }
    for block_id, m in enumerate(pbar):  # pylint: disable=unused-variable
        m.to(dev)
        for i in range(qcfg["n_samples"]):
            if qcfg["cached_alibi"]:
                cached_inp_prev_lay = qcfg["cached_input"][i].unsqueeze(0).to(dev)
                data_mb = {
                    "attention_mask": qcfg["cached_mask"][i].unsqueeze(0).to(dev),
                    "alibi": qcfg["cached_alibi"][i].unsqueeze(0).to(dev),
                }
            else:
                cached_inp_prev_lay = qcfg["cached_input"][i]
                data_mb = {
                    kw_org: move_to(qcfg[kw_new][i], dev)
                    for kw_org, kw_new in kw_to_use.items()
                }

            with patch_torch_bmm(qcfg):
                qcfg["cached_input"][i] = m(cached_inp_prev_lay, **data_mb)[0]

        m.cpu()
        torch.cuda.empty_cache()

    logger.info("All blocks are computed for evaluation")

    nlls = []
    # for i, data_mb in enumerate(dloader): #if using dloader.
    for i in tqdm(range(qcfg["n_samples"]), desc="Final Evaluating..."):
        hidden_states = qcfg["cached_input"][i].to(dev)
        if ln_f is not None:
            ln_f.to(dev)
            hidden_states = ln_f(hidden_states)
        lm_head.to(dev)
        lm_logits = lm_head(hidden_states)

        # Shift so that tokens < n predict n
        shift_logits = lm_logits[:, :-1, :].contiguous().float()
        shift_labels = test_dataset.input_ids[:, (i * seq_len) : ((i + 1) * seq_len)][
            :, 1:
        ].to(dev)
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
        )
        neg_log_likelihood = loss.float() * seq_len
        nlls.append(neg_log_likelihood)
    eval_loss = torch.stack(nlls).sum() / (qcfg["n_samples"] * seq_len)
    ppl = torch.exp(eval_loss)
    logger.info("-" * 50)
    logger.info(f"Quantized Model Type: {qcfg['model_type']} \n")
    logger.info(f"Final Evaluation Loss: {eval_loss.item()} \n")
    logger.info(f"Final Perplexity: {ppl.item()} \n")
    logger.info("-" * 50)
    model.config.use_cache = use_cache


class Evaluator:
    """Evaluates the model perplexity for a model that can fit into 1 GPU node.
    modifed base on : https://github.com/mit-han-lab/smoothquant/blob/main/smoothquant/ppl_eval.py
    """

    def __init__(self, dataset, device, n_samples=160):
        self.dataset = dataset
        self.device = device
        # loading tokenized dataset.
        self.dataset = dataset.input_ids.to(device)
        self.n_samples = n_samples

    @torch.no_grad()
    def evaluate(self, model, block_size=2048):
        """
        Function for evaluating the model inference performance by meausring the model's perplexity
        """
        model.eval()
        nlls = []
        for i in tqdm(range(self.n_samples), desc="Util Evaluating..."):
            batch = self.dataset[:, (i * block_size) : ((i + 1) * block_size)].to(
                model.device
            )
            with torch.no_grad():
                mod_out = model(batch, return_dict=True)
                # for newer transformers, model output could be simply a tuple
                lm_logits = getattr(mod_out, "logits", mod_out[0])
            shift_logits = lm_logits[:, :-1, :].contiguous().float()
            shift_labels = self.dataset[:, (i * block_size) : ((i + 1) * block_size)][
                :, 1:
            ]
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
            )
            neg_log_likelihood = loss.float() * block_size
            nlls.append(neg_log_likelihood)

        return torch.exp(torch.stack(nlls).sum() / (self.n_samples * block_size))
