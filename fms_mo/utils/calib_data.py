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

"""Data preparation functions for GPTQ and directQuant. Get data from different datasets and
tokenize.

This code is modified from: https://github.com/IST-DASLab/gptq/blob/main/zeroShot/datautils.py
"""

# Standard
from pathlib import Path
import json
import os
import random

# Third Party
from datasets import load_dataset, load_from_disk
from transformers import BatchEncoding
import datasets
import torch


def return_tokenized_samples(nsamples, trainenc, seqlen, sequential=False) -> dict[str, torch.int]:
    """Randomly crop nsamples sequence from trainenc, each with the length of seqlen.
    see below functions, e.g. get_wikitext2() for more details.
    """
    traindataset = {
        "input_ids": torch.zeros(size = (nsamples, seqlen), dtype = torch.int), 
        "attention_mask": torch.zeros(size = (nsamples, seqlen), dtype = torch.int)
    }
    i = 0

    for k in range(nsamples):
        if not sequential:
            i = random.randint(0, len(trainenc.input_ids) - seqlen - 1)

        j = i + seqlen
        inp = trainenc.input_ids[i:j]
        mask = trainenc.attention_mask[i:j]
        traindataset["input_ids"][k] = torch.tensor(inp)
        traindataset["attention_mask"][k] = torch.tensor(mask)

        i = j

    return traindataset


def get_wikitext2(
    nsamples, seed, seqlen, tokenizer, sequential=False, gptq_style=False
):
    """Prepare data for GPTQ using wikitext2 dataset.

    Args:
        nsamples (int): number of samples needed
        seed (int): random seed
        seqlen (int): sequence length
        tokenizer (Tokenizer): Tokenizer to be used
        sequential (bool, optional): whether to crop samples sequentially

    Returns:
        list: tokenized random cropped samples
    """
    random.seed(seed)
    traindata = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    testdata = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")

    if gptq_style:
        traindata = "".join([" \n" if s == "" else s for s in traindata["text"]])
    else:
        traindata = "\n\n".join(traindata["text"])

    trainenc = tokenizer(traindata)
    testenc = tokenizer("\n\n".join(testdata["text"]), return_tensors="pt")

    traindataset = return_tokenized_samples(
        nsamples, trainenc, seqlen, sequential=sequential
    )
    testenc = {
                "input_ids": testenc["input_ids"], 
                "attention_mask": testenc["attention_mask"]
    }

    return traindataset, testenc


def get_ptb(nsamples, seed, seqlen, tokenizer, sequential=False, gptq_style=False):
    """Prepare data for GPTQ using PTB dataset.

    Args:
        nsamples (int): number of samples needed
        seed (int): random seed
        seqlen (int): sequence length
        tokenizer (Tokenizer): Tokenizer to be used
        sequential (bool, optional): whether to crop samples sequentially

    Returns:
        list: tokenized random cropped samples
    """
    random.seed(seed)

    traindata = load_dataset("ptb_text_only", "penn_treebank", split="train")
    valdata = load_dataset("ptb_text_only", "penn_treebank", split="validation")
    if gptq_style:
        traindata = "".join([" \n" if s == "" else s for s in traindata["sentence"]])
    else:
        traindata = "\n\n".join(traindata["sentence"])

    trainenc = tokenizer(traindata)
    testenc = tokenizer("\n\n".join(valdata["sentence"]),return_tensors="pt")

    traindataset = return_tokenized_samples(nsamples, trainenc, seqlen, sequential)
    testenc = {
                "input_ids": testenc["input_ids"], 
                "attention_mask": testenc["attention_mask"]
    }

    return traindataset, testenc


def get_c4_train(nsamples, seed, seqlen, tokenizer, sequential=False):
    """Prepare data for GPTQ using C4 dataset.

    Args:
        nsamples (int): number of samples needed
        seed (int): random seed
        seqlen (int): sequence length
        tokenizer (Tokenizer): Tokenizer to be used
        sequential (bool, optional): whether to crop samples sequentially

    Returns:
        list: tokenized random cropped samples
    """
    random.seed(seed)
    traindata = load_dataset(
        "allenai/c4",
        data_files={"train": "en/c4-train.00001-of-01024.json.gz"},
        split="train",
    )
    valdata = load_dataset(
        "allenai/c4",
        data_files={"validation": "en/c4-validation.00000-of-00008.json.gz"},
        split="validation",
    )

    testenc = tokenizer("\n\n".join(valdata["text"]),return_tensors="pt")

    trainloader ={
        "input_ids": torch.zeros(size = (nsamples, seqlen), dtype = torch.int), 
        "attention_mask": torch.zeros(size = (nsamples, seqlen), dtype = torch.int)
    }
    for k in range(nsamples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]["text"])
            if len(trainenc.input_ids) >= seqlen:
                break
        if not sequential:
            i = random.randint(0, max(len(trainenc.input_ids) - seqlen - 1, 0))
        j = i + seqlen
        inp = trainenc.input_ids[i:j]
        mask = trainenc.attention_mask[i:j]
        trainloader["input_ids"][k] = torch.tensor(inp)
        trainloader["attention_mask"][k] = torch.tensor(mask)
        j = i

    testdataset = {
        "input_ids": testenc["input_ids"],
        "attention_mask": testenc["attention_mask"],
    }

    return trainloader, testdataset


def get_c4_new(nsamples, seed, seqlen, tokenizer):
    """Prepare data for GPTQ using C4 dataset.

    Args:
        nsamples (int): number of samples needed
        seed (int): random seed
        seqlen (int): sequence length
        tokenizer (Tokenizer): Tokenizer to be used
        sequential (bool, optional): whether to crop samples sequentially

    Returns:
        list: tokenized random cropped samples
    """
    traindata = load_dataset(
        "allenai/c4",
        data_files={"train": "en/c4-train.00000-of-01024.json.gz"},
        split="train",
    )
    valdata = load_dataset(
        "allenai/c4",
        data_files={"validation": "en/c4-validation.00000-of-00008.json.gz"},
        split="validation",
    )
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]["text"], return_tensors="pt")
            if trainenc.input_ids.shape[1] >= seqlen:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    valenc = tokenizer(" ".join(valdata[:1100]["text"]), return_tensors="pt")
    valenc = valenc.input_ids[:, : (256 * seqlen)]

    return trainloader, valenc


def get_self_instruct_starcoder(
    nsamples, seed, seqlen, tokenizer, split_name="curated"
):  # pylint: disable=unused-argument
    """Prepare data for GPTQ using starcoder dataset.

    Args:
        nsamples (int): number of samples needed
        seed (int): random seed
        tokenizer (Tokenizer): Tokenizer to be used
        sequential (bool, optional): whether to crop samples sequentially

    Returns:
        list: tokenized random cropped samples
    """
    cr_dataset = load_dataset("codeparrot/self-instruct-starcoder", split=split_name)

    eval_dataset = tokenizer(" ".join(cr_dataset[:]["output"]), return_tensors="pt")
    eval_dataset = {
                    "input_ids": eval_dataset["input_ids"], 
                    "attention_mask": eval_dataset["attention_mask"]
    }

    cr_dataset.shuffle(seed)
    nsamples = min(nsamples, len(cr_dataset))

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    trainloader = {
        "input_ids": torch.zeros(size = (nsamples,seqlen), dtype=torch.int), 
        "attention_mask": torch.zeros(size = (nsamples,seqlen), dtype=torch.int)
    }
    for k in range(nsamples):
        tokenized = tokenizer(
            cr_dataset[k]["output"], return_tensors="pt",
            padding="max_length", max_length = seqlen
        )
        trainloader["input_ids"][k] = tokenized.input_ids.squeeze(0)
        trainloader["attention_mask"][k] = tokenized.attention_mask.squeeze(0)

    return trainloader, eval_dataset


def get_cobol_java_supervised(
    nsamples, seed, seqlen=8192, tokenizer = "", split_name="both", file_path=None
):
    """Prepare data for GPTQ using cobol/java dataset.

    Args:
        nsamples (int): number of samples needed
        seed (int): random seed
        seqlen (int): sequence length
        tokenizer (Tokenizer): Tokenizer to be used
        sequential (bool, optional): whether to crop samples sequentially

    Returns:
        list: tokenized random cropped samples
    """
    assert file_path, "Please provide a valid file path to COBOL/JAVA dataset"

    random.seed(seed)
    with open(file_path, encoding="utf-8") as f:
        raw_data = f.readlines()

    data_dict_array = [json.loads(line) for line in raw_data]

    eval_dataset = tokenizer(data_dict_array["content"], return_tensors="pt")
    eval_dataset = {
                    "input_ids": eval_dataset["input_ids"], 
                    "attention_mask": eval_dataset["attention_mask"]
    }

    random.shuffle(data_dict_array)

    nsamples = min(nsamples, len(data_dict_array))

    trainloader = {
        "input_ids": torch.zeros(size = (nsamples,seqlen), dtype=torch.int), 
        "attention_mask": torch.zeros(size = (nsamples,seqlen), dtype=torch.int)
    }
    added_ex = 0

    while added_ex < nsamples:
        sp_idx = random.randint(0, len(data_dict_array) - 1)
        inputs = data_dict_array[sp_idx]["content"]

        if not len(inputs) > seqlen:
            continue

        if split_name != "both":
            inp_split = inputs.split("## Java:")
            if split_name == "cobol":
                inputs = inp_split[0]
            elif split_name == "java":
                inputs = "## Java:" + inp_split[1]
            else:
                raise RuntimeError(
                    "split_name should be one of ['both', 'java', 'cobol']"
                )

        if len(inputs) > seqlen:
            i = random.randint(0, len(inputs) - seqlen - 1)
        else:
            i = 0
        j = i + seqlen
        inputs = inputs[i:j]

        tokenized = tokenizer(inputs, return_tensors="pt")
        trainloader["input_ids"][added_ex] = tokenized.input_ids.squeeze(0)
        trainloader["attention_mask"][added_ex] = tokenized.attention_mask.squeeze(0)

        added_ex += 1

    return trainloader, eval_dataset


def get_tokenized_data(
    name,
    nsamples=128,
    seqlen=2048,
    tokenizer="",
    seed=0,
    gptq_style=False,
    path_to_save=None,
    field_name=None,
):
    """Convenient function to get data. Default to get_wikitext2."""

    # Option 1: User provide a dataset from disk, only need to tokenize and format it.
    if Path(name).is_dir() and len(list(Path(name).glob("*.arrow"))) > 0:
        assert field_name, (
            "User try to tokenize a custom dataset from disk, but did not provide field name.\n"
            "Assuming the data to be loaded is derived from datasets.Dataset.save_to_disk()."
        )
        # Assume it's training data. Make sure not to overwrite the existing tokenized data.
        traindata = load_from_disk(name)

        if gptq_style:
            traindata = "".join(
                [" \n" if s == "" else s for s in traindata[field_name]]
            )
        else:
            traindata = "\n\n".join(traindata[field_name])

        trainenc = tokenizer(traindata)
        traindataset = return_tokenized_samples(nsamples, trainenc, seqlen)

        if path_to_save:
            datasets.Dataset.from_list(traindataset).save_to_disk(
                path_to_save + "_train"
            )
        else:
            return traindataset

    # Option 2: Fetch from public dataset, only a few commonly used ones here. Add more if needed.
    if gptq_style and not ("wiki" in name or "ptb" in name):
        raise NotImplementedError(
            "Dataset {name} with GPTQ style is not implemented yet. Please refer to "
            "get_wikitext2() and implement for your own dataset if needed"
        )

    if "mix" in name:
        wiki_train, _ = get_wikitext2(
            nsamples // 3, seed, seqlen, tokenizer, gptq_style=gptq_style
        )
        ptb_train, _ = get_ptb(
            nsamples // 3, seed, seqlen, tokenizer, gptq_style=gptq_style
        )
        c4_train, _ = get_c4_train(nsamples // 3, seed, seqlen, tokenizer)
        train = wiki_train + ptb_train + c4_train

        return train, None

    get_data_func = get_wikitext2
    if "wiki" in name:
        traindataset, testdataset = get_wikitext2(
            nsamples, seed, seqlen, tokenizer, gptq_style=gptq_style
        )
    elif "ptb" in name:
        traindataset, testdataset = get_ptb(
            nsamples, seed, seqlen, tokenizer, gptq_style=gptq_style
        )
    elif "c4" in name:
        get_data_func = get_c4_new if "new" in name else get_c4_train
        traindataset, testdataset = get_data_func(
            nsamples,
            seed,
            seqlen,
            tokenizer,
        )
    elif "starcoder" in name:
        traindataset, testdataset = get_self_instruct_starcoder(
            nsamples, seed, seqlen, tokenizer, split_name="curated"
        )
    elif "java" in name:
        traindataset, testdataset = get_cobol_java_supervised(
            nsamples, seed, seqlen, tokenizer,
        )
    else:
        raise NotImplementedError(
            f"Dataset {name} is not implemented yet. Please refer to get_wikitext2() and implement"
            "for your own dataset if needed."
        )

    if path_to_save:
        datasets.Dataset.from_dict(traindataset).save_to_disk(path_to_save + "_train")
        if isinstance(testdataset, BatchEncoding):
            if not os.path.exists(path_to_save + "_test"):
                os.mkdir(path_to_save + "_test")
            torch.save(testdataset, path_to_save + "_test/testdataset.pt")
        elif isinstance(testdataset, list):
            datasets.Dataset.from_list(testdataset).save_to_disk(path_to_save + "_test")
        elif isinstance(testdataset, dict):
            datasets.Dataset.from_dict(testdataset).save_to_disk(path_to_save + "_test")
        else:
            raise NotImplementedError(
                "Test dataset is of unknown format. Please check your implementation"
            )
    else:
        return traindataset, testdataset
