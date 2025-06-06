# Third Party
from transformers import BatchEncoding, BertModel, GraniteModel, LlamaModel
import pytest

# Local
from .test_model_utils import check_linear_dtypes, delete_file, load_state_dict
from fms_mo import qmodel_prep
from fms_mo.utils.aiu_utils import save_for_aiu


@pytest.fixture(autouse=True)
def delete_files():
    """
    Delete any known files lingering before starting test
    """
    delete_file("qcfg.json")
    delete_file("keys_to_save.json")
    delete_file("qmodel_for_aiu.pt")


def test_save_model_bert(
    model_tiny_bert: BertModel,
    input_tiny: BatchEncoding,
    qcfg_bert: dict,
    bert_linear_names: list,
):
    """
    Save a BERT state dictionary and attempt to reload it to a fresh model

    Args:
        model_tiny_bert (BertModel): Bert Tiny Model
        input_tiny (BatchEncoding): Fake tiny input
        qcfg_bert (dict): Quantized config for Tiny Bert
        bert_linear_names (list): Names of linear layers for Bert
    """
    # Quantize model and save state dict
    qmodel_prep(model_tiny_bert, input_tiny, qcfg_bert, use_dynamo=True)
    save_for_aiu(model_tiny_bert, qcfg=qcfg_bert, verbose=True)

    # Fetch saved state dict
    state_dict = load_state_dict()
    check_linear_dtypes(state_dict, bert_linear_names)


def test_large_outlier_bert(
    model_tiny_bert: BertModel,
    input_tiny: BatchEncoding,
    qcfg_bert: dict,
    bert_linear_names: list,
):
    """
    Test if the recomputation mode increases standard deviation of a tensor with an outlier.

    Args:
        model_tiny_bert (BertModel): Bert Tiny Model
        input_tiny (BatchEncoding): Bert Tiny config
        qcfg_bert (dict): Quantized config for Tiny Bert
        bert_linear_names (list): Names of linear layers for Bert
    """
    # Third Party
    import torch

    # Break every tensor channel with a large magnitude outlier - should work for per tensor too
    for k, v in model_tiny_bert.state_dict().items():
        if k.endswith(".weight") and any(n in k for n in bert_linear_names):
            v[:, 0] = 1.21

    # Set recomputation for narrow weights and prep
    qcfg_bert["recompute_narrow_weights"] = True
    qmodel_prep(model_tiny_bert, input_tiny, qcfg_bert, use_dynamo=True)

    # Reduce perCh or perTensor
    stddev_dim = -1 if "perCh" in qcfg_bert["qw_mode"] else None

    # Qmax should break the quantization with an outlier to have skinny distribution
    layer2stdev: dict[str, torch.Tensor] = {}
    for k, v in model_tiny_bert.state_dict().items():
        if k.endswith(".weight") and any(n in k for n in bert_linear_names):
            # Collect perCh or perTensor std dev
            layer2stdev[k] = v.to(torch.float32).std(dim=stddev_dim)

    save_for_aiu(model_tiny_bert, qcfg=qcfg_bert, verbose=True)
    state_dict = load_state_dict()

    # Loaded model w/ recomputed SAWB should have widened channel quantization stdev
    for k, v in state_dict.items():
        if k.endswith(".weight") and any(n in k for n in bert_linear_names):
            stddev_model = layer2stdev.get(k)
            stddev_loaded = v.to(torch.float32).std(dim=stddev_dim)

            # SAWB stddev should be at least as good as Qmax stddev w/ outlier
            assert torch.all(stddev_loaded >= stddev_model)


def test_clip_vals_zero_bert(
    model_tiny_bert: BertModel,
    input_tiny: BatchEncoding,
    qcfg_bert: dict,
):
    """
    Test if uninitialized clip vals throws an error

    Args:
        model_tiny_bert (BertModel): Bert Tiny Model
        input_tiny (BatchEncoding): Bert Tiny config
        qcfg_bert (dict): Quantized config for Tiny Bert
    """
    # Turn off calibration -> clip vals are init as 0
    qcfg_bert["qmodel_calibration"] = 0
    qmodel_prep(model_tiny_bert, input_tiny, qcfg_bert, use_dynamo=True)

    with pytest.raises(ValueError):
        save_for_aiu(model_tiny_bert, qcfg=qcfg_bert, verbose=True)


def test_save_model_llama(
    model_tiny_llama: LlamaModel,
    input_tiny: BatchEncoding,
    qcfg_llama: dict,
    llama_linear_names: list,
):
    """
    Save a Llama state dictionary and attempt to reload it to a fresh model

    Args:
        model_tiny_llama (LlamaModel): Llama Tiny Model
        config_tiny_llama (LlamaConfig): Llama Tiny config
        input_tiny (BatchEncoding): Fake tiny input
        qcfg_llama (dict): Quantized config for Llama
    """
    # Quantize model and save state dict
    qmodel_prep(model_tiny_llama, input_tiny, qcfg_llama, use_dynamo=True)
    save_for_aiu(model_tiny_llama, qcfg=qcfg_llama, verbose=True)

    # Fetch saved state dict
    state_dict = load_state_dict()
    check_linear_dtypes(state_dict, llama_linear_names)


def test_save_model_granite(
    model_tiny_granite: GraniteModel,
    input_tiny: BatchEncoding,
    qcfg_granite: dict,
    granite_linear_names: list,
):
    """
    Save a Granite state dictionary and attempt to reload it to a fresh model

    Args:
        model_tiny_granite (GraniteModel): Granite Tiny Model
        config_tiny_granite (GraniteConfig): Granite Tiny config
        input_tiny (BatchEncoding): Fake tiny input
        qcfg_granite (dict): Quantized config for Granite
    """
    # Quantize model and save state dict
    qmodel_prep(model_tiny_granite, input_tiny, qcfg_granite, use_dynamo=True)
    save_for_aiu(model_tiny_granite, qcfg=qcfg_granite, verbose=True)

    # Fetch saved state dict
    state_dict = load_state_dict()
    check_linear_dtypes(state_dict, granite_linear_names)
