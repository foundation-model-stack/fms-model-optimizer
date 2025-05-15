# Third Party
from transformers import BatchEncoding, BertModel, GraniteModel, LlamaModel
import pytest

# Local
from .test_model_utils import check_linear_dtypes, delete_config, load_state_dict
from fms_mo import qmodel_prep
from fms_mo.utils.aiu_utils import save_for_aiu


@pytest.fixture(autouse=True)
def delete_files():
    """
    Delete any known files lingering before starting test
    """
    delete_config("qcfg.json")
    delete_config("keys_to_save.json")
    delete_config("qmodel_for_aiu.pt")


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
        config_tiny_bert (BertConfig): Bert Tiny config
        input_tiny (BatchEncoding): Fake tiny input
        qcfg_bert (dict): Quantized config for Bert
    """
    # Quantize model and save state dict
    qmodel_prep(model_tiny_bert, input_tiny, qcfg_bert, use_dynamo=True)
    save_for_aiu(model_tiny_bert, qcfg=qcfg_bert, verbose=True)

    # Fetch saved state dict
    state_dict = load_state_dict()
    check_linear_dtypes(state_dict, bert_linear_names)


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
