from huggingface_hub import save_torch_state_dict
import json
import os 
import glob
from safetensors.torch import load_file
from torch import nn
import torch
from fms_mo.utils.qconfig_utils import get_recipe
from fms_mo.quant.quantizers import to_fp8_scaled_perCh

def check_quantization_setting(inference :dict = None):
    """
    function checks if the checkpoint is from fp8 quantization
    """
    status= False
    if inference["config_groups"]["group_0"]["input_activations"]["num_bits"]== 8 \
        and inference["config_groups"]["group_0"]["weights"]["num_bits"] == 8 \
        and inference["config_groups"]["group_0"]["weights"]["type"] == "float" \
        and inference["config_groups"]["group_0"]["input_activations"]["type"] == "float":

        status = True
    return status

#def rename_fms_dict_to_vllm_dict (model_dict : dict= None, qcfg : dict = None):
def rename_fms_dict_to_vllm_dict (model_dict : dict= None):
    """
    Function to rename the dict in fms_mo format to vllm_format.
    """
    st_dict={}
    fms_dict={}
    keys = model_dict.keys()
    print(keys)
    count=0
    for k,v in model_dict.items():
        if ".weight" in k:
            count+=1
            key= k.split("weight")[0]
            if key+"quantize_weight.scale" in keys:
                weight, scale = to_fp8_scaled_perCh(v,emulate=False)
                st_dict[key+"weight"]= weight
                st_dict[key + "weight_scale"] = 1/scale
            else:
                st_dict[k]= v
        else:
            fms_dict[k] = v
    return st_dict, fms_dict

def update_config(model_config_file : dict = None, qcfg : dict = None):
    """
    Function to update the model config file with quantization configuration
    """
    data = get_recipe("quant")
    if "perCh" not in qcfg["qw_mode"]:
        data["quantization_config"]["config_groups"]["group_0"]["weights"] = \
        "{num_bits: 8, type: float, symmetric: true, strategy: tensor}"

    model_config_file.update(data)
    return model_config_file

def save_vllm_fp8(model: nn.Module, qcfg: dict, tokenizer = None, folder: str = None):
    """
    Function to save fp8 DQ model in vllm fp8 format
    """
    model_dict = model.state_dict()
    vllm_dict, fms_dict = rename_fms_dict_to_vllm_dict(model_dict=model_dict)
    config = model.config.to_dict()
    config = update_config( config, qcfg)

    save_torch_state_dict(vllm_dict, folder)
    save_torch_state_dict(fms_dict, folder, filename_pattern="fms_mo{suffix}.safetensors")
    tokenizer.save_pretrained(folder)

    with open(folder+"/config.json", "w+") as f:
        json.dump(config, f, indent=4)
    
def convert_fms_mo_to_vllm_fp8_format(checkpoint : str = None, folder: str = None):
    """
    Function to convert fp8 fms_mo DQ model checkpoint to vllm fp8 format
    """
    folder = checkpoint+"/" + folder
    if os.path.isdir(folder):
        print(f"The folder '{folder}' exists.")
    else:
        os.mkdir(folder)
        print(f"The folder '{folder}' created.")

    qcfg = get_recipe(checkpoint+"/qcfg")
    config = get_recipe(checkpoint+"/config")
    files = find_file_glob("model-*",checkpoint)
    merged_files_dict={}

    for file in files:
        temp_dict = load_file(file)
        merged_files_dict.update(temp_dict)

    vllm_dict, fms_dict = rename_fms_dict_to_vllm_dict(merged_files_dict)
    config = update_config(config, qcfg)

    save_torch_state_dict(vllm_dict, folder)
    save_torch_state_dict(fms_dict, folder, filename_pattern="fms_mo{suffix}.safetensors")
    with open(folder+"/config.json", "w+") as f:
        json.dump(config, f, indent=4)

def find_file_glob(pattern: str , search_path: str):
    """
    Finds files matching a pattern within a directory and its subdirectories.
    """
    # Use '**' for recursive search in modern Python versions (3.5+)
    full_pattern = os.path.join(search_path, "**", pattern)
    found_files = glob.glob(full_pattern, recursive=True)
    return sorted(found_files)

def convert_fp8_vllm_dict_to_fms_mo_dict(checkpoint: str=None, output_dir : str=None):
    """
    Function to help convert vllm fp8 checkpoint into fms_mo fp8 format
    """
    merged_files_dict={}
    files = find_file_glob("model-*",checkpoint)
    for file in files:
        temp_dict = load_file(file)
        merged_files_dict.update(temp_dict)

    fms_mo_dict = rename_vllm_dict_to_fms_mo(merged_files_dict)
    save_torch_state_dict(fms_mo_dict, output_dir)

def rename_vllm_dict_to_fms_mo(vllm_dict : dict = None):
    """
    Function to help rename vllm dict format to fms_mo dict format
    """
    fms_mo_dict ={}
    for k,v in vllm_dict.items():
        if "weight_scale" in k:
            key = k.split("weight")[0]
            fms_mo_dict[key+"weight"]=vllm_dict[key+"weight"].to(torch.float16) *v
            fms_mo_dict[k]= v
        else:
            key = k.split("weight")[0]
            if key+"weight_scale" in vllm_dict.keys():
                pass
            else:
                fms_mo_dict[k]= v
    return fms_mo_dict

def convert_fp8_vllm_to_fms_mo(model: nn.Module = None):
    """
    Function to help convert fp8 vllm model dict format to fms_mo fp8 format
    """
    model_dict = model.state_dict()
    fms_dict = rename_vllm_dict_to_fms_mo(model_dict)
    model = model.to(torch.float16)
    model.load_state_dict(fms_dict)
    return model
