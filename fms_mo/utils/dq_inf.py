import torch
import fms_mo
from fms_mo.quant.quantizers import to_fp8_scaled_perCh as fp8
from huggingface_hub import save_torch_state_dict
import json
import os 
import glob
from fms_mo.utils.qconfig_utils import get_recipe
from safetensors.torch import load_file, save_file
from torch import nn

def save_vllm_fp8(model: nn.Module, qcfg: dict, tokenizer = None, folder: str = None):   
    """
    Function to save fms_mo fp8 checkpoint in vllm fp8 format
    """

    st_dict={}
 
    for k,v in model.state_dict().items():
        if k[-11:] == "proj.weight":
            weight, scale = fp8(v,emulate=False)
            st_dict[k]= weight

            if k[:-7] in qcfg["qskip_layer_name"]:
                pass
            else:
                st_dict[k + "_scale"] = 1/scale 

        elif k[-6:] == "weight":
            st_dict[k]=v
        else:
            pass

    config = model.config.to_dict()

    #TO DO: To support multiple recipes, check qconfig arguments and update data loaded from quant.json
    data = get_recipe('quant')
    
    config.update(data)

    save_torch_state_dict(st_dict, folder)

    tokenizer.save_pretrained(folder)

    with open(folder+'/config.json', 'a') as f:
        json.dump(config, f, indent=4)
    


def find_file_glob(pattern: str , search_path: str):
    """
    Finds files matching a pattern within a directory and its subdirectories.
    """
    # Use '**' for recursive search in modern Python versions (3.5+)
    full_pattern = os.path.join(search_path, '**', pattern)
    found_files = glob.glob(full_pattern, recursive=True)
    return sorted(found_files)

def load_fp8_vllm(model: nn.Module = None, checkpoint: str=None):
    """
    Function to help load vllm fp8 checkpoint into fms_mo
    """   

    merged_files_dict={}

    files = find_file_glob('*.safetensors',checkpoint)

    model_dict = model.state_dict()

    for file in files:
        merged_files_dict = load_file(file)

        for k,v in merged_files_dict.items():

            if k[-11:] == "proj.weight":
                scale = merged_files_dict[k+ "_scale"].reshape(-1,1)
                model_dict[k]= merged_files_dict[k].to(torch.float16) * scale

            elif k[-6:] == "weight":
                model_dict[k]=v

            else:
                pass

    return model




