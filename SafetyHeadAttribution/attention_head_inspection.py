import pandas as pd
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from lib.utils.custommodel import CustomLlamaModelForCausalLM
from lib.utils.batch_inference import surgery, inference
from lib.utils.get_model import get_model, get_tokenizer
import json
import os
import torch
import accelerate

#accelerator = accelerate.Accelerator()
model_path = "meta-llama/Llama-3.1-8B-Instruct"
data = "drug_abuse,weapons,banned_substance_CoT"
data_path = f"./SafetyHeadAttribution/exp_data/unused_beaver/{data}.csv"


model, accelerator = get_model(model_path, get_custom=True)

with open(
        f"./SafetyHeadAttribution/exp_res/sahara/llama-8B-search1-mean/drug_abuse,weapons,banned_substance_CoT.csv_0.jsonl",
        "r") as f:
    res = [json.loads(line) for line in f]
    attr = pd.Series(res[0])
    layer, head = attr.idxmax().split('-')

head_dim = model.model.layers[int(layer)].self_attn.head_dim
start_index = int(head) * head_dim
end_index = start_index + head_dim
torch.save(model.model.layers[int(layer)].self_attn.q_proj.weight[start_index:end_index, :],
         f"./SafetyHeadAttribution/exp_data/Llama-drug_abuse,weapons,banned_substance_CoT/standard_model_attention_{layer},{head}.pt")

mask_qkv = ['q']
scale_factor = 1e-4
mask_type = "scale_mask"

head_mask = {
    (layer, head): mask_qkv
    }
tokenizer = AutoTokenizer.from_pretrained(model_path)

surgery_path = f"./Llama-{data}"
# storage_path
surgery_model = surgery(model,
                        tokenizer,
                        head_mask,
                        mask_type,
                        scale_factor,
                        surgery_path)

head_dim = surgery_model.model.layers[int(layer)].self_attn.head_dim
start_index = int(head) * head_dim
end_index = start_index + head_dim
torch.save(surgery_model.model.layers[int(layer)].self_attn.q_proj.weight[start_index:end_index, :],
         f"./SafetyHeadAttribution/exp_data/Llama-drug_abuse,weapons,banned_substance_CoT/surg_model_attention_{layer},{head}.pt")

