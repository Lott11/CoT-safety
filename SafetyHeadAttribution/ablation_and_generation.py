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

mask_qkv = ['q']
scale_factor = 1e-5
mask_type = "mean_mask"

#accelerator = accelerate.Accelerator()
model_path = "meta-llama/Llama-3.1-8B-Instruct"
data = "drug_abuse,weapons,banned_substance_CoT"
data_path = f"./SafetyHeadAttribution/exp_data/unused_beaver/{data}.csv"

default_generate_config = {
    "max_new_tokens": 256,
    "top_k": 1,
    "top_p": 0.9,
    "do_sample": False,
}


default_inference_config = {
    "use_conv": True,
    "store_path": f"./exp_res/{data}/",
    "batch_size": 2,
}

with open(
        f"./SafetyHeadAttribution/exp_res/sahara/llama-8B-search1-mean/drug_abuse,weapons,banned_substance_CoT.csv_0.jsonl",
        "r") as f:
    res = [json.loads(line) for line in f]
    attr = pd.Series(res[0])
    layer, head = attr.idxmax().split('-')

head_mask = {
    (layer, head): mask_qkv
    }

model, accelerator = get_model(model_path, get_custom=True)
model.to(device="cuda:0")
tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.padding_side = "left"
tokenizer.pad_token = tokenizer.eos_token

df = pd.read_csv(data_path)
res_store_file = f"{default_inference_config['store_path']}/Llama-{data}/"
os.makedirs(res_store_file, exist_ok=True)
res_store_file += "fowardpass_mean_generation.json"
with open(res_store_file, "w+") as fp:
    for _,i in df.iterrows():
      inputs = tokenizer([i["input"]], truncation=True, padding = True, return_tensors="pt").to("cuda:0")

      generate_ids = model.generate(inputs.input_ids, attn_mask = , head_mask = head_mask, mask_type = mask_type, scale_factor = scale_factor,
                            mask_para = True, head_dim = model.config.hidden_size // model.config.num_attention_heads, max_new_tokens =256,
                            pad_token_id = tokenizer.pad_token_id)
      output = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)

      #with open(f"exp_data/beaver_results/{cat}_outputs.json", "w") as fp:
      fp.write(json.dumps(output) + '\n')



'''
surgery_model = surgery(model,
                        tokenizer,
                        head_mask,
                        mask_type,
                        scale_factor,
                        surgery_path)



inference(surgery_path,data_path,accelerator, generate_cfg=default_generate_config,
         inference_cfg=default_inference_config)

head_mask = {}
for i in np.arange(3):
    with open(f"./SafetyHeadAttribution/exp_res/sahara/privacy_violation-llama-8B-search3-mean/privacy_violation_CoT.csv_{i}.jsonl", "r") as f:
        res = [json.loads(line) for line in f]
    attr = pd.Series(res[0])
    layer, head = attr.idxmax().split('-')
    head_mask[(layer, head)] =  mask_qkv


data = "privacy_violation_CoT"
data_path = f"./SafetyHeadAttribution/exp_data/unused_beaver/{data}.csv"

default_inference_config = {
    "use_conv": True,
    "store_path": f"./exp_res/{data}/",
    "batch_size": 2,
}

surgery_path = f"./Llama-{data}"
# storage_path
surgery_model = surgery(model,
                        tokenizer,
                        head_mask,
                        mask_type,
                        scale_factor,
                        surgery_path)

inference(surgery_path,data_path,accelerator, generate_cfg=default_generate_config,
         inference_cfg=default_inference_config)
'''