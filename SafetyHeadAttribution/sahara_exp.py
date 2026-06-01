import pandas as pd
import json
from lib.Sahara.attribution import safety_head_attribution
#from lib.SHIPS.get_ships import SHIPS
import os
#from accelerate import Accelerator

cat_list = [#'animal_abuse',
            #'child_abuse',
            #'controversial_topics,politics',
            #'discrimination,stereotype,injustice',
            'drug_abuse,weapons,banned_substance',
            #'financial_crime,property_crime,theft',
            #'hate_speech,offensive_language',
            #'misinformation_regarding_ethics,laws_and_safety',
            #'non_violent_unethical_behavior',
            'privacy_violation',
            #'self_harm',
            #'sexually_explicit,adult_content',
            #'terrorism,organized_crime',
            #'violence,aiding_and_abetting,incitement'
            ]

default_search_cfg = {
    "search_step": 5,
    "mask_qkv": ['q'],
    "scale_factor": 1e-4,
    "mask_type": "scale_mask"
}

model_path = "meta-llama/Llama-3.1-8B-Instruct"
for cat in cat_list:
    data_path = f"./SafetyHeadAttribution/exp_data/unused_beaver/{cat}_CoT.csv"
    storage_path = f"./SafetyHeadAttribution/exp_res/sahara/{cat}-mistral-search5-scale1e-4/"
    if not os.path.exists(storage_path):
        os.makedirs(storage_path)

    safety_head_attribution(
            model_name=model_path,
            data_path=data_path,
            search_cfg=default_search_cfg,
            storage_path=storage_path,
            device='cuda:0'
    )