import pandas as pd
import json
from lib.Sahara.attribution import safety_head_attribution
from lib.SHIPS.get_ships import SHIPS
import os
from accelerate import Accelerator


mask_config = {
    "mask_qkv": ['q'],
    "scale_factor": 1e-5,
    "mask_type": "scale_mask",
}
cat_list = [#'animal_abuse',
            #'child_abuse',
            #'controversial_topics,politics',
            #'discrimination,stereotype,injustice',
            #'drug_abuse,weapons,banned_substance',
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

test_accelerator = Accelerator()

for cat in cat_list:
    test = SHIPS(f"./SafetyHeadAttribution/exp_data/unused_beaver/{cat}_CoT.csv",
                "mistralai/Mistral-7B-Instruct-v0.3",
                 mask_cfg=mask_config,
                 device="cuda:0")

    test.model.to("cuda:0")
    test.main(f"./SafetyHeadAttribution/exp_data/ships-llama-{cat}_scale1e-5.jsonl",
              generate_flag=False)

    #for mask_n in range(1, 2):
    #    test.ships_test(f"./SafetyHeadAttribution/exp_data/ships-llama-{cat}.jsonl", top_k=1, mask_num=mask_n, max_new_tokens=256)