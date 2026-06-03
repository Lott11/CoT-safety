import pandas as pd
import json
from lib.Sahara.attribution import safety_head_attribution
from lib.SHIPS.get_ships import SHIPS
import os
from accelerate import Accelerator
import tqdm
import numpy as np


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

default_search_cfg = {
    "search_step": 5,
    "mask_qkv": ['q'],
    "scale_factor": 1e-5,
    "mask_type": "scale_mask"
}

model_path = "mistralai/Mistral-7B-Instruct-v0.3"

'''for cat in cat_list:
    data_path = f"./SafetyHeadAttribution/exp_data/unused_beaver/{cat}_CoT.csv"
    storage_path = f"./SafetyHeadAttribution/exp_res/sahara/{cat}-llama-search5-scale1e-5/"
    if not os.path.exists(storage_path):
        os.makedirs(storage_path)

    safety_head_attribution(
            model_name=model_path,
            data_path=data_path,
            search_cfg=default_search_cfg,
            storage_path=storage_path,
            device='cuda:0'
    )'''


for cat in cat_list:

    test = SHIPS(f"./SafetyHeadAttribution/exp_data/unused_beaver/{cat}_CoT.csv",
                model_path,
                 mask_cfg=mask_config,
                 device="cuda:0")

    test.model.to("cuda:0")

    '''top_heads = {}
    for i in range(5):
        with open(
                f"./SafetyHeadAttribution/exp_res/sahara/{cat}-llama-search5-scale1e-5/{cat}_CoT.csv_{i}.jsonl",
                "r") as f:
            res = [json.loads(line) for line in f]
            attr = pd.Series(res[0])
            largest = attr.nlargest(6).index.tolist()
            indx = 0
            layer, head = largest[indx].split('-')
            # Loop makes sure that we dont include gibberish-heads or duplicate any heads (0,4 and 0,13 in mistral)
            #while (layer == "0" and (head == "4" or head == "13")) or str(layer) + "-" + str(head) in top_heads.keys():
            #    indx += 1
            #    layer, head = largest[indx].split('-')
            top_heads[str(layer) + "-" + str(head)] = i'''
    #Choosing random heads to ablate, that have not been found by sahara and are not known consistency-heads
    taken = np.array([#[0, 21],
                      #[0, 12],
                      #[31, 2],
                      #[0, 9],
                      #[31, 3],
                      #[0, 7],
                      #[31, 1],
                      [0, 4],
                      [0, 13]])
    with open(f"SafetyHeadAttribution/exp_data/ships-mistral-{cat}_scale1e-4.jsonl",
              "r") as f:
        sample_level_heads = [json.loads(file) for file in f]
    with open(f"./SafetyHeadAttribution/exp_res/{cat}-mistral-search5-scale1e-5/sample_ships_ablated.jsonl",
              "w+") as fp:
        for i in sample_level_heads:
            ships_heads = list(list(i.values())[0].keys())
            layer, head = ships_heads[0].split('-')
            indx = 0
            top_heads = {}
            while (layer == "0" and (head == "4" or head == "13")) or str(layer) + "-" + str(head) in top_heads.keys():
                indx += 1
                layer, head = ships_heads[indx].split('-')
            top_heads[str(layer) + "-" + str(head)] = 1
            mask_n = 1
        #for mask_n in range(1, 6):

            #for idx, data in tqdm.tqdm(test.data.iterrows()):
            generated_text = test.ships_generate(input_text=list(i.keys())[0],
                                                top_ships=top_heads,
                                                top_k=1,
                                                mask_num=mask_n,
                                                max_new_tokens=256)
            output = {list(i.keys())[0]: generated_text[len(list(i.keys())[0]):], "head" : str(layer) + "-" + str(head)}
            fp.write(json.dumps(output) + '\n')