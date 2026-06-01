import numpy as np
import torch
import pandas as pd

from load_script import load_model
from transformers import TrainingArguments, Trainer, DataCollatorWithPadding, AutoTokenizer
import evaluate
from datasets import load_dataset, Dataset, Sequence, ClassLabel

cats = np.array(["ADDING_INTENTION",
                 "REASONING_ABOUT_SAFETY",
                 "OTHER",
                 "HARMFUL_RESPONSE"])

model_name = "google/gemma-2-2b-it"

model, _ = load_model(
            model_name,
            lora_alpha=109,
            cache_dir='/home/vlh769/.cache/huggingface/transformers/',
            lora_r=51,
            finetuned_path=None,
            quantize=False,
            peft_task="SEQ_CLS",
            problem_type="multi_label_classification",
            num_labels=4
        )

cache_dir='/home/vlh769/.cache/huggingface/transformers/'
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding="longest")

tokenized_train = load_dataset("json", data_files="finetuning/tokenized_train.jsonl")["train"]
tokenized_val = load_dataset("json", data_files="finetuning/tokenized_val.jsonl")["train"]

metric = evaluate.load("f1", "multilabel")
sigmoid = torch.nn.Sigmoid()
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    print("predictions: ", predictions)
    print("eval_pred: ", eval_pred, "\n\n")
    predictions = sigmoid(torch.tensor(predictions)).round()
    output_dict = metric.compute(predictions=predictions, references=labels, average="macro")
    print(output_dict)
    #for key in output_dict.keys():
    #    output_dict[key] = output_dict[key].tolist()
    return output_dict
training_args = TrainingArguments(
    output_dir="exp_data/judge_results/finetuned_judge",
    learning_rate = 5.798439376667845e-05,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    num_train_epochs=5,
    weight_decay=0.01,
    eval_strategy="steps",
    save_strategy="steps",
    eval_steps=25,
    load_best_model_at_end=True,
    gradient_accumulation_steps = 100,
    metric_for_best_model="eval_f1",
)

def preprocess_logits(logits, labels):
    #print("logits: ", logits)
    return logits

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    #model_init = model_init,
    processing_class=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    preprocess_logits_for_metrics=preprocess_logits,
)

trainer.train()
trainer.save_model("finetuning")


'''
def compute_objective(metrics):
    return metrics["eval_f1"]

def optuna_hp_space(trial):
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
        "lora_alpha": trial.suggest_int("lora_alpha", 32, 128, log=True),
        "lora_r": trial.suggest_int("lora_r", 32, 128, log=True),
    }

best_trials = trainer.hyperparameter_search(
    direction=["maximize"],
    backend="optuna",
    hp_space=optuna_hp_space,
    n_trials=9,
    study_name="gemma-2-2b-it-study",
    compute_objective=compute_objective,
)

print(best_trials)
'''