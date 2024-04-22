from utils import *
from transformers import AutoModel, AutoTokenizer
import torch
import sys
import pickle

prune_tasks = (
    "paws_en",
    "hellaswag",
    "arc_easy",
    "blimp_ellipsis_n_bar_1",
    "blimp_irregular_plural_subject_verb_agreement_1"
)
path = "/home/data_shares/mapillary/prompts/opt-13b"
model_name="facebook/opt-13b"
model = AutoModel.from_pretrained(model_name, output_attentions=True)
model.eval()
tokenizer = AutoTokenizer.from_pretrained(model_name)
for task in prune_tasks:
    
    prompts = get_prompts_from_file(task)
    attention_maps = get_batched_attention(prompts, model, tokenizer, first_token=True)
    with open(f"{path}/{task}_attention_maps.pkl", "wb") as f:
        pickle.dump(attention_maps, f)
