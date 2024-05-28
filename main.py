from utils import *
from prune import duplicate_prune_model, prune
from lm_eval.models import huggingface
import eval_f
import eval
import sys
import pandas as pd 
from transformers import AutoModel, AutoTokenizer
import torch
from collections import defaultdict
from pathlib import Path
import os
import gc

SAVE_MODEL=False

def main(model_name, path, metric, group_metric, prune_task, prune_method,):
    tasks=(
    "paws_en",
    "hellaswag",
    "blimp_ellipsis_n_bar_1",
    "blimp_irregular_plural_subject_verb_agreement_1",
    "arc_easy",
)
    prune_percents=(
        0, 
        0.1,
        0.2,
        0.25,
        0.3,
        0.4,
        0.5,
        0.75,
    )
    prompts = get_prompts_from_file(prune_task)
    for prune_percent in prune_percents:
        model = AutoModel.from_pretrained(model_name, output_attentions=True)
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model_basename = os.path.basename(model_name)
        model = duplicate_prune_model(prompts=prompts, 
                                        path=path,
                                        model=model,
                                        model_name=model_name,
                                        tokenizer=tokenizer,
                                        prune_percent=prune_percent,
                                        prune_task=prune_task,
                                        metric=metric,
                                        group_metric=group_metric,
                                        prune_method=prune_method,
                                        verbose=True,
                                        save=SAVE_MODEL,)
        print("Pruning done for: ", model_name, prune_method, metric, prune_task, prune_percent)
        # path_log = Path("pruning_logs") / model / prune_method / prune_task / metric / prune_percent / "pruning_log.txt"
        prune_method_path = prune_method # + "_mask"
        metric_path = metric + "_" + group_metric
        model_path = Path(model_basename) / prune_method_path / prune_task / metric_path / str(prune_percent)
        # layers, heads = get_model_layers_and_heads(model.config)

        model_args = {
                        "pretrained": str(model_name),
                        "dtype":"float16",
                        "device": "cuda:0"
                        }   
        model_lm = huggingface.HFLM(**model_args)
        if model_basename == "opt-13b":
            model_lm.model.model = model
        elif model_basename == "bloom-7b1":
            model_lm.model.transformer = model
        eval_f.evaluate(model_lm, tasks, model_path)    
        print("Evaluted: ", model_name, prune_method, metric_path, prune_task, prune_percent)
        ## Make sure models aren't hanging around when no longer needed
        del model_lm
        model.to('cpu')
        del model
        gc.collect()
        
        if prune_percent == 0:
            continue
        ### masked pruning
        print("Starting mask pruning")
        model = AutoModel.from_pretrained(model_name, output_attentions=True)
        model.eval()
        prune_method_path = prune_method + "_mask"
        model_path = Path(model_basename) / prune_method_path / prune_task / metric_path / str(prune_percent)
        pruning_log = []
        pruning_dict = defaultdict(list)
        metric_path = metric + "_" + group_metric
        path_log = Path("pruning_logs") / model_basename / prune_method / prune_task / metric_path / str(prune_percent) / "pruning_log.txt"

        with open (path_log, "r") as f:
            lines = f.readlines()
            for line in lines:
                if prune_method == "imbalanced":
                    layer_keep, head_to_keep, layer, head_to_prune = map(int,line.strip().split(","))
                else:
                    layer, head_to_keep, head_to_prune = map(int,line.strip().split(","))
                pruning_log.append((layer, head_to_prune))
        print("Pruning log found for: ", model, prune_method, metric, prune_task, prune_percent)
            
        for layer, head in pruning_log:
            pruning_dict[layer].append(head)
        
        model_lm = huggingface.HFLM(**model_args) ## model_args are unchanged
        if model_basename == "opt-13b":
            model_lm.model.model = prune(model_lm.model.model, pruning_dict)
        else:
            model_lm.model.transformer = prune(model_lm.model.transformer, pruning_dict)
        
        eval_f.evaluate(model_lm, tasks, model_path)
        print("Mask evaluated")
        del model_lm
        model.to('cpu')
        del model
        gc.collect()
        
        

if __name__ == "__main__":
    # prune_percent = float(sys.argv[1]) # 0.25 0.5 0.75
    metric = sys.argv[1] # euclidean cosine random
    model_name = sys.argv[2] # i.e "bigscience/bloom-560m"
    path = sys.argv[3] # i.e path/bloom-560m-pruned
    prompt = sys.argv[4] # i.e pawsx_en
    group_metric = sys.argv[5] # i.e "cosine if metric is cosine otherwise random"
    prune_method = sys.argv[6] # i.e balanced or imbalanced
    main(model_name=model_name,
            path=path,
            metric=metric,
            prune_task=prompt,
            group_metric=group_metric,
            prune_method=prune_method,
            )
           
            
