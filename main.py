from utils import *
from prune import duplicate_prune_model
import eval
import sys
import pandas as pd 
from transformers import AutoModel, AutoTokenizer
import torch

def main(model_name, metric, group_metric, prune_percent, prune_task, prune_method):
    prompts = get_prompts_from_file(prune_task)
    model = AutoModel.from_pretrained(model_name, output_attentions=True)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model_path = duplicate_prune_model(prompts=prompts, 
                                       path=path,
                                       model=model,
                                       model_name=model_name,
                                       tokenizer=tokenizer,
                                       prune_percent=prune_percent,
                                       prune_task=prune_task,
                                       metric=metric,
                                       group_metric=group_metric,
                                       prune_method=prune_method,
                                       verbose=True)
    print(model_path)

if __name__ == "__main__":
    prune_percent = float(sys.argv[1]) # 0.25 0.5 0.75
    metric = sys.argv[2] # euclidean cosine random
    model_name = sys.argv[3] # i.e "bigscience/bloom-560m"
    path = sys.argv[4] # i.e path/bloom-560m-pruned
    prompt = sys.argv[5] # i.e pawsx_en
    group_metric = sys.argv[6] # i.e "cosine if metric is cosine otherwise random"
    prune_method = sys.argv[7] # i.e balanced or imbalanced
    main(model_name=model_name,
            pruned_model_name=path,
            metric=metric,
            prune_percent=prune_percent,
            prune_task=prompt,
            group_metric=group_metric,
            prune_method=prune_method,
            )
           
            
