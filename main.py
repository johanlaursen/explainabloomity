from utils import *
import eval
import sys
import pandas as pd 
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

def main(model_name, pruned_model_name, metric, prune_percent):
    df = pd.read_csv('pawsx_en_train_sample.tsv', sep="\t", header=None)
    prompts = [x for x in df[0]]
    
    
    
    model = AutoModel.from_pretrained(model_name, output_attentions=True)#.to(device)  # Configure model to return attention values
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model_path = duplicate_prune_model(prompts, pruned_model_name, model, tokenizer, prune_percent=prune_percent,metric=metric, verbose=True)
    print(model_path)

if __name__ == "__main__":
    pbar = tqdm(total=3*3, desc="Progress")
    for prune_percent in [0.25, 0.5, 0.75]:
        for metric in ["euclidean", "cosine", "random"]:
            main(model_name="bigscience/bloom-560m",
                 pruned_model_name=f"models/bloom-560m_clusterpruned",
                 metric=metric,
                 prune_percent=prune_percent)
            pbar.update(1)
            
    pbar.close()