from utils import *
import eval
import sys
import pandas as pd 
from transformers import AutoModel, AutoTokenizer

def main(model_name, pruned_model_name, metric, prune_percent):
    df = pd.read_csv('pawsx_en_train_sample.tsv', sep="\t", header=None)
    prompts = [x for x in df[0]]
    
    
    
    model = AutoModel.from_pretrained(model_name, output_attentions=True)#.to(device)  # Configure model to return attention values
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model_path = duplicate_prune_model(prompts, pruned_model_name, model, tokenizer, prune_percent=prune_percent,metric=metric, verbose=True)
    print(model_path)

if __name__ == "__main__":
    prune_percent = float(sys.argv[1]) # 0.25 0.5 0.75
    metric = sys.argv[2] # euclidean cosine random
    model_name = sys.argv[3] # i.e "bigscience/bloom-560m"
    path = sys.argv[4] # i.e path/bloom-560m-pruned

    main(model_name=model_name,
            pruned_model_name=path,
            metric=metric,
            prune_percent=prune_percent)
           
            
