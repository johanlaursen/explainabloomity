import json
import random
import seaborn as sns
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModel, utils
import torch
from tqdm import tqdm
from utils import visualize_single
from bertviz import model_view, head_view

with open('pawsx_en_write_out_info.json', 'r', encoding="utf-8") as file:
    data = json.load(file)


model_name = "bigscience/bloom-7b1"
model = AutoModel.from_pretrained(model_name, output_attentions=True)#.to(device)  # Configure model to return attention values
tokenizer = AutoTokenizer.from_pretrained(model_name)


def viz_bert(prompt, model=model, tokenizer=tokenizer):
    inputs = tokenizer.encode(prompt, return_tensors='pt')#.to(device)
    with torch.no_grad():
        outputs = model(inputs) 
        attention = outputs[-1]  # Retrieve attention from model outputs
    tokens = tokenizer.convert_ids_to_tokens(inputs[0])
    model_view(attention, tokens) 
    

def random_sample(data):
# Step 2: Randomly select 10 dictionaries
    num_samples = 10
    selected_dicts = random.sample(data, num_samples)

    # Step 3: Print `doc_id` and `prompt_0`
    for d in selected_dicts:
        doc_id = d.get('doc_id', 'No doc_id')  # Default to 'No doc_id' if not found
        prompt_0 = d.get('prompt_0', 'No prompt_0')  # Default to 'No prompt_0' if not found
        print(f"doc_id: {doc_id}, prompt_0: {prompt_0}")

def shortest_sample(data, n=5):
    sorted_dicts = sorted(data, key=lambda x: len(x.get('prompt_0', '')))

    # select 5 shortest samples
    top_dicts = sorted_dicts[:n]
    prompts = []
    for d in tqdm(top_dicts):
        # doc_id = d.get('doc_id', 'No doc_id')  # Default to 'No doc_id' if not found
        prompt_0 = d.get('prompt_0', 'No prompt_0')  # Default to 'No prompt_0' if not found
        prompt_1 = d.get('prompt_1', 'No prompt_1')  # Default to 'No prompt_0' if not found
        # print(f"doc_id: {doc_id}, prompt_0: {prompt_0}")
        # print(f"doc_id: {doc_id}, prompt_1: {prompt_1}")
        prompts += [prompt_0, prompt_1]
    return prompts

def visualize_singles(prompts, model, tokenizer, layer=16, head=27):
    
    for idx, prompt in enumerate(prompts):
            inputs = tokenizer.encode(prompt, return_tensors='pt')
            with torch.no_grad():
                outputs = model(inputs)  # Run model
                attention = outputs[-1]  # Retrieve attention from model outputs
            attention_weights = attention[layer][:, head, :, :] # Get the first layer [0], and the first attention head's attention

            tokens = tokenizer.convert_ids_to_tokens(inputs[0])  # Convert input ids to token strings
            #head_view(attention, tokens)  # Display model view
            # attention_weights = attention[18][:, 13, :, :] # Get the first layer [0], and the first attention head's attention
            visualize_single(attention_weights[0],tokens, figname=f"attention_doc_{idx}_prompt.png")

if __name__ == "__main__":
    #random_sample(data)
    # shortest_sample(data)
    pass