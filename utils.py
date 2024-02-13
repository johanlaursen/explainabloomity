import matplotlib.pyplot as plt
import torch
import random
import numpy as np
from transformers import AutoTokenizer, AutoModel, utils

model_name = "bigscience/bloom-560m"
model = AutoModel.from_pretrained(model_name, output_attentions=True)#.to(device)  # Configure model to return attention values
tokenizer = AutoTokenizer.from_pretrained(model_name)

def visualize_single(att_map, sentence, figname):
    """
    Attention map for a given layer and head
    """
    
    plt.figure(figsize=(16, 12))
    plt.imshow(att_map, cmap='Reds')
    plt.xticks(range(len(sentence)), sentence, rotation=60, fontsize=12)
    plt.yticks(range(len(sentence)), sentence, fontsize=12)

    plt.grid(False)
    plt.savefig(figname, dpi=400)

def visualize_some(attention, tokens, head_list, random_sample=0, cols=4):
    """
    Visualize attention maps for specified heads.
    
    Parameters:
    - attention: A nested list (or array) where each element is a layer and each layer contains
                 head matrices with dimensions [num_heads, seq_len, seq_len].
    - tokens: A list of tokens corresponding to the sequence length.
    - head_list: A list of tuples (layer, head) to visualize.
    """

    # If there is only one head to plot, plot single.
    if type(head_list) == tuple:
        visualize_single(attention[head_list[0]][:, head_list[1], :, :], tokens)
        return

    if random_sample > len(head_list):
        head_list = random.sample(head_list, random_sample)

    num_heads = len(head_list)

    num_columns = min(num_heads, cols)
    num_rows = (num_heads + 3) // cols

    fig_width = 5 * num_columns
    fig_height = 5 * num_rows
    
    fig, axes = plt.subplots(num_rows, num_columns, figsize=(fig_width, fig_height))

    if num_rows > 1 or num_columns > 1:
        axes = axes.flatten()

    for i, head in enumerate(head_list):
        att_map = attention[head[0]][0, head[1], :, :]
        ax = axes[i]
        ax.imshow(att_map, cmap='Reds')
        ax.set_title(f"Layer {head[0]} Head {head[1]}")
        ax.set_xticks(range(len(tokens)))
        ax.set_yticks(range(len(tokens)))
        ax.set_xticklabels(tokens, rotation=60, fontsize=10)
        ax.set_yticklabels(tokens, fontsize=10)
        ax.grid(False)

    for ax in axes[num_heads:]:
        ax.axis('off')        
    
    plt.tight_layout()
    plt.show()

def visualize_n_inputs(attention_dict, layer_head_combinations, n=5):
    """
    Visualize attention maps for specified layer-head combinations, with each column showing the same input for different heads.
    
    Parameters:
    - attention_dict: A dictionary where keys are tuples (layer_id, head_id)
      and values are lists of attention maps. Output of attention_dict_multiple_inputs()
    - layer_head_combinations: A list of tuples in the form (layer_id, head_id)
    - n: number of inputs to visualize per column
    """

    first_key = list(attention_dict.keys())[0]
    if len(attention_dict[first_key]) > n:
        indices = random.sample(range(len(attention_dict[first_key])), n)
    else:
        indices = range(len(attention_dict[first_key]))

    num_rows = len(layer_head_combinations)
    num_columns = n 

    fig_width = 5 * num_columns
    fig_height = 5 * num_rows

    _, axes = plt.subplots(num_rows, num_columns, figsize=(fig_width, fig_height), layout="constrained")
    
    if num_rows > 1 or num_columns > 1:
        axes = axes.flatten()
    else:
        axes = [axes]

    for row, (layer_id, head_id) in enumerate(layer_head_combinations):
        for col, idx in enumerate(indices):
            ax_idx = row * num_columns + col
            ax = axes[ax_idx]
            attention_map = attention_dict[(layer_id, head_id)][idx]
            ax.imshow(attention_map, cmap='Reds')
            ax.grid(False)
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(f"Input: {idx}", fontsize=20)
        axes[row * num_columns].set_ylabel(f'Layer {layer_id} - Head {head_id}', fontsize=20, rotation=90)

    #plt.tight_layout(rect=[0, 0.01, 1, 1])  # Leave space for the super title
    #fig.suptitle("Attention Maps Visualization", fontsize=16, va='bottom')

    plt.show()

def tensor_to_vector(tensor):
    """Converts 2-D tensor into a vector. 
    Will only keep elements on or below the diagonal"""
    assert tensor.size(0) == tensor.size(1), "Tensor must be square"
    # Create a mask for the lower triangular part, including the diagonal
    mask = torch.tril(torch.ones_like(tensor)).bool()
    return tensor[mask]

def vector_to_tensor(vector, size):
    """Converts a vector back into a square 2-D tensor
    Will only fill the lower triangular part, including the diagonal"""
    tensor = torch.zeros((size, size), dtype=vector.dtype)
    # Create the same mask used for selecting the lower triangular part
    mask = torch.tril(torch.ones(size, size)).bool()
    tensor[mask] = vector
    return tensor

def random_sample(data, num_samples=100, random_seed=42):
    random.seed(random_seed)
    selected_dicts = random.sample(data, num_samples)
    out = []
    for d in selected_dicts:
        if random.random() < 0.5:
            out.append(d['prompt_0'])
        else:
            out.append(d['prompt_1'])
    return out

def delete_first_token(attention):
    for layer in attention:
        for i in range(layer.shape[1]):
            head = layer[:, i, :, :]
            head[:, :, 0] = 0
    return attention

def get_attention(prompt, model=model, tokenizer=tokenizer, first_token=True):
    inputs = tokenizer.encode(prompt, return_tensors='pt')#.to(device)
    tokens = tokenizer.convert_ids_to_tokens(inputs[0])
    with torch.no_grad():
        outputs = model(inputs) 
        attention = outputs[-1]
    if not first_token:
        attention = delete_first_token(attention)    
    return attention, tokens

def attention_vector_multiple_inputs(attention_maps):
    head_attentions = dict()
    for attention_map in attention_maps:
        for layer_idx, layer in enumerate(attention_map):
            for head_idx in range(layer.shape[1]):
                att_map_vector = tensor_to_vector(layer[0, head_idx, :, :])
                if (layer_idx, head_idx) not in head_attentions:
                    head_attentions[(layer_idx, head_idx)] = att_map_vector
                else:
                    head_attentions[(layer_idx, head_idx)] = torch.cat((head_attentions[(layer_idx, head_idx)], att_map_vector))
    
    head_attentions = np.array(list(head_attentions.values()))
    return head_attentions

def attention_dict_multiple_inputs(attention_maps):
    head_attentions = dict()
    for attention_map in attention_maps:
        for layer_idx, layer in enumerate(attention_map):
            for head_idx in range(layer.shape[1]):
                att_map = layer[0, head_idx, :, :]
                if (layer_idx, head_idx) not in head_attentions:
                    head_attentions[(layer_idx, head_idx)] = [att_map]
                else:
                    head_attentions[(layer_idx, head_idx)].append(att_map)
    
    return head_attentions

def get_group_dict(clusters, n_layers=24, n_heads=16):
    """
    Takes in output of fcluster(hc_linkage)
    Returns a dict with group number as key and list of tuples, where each tuple is (layer_id, head_id) as value
    """
    group_dict = dict()
    for i in range(n_layers):
        for j in range(n_heads):
            if clusters[i*n_heads + j] not in group_dict:
                group_dict[clusters[i*n_heads + j]] = [(i, j)]
            else:
                group_dict[clusters[i*n_heads + j]].append((i, j))
    return group_dict