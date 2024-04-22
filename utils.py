import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
import random
import json
import numpy as np
import pandas as pd
from pathlib import Path
import pickle
import os
import re
from collections import defaultdict, Counter
from transformers import AutoTokenizer, AutoModel, utils
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
# from datasets import load_dataset


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

def get_attention(prompt, model, tokenizer, first_token=True):
    inputs = tokenizer.encode(prompt, return_tensors='pt')#.to(device)
    tokens = tokenizer.convert_ids_to_tokens(inputs[0])
    with torch.no_grad():
        outputs = model(inputs) 
        attention = outputs[-1]
    if not first_token:
        attention = delete_first_token(attention)    
    return attention, tokens

def get_attention_multiple_inputs(prompts, model, tokenizer, first_token=True):
    """Returns tuple of len(layers) attention maps for each layer 
    each attention map is of shape (len(prompts), num_heads, max_seq_len, max_seq_len)"""
    inputs = tokenizer(prompts, return_tensors='pt', padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
        attention = outputs[-1]
    return attention

def get_batched_attention(prompts, model, tokenizer, batch_size=10, first_token=True, prune_task="paws_en"):
    """Returns tuple of len(layers) attention maps for each layer 
    each attention map is of shape (total_prompts, num_heads, max_seq_len, max_seq_len)"""
    model_name = os.path.basename(model.config._name_or_path)
    path = f"/home/data_shares/mapillary/prompts/{model_name}/{prune_task}_attention_maps.pkl"
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    # Calculate the maximum length after tokenization
    max_length = max(len(tokenizer.encode(prompt)) for prompt in prompts)

    # Create a DataLoader for batch processing
    prompts_dataloader = DataLoader(prompts, batch_size=batch_size)

    all_attention_maps = []

    for batch_prompts in prompts_dataloader:
        inputs = tokenizer(list(batch_prompts), return_tensors='pt', padding='max_length', truncation=True, max_length=max_length)
        with torch.no_grad():
            outputs = model(**inputs)
            attention = outputs[-1]

        all_attention_maps.append(attention)

    # Concatenate attention maps across all batches
    concatenated_attention_maps = [torch.cat([batch[i] for batch in all_attention_maps], dim=0) for i in range(len(all_attention_maps[0]))]

    return concatenated_attention_maps

def attention_vector_multiple_inputs(attention_maps):
    head_attentions = dict()
    num_prompts = attention_maps[0].shape[0]  # Get the number of prompts

    for prompt_idx in range(num_prompts):
        for layer_idx, layer in enumerate(attention_maps):
            for head_idx in range(layer.shape[1]):  # Number of heads
                att_map_vector = tensor_to_vector(layer[prompt_idx, head_idx, :, :])
                key = (layer_idx, head_idx)

                if key not in head_attentions:
                    head_attentions[key] = att_map_vector
                else:
                    head_attentions[key] = torch.cat((head_attentions[key], att_map_vector))
    head_attentions = {key: tensor.cpu().numpy() for key, tensor in head_attentions.items()}  # Convert each tensor to a numpy array
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

def get_clustering_dict(prompts, model, tokenizer, n_groups=8, metric='cosine', n_layers=24, n_heads=16, by_layer=True, prune_percent=0.5, prune_task="paws_en"):
    """
    n_groups is a int if pruning same number of heads from each layer, a list of ints if pruning is variable and None if determining ourself
    Returns a dictionary with layer number as key and list of attention heads as value when by_layer = True

    Otherwise returns a dictionary with group number as key and list of (layer, head) tuples as value
    """

    attention_maps = get_batched_attention(prompts, model, tokenizer, first_token=True, prune_task=prune_task)
    attention_vectors = attention_vector_multiple_inputs(attention_maps)

    if n_groups is None:
        distances_list = []
        heads_to_prune = prune_percent * n_layers * n_heads
        for layer_number in range(n_layers):
            layer_heads = attention_vectors[layer_number*n_heads:(layer_number+1)*n_heads]
            squaref = squareform(pdist(layer_heads, metric=metric))
            for head_id, squaref_row in enumerate(squaref):
                squaref_row_updated = []
                for target_head_id, distance in enumerate(squaref_row):
                    squaref_row_updated.append((target_head_id, distance))
                squaref_row_updated.pop(head_id) # Remove the distance to itself
                squaref_row_updated.sort(key=lambda x: x[1])
                distances_list.append([layer_number, head_id, squaref_row_updated])
        heads_pruned = set()
        while heads_to_prune != 0:
            distances_list.sort(key=lambda x: x[2][0][1])
            # Remove the head with the smallest distance
            layer_number, head_id, squaref_row = distances_list[0]
            min_dist_head = squaref_row[0][0]
            if (layer_number, min_dist_head) not in heads_pruned:
                heads_pruned.add((layer_number, min_dist_head))
                heads_to_prune -= 1
                distances_list.pop(0)
            else:
                distances_list[0][2].pop(0)
        n_groups = [n_heads] * n_layers
        for head in heads_pruned:
            n_groups[head[0]] -= 1

    if by_layer:
        clusters = dict()
        for i in range(n_layers):
            layer_heads = attention_vectors[i*n_heads:(i+1)*n_heads]
            distance_matrix = pdist(layer_heads, metric=metric)
            hc_linkage = linkage(distance_matrix, method='ward')
            if type(n_groups) == int:
                clusters[i] = fcluster(hc_linkage, n_groups, criterion='maxclust')
            elif type(n_groups) == list:
                clusters[i] = fcluster(hc_linkage, n_groups[i], criterion='maxclust')
            else:
                raise ValueError("n_groups must be an int or a list")

        layer_clusters_dict = dict()
        for i in range(n_layers):
            group_indices = defaultdict(list)
            for index, group in enumerate(clusters[i]):
                group_indices[group].append(index)

            layer_clusters_dict[i] = list(group_indices.values())
        return layer_clusters_dict, attention_maps, attention_vectors
    else:
        # clusters = dict()
        distance_matrix = pdist(attention_vectors, metric=metric)
        hc_linkage = linkage(distance_matrix, method='ward')
        clusters = fcluster(hc_linkage, n_groups, criterion='maxclust')
        group_indices = defaultdict(list)

        for layer in range(n_layers):
            for index, group in enumerate(clusters[layer*n_heads:(layer+1)*n_heads]):
                group_indices[group].append((layer,index))
        return group_indices, attention_maps, attention_vectors

def get_bloom_attention_weights(model,layer,head):
    """Get attention weights for a specific layer and head
    returns:
        head_attention: tensor of shape (3, head_dim, embedding_dim)
        head_bias: tensor of shape (3, head_dim)"""
    bloomblock = model.h[layer]
    attention_weight = bloomblock.self_attention.query_key_value.weight
    attention_bias = bloomblock.self_attention.query_key_value.bias
    hidden_size = model.config.hidden_size
    n_head = model.config.n_head
    hidden_size = model.config.hidden_size
    # hiddensize, 3, n_head, head size
    # query key value 
    attention_weight = attention_weight.view(n_head, 3,hidden_size//n_head, hidden_size)
    attention_bias = attention_bias.view(n_head, 3, hidden_size//n_head)
    head_attention = attention_weight[head,:,:,:]
    head_bias = attention_bias[head,:,:]
    return head_attention, head_bias

def get_opt_attention_weights(model, layer, head):
    """Get attention weights for a specific layer and head
    returns:
        k: tensor of shape (head_dim, embedding_dim)
        k_b: tensor of shape (head_dim)
        v: tensor of shape (head_dim, embedding_dim)
        v_b: tensor of shape (head_dim)
        q: tensor of shape (head_dim, embedding_dim)
        q_b: tensor of shape (head_dim)
        out: tensor of shape (head_dim, embedding_dim)
        out_b: tensor of shape (head_dim)
        """
    n_layers, n_head = get_model_layers_and_heads(model.config)
    hidden_size = model.config.hidden_size
    optdecoderlayer = model.decoder.layers[layer]
    k = optdecoderlayer.self_attn.k_proj.weight.view(n_head, hidden_size//n_head, hidden_size)[head,:,:]
    k_b = optdecoderlayer.self_attn.k_proj.bias.view(n_head, hidden_size//n_head)[head, :]
    v = optdecoderlayer.self_attn.v_proj.weight.view(n_head, hidden_size//n_head, hidden_size)[head,:,:]
    v_b = optdecoderlayer.self_attn.v_proj.bias.view(n_head, hidden_size//n_head)[head, :]
    q = optdecoderlayer.self_attn.q_proj.weight.view(n_head, hidden_size//n_head, hidden_size)[head,:,:]
    q_b = optdecoderlayer.self_attn.q_proj.bias.view(n_head, hidden_size//n_head)[head, :]
    out = optdecoderlayer.self_attn.out_proj.weight.view(n_head, hidden_size//n_head, hidden_size)[head,:,:]
    out_b = optdecoderlayer.self_attn.out_proj.bias.view(n_head, hidden_size//n_head)[head, :]

    return k, k_b, v, v_b, q, q_b, out, out_b

def get_model_layers_and_heads(config):
    if type(config) == str:
        if config == "opt-13b":
            layers = 40
            heads = 40
            return layers, heads
        else:
            raise ValueError("Model not supported")
    if "bloom" in config._name_or_path:
        heads = config.n_head
        layers = config.n_layer
    elif "opt" in config._name_or_path:
        heads = config.num_attention_heads
        layers = config.num_hidden_layers
    else:
        raise ValueError("Model not supported")
    return layers, heads

def extract_metrics(results_dict):
    # Extracting the 'results' dictionary
    metrics = results_dict.get('results', {})
    # Flattening the nested structure
    flat_metrics = {}
    for test, scores in metrics.items():
        for metric, value in scores.items():
            flat_metrics[f'{test}_{metric}'] = value
    return flat_metrics

def get_dataframe(results_dict):
    return pd.DataFrame([extract_metrics(results_dict)])

def save_results(name, results_dict):
    print(name, " ", results_dict)
    df = get_dataframe(results_dict)
    df.to_csv("results/" + name + '.csv')

def get_paws_data(n_samples=100, save_to_file=False, file_name=None):
    """
    Gets a random sample of training data from a given dataset.
    Returns a list of tuples, where each tuple contains a modified sample and its ID.
    """
    dataset = load_dataset("paws-x", "en", split="train")
    random_sample = dataset.shuffle(seed=42)[:n_samples]

    QUESTION_WORD = "right"
    YES = "Yes"
    NO = "No"

    training_samples = []
    for i in range(len(random_sample["id"])):
        sentence1 = random_sample["sentence1"][i]
        sentence2 = random_sample["sentence2"][i]
        if random_sample["label"][i]:
            modified_sample = f"{sentence1}, {QUESTION_WORD}? {YES}, {sentence2}"
        else:
            modified_sample = f"{sentence1}, {QUESTION_WORD}? {NO}, {sentence2}"
        training_samples.append((modified_sample,random_sample["id"][i]))

    if save_to_file:
        if file_name is None:
            file_name = f"tasks/paws_en.tsv"
        df = pd.DataFrame(training_samples, columns=["prompt", "id"])
        df.to_csv(file_name, sep="\t", index=False, header=False)
        
    return training_samples

def get_arc_data(n_samples=100, save_to_file=False, file_name=None):
    """
    Gets a random sample of training data from a given dataset.
    Returns a list of tuples, where each tuple contains a modified sample and its ID.
    """
    dataset = load_dataset("allenai/ai2_arc", "ARC-Easy", split="train")
    random_sample = dataset.shuffle(seed=42)[:n_samples]

    training_samples = []
    for i in range(len(random_sample["id"])):
        question = random_sample["question"][i]
        choices = random_sample["choices"][i]["text"]
        option_a = choices[0]
        option_b = choices[1]
        option_c = choices[2]
        option_d = choices[3]
        modified_sample = f'''Question: {question}
        A. {option_a}
        B. {option_b}
        C. {option_c}
        D. {option_d}
        Answer:'''

        training_samples.append((modified_sample,random_sample["id"][i]))

    if save_to_file:
        if file_name is None:
            file_name = f"tasks/arc_easy.tsv"
        df = pd.DataFrame(training_samples, columns=["prompt", "id"])
        df.to_csv(file_name, sep="\t", index=False, header=False)
        
    return training_samples

def preprocess(text):
    text = text.strip()
    # NOTE: Brackets are artifacts of the WikiHow dataset portion of HellaSwag.
    text = text.replace(" [title]", ". ")
    text = re.sub("\\[.*?\\]", "", text)
    text = text.replace("  ", " ")
    return text

def get_hellaswag_data(n_samples=100, save_to_file=False, file_name=None):
    """
    Gets a random sample of training data from a given dataset.
    Returns a list of tuples, where each tuple contains a modified sample and its ID.
    """
    dataset = load_dataset("Rowan/hellaswag", split="train")
    random_sample = dataset.shuffle(seed=42)[:n_samples]

    training_samples = []
    for i in range(len(random_sample["ind"])):
        label = int(random_sample["label"][i])
        text = preprocess(random_sample["ctx"][i])
        ending = preprocess(random_sample["endings"][i][label])
        modified_sample = text+ending
        training_samples.append((modified_sample,random_sample["ind"][i]))

    if save_to_file:
        if file_name is None:
            file_name = f"tasks/hellaswag.tsv"
        df = pd.DataFrame(training_samples, columns=["prompt", "id"])
        df.to_csv(file_name, sep="\t", index=False, header=False)
        
    return training_samples

def get_model_percentage_weight(model, verbose_attn=False, verbose_non_attn=False):
    total_params = 0
    attention_params = 0
    for name, parameter in model.named_parameters():
        # Count the total number of parameters
        param_count = parameter.numel()
        total_params += param_count
        # Check if the layer is an attention layer
        if "attention" in name or "attn" in name:
            attention_params += param_count
            if verbose_attn:
                print(f"{name}: {param_count:,} parameters")
        else:
            if verbose_non_attn:
                print(f"{name}: {param_count:,} parameters")


    # Calculate the percentage of parameters in attention layers
    attention_percentage = (attention_params / total_params) * 100
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Attention parameters: {attention_params:,}")
    print(f"Attention parameters make up {attention_percentage:.2f}% of the total parameters")
    
def save_pruning_log(path, pruning_log):
    file_path = f"pruning_logs/{path}/pruning_log.txt"
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w") as f:
        # For pruning log for balanced pruning
        if len(pruning_log[0]) == 3:
            for layer, head_to_keep, head_to_remove in pruning_log:
                f.write(f"{layer}, {head_to_keep}, {head_to_remove}\n")
        # For pruning log for imbalanced pruning
        else:
            for head_to_keep, head_to_remove in pruning_log:
                f.write(f"{head_to_keep[0]}, {head_to_keep[1]}, {head_to_remove[0]}, {head_to_remove[1]}\n")
            

            
def get_prompts_from_file(prune_task):
    file_path = "tasks/" + prune_task + ".tsv"
    df = pd.read_csv(file_path, sep="\t", header=None)
    prompts = [x for x in df[0]]
    return prompts

def get_results(path="/mnt/c/github/explainabloomity/results"):
    models = ("opt-13b", )
    prune_methods=(
        "balanced",
        # "imbalanced" ,
    )
    metrics=(
        "cosine_cosine" ,
        "euclidean_euclidean" ,
        "cosine_random",
        "euclidean_random",
    )
    prunetasks=(
        "paws_en",
        "hellaswag",
        "arc_easy",
        "blimp_ellipsis_n_bar_1",
    )
    prune_percents=(
        "0.25",
        "0.5",
        "0.75",
    )
    tasks=(
        "lambada_openai",
        "paws_en",
        "hellaswag",
        "arc_easy",
        "blimp_ellipsis_n_bar_1",
        "blimp_irregular_plural_subject_verb_agreement_1")
    rows = []
    for model in models:
        for prune_method in prune_methods:
            for prune_task in prunetasks:
                for prune_metric in metrics:
                    for prune_percent in prune_percents:
                        for task in tasks:
                            path_model = Path("./results")/ task / model / prune_method / prune_task / prune_metric / prune_percent / "results.json"
                            if path_model.exists():
                                with open(path_model, "r") as f:
                                    data = json.load(f)
                                    norm_accuracy = data["results"][task]["acc,none"]
                                    row = {"model": model, "prune_method": prune_method, "prune_task": prune_task, "metric": prune_metric, "percent": prune_percent, "task": task, "norm_accuracy": norm_accuracy}
                                    rows.append(row)

    for task in tasks:
        for model in models:
            model += "_base"
            path_model = Path("./results")/ task / model / "results.json"
            if path_model.exists():
                with open(path_model, "r") as f:
                    data = json.load(f)
                    norm_accuracy = data["results"][task]["acc,none"]
                    row = {"model": model, "prune_method": None, "prune_task": None, "metric": None, "percent": None, "task": task, "norm_accuracy": norm_accuracy}
                    rows.append(row)

    df = pd.DataFrame(rows)
    return df

def get_amazon_attention_mask(path = 'head_importance/0shot_hellaswag.pkl', head_percent_mask = 50):
    head_importance = pickle.load(open(path, 'rb'))
    num_hidden_layers = head_importance.shape[0]
    num_heads= head_importance.shape[1]
    head_mask = torch.ones(num_hidden_layers * num_heads, dtype = torch.half)

    _, head_indices = torch.sort(head_importance.view(-1))
    head_indices = list(head_indices.numpy())
    head_indices = head_indices[: int(head_percent_mask) * len(head_indices) // 100]
    head_mask[head_indices] = 0.
    return head_mask

def get_amazon_prune_groups(path="head_importance/0shot_hellaswag.pkl", head_percent=0.5):
    head_percent_mask = head_percent * 100
    head_mask = get_amazon_attention_mask(path, head_percent_mask)
    layers = 40
    n_head = 40
    n_groups = []
    for layer_number in range(layers):
        layer_heads = head_mask[layer_number*n_head:(layer_number+1)*n_head]
        n_groups.append(int(layer_heads.sum().item()))
    return n_groups

def get_amazon_prune_heads(path="head_importance/0shot_hellaswag.pkl", head_percent=0.5):
    head_percent_mask = head_percent * 100
    head_mask = get_amazon_attention_mask(path, head_percent_mask)
    layers = 40
    n_head = 40
    prune_heads = []
    for layer_number in range(layers):
        layer_heads = head_mask[layer_number*n_head:(layer_number+1)*n_head]
        for i, head in enumerate(layer_heads):
            if head == 0:
                prune_heads.append((layer_number, i))
    return prune_heads
