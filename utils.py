import matplotlib.pyplot as plt
import torch
import random
import numpy as np
import pandas as pd
from collections import defaultdict, Counter
from transformers import AutoTokenizer, AutoModel, utils
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from datasets import load_dataset


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

def get_clustering_dict(prompts, model, tokenizer, n_groups=8, metric='cosine', n_layers=24, n_heads=16):
    """
    Returns a dictionary with layer number as key and list of attention heads as value
    """
    attention_maps = get_attention_multiple_inputs(prompts, model, tokenizer, first_token=True)
    attention_vectors = attention_vector_multiple_inputs(attention_maps)
    clusters = dict()
    for i in range(n_layers):
        layer_heads = attention_vectors[i*n_heads:(i+1)*n_heads]
        distance_matrix = squareform(pdist(layer_heads, metric=metric))
        hc_linkage = linkage(distance_matrix, method='ward')
        clusters[i] = fcluster(hc_linkage, n_groups, criterion='maxclust')

    layer_clusters_dict = dict()
    for i in range(n_layers):
        group_indices = defaultdict(list)
        for index, group in enumerate(clusters[i]):
            group_indices[group].append(index)

        layer_clusters_dict[i] = list(group_indices.values())

    return layer_clusters_dict

def get_attention_weights(model,layer,head):
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

def duplicate_prune(model, source_layer, source_head, target_layer, target_head):
    """Given source layer and head, duplicate the attention weights and bias to the target layer and head
    Args: 
        source_layer: int,
        source_head: int,
        target_layer: int,
        target_head: int
    Returns:
        model: updated model with attention weights and bias duplicated to the target layer and head
    """
    
    source_weight, source_bias = get_attention_weights(model, source_layer, source_head)
    target_bloom_block = model.h[target_layer]
    attention_weight = target_bloom_block.self_attention.query_key_value.weight
    attention_bias = target_bloom_block.self_attention.query_key_value.bias
    hidden_size = model.config.hidden_size
    n_head = model.config.n_head
    hidden_size = model.config.hidden_size
    # hiddensize, 3, n_head, head size
    # query key value 
    attention_weight = attention_weight.view(n_head, 3,hidden_size//n_head, hidden_size)
    attention_bias = attention_bias.view(n_head, 3, hidden_size//n_head)

    with torch.no_grad():
        attention_weight[target_head] = source_weight
        attention_bias[target_head] = source_bias
        
    model.h[target_layer].self_attention.query_key_value.weight = torch.nn.Parameter(attention_weight.view_as(target_bloom_block.self_attention.query_key_value.weight))
    model.h[target_layer].self_attention.query_key_value.bias = torch.nn.Parameter(attention_bias.view_as(target_bloom_block.self_attention.query_key_value.bias))
    return model

def duplicate_prune_model(prompts, model_name, model, tokenizer, prune_percent=0.5, metric='euclidean', verbose=True):
    '''
    Duplicate prunes a model based on the attention scores of the heads.
    The attention scores are calculated for the prompts and the heads are clustered based on cosine similarity.
    The heads within groups are then compared using metric and the head with the highest similarity to other heads in cluster is kepts
    
    Args:
        prompts: list of strings, prompts to calculate attention scores for
        model_name: str, filename of the pruned model in models folder
        model: model to prune (currently only implemented for bloom models)
        tokenizer: tokenizer for the model
        prune_percent: float, percentage of heads to prune
        metric: str, metric to use for comparing heads within clusters. Options are 'euclidean', 'cosine' and 'random'
        verbose: bool, whether to print the number of clusters of each size
    Returns:
        path: str, path where the model is saved. Also
        saves the model to model folder
    '''
    
    attentions = get_attention_multiple_inputs(prompts, model, tokenizer)
    n_head = model.config.n_head
    n_layers = model.config.n_layer
    n_groups = n_head - int(n_head * prune_percent)
    # attention is tuple of len(layers) where 
    # each element is a tensor of shape 
    # (num_prompts, num_heads, num_tokens, num_tokens)

    layers_clustering_dict = get_clustering_dict(prompts, model, tokenizer,n_layers=n_layers, n_groups=n_groups)
    counter = Counter()
    for layer_number in layers_clustering_dict.keys():
        squaref = squareform(pdist(attentions[layer_number].view(n_head, -1), metric=metric))
        layer_clusters = layers_clustering_dict[layer_number]
        for group in layer_clusters:
            group_scores = defaultdict(int)
            counter.update([len(group)])
            if len(group) <= 1:
                continue
            if len(group) == 2:
                # with 2 heads just keep the first 1
                head_to_keep = group[0]
            else:
                if metric == 'random':
                    head_to_keep = random.choice(group)
                else:
                    for head_id in group:
                        for head_id_2 in group:
                            if head_id == head_id_2:
                                continue
                            group_scores[head_id] += squaref[head_id, head_id_2]
                    if metric == 'euclidean':
                        head_to_keep = min(group_scores, key=lambda k: group_scores[k])
                    elif metric == 'cosine':
                        head_to_keep = max(group_scores, key=lambda k: group_scores[k])
                    
            for head in group:
                if head == head_to_keep:
                    continue
                head_to_remove = head
                model = duplicate_prune(model, source_layer=layer_number, source_head=head_to_keep, target_layer=layer_number, target_head=head_to_remove)
                
    if verbose:
        print(counter)
    path = f'./models/{model_name}_{metric}_{prune_percent}'
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)
    return path


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

def get_training_data(dataset_name="paws-x", n_samples=100, save_to_file=False, file_name=None):
    """
    Gets a random sample of training data from a given dataset.
    Returns a list of tuples, where each tuple contains a modified sample and its ID.
    """
    dataset = load_dataset(dataset_name, "en", split="train")
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
            file_name = f"{dataset_name}_training_data.csv"
        df = pd.DataFrame(training_samples, columns=["prompt", "id"])
        df.to_csv(file_name, sep="\t", index=False, header=False)
        
    return training_samples