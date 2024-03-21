import matplotlib.pyplot as plt
import torch
import random
import numpy as np
import pandas as pd
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

def duplicate_prune_bloom(model, source_layer, source_head, target_layer, target_head):
    """Given source layer and head, duplicate the attention weights and bias to the target layer and head
    Args: 
        source_layer: int,
        source_head: int,
        target_layer: int,
        target_head: int
    Returns:
        model: updated model with attention weights and bias duplicated to the target layer and head
    """
    
    source_weight, source_bias = get_bloom_attention_weights(model, source_layer, source_head)
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

def duplicate_prune_opt(model, source_layer, source_head, target_layer, target_head):

    target_opt_block = model.decoder.layers[target_layer]
    n_layer, n_head = get_model_layers_and_heads(model.config)
    k, k_b, v, v_b, q, q_b, out, out_b = get_opt_attention_weights(model, source_layer, source_head)
    hidden_size = model.config.hidden_size
    t_k = target_opt_block.self_attn.k_proj.weight.view(n_head, hidden_size//n_head, hidden_size)
    t_k_b = target_opt_block.self_attn.k_proj.bias.view(n_head, hidden_size//n_head)
    t_v = target_opt_block.self_attn.v_proj.weight.view(n_head, hidden_size//n_head, hidden_size)
    t_v_b = target_opt_block.self_attn.v_proj.bias.view(n_head, hidden_size//n_head)
    t_q = target_opt_block.self_attn.q_proj.weight.view(n_head, hidden_size//n_head, hidden_size)
    t_q_b = target_opt_block.self_attn.q_proj.bias.view(n_head, hidden_size//n_head)
    t_out = target_opt_block.self_attn.out_proj.weight.view(n_head, hidden_size//n_head, hidden_size)
    t_out_b = target_opt_block.self_attn.out_proj.bias.view(n_head, hidden_size//n_head)

    with torch.no_grad():
        t_k[target_head] =  k
        t_k_b[target_head] =  k_b
        t_v[target_head] =  v
        t_v_b[target_head] = v_b 
        t_q[target_head] =  q
        t_q_b[target_head] = q_b 
        t_out[target_head] =  out
        t_out_b[target_head] =  out_b

    model.decoder.layers[target_layer].self_attn.k_proj.weight = torch.nn.Parameter(t_k.view_as(target_opt_block.self_attn.k_proj.weight))
    model.decoder.layers[target_layer].self_attn.k_proj.bias = torch.nn.Parameter(t_k_b.view_as(target_opt_block.self_attn.k_proj.bias))
    model.decoder.layers[target_layer].self_attn.v_proj.weight = torch.nn.Parameter(t_v.view_as(target_opt_block.self_attn.v_proj.weight))
    model.decoder.layers[target_layer].self_attn.v_proj.bias = torch.nn.Parameter(t_v_b.view_as(target_opt_block.self_attn.v_proj.bias))
    model.decoder.layers[target_layer].self_attn.q_proj.weight = torch.nn.Parameter(t_q.view_as(target_opt_block.self_attn.q_proj.weight))
    model.decoder.layers[target_layer].self_attn.q_proj.bias = torch.nn.Parameter(t_q_b.view_as(target_opt_block.self_attn.q_proj.bias))
    model.decoder.layers[target_layer].self_attn.out_proj.weight = torch.nn.Parameter(t_out.view_as(target_opt_block.self_attn.out_proj.weight))
    model.decoder.layers[target_layer].self_attn.out_proj.bias = torch.nn.Parameter(t_out_b.view_as(target_opt_block.self_attn.out_proj.bias))
    return model

def duplicate_prune_model(prompts, model_name, model, tokenizer, prune_percent=0.5, metric='euclidean', verbose=True):
    '''
    Duplicate prunes a model based on the attention scores of the heads.
    The attention scores are calculated for the prompts and the heads are clustered based on cosine similarity.
    The heads within groups are then compared using a metric and the head with the highest similarity to other heads in cluster is kept
    
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
    n_layers, n_head = get_model_layers_and_heads(model.config)
    n_groups = n_head - int(n_head * prune_percent)
    # attention is tuple of len(layers) where 
    # each element is a tensor of shape 
    # (num_prompts, num_heads, num_tokens, num_tokens)

    layers_clustering_dict = get_clustering_dict(prompts, model, tokenizer,n_layers=n_layers, n_groups=n_groups, n_heads=n_head)
    counter = Counter()
    for layer_number in layers_clustering_dict.keys():
        if metric != 'random':
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
                if "bloom" in model.config._name_or_path:
                    model = duplicate_prune_bloom(model, source_layer=layer_number, source_head=head_to_keep, target_layer=layer_number, target_head=head_to_remove)
                elif "opt" in model.config._name_or_path:
                    model = duplicate_prune_opt(model, source_layer=layer_number, source_head=head_to_keep, target_layer=layer_number, target_head=head_to_remove)
                else:
                    raise ValueError(f"Model {model.config._name_or_path} not supported")
                
    if verbose:
        print(counter)
    path = f'{model_name}_{metric}_{prune_percent}'
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)
    return path

def duplicate_prune_model_imbalanced(prompts, model_name, model, tokenizer, prune_percent=0.5, metric='euclidean', verbose=True):
    attentions = get_attention_multiple_inputs(prompts, model, tokenizer)
    n_layers, n_head = get_model_layers_and_heads(model.config)
    total_heads = n_head * n_layers
    heads_to_prune = int(total_heads * prune_percent)

    head_similarity_scores = []

    for layer_number in range(n_layers):
        if metric != 'random':
            squaref = squareform(pdist(attentions[layer_number].view(n_head, -1), metric=metric))
        for head_id in range(n_head):
            if metric == 'random':
                score = random.random()
            else:
                score = sum(squaref[head_id]) / (n_head - 1)  # Average similarity with other heads
            head_similarity_scores.append((layer_number, head_id, score))

    # Sort by similarity score (lower is better for euclidean, higher for cosine or random)
    head_similarity_scores.sort(key=lambda x: x[2], reverse=(metric != 'euclidean'))

    # Prune the required number of heads based on global ranking
    pruned_heads = head_similarity_scores[:heads_to_prune]

    for layer_number, head_id, _ in pruned_heads:
        # Find the most similar head in the same layer to duplicate
        similar_heads = [x for x in head_similarity_scores if x[0] == layer_number and x[1] != head_id]
        if similar_heads:
            # Sort by similarity (most similar first)
            similar_heads.sort(key=lambda x: x[2], reverse=(metric != 'euclidean'))
            target_head = similar_heads[0][1]
            model = duplicate_prune_bloom(model, source_layer=layer_number, source_head=target_head, target_layer=layer_number, target_head=head_id)

    if verbose:
        print(f'Pruned {len(pruned_heads)} heads out of {total_heads}')

    path = f'{model_name}_{metric}_{prune_percent}'
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)
    return path

def get_model_layers_and_heads(config):
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


def get_model_percentage_weight(model, verbose_attn=False, verbose_non_attn=False):
    total_params = 0
    attention_params = 0
    for name, parameter in model_bloom.named_parameters():
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