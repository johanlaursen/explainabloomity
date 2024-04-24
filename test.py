import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from transformers import AutoTokenizer, AutoModel, utils
import torch
import json
import random
import pandas as pd
from tqdm import tqdm
from bertviz import model_view, head_view
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from sklearn.cluster import DBSCAN
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from utils import tensor_to_vector
utils.logging.set_verbosity_error()  # Suppress standard warnings
from utils import random_sample
import json
from scipy.spatial.distance import pdist, squareform
from datasets import load_dataset
from collections import defaultdict, Counter
from prune import duplicate_prune_model
from utils import *
import pickle

head_percent = 0.5
path = "head_importance/0shot_hellaswag.pkl"
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
len(prune_heads)

chosen_clusters = set()
head_indices = get_top_heads(top_n = 800)
for head in head_indices:
    for i, cluster in enumerate(clusters):
        if head in cluster:
            chosen_clusters.add(tuple(cluster))
len(set(chosen_clusters))/len(chosen_clusters)
chosen_clusters
# sort counter by key
sorted(Counter([layer for layer, head in head_indices]).items())
import random
tuple_list = []
for i in range(40):
    for j in range(40):
        tuple_list.append((i, j))
random_heads = random.choices(tuple_list, k=800)

#model_name = "bigscience/bloom-560m"
model_name = "facebook/opt-13b"
model = AutoModel.from_pretrained(model_name, output_attentions=True)#.to(device)  # Configure model to return attention values
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
#prompt = "A man is riding on a horse. he runs after a calf and ties its legs."
prompt = "Write once, run anywhere, right? Yes, Write anywhere, once run"

def get_attention(prompt, model=model, tokenizer=tokenizer, first_token=True):
    inputs = tokenizer.encode(prompt, return_tensors='pt')#.to(device)
    tokens = tokenizer.convert_ids_to_tokens(inputs[0])
    with torch.no_grad():
        outputs = model(inputs) 
        attention = outputs[-1]
    if not first_token:
        attention = delete_first_token(attention)    
    return attention, tokens

att, tok = get_attention(prompt)
#pickle attention
with open('attention.pkl', 'wb') as f:
    pickle.dump(att, f)

def visualize_single(att_map, sentence):
    """
    Attention map for a given layer and head
    """
    
    plt.figure(figsize=(16, 12))
    plt.imshow(att_map, cmap='Reds')
    plt.xticks(range(len(sentence)), sentence, rotation=60, fontsize=12)
    plt.yticks(range(len(sentence)), sentence, fontsize=12)

    plt.grid(False)

inputs = tokenizer.encode(prompt, return_tensors='pt')
tok= tokenizer.convert_ids_to_tokens(inputs[0])
att = pickle.load(open('attention_write_once.pkl', 'rb'))
model_view(att, tok)  # Display model view
#head_view(att, tok, layer=15, heads=[1])

visualize_single(att[0][0, 21, :, :], tok)

clustering = pickle.load(open('clustering.pkl', 'rb'))
clustering

#visualize_all(attention, n_layers=24, n_heads=16, figname='test.png')
#attention_weights = attention[20][:, 12, :, :] # Get the first layer [0], and the first attention head's attention
#visualize_single(attention_weights, tokens)

task = get_prompts_from_file("arc_easy")
model_path = duplicate_prune_model(prompts=task, path="models/", model=model, model_name=model_name, tokenizer=tokenizer, prune_percent=0.5, prune_task="arc_easy", metric="cosine", group_metric="cosine", prune_method="imbalanced", verbose=True)
blimp = load_dataset("nyu-mll/blimp", "ellipsis_n_bar_1", split="train")


#if torch.cuda.is_available():
#    device = torch.device("cuda")
#else:
#    device = torch.device("cpu")
#print(f"Using device: {device}")
model_name = "bigscience/bloom-560m"
model = AutoModel.from_pretrained(model_name, output_attentions=True)#.to(device)  # Configure model to return attention values
tokenizer = AutoTokenizer.from_pretrained(model_name)

with open('/mnt/d/github/wanda/pawsx_en_train_sample.tsv', 'r') as file:
    lines = file.readlines()
    lines = [line.strip().split('\t')[0] for line in lines]

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

test = get_training_data()





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

def visualize_all(attn, n_layers=12, n_heads=12, title="", figname='attention.png'):
    """
    Full grid of attention maps [layers x heads]
    """
    fig, axes = plt.subplots(n_layers, n_heads, figsize=(40, 30), sharex=True, sharey=True)
    
    for i in tqdm(range(n_layers)):
        for j in range(n_heads):
            im = axes[i, j].imshow(attn[i][0, j, :, :], cmap='Oranges')
            axes[i, j].axis('off')
    fig.colorbar(im, ax=axes.ravel().tolist())
    fig.suptitle(title, fontsize=20)
    plt.savefig(figname, dpi=400)

def random_sample(data, num_samples=100):
    random.seed(42)
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

def visualize_n_inputs(attention_dict, layer_head_combinations, n=5, figname='attention.png'):
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

    #fig.suptitle("Attention Maps Visualization", fontsize=16, va='bottom')
    plt.savefig(figname, dpi=400)
    #plt.show()

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

def get_attention_multiple_inputs(prompts, model, tokenizer, first_token=True):
    """Returns tuple of len(layers) attention maps for each layer 
    each attention map is of shape (len(prompts), num_heads, max_seq_len, max_seq_len)"""
    inputs = tokenizer(prompts, return_tensors='pt', padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
        attention = outputs[-1]
    return attention

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
    n_head = model.config.n_head
    n_layers = model.config.n_layer
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
                model = duplicate_prune(model, source_layer=layer_number, source_head=head_to_keep, target_layer=layer_number, target_head=head_to_remove)
                
    if verbose:
        print(counter)
    path = f'{model_name}_{metric}_{prune_percent}'
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)
    return path

def duplicate_prune_model_imbalanced(prompts, model_name, model, tokenizer, prune_percent=0.5, metric='euclidean', verbose=True):
    attentions = get_attention_multiple_inputs(prompts, model, tokenizer)
    n_head = model.config.n_head
    n_layers = model.config.n_layer
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
            model = duplicate_prune(model, source_layer=layer_number, source_head=target_head, target_layer=layer_number, target_head=head_id)

    if verbose:
        print(f'Pruned {len(pruned_heads)} heads out of {total_heads}')

    path = f'{model_name}_{metric}_{prune_percent}'
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)
    return path

with open('pawsx_en_write_out_info.json', 'r', encoding="utf-8") as file:
    data = json.load(file)
prompts = random_sample(data)
# to tsv
with open('pawsx_en_test_sample.tsv', 'w') as file:
    for prompt in prompts:
        file.write(prompt + "\n")

with open('/mnt/d/github/wanda/pawsx_en_train_sample.tsv', 'r') as file:
    lines = file.readlines()
    train = [line.strip().split('\t')[0] for line in lines]

#att, tok = get_attention(prompts[2])
#model_view(attention, tokens)  # Display model view
#head_view(att, tok, layer=15, heads=[1])

#visualize_all(attention, n_layers=24, n_heads=16, figname='test.png')
#attention_weights = attention[20][:, 12, :, :] # Get the first layer [0], and the first attention head's attention
#visualize_single(attention_weights, tokens)
    
pruned_balanced_model = duplicate_prune_model(prompts, 'balanced-bloom-560m', model, tokenizer)
pruned_model = duplicate_prune_model_imbalanced(prompts, 'bloom-560m', model, tokenizer)



attention_maps = [x[0] for x in [get_attention(prompt) for prompt in prompts]]
attention_vectors = attention_vector_multiple_inputs(attention_maps)
distance_matrix = squareform(pdist(attention_vectors, metric='cosine'))
hc_linkage = linkage(distance_matrix, method='ward')
clusters = fcluster(hc_linkage, 5, criterion='distance')
group_dict = get_group_dict(clusters)
attention_dict = attention_dict_multiple_inputs(attention_maps)

for key in group_dict.keys():
    visualize_n_inputs(attention_dict, group_dict[key], figname=f'{key}.png')


# Hierarchical Clustering
plt.figure(figsize=(10, 7))
dendrogram(hc_linkage)
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Head index')
plt.ylabel('Distance')
plt.show()

dbscan = DBSCAN(eps=0.15, min_samples=4, metric='precomputed')
clusters = dbscan.fit_predict(distance_matrix)
print(clusters)

attention_vectors_np = np.array(attention_vectors)
scaler = StandardScaler()
normalized_attention_vectors = scaler.fit_transform(attention_vectors_np)


# Create a t-SNE instance
tsne = TSNE(n_components=2, perplexity=10, n_iter=1000, random_state=42)

# Fit and transform the data
tsne_results = tsne.fit_transform(normalized_attention_vectors)

# Plotting
plt.figure(figsize=(12, 8))
plt.scatter(tsne_results[:, 0], tsne_results[:, 1], alpha=0.5)
plt.title('t-SNE Visualization of Attention Heads')
plt.xlabel('t-SNE Feature 1')
plt.ylabel('t-SNE Feature 2')
plt.show()


###############################




num_heads = 16
#tokens = tokenizer.tokenize(input_text)

fig, axes = plt.subplots(4, 4, figsize=(20, 16))
axes = axes.flatten()  # Flatten the 4x4 grid into a linear array for easy indexing

for i in range(num_heads):
    # Extract the attention weights for the ith head
    attention_weights = attention[20][:, i, :, :].detach().numpy()

    # Plot the heatmap
    sns.heatmap(attention_weights[0], ax=axes[i], annot=False, cbar=False,
                 xticklabels=tokens, yticklabels=tokens, cmap='Reds', linewidths=.5)
    axes[i].set_title(f"Head {i+1}")
    axes[i].set_xlabel("Key Positions")
    axes[i].set_ylabel("Query Positions")

# Adjust layout for better visualization
plt.tight_layout()
plt.show()

fig, axes = plt.subplots(4, 4, figsize=(20, 16))
axes = axes.flatten()  # Flatten the 4x4 grid into a linear array for easy indexing

for i in range(1):
    # Extract the attention weights for the ith head
    attention_weights = attention[0][:, i, :, :].detach().numpy()

    # Plot the heatmap
    ax = sns.heatmap(attention_weights[0], ax=axes[i], annot=False, cbar=False,
                     xticklabels=tokens, yticklabels=tokens,
                     cmap='Reds', linewidths=.5)
    ax.set_title(f"Head {i+1}")
    ax.set_xlabel('')
    ax.set_ylabel('')
    # Only show the labels for [CLS] and [SEP] tokens
    ax.set_xticklabels(tokens, rotation=90, ha='center')
    ax.set_yticklabels(tokens, rotation=0, ha='right')

# Adjust layout for better visualization
plt.tight_layout()
plt.show()


#tokenizer.tokenize(input_text)



##############################

import inseq
from inseq.data.aggregator import SubwordAggregator


model = inseq.load_model("bigscience/bloom-560m", "attention")

out = model.attribute(
    "Hello ladies and",
    generation_args={"max_new_tokens": 9},
    n_steps=500,
    internal_batch_size=50
)

#attention weights associated to tokens by the 4th attention head of the 3rd layer, produced when generating the 2nd target token
out[0].target_attributions[:, 8, 0, 0].show()
out[0].show()

model

inseq.list_feature_attribution_methods()
#token_strings = [token_with_id.token for token_with_id in attribution_output[0].target]
#attribution_output[0].target = model.clean_tokens(tokens=token_strings)
#attribution_output.show()

model = inseq.load_model("gpt2", "integrated_gradients")
model.attribute(
    "Hello ladies and",
    generation_args={"max_new_tokens": 9},
    n_steps=500,
    internal_batch_size=50
).show()