import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModel, utils
import torch
import json
import random
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

#if torch.cuda.is_available():
#    device = torch.device("cuda")
#else:
#    device = torch.device("cpu")
#print(f"Using device: {device}")
model_name = "bigscience/bloom-560m"
model = AutoModel.from_pretrained(model_name, output_attentions=True)#.to(device)  # Configure model to return attention values
tokenizer = AutoTokenizer.from_pretrained(model_name)

def visualize_single(att_map, sentence):
    """
    Attention map for a given layer and head
    """
    
    plt.figure(figsize=(16, 12))
    plt.imshow(att_map[0], cmap='Reds')
    plt.xticks(range(len(sentence)), sentence, rotation=60, fontsize=12)
    plt.yticks(range(len(sentence)), sentence, fontsize=12)

    plt.grid(False)


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

with open('pawsx_en_write_out_info.json', 'r', encoding="utf-8") as file:
    data = json.load(file)
prompts = random_sample(data)

#att, tok = get_attention(prompts[2])
#model_view(attention, tokens)  # Display model view
#head_view(att, tok, layer=15, heads=[1])

#visualize_all(attention, n_layers=24, n_heads=16, figname='test.png')
#attention_weights = attention[20][:, 12, :, :] # Get the first layer [0], and the first attention head's attention
#visualize_single(attention_weights, tokens)

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
