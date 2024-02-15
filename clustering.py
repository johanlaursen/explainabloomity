from utils import tensor_to_vector
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
import matplotlib.pyplot as plt
import random

def clustering(attention, num_clusters=5):
    num_heads = attention[0].shape[1]
    vectors = []
    ids = []

    for layer_idx, tensor in enumerate(attention):
        for head_idx in range(num_heads):
            attention_map = tensor[0, head_idx]
            vector = tensor_to_vector(attention_map)
            vectors.append(vector)
            ids.append((layer_idx, head_idx))

    Z = linkage(vectors, 'ward')

    max_d = num_clusters
    clusters = fcluster(Z, max_d, criterion='maxclust')

    cluster_map = {}
    for head_id, cluster_id in zip(ids, clusters):
        if cluster_id not in cluster_map:
            cluster_map[cluster_id] = []
        cluster_map[cluster_id].append(head_id)

    return cluster_map

def plot_attention_maps(sampled_heads, attention_maps):
    fig, axs = plt.subplots(1, len(sampled_heads), figsize=(20, 4))
    for ax, (layer_idx, head_idx) in zip(axs, sampled_heads):
        attention_map = attention_maps[layer_idx][0, head_idx].numpy()
        ax.imshow(attention_map, cmap='viridis')
        ax.set_title(f'Layer {layer_idx+1} Head {head_idx+1}')
    plt.show()

def sample_heads_from_clusters(cluster_map, n=3):
    sampled_heads = []
    for cluster_id in cluster_map:
        cluster_size = len(cluster_map[cluster_id])
        if cluster_size >= n:
            sampled_heads.append(random.sample(cluster_map[cluster_id], n))
        else:
            # If the cluster has fewer than n elements, take what is available
            sampled_heads.append(cluster_map[cluster_id])
    return sampled_heads