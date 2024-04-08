from utils import *
import os
from tqdm import tqdm

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

def prune_opt(model, layer, head):
    target_opt_block = model.decoder.layers[layer]
    n_layer, n_head = get_model_layers_and_heads(model.config)
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
        t_k[head] =  0
        t_k_b[head] = 0
        t_v[head] = 0
        t_v_b[head] = 0
        t_q[head] = 0
        t_q_b[head] = 0
        t_out[head] = 0
        t_out_b[head] = 0

    model.decoder.layers[layer].self_attn.k_proj.weight = torch.nn.Parameter(t_k.view_as(target_opt_block.self_attn.k_proj.weight))
    model.decoder.layers[layer].self_attn.k_proj.bias = torch.nn.Parameter(t_k_b.view_as(target_opt_block.self_attn.k_proj.bias))
    model.decoder.layers[layer].self_attn.v_proj.weight = torch.nn.Parameter(t_v.view_as(target_opt_block.self_attn.v_proj.weight))
    model.decoder.layers[layer].self_attn.v_proj.bias = torch.nn.Parameter(t_v_b.view_as(target_opt_block.self_attn.v_proj.bias))
    model.decoder.layers[layer].self_attn.q_proj.weight = torch.nn.Parameter(t_q.view_as(target_opt_block.self_attn.q_proj.weight))
    model.decoder.layers[layer].self_attn.q_proj.bias = torch.nn.Parameter(t_q_b.view_as(target_opt_block.self_attn.q_proj.bias))
    model.decoder.layers[layer].self_attn.out_proj.weight = torch.nn.Parameter(t_out.view_as(target_opt_block.self_attn.out_proj.weight))
    model.decoder.layers[layer].self_attn.out_proj.bias = torch.nn.Parameter(t_out_b.view_as(target_opt_block.self_attn.out_proj.bias))
    return model

def prune_bloom(model, layer, head):
    target_bloom_block = model.h[layer]
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
        attention_weight[head] = 0
        attention_bias[head] = 0
        
    model.h[layer].self_attention.query_key_value.weight = torch.nn.Parameter(attention_weight.view_as(target_bloom_block.self_attention.query_key_value.weight))
    model.h[layer].self_attention.query_key_value.bias = torch.nn.Parameter(attention_bias.view_as(target_bloom_block.self_attention.query_key_value.bias))
    return model

def prune(model, prune_dict):
    if "bloom" in model.config._name_or_path:
        prune_fn = prune_bloom
    elif "opt" in model.config._name_or_path:
        prune_fn = prune_opt
    else:
        raise ValueError(f"Model {model.config._name_or_path} not supported")
    
    for layer in prune_dict.keys():
        for head in prune_dict[layer]:
            model = prune_fn(model, layer, head)
    return model

def duplicate_prune(model, source_layer, source_head, target_layer, target_head):
    if "bloom" in model.config._name_or_path:
        model = duplicate_prune_bloom(model, source_layer, source_head, target_layer, target_head)
    elif "opt" in model.config._name_or_path:
        model = duplicate_prune_opt(model, source_layer, source_head, target_layer, target_head)
    else:
        raise ValueError(f"Model {model.config._name_or_path} not supported")
    return model

def duplicate_prune_model(prompts, path, model, model_name, tokenizer, prune_method, prune_task, prune_percent=0.5, metric='euclidean', group_metric='euclidean', verbose=False):
    '''
    Duplicate prunes a model based on the attention scores of the heads.
    The attention scores are calculated for the prompts and the heads are clustered based on cosine similarity.
    The heads within groups are then compared using a metric and the head with the highest similarity to other heads in cluster is kept
    
    Args:
        prompts: list of strings, prompts to calculate attention scores for
        path: str, path to folder where the model is saved
        model: model to prune
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

    # TODO get_clustering dict should use metric
    counter = Counter()
    pruning_log = []
    if prune_method == "balanced":
        layers_clustering_dict = get_clustering_dict(prompts, model, tokenizer,n_layers=n_layers, n_groups=n_groups, n_heads=n_head, metric=metric)
        for layer_number in tqdm(layers_clustering_dict.keys()):
            if group_metric != 'random':
                squaref = squareform(pdist(attentions[layer_number].view(n_head, -1), metric=group_metric))
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
                    if group_metric == 'random':
                        head_to_keep = random.choice(group)
                    else:
                        for head_id in group:
                            for head_id_2 in group:
                                if head_id == head_id_2:
                                    continue
                                group_scores[head_id] += squaref[head_id, head_id_2]
                        head_to_keep = min(group_scores, key=lambda k: group_scores[k])
                        
                for head in group:
                    if head == head_to_keep:
                        continue
                    head_to_remove = head
                    pruning_log.append((layer_number, head_to_keep, head_to_remove))
                    model = duplicate_prune(model, source_layer=layer_number, source_head=head_to_keep, target_layer=layer_number, target_head=head_to_remove)
        if verbose:
            print("size of groups: ", counter)

    elif prune_method == "imbalanced":
        total_heads = n_head * n_layers
        heads_to_prune = int(total_heads * prune_percent)
        head_similarity_scores = []
        for layer_number in range(n_layers):
            if group_metric != 'random':
                squaref = squareform(pdist(attentions[layer_number].view(n_head, -1), metric=metric))
            for head_id in range(n_head):
                if group_metric == 'random':
                    score = random.random()
                else:
                    score = sum(squaref[head_id]) / (n_head - 1)  # Average similarity with other heads
                head_similarity_scores.append((layer_number, head_id, score))

        # Sort by similarity score (lower is better for euclidean, higher for cosine or random)
        head_similarity_scores.sort(key=lambda x: x[2])

        # Prune the required number of heads based on global ranking
        pruned_heads = head_similarity_scores[:heads_to_prune]

        for layer_number, head_id, _ in pruned_heads:
            # Find the most similar head in the same layer to duplicate
            similar_heads = [x for x in head_similarity_scores if x[0] == layer_number and x[1] != head_id]
            if similar_heads:
                # Sort by similarity (most similar first)
                similar_heads.sort(key=lambda x: x[2])
                target_head = similar_heads[0][1]
                model = duplicate_prune(model, source_layer=layer_number, source_head=target_head, target_layer=layer_number, target_head=head_id)
        
        if verbose:
            print(f'Pruned {len(pruned_heads)} heads out of {total_heads}')
                
    # met = metric[:3]
    prune_metric = metric + "_" + group_metric
    model_name = os.path.basename(model_name)
    path_model = f"{model_name}/{prune_method}/{prune_task}/{prune_metric}/{prune_percent}"
    
    model.half()
    model.save_pretrained(path+path_model+"/model")
    save_pruning_log(path_model, pruning_log)
    tokenizer.save_pretrained(path+path_model+"/model")
    return path+path_model