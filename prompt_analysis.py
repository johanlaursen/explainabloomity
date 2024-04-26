from utils import *
from tqdm import tqdm
from collections import Counter

# model = AutoModel.from_pretrained("facebook/opt-13b", output_attentions=True)
# model.eval()
# tokenizer = AutoTokenizer.from_pretrained("facebook/opt-13b")

model_name = "opt-13b"
prune_task = "hellaswag"
path = f"/home/data_shares/mapillary/prompts/{model_name}/{prune_task}_attention_maps.pkl"
with open(path, "rb") as f:
    attention_maps = pickle.load(f)

counter = Counter()
attention_maps_list = []
for i in range(100):
    inner_list = [] 
    for tensor in attention_maps:
        sliced_tensor = tensor[i:i+1]  
        inner_list.append(sliced_tensor)
    attention_maps_list.append(inner_list)
n_head = 40
percents = [0.25, 0.5]
for prune_percent in percents:
    n_groups = get_amazon_prune_groups(head_percent=prune_percent)
    for idx, attention_map in enumerate(attention_maps_list):
        pruning_log = []
        # Don't need to provide prompts, model or tokenizer
        layers_clustering_dict, attentions, attention_vectors = get_clustering_dict([], None, None, n_layers=40, n_heads=40, n_groups=n_groups, metric="cosine",prune_percent=prune_percent, prompt_analysis=True, attention_maps=attention_map)
        for layer_number in tqdm(layers_clustering_dict.keys()):
            layer_heads = attention_vectors[layer_number*n_head:(layer_number+1)*n_head]
            squaref = squareform(pdist(layer_heads, metric="cosine"))
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
        
        
        path_model = f"{model_name}/imbalanced_amazon/{prune_task}/cosine/{prune_percent}/{idx}"
        save_pruning_log(path_model, pruning_log)

        print(f"size of groups for prompt {idx}: ", counter)