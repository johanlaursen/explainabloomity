import matplotlib.pyplot as plt
import torch

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

def attention_multiple_inputs(prompts):
    """
    Takes in a list of prompts and returns the attention maps for each prompt
    f.x. prompts = random_sample(data)
    """
    attention_maps = [x[0] for x in [get_attention(prompt) for prompt in prompts]]
    head_attentions = dict()
    for attention_map in attention_maps:
        for layer_idx, layer in enumerate(attention_map):
            for head_idx in range(layer.shape[1]):
                att_map_vector = tensor_to_vector(layer[0, head_idx, :, :])
                if (layer_idx, head_idx) not in head_attentions:
                    head_attentions[(layer_idx, head_idx)] = att_map_vector
                else:
                    head_attentions[(layer_idx, head_idx)] = torch.cat((head_attentions[(layer_idx, head_idx)], att_map_vector))
    # Convert head_attentions dict to a numpy array of np.arrays
    head_attentions = np.array(list(head_attentions.values()))
    return head_attentions