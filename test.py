import seaborn as sns
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModel, utils
import torch
from tqdm import tqdm
#from bertviz import model_view, head_view
utils.logging.set_verbosity_error()  # Suppress standard warnings


#if torch.cuda.is_available():
#    device = torch.device("cuda")
#else:
#    device = torch.device("cpu")
#print(f"Using device: {device}")
model_name = "bigscience/bloom-560m"
input_text = "Mrs. Dursley, Mr. Dursley, Dudley Dursley"  
model = AutoModel.from_pretrained(model_name, output_attentions=True)#.to(device)  # Configure model to return attention values
tokenizer = AutoTokenizer.from_pretrained(model_name)
inputs = tokenizer.encode(input_text, return_tensors='pt')#.to(device)
with torch.no_grad():
    outputs = model(inputs)  # Run model
    attention = outputs[-1]  # Retrieve attention from model outputs
tokens = tokenizer.convert_ids_to_tokens(inputs[0])  # Convert input ids to token strings
#head_view(attention, tokens)  # Display model view
attention_weights = attention[18][:, 13, :, :] # Get the first layer [0], and the first attention head's attention

def visualize_single(att_map, sentence):
    """
    Attention map for a given layer and head
    """
    
    plt.figure(figsize=(16, 12))
    plt.imshow(att_map, cmap='Reds')
    plt.xticks(range(len(sentence)), sentence, rotation=60, fontsize=12)
    plt.yticks(range(len(sentence)), sentence, fontsize=12)

    plt.grid(False)

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

visualize_all(attention, n_layers=24, n_heads=16)
visualize_single(attention_weights[0], tokens)







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
