import inseq
from inseq.data.aggregator import SubwordAggregator
from transformers import AutoTokenizer
import seaborn as sns
import matplotlib.pyplot as plt

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


from transformers import AutoTokenizer, AutoModel, utils
from bertviz import model_view, head_view
utils.logging.set_verbosity_error()  # Suppress standard warnings

model_name = "bigscience/bloom-560m"
input_text = "States must show reasonable progress in their state implementation plans toward the congressionally mandated goal of returning to natural conditions in national parks and wilderness areas."  
model = AutoModel.from_pretrained(model_name, output_attentions=True)  # Configure model to return attention values
tokenizer = AutoTokenizer.from_pretrained(model_name)
inputs = tokenizer.encode(input_text, return_tensors='pt')  # Tokenize input text
outputs = model(inputs)  # Run model
attention = outputs[-1]  # Retrieve attention from model outputs
tokens = tokenizer.convert_ids_to_tokens(inputs[0])  # Convert input ids to token strings
head_view(attention, tokens)  # Display model view
#attention_weights = attention[0][:, 0, :, :] # Get the first layer [0], and the first attention head's attention


num_heads = 16
text = input_text.split()

fig, axes = plt.subplots(4, 4, figsize=(20, 16))
axes = axes.flatten()  # Flatten the 4x4 grid into a linear array for easy indexing

for i in range(num_heads):
    # Extract the attention weights for the ith head
    attention_weights = attention[20][:, i, :, :].detach().numpy()

    # Plot the heatmap
    sns.heatmap(attention_weights[0], ax=axes[i], annot=True, fmt=".2f", xticklabels=text, yticklabels=text)
    axes[i].set_title(f"Head {i+1}")
    axes[i].set_xlabel("Key Positions")
    axes[i].set_ylabel("Query Positions")


# Adjust layout for better visualization
plt.tight_layout()
plt.show()