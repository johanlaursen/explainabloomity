import json
import matplotlib.pyplot as plt
import os
from collections import defaultdict

def load_results(directory):
    all_results = {}
    for folder_name in os.listdir(directory):
        folder_path = os.path.join(directory, folder_name)
        if os.path.isdir(folder_path):
            for file_name in os.listdir(folder_path):
                if file_name.endswith('.json'):
                    file_path = os.path.join(folder_path, file_name)
                    with open(file_path, 'r') as file:
                        try:
                            data = json.load(file)
                            all_results[folder_name] = data
                        except:
                            print(f'Error loading {file_path}')
    return all_results

results_directory = 'results'
results = load_results(results_directory)

model = 'bloom-560m'
datasets = results[model]['results'].keys()
dataset_accuracies = defaultdict(dict)

for model in results.keys():
    for dataset in datasets:
        dataset_accuracies[dataset][model] = results[model]['results'][dataset]['acc,none']

# remove 'bloom-560m' as key from the dictionary
dataset_accuracies = {k: v for k, v in dataset_accuracies.items() if k != 'bloom-560m'}

pruning_percentages = [0, 0.25, 0.5, 0.75]
pruning_methods = ['clusterpruned_cosine', 
                   'clusterpruned_random', 
                   #'clusterpruned_euclidean', 
                   'wanda_unstructured', 
                   'wanda_2-4', 
                   'wanda_4-8']

model_accuracies = {model: [] for model in pruning_methods if model != 'bloom-560m'}


fig, axes = plt.subplots(len(datasets), 1, figsize=(10, 15), sharex=True)
for i, (dataset, models) in enumerate(dataset_accuracies.items()):
    original_accuracy = dataset_accuracies[dataset]['bloom-560m']
    axes[i].axhline(y=original_accuracy, color='r', linestyle='dotted', label='bloom-560m')

    # Initialize the dictionary for each dataset
    for model in model_accuracies.keys():
        model_accuracies[model] = []

    # Collect accuracies for each model
    for model, accuracy in models.items():
        if model != 'bloom-560m':
            pruning_percentage = float(model.split('_')[2])
            model_name = f"{model.split('_')[1]}_{model.split('_')[3]}"
            try:
                model_accuracies[model_name].append((pruning_percentage, accuracy))
            except:
                pass
    # Plot lines or points
    for model_name, values in model_accuracies.items():
        values.sort(key=lambda x: x[0])  # Sort by pruning percentage
        if len(values) > 1:
            # If multiple values, plot a line
            x_values, y_values = zip(*values)
            axes[i].plot(x_values, y_values, label=model_name)
        elif values:
            # If only one value, plot a point
            x_value, y_value = values[0]
            axes[i].plot(x_value, y_value, 'o', label=model_name)

    axes[i].set_title(f'{dataset} Dataset')
    axes[i].set_ylabel('Accuracy')

axes[-1].set_xlabel('Pruning Percentage')
plt.xticks(pruning_percentages, ['0%', '25%', '50%', '75%'])
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='center right')

plt.tight_layout()
plt.show()