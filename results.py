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

##########


import json
import matplotlib.pyplot as plt
import os
from collections import defaultdict

def load_results(directory):
    all_results = defaultdict(lambda: defaultdict(dict))
    # Directory names are like '0shot_arc_easy', '0shot_hellaswag', etc.
    for dir_name in os.listdir(directory):
        dir_path = os.path.join(directory, dir_name)
        if os.path.isdir(dir_path):
            # File names are like '0shot_25percent', '0shot_50percent', etc.
            for file_name in os.listdir(dir_path):
                file_path = os.path.join(dir_path, file_name)
                try:
                    with open(file_path, 'r') as file:
                        data = json.load(file)
                        # Extract pruning percentage from file name
                        pruning_percent = file_name.split('_')[1].replace('percent', '')
                        # Store data based on dir_name and pruning percent
                        for dataset, metrics in data['results'].items():
                            if 'acc' in metrics:
                                all_results[dataset][dir_name][pruning_percent] = metrics['acc']
                except Exception as e:
                    print(f'Error loading {file_path}: {e}')
    return all_results

# Path to the directory containing the results
results_directory = 'results/opt-13b_amazon'
all_data = load_results(results_directory)

# Plotting
fig, axes = plt.subplots(len(all_data), 1, figsize=(10, 15), sharex=True)

for i, (dataset, dir_data) in enumerate(all_data.items()):
    for dir_name, pruning_data in dir_data.items():
        x_values = sorted(pruning_data.keys(), key=lambda x: float(x))
        y_values = [pruning_data[x] for x in x_values]

        # Converting string percentages to float for plotting
        x_values = [float(x)/100 for x in x_values]

        axes[i].plot(x_values, y_values, label=dir_name)

    axes[i].set_title(f'{dataset} Dataset')
    axes[i].set_ylabel('Accuracy')

axes[-1].set_xlabel('Pruning Percentage')
plt.xticks([0, 0.25, 0.5, 0.75], ['0%', '25%', '50%', '75%'])

# Creating a unified legend
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='center right')

plt.tight_layout()
plt.show()


