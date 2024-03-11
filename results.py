import json
import matplotlib.pyplot as plt
import os

def load_results(directory):
    all_results = {}
    for folder_name in os.listdir(directory):
        folder_path = os.path.join(directory, folder_name)
        if os.path.isdir(folder_path):
            for file_name in os.listdir(folder_path):
                if file_name.endswith('.json'):
                    file_path = os.path.join(folder_path, file_name)
                    with open(file_path, 'r') as file:
                        data = json.load(file)
                        all_results[folder_name] = data
    return all_results

results_directory = 'results'
results = load_results(results_directory)

num_models = len(results)
num_cols = 2 
num_rows = num_models // num_cols + (num_models % num_cols > 0)
plt.figure(figsize=(30, 5 * num_rows))


for i, (model, data) in enumerate(results.items(), 1):
    plt.subplot(num_rows, num_cols, i)
    
    datasets = data['results'].keys()
    accuracies = [data['results'][dataset]['acc,none'] for dataset in datasets]

    plt.bar(datasets, accuracies, color='skyblue')
    plt.xlabel('Dataset')
    plt.ylabel('Accuracy')
    plt.title(f'Accuracy for {model}')
    plt.ylim(0, 1)  # Assuming accuracy values are between 0 and 1

plt.tight_layout()
plt.show()

model = 'bloom-560m_clusterpruned_0.25_cosine_cosine_0.25'

datasets = results[model]['results'].keys()
accuracies = [results[model]['results'][dataset]['acc,none'] for dataset in datasets]

plt.figure(figsize=(10, 6))
plt.bar(datasets, accuracies, color='skyblue')
plt.xlabel('Dataset')
plt.ylabel('Accuracy')
plt.title('Accuracy for Each Dataset')
plt.ylim(0, 1)  # Assuming accuracy values are between 0 and 1
plt.show()