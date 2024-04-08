import os
import json
import csv

def load_results_and_aggregate(directory):
    all_metrics = set()
    all_results = {}

    # Collect all metrics and aggregate results
    for folder_name in os.listdir(directory):
        folder_path = os.path.join(directory, folder_name)
        if os.path.isdir(folder_path):
            for file_name in os.listdir(folder_path):
                if file_name.endswith('.json'):
                    file_path = os.path.join(folder_path, file_name)
                    with open(file_path, 'r') as file:
                        data = json.load(file)
                        results = data.get("results", {})
                        for metric, values in results.items():
                            all_metrics.add(metric)
                            key = (folder_name, file_name)
                            if key not in all_results:
                                all_results[key] = {}
                            all_results[key][metric] = values.get("acc,none", "N/A")

    return all_metrics, all_results

def print_aggregated_table(all_metrics, all_results):
    if not all_results:
        print("No results to display.")
        return

    folder_name_width = max(len(result[0]) for result in all_results) + 2


    # Prepare the header
    header_parts = [f"{'Folder Name':<{folder_name_width}}"] + sorted(all_metrics)
    header = "\t".join(header_parts)
    print(header)
    print('-' * len(header))

    # Print each row
    for (folder_name, file_name), metrics in sorted(all_results.items()):
        row_parts = [f"{folder_name:<{folder_name_width}}"] + [str(metrics.get(metric, "N/A")) for metric in sorted(all_metrics)]
        print("\t".join(row_parts))
        
        
        
def save_results_to_csv(all_metrics, all_results, output_file='aggregated_results.csv'):
    if not all_results:
        print("No results to save.")
        return

    with open(output_file, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        # Write the header
        header = ["Folder Name", "File Name"] + sorted(all_metrics)
        writer.writerow(header)

        # Write each row
        for (folder_name, file_name), metrics in sorted(all_results.items()):
            row = [folder_name, file_name] + [metrics.get(metric, "N/A") for metric in sorted(all_metrics)]
            writer.writerow(row)

    print(f"Results saved to {output_file}")

# Example usage
directory = "results"
all_metrics, all_results = load_results_and_aggregate(directory)
print_aggregated_table(all_metrics, all_results)
save_results_to_csv(all_metrics, all_results)