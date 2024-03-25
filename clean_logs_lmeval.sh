#!/bin/bash

# Directory containing the log files
log_dir="./logs/lmeval"

# Navigate to the directory
cd "$log_dir" || exit

# Create an associative array to store the highest job number for each base name
declare -A base_names

# Loop through the log files
for file in R-prune_*; do
    # Extract the base name and job number
    base_name=$(echo "$file" | sed 's/\(.*\)\.[0-9]*\.(out|err)$/\1/')
    job_number=$(echo "$file" | grep -oE '[0-9]+\.(out|err)$$' | sed 's/\.(out|err)$//')

    # Check if this base name has been seen before and if the current job number is higher
    if [[ -z ${base_names[$base_name]} ]] || [[ ${job_number} -gt ${base_names[$base_name]} ]]; then
        # Update the highest job number for this base name
        base_names[$base_name]=$job_number
    fi
done

# Now loop through the files again to delete those with smaller job numbers
for file in R-prune_*; do
    # Extract the base name and job number again
    base_name=$(echo "$file" | sed 's/\(.*\)\.[0-9]*\.(out|err)$/\1/')
    job_number=$(echo "$file" | grep -oE '[0-9]+\.(out|err)$$' | sed 's/\.(out|err)$//')

    # If the job number is smaller than the highest job number for this base name, delete the file
    if [[ ${job_number} -lt ${base_names[$base_name]} ]]; then
        echo "Deleting $file"
        rm "$file"
    fi
done
