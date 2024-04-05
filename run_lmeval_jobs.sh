#!/bin/bash

path="/home/data_shares/mapillary/thesis_models/pruned_models/"
base_models=(
    "opt-13b"
    
)
prunetasks=(
    "paws_en"
    # "hellaswag"
    # "arc_easy"
)
pruning_method=(
    "balanced"
    # "imbalanced"
)
metrics=(
    # "cosine_cosine"
    # "euclidean_euclidean"
    # "cosine_random"
    "euclidean_random"
)
prune_ratios=(
    # "0.25"
    # "0.5"
    "0.75"
)

for model in "${base_models[@]}"
do
    for prune_percent in "${prune_ratios[@]}"
    do
        for prunetask in "${prunetasks[@]}"
        do
            for prunemethod in "${prune_methods[@]}"
            do
                for metric in "${metrics[@]}"
                do
                    sbatch --job-name="lmeval_${model}_${prunetask}" lmeval.job $prunetask $prunemethod
                    echo "${path}${model}/${prunemethod}/${prunetask}/${metric}/${prune_percent}/model"
                done
            done
        done
    done
done

# sbatch --job-name="lmeval_opt13b_base" lmeval.job "facebook/opt-13b" "opt-13b_base" $tasks