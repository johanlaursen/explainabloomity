#!/bin/bash

path="/home/data_shares/mapillary/thesis_models/pruned_models/"
base_models=(
    "opt-13b"
    
)
prunetasks=(
    # "paws_en"
    "hellaswag"
    # "arc_easy"
    # "blimp_ellipsis_n_bar_1"
)
pruning_methods=(
    # "balanced"
    # "imbalanced"
    # "imbalanced_amazon"
    "imbalanced_correct"
)

for model in "${base_models[@]}"
do
    for prunetask in "${prunetasks[@]}"
    do
        for prunemethod in "${pruning_methods[@]}"
        do
            
            sbatch --job-name="lmeval_${model}_${prunetask}" lmeval.job $prunetask $prunemethod
            # echo "${path}${model}/${prunemethod}/${prunetask}/${metric}/${prune_percent}/model" 
        done
    done
done

# sbatch --job-name="lmeval_opt13b_base" lmeval.job "facebook/opt-13b" "opt-13b_base" $tasks