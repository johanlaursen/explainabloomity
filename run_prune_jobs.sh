#!/bin/bash


# model_name="bigscience/bloom-7b1"
# model_name="bigscience/bloom-7b1"
# model_name="facebook/opt-6.7b"
model_name="facebook/opt-13b"
model_name="facebook/opt-13b"

model_basename="${model_name##*/}"
path="/home/data_shares/mapillary/thesis_models/pruned_models/"

# Note if it is balanced or unbalanced need to change prune.job
# prunetype="unbalanced_prune" # unbalanced_prune or pruned
prune_methods=(
    "balanced"
    # "imbalanced"
)
metrics=(
    # "cosine"
    "euclidean"
)
prunetasks=(
    #"paws_en"
    "hellaswag"
    # "arc_easy"
)
prune_percents=(
    "0.25"
    # "0.5"
    # "0.75"
)

for metric in "${metrics[@]}"
do
    for prune_percent in "${prune_percents[@]}"
    do
        for prunetask in "${prunetasks[@]}"
        do
            for prunemethod in "${prune_methods[@]}"
            do
                sbatch --job-name="${prunemethod}_${model_basename}_${metric}_${prune_percent}_${prompt}" prune.job $prune_percent $metric $model_name $path $prunetask $metric $prunemethod
                # sbatch --job-name="${prunetype}_${model_basename}_${metric}_${prune_percent}_${prompt}" prune.job $prune_percent $metric $model_name $path $prunetask "random" $prunemethod
            done
        done
        # Note if it is balanced or unbalanced
    done
done