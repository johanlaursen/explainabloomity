#!/bin/bash


# model_name="bigscience/bloom-7b1"
# model_name="bigscience/bloom-7b1"
# model_name="facebook/opt-6.7b"
model_name="facebook/opt-13b"

model_basename="${model_name##*/}"
# path="/home/data_shares/mapillary/thesis_models/pruned_models/"


prune_methods=(
    # "balanced"
    # "imbalanced"
    # "imbalanced_amazon"
    "imbalanced_correct"
)
metrics=(
    "cosine"
    # "euclidean"
)
prunetasks=(
    # "paws_en"
    "hellaswag"
    # "arc_easy"
    # "blimp_ellipsis_n_bar_1"
)

for metric in "${metrics[@]}"
do
    for prunetask in "${prunetasks[@]}"
    do
        for prunemethod in "${prune_methods[@]}"
        do
            sbatch --job-name="${prunemethod}_${model_basename}_${metric}_${prunetask}" prune_lmeval.job $metric $model_name $prunetask $metric $prunemethod
            # sbatch --job-name="${prunemethod}_${model_basename}_${metric}_${prune_percent}_${prompt}" prune.job $prune_percent $metric $model_name $path $prunetask "random" $prunemethod
        done
    done
done