#!/bin/bash


# model_name="bigscience/bloom-7b1"
# model_name="facebook/opt-6.7b"
model_name="facebook/opt-13b"

model_basename="${model_name##*/}"
path="/home/data_shares/mapillary/thesis_models/pruned_models"

# Note if it is balanced or unbalanced need to change prune.job
# prunetype="unbalanced_prune" # unbalanced_prune or pruned
prunetype="prune"
pathpruned="${path}/${model_basename}_${prunetype}"

metrics=(
    "cosine"
    # "euclidean"
    # "random"
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
        case "$prunetype" in
            prune)
                sbatch --job-name="${prunetype}_${model_basename}_${metric}_${prune_percent}" prune.job $prune_percent $metric $model_name $pathpruned
                ;;
            unbalanced_prune)
                sbatch --job-name="${prunetype}_${model_basename}_${metric}_${prune_percent}" unbalanced_prune.job $prune_percent $metric $model_name $pathpruned
                ;;
        esac
        # Note if it is balanced or unbalanced
    done
done