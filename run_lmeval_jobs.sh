#!/bin/bash


# model_path model_name  tasks
# model_path is the path to the model
# model_name is name to save results as
# tasks is a comma separated list of tasks to evaluate on
tasks="lambada_openai,paws_en,hellaswag"


path="/home/data_shares/mapillary/thesis_models/pruned_models/"
models=(
    "opt-13b_prune_cosine_0.25"
    "opt-13b_prune_cosine_0.5"
    "opt-13b_prune_cosine_0.75"
    "opt-13b_prune_euclidean_0.25"
    "opt-13b_prune_euclidean_0.5"
    "opt-13b_prune_euclidean_0.75"
    "opt-13b_prune_random_0.25"
    "opt-13b_prune_random_0.5"
    "opt-13b_prune_random_0.75"
    "opt-13b_unbalanced_prune_cosine_0.25"
    "opt-13b_unbalanced_prune_cosine_0.5"
    "opt-13b_unbalanced_prune_cosine_0.75"
)

for model in "${models[@]}"
do
    sbatch --job-name="lmeval_${model}" lmeval.job "${path}${model}" "${model}" $tasks
done

sbatch --job-name="lmeval_opt13b_base" lmeval.job "facebook/opt-13b" "opt-13b_base" $tasks