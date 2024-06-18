#!/bin/bash

#SBATCH --account=researchers
#SBATCH --output=logs/lmeval/R-%x.%j.out      # Name of output file (%j expands to jobId)
#SBATCH --error=logs/lmeval/R-%x.%j.err     # Error handling
#SBATCH --nodes=1                # Total number of nodes requested
#SBATCH --cpus-per-task=8        # Schedule 8 cores (includes hyperthreading)
#SBATCH --mem=64G
#SBATCH --constraint="gpu_rtx8000|gpu_a100_40gb|gpu_v100" # Use either a v100 or a100
#SBATCH --gres=gpu:1      #v100:1 or a100_40gb:1 on brown
#SBATCH --time=0-12:15:00          # Run time (hh:mm:ss) 
#SBATCH --partition=red,brown    # Run on either the red or brown queue
#SBATCH --dependency=afterok:176233:176234:176235

#srun hostname

nvidia-smi
module load Anaconda3
conda activate lmeval

# model_path="bigscience/bloom-560m"
# model_name="bloom-560m"
# tasks="lambada_openai,paws_en,hellaswag"

# python eval.py $1 $2
# model=$1
# model="facebook/opt-13b"
# model_args="pretrained=${model},dtype=float16"
tasks=(
    "lambada_openai"
    "paws_en"
    "hellaswag"
    "arc_easy"
    "blimp_ellipsis_n_bar_1"
    "blimp_irregular_plural_subject_verb_agreement_1"
)
output_path=""

metrics=(
    "cosine_cosine"
    # "euclidean_euclidean"
    # "cosine_random"
    # "euclidean_random"
)
prune_ratios=(
    "0.25"
    "0.5"
    "0.75"
)
path="/home/data_shares/mapillary/thesis_models/pruned_models/"
model="opt-13b"
prunetask=$1
prunemethod=$2

echo "model=${model}"
echo "prunetask=${prunetask}"
echo "prune_method=${prunemethod}"

for task in "${tasks[@]}"
do
    for metric in "${metrics[@]}"
    do
        for prune_percent in "${prune_ratios[@]}"
        do
            model_path="${model}/${prunemethod}/${prunetask}/${metric}/${prune_percent}"
            model_args="pretrained=${path}${model_path}/model,dtype=float16"
            echo "path=${path}${model_path}/model"
            lm_eval --model "hf" \
            --model_args $model_args  \
            --tasks $task  \
            --batch_size "auto" \
            --device "cuda:0" \
            --output_path "results/${task}/${model_path}" \
            --num_fewshot "0" \
            --write_out
        done
    done
done