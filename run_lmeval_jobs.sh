#!/bin/bash


# model_path model_name  tasks
# model_path is the path to the model
# model_name is name to save results as
# tasks is a comma separated list of tasks to evaluate on
tasks="lambada_openai,paws_en,hellaswag"

sbatch --job-name="lmeval_bloom560m" lmeval.job "bigscience/bloom-560m" "bloom-560m" $tasks
sbatch --job-name="lmeval_bloom560m_0_25-unstructured" lmeval.job "/home/data_shares/mapillary/thesis_models/bloom-560m/0.25-unstructured" "bloom560m_0_25-unstructured" $tasks
sbatch --job-name="lmeval_bloom560m_0_5-unstructured" lmeval.job "/home/data_shares/mapillary/thesis_models/bloom-560m/0.5-unstructured" "bloom560m_0_5-unstructured" $tasks
sbatch --job-name="lmeval_bloom560m_0_5-2:4" lmeval.job "/home/data_shares/mapillary/thesis_models/bloom-560m/0.5-2:4" "bloom560m_0_5-2:4" $tasks
sbatch --job-name="lmeval_bloom560m_0_5-4:8" lmeval.job "/home/data_shares/mapillary/thesis_models/bloom-560m/0.5-4:8" "bloom560m_0_5-4:8" $tasks

sbatch --job-name="lmeval_bloom560m_clusterpruned_0.5_cosine_cosine_0.5" lmeval.job "models/bloom-560m_clusterpruned_0.5_cosine_cosine_0.5" "bloom-560m_clusterpruned_0.5_cosine_cosine_0.5" $tasks
sbatch --job-name="lmeval_bloom560m_clusterpruned_0.5_euclidean_euclidean_0.5" lmeval.job "models/bloom-560m_clusterpruned_0.5_euclidean_euclidean_0.5" "bloom-560m_clusterpruned_0.5_euclidean_euclidean_0.5" $tasks
sbatch --job-name="lmeval_bloom560m_clusterpruned_0.5_random_random_0.5" lmeval.job "models/bloom-560m_clusterpruned_0.5_random_random_0.5" "bloom-560m_clusterpruned_0.5_random_random_0.5" $tasks
sbatch --job-name="lmeval_bloom560m_clusterpruned_0.25_cosine_cosine_0.25" lmeval.job "models/bloom-560m_clusterpruned_0.25_cosine_cosine_0.25" "bloom-560m_clusterpruned_0.25_cosine_cosine_0.25" $tasks
sbatch --job-name="lmeval_bloom560m_clusterpruned_0.25_euclidean_euclidean_0.25" lmeval.job "models/bloom-560m_clusterpruned_0.25_euclidean_euclidean_0.25" "bloom-560m_clusterpruned_0.25_euclidean_euclidean_0.25" $tasks
sbatch --job-name="lmeval_bloom560m_clusterpruned_0.25_random_random_0.25" lmeval.job "models/bloom-560m_clusterpruned_0.25_random_random_0.25" "bloom-560m_clusterpruned_0.25_random_random_0.25" $tasks
sbatch --job-name="lmeval_bloom560m_clusterpruned_0.75_cosine_cosine_0.75" lmeval.job "models/bloom-560m_clusterpruned_0.75_cosine_cosine_0.75" "bloom-560m_clusterpruned_0.75_cosine_cosine_0.75" $tasks
sbatch --job-name="lmeval_bloom560m_clusterpruned_0.75_euclidean_euclidean_0.75" lmeval.job "models/bloom-560m_clusterpruned_0.75_euclidean_euclidean_0.75" "bloom-560m_clusterpruned_0.75_euclidean_euclidean_0.75" $tasks
sbatch --job-name="lmeval_bloom560m_clusterpruned_0.75_random_random_0.75" lmeval.job "models/bloom-560m_clusterpruned_0.75_random_random_0.75" "bloom-560m_clusterpruned_0.75_random_random_0.75" $tasks