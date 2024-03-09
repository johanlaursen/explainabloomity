#!/bin/bash


# model_path model_name tasks

tasks="lambada_openai,paws_en,hellaswag"

sbatch --job-name="lmeval_bloom560m" lmeval.job "bigscience/bloom-560m" "bloom-560m" $tasks
sbatch --job-name="lmeval_bloom560m_0_25-unstructured" lmeval.job "bloom560m_0_25-unstructured" "/home/data_shares/mapillary/thesis_models/bloom-560m/0.25-unstructured" $tasks
sbatch --job-name="lmeval_bloom560m_0_5-unstructured" lmeval.job "bloom560m_0_5-unstructured" "/home/data_shares/mapillary/thesis_models/bloom-560m/0.5-unstructured" $tasks
sbatch --job-name="lmeval_bloom560m_0_5-2:4" lmeval.job "bloom560m_0_5-2:4" "/home/data_shares/mapillary/thesis_models/bloom-560m/0.5-2:4" $tasks
sbatch --job-name="lmeval_bloom560m_0_5-4:8" lmeval.job "bloom560m_0_5-4:8" "/home/data_shares/mapillary/thesis_models/bloom-560m/0.5-4:8" $tasks

