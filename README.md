# explainabloomity

Exploring attention output similarity for duplication pruning of BLOOM and OPT.

This repository contains the code, data, and analysis for the Master's thesis:
[Exploring attention output similarity for duplication pruning of BLOOM and OPT](Attention_Head_Pruning.pdf).

## Table of Contents

- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [Directory Structure](#directory-structure)
- [Results and Analysis](#results-and-analysis)

## Prerequisites

- Conda (Anaconda or Miniconda)
- CUDA-enabled GPU (for model evaluation)

## Installation

1. Create and activate the Conda environment:

   ```bash
   conda env create -f environment.yml
   conda activate thesis
   ```

2. (Optional) Install additional Python dependencies via pip:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Pruning & Evaluation Pipeline

The primary pipeline is implemented in `main.py`. It performs duplication pruning, optional masked pruning,
and evaluation on benchmark tasks using the EleutherAI LM Evaluation Harness.

```bash
python main.py <metric> <model_name> <output_path> <task> <group_metric> <prune_method>
```

- `<metric>`: distance metric for pruning (`euclidean`, `cosine`, or `random`)
- `<model_name>`: HuggingFace model ID (e.g., `bigscience/bloom-560m`, `facebook/opt-13b`)
- `<output_path>`: directory to save pruned models and evaluation logs
- `<task>`: evaluation task (see `tasks/` for available TSVs)
- `<group_metric>`: metric to group prompts (e.g., same as `<metric>` or `random`)
- `<prune_method>`: strategy (`balanced` or `imbalanced`)

Example:

```bash
python main.py euclidean bigscience/bloom-560m ./results/bloom-560m euclidean balanced
```

### Notebooks

View and run Jupyter notebooks for data exploration and result visualization:

```bash
jupyter lab notebooks/
```

### Utility Scripts

Simplify job submission and cleanup on HPC clusters using the `scripts/` directory:

- `scripts/run_prune_jobs.sh`
- `scripts/run_lmeval_jobs.sh`
- `scripts/clean_logs_prune.sh`
- `scripts/clean_logs_lmeval.sh`

## Directory Structure

```text
├── Attention_Head_Pruning.pdf      # Master's thesis document
├── environment.yml                 # Conda environment specification
├── main.py                         # Pipeline for pruning & evaluation
├── prune.py                        # Core pruning algorithms
├── prompt.py                       # Prompt parsing utilities
├── utils.py                        # Common utility functions
├── clustering.py                   # Head clustering routines
├── eval_mask*.py                   # Masked pruning evaluation scripts
├── eval_f.py                       # Extended evaluation helpers
├── prompt_testing.py               # Initial prompt experiments
├── resultstable.py                 # Result table I/O
├── head_importance/                # Pickle files of precomputed head importance
├── tasks/                          # TSV files defining tasks/prompts
├── images/                         # Generated figures and plots
├── notebooks/                      # Jupyter notebooks for analyses
├── pruning_logs/                   # Logs detailing pruned heads
├── scripts/                        # Job submission & cleanup scripts
└── jobs/                           # SLURM job scripts for experiments
```

## Results and Analysis

- Figures in `images/` illustrate pruning impacts across layers, models, and metrics.
- Notebooks in `notebooks/` provide in-depth explorations.
