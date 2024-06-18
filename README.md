# explainabloomity


## Folders

### head_importance

Pickle files containing head importance for OPT model

### images

saved figures of results and analysis

### lm-evaluation-harness

github of lm-eval [Github link](https://github.com/EleutherAI/lm-evaluation-harness/tree/master)

### logs

Folder containing all SLURM logs of various runs. Kept locally due to file size.

### notebooks

Contains all jupyter notebook used for analysis and exploration

### pruning_logs

All pruning logs for each model experiment run to keep track of which heads were pruned. Used for analysis as well as determining which heads to prune using masked pruning.

### results

Folder of results. File path to find specific result is:
    - Lmeval metric used for evaluation
    - pruning method i.e balanced/imbalanced
    - Lmeval metric used for prompts for pruning
    - Model pruned

### scripts

Contains bash scripts used to run jobs and clean up logs.

### tasks

tsv files containing prompts used for pruning

## Files

`clustering.py` contains a couple of utility functions

`eval.py` deprecated contains utility function used to run lmeval library

`eval_mask.py` script used to prune model using masked method by loading pruning log. Not used in final results see `main.py`

`eval_mask_random.py` script used to randomely prune heads in model

`main.py` script used to duplicate prune and mask prune for final results

`prompt_testing.py` script used for initial exploration

`prune.py` contains all pruning functions for bloom and opt

`resultstable.py` Unused file for loading and saving results to csv from results folder

`utils.py` contains all util functions used for pruning/visualization/analysis

