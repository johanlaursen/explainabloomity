import subprocess


# Install lm-evaluation-harness from EleutherAI using the following commands:
# git clone https://github.com/EleutherAI/lm-evaluation-harness
# cd lm-evaluation-harness
# pip install -e .


def evaluate(model_path,
            model_name,
            task="", # TODO figure out list of tasks to use
            device = "cuda:0",
            ):
    """
    Calls lm_eval and saves results to results folder with model_name

    ARGS:
        - model_path: str, path to model folder
        - model_name: str, name of model for saving lm_cache and results
        - task: str, task or list of tasks to evaluate on in format "task1,task2,task3"
    RETURNS:
        - None"""

    command = [
    "lm_eval",
    "--model", "hf",
    "--model_args", f"pretrained={model_path}",
    "--tasks", task,
    "--batch_size", "auto",
    "--max_batch_size", 64,
    "--use_cache", f"lm_cache/{model_name}",
    "--device", device,
    "--output_path", f"results/{model_name}",
    "--num_fewshot", 0
    ]
    try:
        subprocess.run(command)
    except:
        print("Error running lm_eval")
