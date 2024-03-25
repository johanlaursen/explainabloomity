import subprocess
import sys

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
        - model_name: str, name of model for saving results
        - task: str, task or list of tasks to evaluate on in format "task1,task2,task3"
    RETURNS:
        - None"""

    command = [
    "lm_eval",
    "--model", "hf",
    "--model_args", f"pretrained={model_path},dtype=float16",
    "--tasks", task,
    "--batch_size", "auto",
    "--max_batch_size", "64",
    # "--use_cache", f"lm_cache/{model_name}", # Massive slowdown from this
    "--device", device,
    "--output_path", f"results/{model_name}",
    "--num_fewshot", "0",
    "--write_out"
    ]
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running lm_eval: {e}")
        print(f"Command '{' '.join(e.cmd)}' returned non-zero exit status {e.returncode}.")
        if e.output:
            print(f"Output: {e.output}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    model_path = sys.argv[1]
    model_name = sys.argv[2]
    task = sys.argv[3]
    evaluate(model_path, model_name, task)
    