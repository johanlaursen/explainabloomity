from lm_eval.evaluator import simple_evaluate
from lm_eval.models import huggingface
from utils import *
from transformers import AutoModel
import torch
from pathlib import Path
import json
from collections import defaultdict
from prune import prune

def _handle_non_serializable(o):
    if isinstance(o, np.int64) or isinstance(o, np.int32):
        return int(o)
    elif isinstance(o, set):
        return list(o)
    else:
        return str(o)


path = "/home/data_shares/mapillary/thesis_models/pruned_models"
models = (
    "opt-13b",
    # "bloom-7b1",
    )
prune_methods=(
    # "balanced",
    "imbalanced_correct" ,
)
metrics=(
    "cosine_cosine" ,
    # "euclidean_euclidean" ,
    # "cosine_random",
    # "euclidean_random",
)
prunetasks=(
    # "paws_en",
    "hellaswag",
    # "arc_easy",
    # "blimp_ellipsis_n_bar_1",
)

prune_percents=(
    # "0.25",
    # "0.5",
    # "0.75",
    "0.1",
    "0.2",
    "0.3",
    "0.4",
    "0.5",
)
tasks=(
    # "lambada_openai",
    # "paws_en",
    "hellaswag",
    "arc_easy",
    # "blimp_ellipsis_n_bar_1",
    # "blimp_irregular_plural_subject_verb_agreement_1"
)
for model in models:
    for prune_method in prune_methods:
        for metric in metrics:
            for prunetask in prunetasks:
                for prune_percent in prune_percents:
                    path_log = Path("pruning_logs") / model / prune_method / prunetask / metric / prune_percent / "pruning_log.txt"
                    prune_method_path = prune_method + "_mask"
                    model_path = Path(model) / prune_method_path / prunetask / metric / prune_percent
                    if "opt" in model:
                        model_name = f"facebook/{model}"
                    elif "bloom" in model:
                        model_name = f"bigscience/{model}"
                    pruning_log = []
                    pruning_dict = defaultdict(list)
                    layers, heads = get_model_layers_and_heads(model)
                    try:
                        with open (path_log, "r") as f:
                            lines = f.readlines()
                            for line in lines:
                                if prune_method == "imbalanced":
                                    layer_keep, head_to_keep, layer, head_to_prune = map(int,line.strip().split(","))
                                else:
                                    layer, head_to_keep, head_to_prune = map(int,line.strip().split(","))
                                pruning_log.append((layer, head_to_prune))
                    except:
                        print("No pruning log found for: ", model, prune_method, metric, prunetask, prune_percent)
                    # mask = torch.ones((layers, heads))
                    for layer, head in pruning_log:
                        pruning_dict[layer].append(head)
                        
                        # mask[layer, head] = 0
                        
                    # model_auto = AutoModel.from_pretrained(model_name)
                    model_args = {
                    "pretrained": str(model_name),
                    "dtype":"float16",
                    # "head_mask": mask,
                    "device": "cuda:0"
                    }
                    model_lm = huggingface.HFLM(**model_args)
                    if model == "opt-13b":
                        model_lm._model.model= prune(model_lm._model.model, pruning_dict)
                    else:
                        model_lm._model.transformer= prune(model_lm._model.transformer, pruning_dict)
                    for task in tasks:
                        model_lm._model.to("cuda:0")
                        print("Pruning done for: ", model, prune_method, metric, prunetask, prune_percent, task)
                        results = simple_evaluate(
                            model = model_lm,
                            batch_size = "auto",
                            device = "cuda:0",
                            num_fewshot = 0,
                            tasks=[task],
                            log_samples=False,
                        )
                        model_lm._model.to('cpu')
                        torch.cuda.empty_cache()
                        output_path = Path(f"results/{task}/{model_path}")
                        output_path.mkdir(parents=True, exist_ok=True)
                        output_path_file = output_path.joinpath("results.json")
                        dumped = json.dumps(results, indent=2, default=_handle_non_serializable, ensure_ascii=False)
                        output_path_file.open("w", encoding="utf-8").write(dumped)
                        print("Evaluted: ", model, prune_method, metric, prunetask, prune_percent, task)