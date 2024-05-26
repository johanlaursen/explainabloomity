from lm_eval.evaluator import simple_evaluate
from lm_eval.models import huggingface
from utils import *
from transformers import AutoModel
import torch
from pathlib import Path
import json
from collections import defaultdict

def _handle_non_serializable(o):
    if isinstance(o, np.int64) or isinstance(o, np.int32):
        return int(o)
    elif isinstance(o, set):
        return list(o)
    else:
        return str(o)

def evaluate(model_lm, tasks, model_path, ):
    model_lm.model.half()
    for task in tasks:
        print("Evaluating: ", task, model_path)
        model_lm._model.to("cuda:0")
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