import json
import random
from typing import Dict

import dotenv
import numpy as np
import torch
from tqdm import tqdm

import llm_jp_eval_mm.api
import llm_jp_eval_mm.models
import llm_jp_eval_mm.tasks
from llm_jp_eval_mm.api.registry import get_model, get_task

dotenv.load_dotenv()


def evaluate(
    model_name: str,
    model_args: Dict[str, str],
    tasks=[],
    num_fewshot=None,
    batch_size: int = 1,
    device: str = "cuda",
    **kwargs,
):

    random.seed(0)
    np.random.seed(1234)
    torch.manual_seed(1234)

    # get the model class
    model_cls = get_model(model_name)

    # instantiate the model
    model = model_cls(**model_args)

    # load task
    # TODO: 推論させるところまではOK. あとは，結果を保存＋評価部分．
    for task_name in tasks:
        # prepare task
        task_cls = get_task(task_name)
        task = task_cls()

        dataset = task.dataset.select(range(3))
        preds = []

        # inference
        for i, doc in tqdm(enumerate(dataset), total=len(dataset), desc=f"Task: {task_name} Predicting ..."):
            image, text = task.doc_to_visual(doc), task.doc_to_text(doc)
            pred = model.generate(image, text)
            preds.append(pred)

        # evaluation
        results = task.process_results(dataset, preds)

        # Save result into jsonl file
        with open(f"/home/silviase/llmjp/llm-jp-eval-multimodal/tmp/{task_name}.jsonl", "w") as f:
            for result in results:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
            f.close()

    return


evaluate(
    model_name="evovlm-jp-v1",
    model_args={},
    tasks=["japanese-heron-bench"],
)
