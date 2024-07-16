import random
from typing import Dict

import numpy as np
import torch

import llm_jp_eval_mm.api
import llm_jp_eval_mm.models
import llm_jp_eval_mm.tasks
from llm_jp_eval_mm.api.registry import get_model, get_task


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
        task_cls = get_task(task_name)
        task = task_cls()
        dataset = task.dataset
        for doc in dataset:
            image, text = task.doc_to_visual(doc), task.doc_to_text(doc)
            pred = model.generate(image, text)
            print("pred:", pred)
    return


evaluate(
    model_name="evovlm-jp-v1",
    model_args={},
    tasks=["japanese-heron-bench"],
)
