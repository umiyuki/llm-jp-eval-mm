import random
from typing import Dict

import dotenv
import numpy as np
import torch

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

        results = []
        # inference
        for i, doc in enumerate(dataset):
            print(f"doc: {doc}")
            image, text = task.doc_to_visual(doc), task.doc_to_text(doc)
            pred = model.generate(image, text)
            print("pred:", pred)
            results.append(pred)

        # evaluation
        task.process_results(dataset, results)

    return


evaluate(
    model_name="evovlm-jp-v1",
    model_args={},
    tasks=["japanese-heron-bench"],
)
