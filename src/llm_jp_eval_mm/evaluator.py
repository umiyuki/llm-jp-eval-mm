import json
import os
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
from llm_jp_eval_mm.configs import config
from llm_jp_eval_mm.utils import log

dotenv.load_dotenv()


def evaluate():

    # Random seed settings
    random.seed(0)
    np.random.seed(1234)
    torch.manual_seed(1234)

    # TODO: STOP loading models if not needed
    # get the model class
    model_cls = get_model(config.model.model_family)
    # instantiate the model
    model = model_cls(config.model)

    # prepare task
    task_name = config.task.task_id
    task_cls = get_task(task_name)
    task = task_cls()
    dataset = task.dataset
    save_dir = os.path.join(config.path.result_dir, task_name, config.model.model_id)
    os.makedirs(save_dir, exist_ok=True)

    log(f"Saving results to {save_dir}")

    if config.task.do_inference:
        log(f"Task: {task_name} Inference ...")
        preds = []
        # inference
        for i, doc in tqdm(enumerate(dataset), total=len(dataset), desc=f"Task: {task_name} Predicting ..."):
            image, text = task.doc_to_visual(doc), task.doc_to_text(doc)
            pred = model.generate(image, text)
            processed_pred = task.process_pred(pred, doc)
            preds.append(processed_pred)

        log(f"Saving inference results to {save_dir}")

        # save inference result into jsonl file
        with open(os.path.join(save_dir, config.task.inference_result_path), "w") as f:
            for pred in preds:
                f.write(json.dumps(pred, ensure_ascii=False) + "\n")
            f.close()

    log(f"Loading inference results from {save_dir}")

    # load inference result from jsonl file
    with open(os.path.join(save_dir, config.task.inference_result_path), "r") as f:
        preds = [json.loads(line) for line in f.readlines()]
        f.close()

    if config.task.do_eval:
        log(f"Task: {task_name} Evaluation ...")
        # evaluation
        results, verbose_reuslts = task.process_results(dataset, preds)

        # Save result into json file
        with open(os.path.join(save_dir, config.task.eval_result_path), "w") as f:
            json.dump(results, f, ensure_ascii=False)
            f.close()

        # Save verbose result into jsonl file
        if config.task.do_verbose_eval:
            with open(os.path.join(save_dir, config.task.verbose_eval_result_path), "w") as f:
                for verbose_result in verbose_reuslts:
                    f.write(json.dumps(verbose_result, ensure_ascii=False) + "\n")
                f.close()

    log("Finished task: ", task_name)

    return


if __name__ == "__main__":
    evaluate()
