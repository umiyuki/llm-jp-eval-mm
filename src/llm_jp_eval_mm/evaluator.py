import random
from typing import Dict

import numpy as np
import torch

import src.llm_jp_eval_mm.models
from src.llm_jp_eval_mm.api.registry import get_model


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

    # Print the config of the model
    print("Successfully loaded the model!")
    print(model.config)

    # load datasets
    # from datasets import get_dataset

    # dataset_config = config.dataset
    # dataset_name = dataset_config.dataset_name

    # # TODO: Implement after loading the dataset
    # _ = get_dataset(dataset_config)

    # # Celebrate the successful loading of the model!
    # print(f"Successfully loaded the dataset -> {dataset_name}")
    # print(dataset_config)


evaluate(
    model_name="code-llama",
    model_args={"model_id": "codellama/CodeLlama-7b-hf"},
)
