import random
from typing import Dict

import numpy as np
import torch

import llm_jp_eval_mm.api
import llm_jp_eval_mm.models
from llm_jp_eval_mm.api.registry import get_model


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

    # TODO: This is a sample generation, please replace it with the task loading and generation code.
    import requests
    from PIL import Image

    url = "https://images.unsplash.com/photo-1694831404826-3400c48c188d?q=80&w=2070&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D"
    image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
    # <image> represents the input image. Please make sure to put the token in your text.
    text = "<image>\nこの信号機の色は何色ですか?"

    generated_text = model.generate(image, text)
    print(generated_text)

    return


evaluate(
    model_name="evovlm-jp-v1",
    model_args={},
)
