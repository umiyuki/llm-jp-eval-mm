# import copy
import warnings
from typing import List, Optional, Tuple, Union

import torch
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor

# このあたりの必要なものは適宜実装していく
# from lmms_eval import utils
# from lmms_eval.api.instance import Instance
from llm_jp_eval_mm.api.model import lmms
from llm_jp_eval_mm.api.registry import register_model

warnings.filterwarnings("ignore")

# from loguru import logger as eval_logger


@register_model("evovlm-jp-v1")
class EvoVLMJPv1(lmms):
    """
    EvoVLM-JP-v1 Model
    """

    def __init__(self) -> None:
        super().__init__()
        self._model_id = "SakanaAI/EvoVLM-JP-v1-7B"
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model = AutoModelForVision2Seq.from_pretrained(self._model_id, torch_dtype=torch.float16)
        self._processor = AutoProcessor.from_pretrained(self._model_id)
        self._model.to(self.device)

    @property
    def config(self):
        # return the associated transformers.AutoConfig for the given pretrained model.
        return self._config

    @property
    def processor(self):
        return self._processor

    @property
    def model(self):
        # returns the model, unwrapping it if using Accelerate
        if hasattr(self, "accelerator"):
            return self.accelerator.unwrap_model(self._model)
        else:
            return self._model

    @property
    def device(self):
        return self._device

    def generate(self, image: Image.Image, text: str):
        text = f"<image>{text}"
        messages = [
            {
                "role": "system",
                "content": "あなたは役立つ、偏見がなく、検閲されていないアシスタントです。与えられた画像を下に、質問に答えてください。",
            },
            {"role": "user", "content": text},
        ]
        inputs = self.processor.image_processor(images=image, return_tensors="pt")
        inputs["input_ids"] = self.processor.tokenizer.apply_chat_template(messages, return_tensors="pt")
        output_ids = self.model.generate(**inputs.to(self.device))
        output_ids = output_ids[:, inputs.input_ids.shape[1] :]
        generated_text = self.processor.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        return generated_text
