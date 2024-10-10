import requests
from PIL import Image
import torch
from mantis.models.conversation import Conversation, SeparatorStyle
from mantis.models.mllava import (
    chat_mllava,
    LlavaForConditionalGeneration,
    MLlavaProcessor,
)
from mantis.models.mllava.utils import conv_templates


# 1. Set the system prompt
conv_llama_3_elyza = Conversation(
    system="<|start_header_id|>system<|end_header_id|>\n\nあなたは誠実で優秀な日本人のアシスタントです。特に指示が無い場合は、常に日本語で回答してください。",
    roles=("user", "assistant"),
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.LLAMA_3,
    sep="<|eot_id|>",
)
conv_templates["llama_3"] = conv_llama_3_elyza


class VLM:
    model_id = "SakanaAI/Llama-3-EvoVLM-JP-v2"

    def __init__(self) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = LlavaForConditionalGeneration.from_pretrained(
            self.model_id, torch_dtype=torch.float16, device_map=self.device
        ).eval()
        self.processor = MLlavaProcessor.from_pretrained(
            "TIGER-Lab/Mantis-8B-siglip-llama3"
        )
        self.processor.tokenizer.pad_token = self.processor.tokenizer.eos_token

    def generate(self, image, text: str, max_new_tokens: int = 256):
        generation_kwargs = {
            "max_new_tokens": max_new_tokens,
            "num_beams": 1,
            "do_sample": False,
            "no_repeat_ngram_size": 3,
        }
        if isinstance(image, list):
            text = "<image> " * len(image) + "\n" + text
            images = image
        else:
            text = "<image>\n" + text
            images = [image]
        response, history = chat_mllava(
            text, images, self.model, self.processor, **generation_kwargs
        )
        return response


if __name__ == "__main__":
    model = VLM()
    image_file = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(image_file, stream=True).raw)
    print(model.generate(image, "What is in the image?"))
    print(model.generate([image, image], "What is in the image?"))
