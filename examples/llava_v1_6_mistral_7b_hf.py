import requests
import torch
from PIL import Image
from transformers import LlavaNextForConditionalGeneration, LlavaNextProcessor
from typing import Union


class VLM:
    model_id: str = "llava-hf/llava-v1.6-mistral-7b-hf"

    def __init__(self) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = LlavaNextForConditionalGeneration.from_pretrained(
            self.model_id, torch_dtype=torch.float16, low_cpu_mem_usage=True
        )
        self.processor = LlavaNextProcessor.from_pretrained(self.model_id)
        self.model.to(self.device)

    def generate(
        self,
        images: Union[Image.Image, list[Image.Image]],
        text: str,
        max_new_tokens: int = 256,
    ):
        num_images = 1
        if isinstance(images, list):
            num_images = len(images)
        content = [{"type": "image"} for _ in range(num_images)]
        content.extend([{"type": "text", "text": text}])
        messages = [
            {
                "role": "user",
                "content": content,
            }
        ]
        input_text = self.processor.apply_chat_template(
            messages, add_generation_prompt=True
        )

        inputs = self.processor(images=images, text=input_text, return_tensors="pt").to(
            "cuda"
        )

        # autoregressively complete prompt
        output = self.model.generate(**inputs, max_new_tokens=max_new_tokens)[0]

        generated_text = self.processor.decode(output, skip_special_tokens=True)
        # split [INST] and return the last part
        generated_text = generated_text.split("[/INST]")[-1].strip()
        return generated_text


if __name__ == "__main__":
    model = VLM()
    image_file = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(image_file, stream=True).raw)
    print(model.generate(image, "What is in the image?"))

    multi_images = [image for _ in range(3)]
    print(model.generate(multi_images, "What is the difference between these images?"))
