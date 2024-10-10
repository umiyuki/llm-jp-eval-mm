import requests
import torch
from PIL import Image
from transformers import MllamaForConditionalGeneration, AutoProcessor
from typing import Union


class VLM:
    model_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"

    def __init__(self) -> None:
        self.model = MllamaForConditionalGeneration.from_pretrained(
            self.model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        self.processor = AutoProcessor.from_pretrained(self.model_id)

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
        inputs = self.processor(
            images, input_text, add_special_tokens=False, return_tensors="pt"
        ).to(self.model.device)
        output_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        generated_ids = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(inputs.input_ids, output_ids)
        ]
        return self.processor.decode(
            generated_ids[0],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )


if __name__ == "__main__":
    model = VLM()
    image_file = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(image_file, stream=True).raw)
    print(model.generate(image, "What is in the image?"))

    multi_images = [image for _ in range(3)]
    print(model.generate(multi_images, "What is the difference between these images?"))
