from PIL import Image
import requests
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor


class VLM:
    model_id = "Qwen/Qwen2-VL-7B-Instruct"

    def __init__(self) -> None:
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.model_id, torch_dtype="bfloat16", device_map="auto"
        )
        min_pixels = 256 * 28 * 28
        max_pixels = 1280 * 28 * 28
        self.processor = AutoProcessor.from_pretrained(
            self.model_id, min_pixels=min_pixels, max_pixels=max_pixels
        )

    def generate(self, image, text: str, max_new_tokens: int = 256):
        conversation = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                    },
                    {"type": "text", "text": text},
                ],
            }
        ]

        text_prompt = self.processor.apply_chat_template(
            conversation, add_generation_prompt=True
        )
        inputs = self.processor(
            text=[text_prompt], images=[image], padding=True, return_tensors="pt"
        )
        inputs = inputs.to(self.model.device)
        print("Input token length:", inputs.input_ids.shape[1])
        output_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        generated_ids = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(inputs.input_ids, output_ids)
        ]
        generated_text = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )[0]
        return generated_text


if __name__ == "__main__":
    model = VLM()
    image_file = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(image_file, stream=True).raw)
    print(model.generate(image, "What is in the image?"))
