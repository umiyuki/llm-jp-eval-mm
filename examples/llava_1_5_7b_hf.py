import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration


class VLM:
    model_id: str = "llava-hf/llava-1.5-7b-hf"

    def __init__(self) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = LlavaForConditionalGeneration.from_pretrained(
            self.model_id,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        )
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.model.to(self.device)

    def generate(self, image, text: str, max_new_tokens: int = 256):
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": text},
                    {"type": "image"},
                ],
            },
        ]

        prompt = self.processor.apply_chat_template(
            conversation, add_generation_prompt=True
        )

        inputs = self.processor(images=image, text=prompt, return_tensors="pt").to(
            "cuda:0"
        )

        # autoregressively complete prompt
        output = self.model.generate(**inputs, max_new_tokens=max_new_tokens)[0]

        generated_text = self.processor.decode(output, skip_special_tokens=True)
        # extract the answer
        answer = generated_text.split("ASSISTANT:")[-1].strip()
        return answer


if __name__ == "__main__":
    import requests
    from PIL import Image

    model = VLM()
    image_file = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(image_file, stream=True).raw)
    print(model.generate(image, "What is in the image?"))
