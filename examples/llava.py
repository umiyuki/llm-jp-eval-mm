
import torch

from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
import torch

class VLM:
    def __init__(self) -> None:
        self.model_id = "llava-hf/llava-v1.6-mistral-7b-hf"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = LlavaNextForConditionalGeneration.from_pretrained(
            self.model_id, torch_dtype=torch.float16, low_cpu_mem_usage=True
        )
        self.processor = LlavaNextProcessor.from_pretrained(self.model_id)
        self.model.to(self.device)

    def generate(self, image, text: str):
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": text},
                    {"type": "image"},
                ],
            },
        ]

        prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)

        inputs = self.processor(images=image, text=prompt, return_tensors="pt").to("cuda:0")

        # autoregressively complete prompt
        output = self.model.generate(**inputs, max_new_tokens=100)

        generated_text = self.processor.decode(output[0], skip_special_tokens=True)
        # split [INST] and return the last part
        generated_text = generated_text.split("[/INST]")[-1].strip()
        return generated_text

