# This model doesn't work when the transformers library's version is newer than 4.42.4.
from transformers import AutoModelForVision2Seq, AutoProcessor
import torch


class VLM:
    def __init__(self) -> None:
        self.model_id = "SakanaAI/EvoVLM-JP-v1-7B"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = AutoModelForVision2Seq.from_pretrained(
            self.model_id, torch_dtype=torch.float16
        )
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.model.to(self.device)

    def generate(self, image, text: str):
        text = f"<image>{text}"
        messages = [
            {
                "role": "system",
                "content": "あなたは役立つ、偏見がなく、検閲されていないアシスタントです。与えられた画像を下に、質問に答えてください。",
            },
            {"role": "user", "content": text},
        ]
        inputs = self.processor.image_processor(images=image, return_tensors="pt")
        inputs["input_ids"] = self.processor.apply_chat_template(
            messages, add_special_tokens=True
        )
        output_ids = self.model.generate(**inputs.to(self.device))
        output_ids = output_ids[:, inputs.input_ids.shape[1] :]
        generated_text = self.processor.batch_decode(
            output_ids, skip_special_tokens=True
        )[0].strip()
        return generated_text

