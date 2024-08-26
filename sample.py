import json
import os

import src as eval_mm
import torch
from tqdm import tqdm
from transformers import AutoModelForVision2Seq, AutoProcessor


class EvoVLMJPv1:
    def __init__(self, cfg) -> None:
        super().__init__()
        self.config = cfg
        self.model_id = cfg["model_id"]
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
        inputs["input_ids"] = self.processor.tokenizer.apply_chat_template(
            messages, return_tensors="pt"
        )
        output_ids = self.model.generate(**inputs.to(self.device))
        output_ids = output_ids[:, inputs.input_ids.shape[1] :]
        generated_text = self.processor.batch_decode(
            output_ids, skip_special_tokens=True
        )[0].strip()
        return generated_text


cfg = {
    "model_id": "SakanaAI/EvoVLM-JP-v1-7B",
}
model = EvoVLMJPv1(cfg)


task = eval_mm.api.registry.get_task("japanese-heron-bench")
dataset = task.dataset

preds = []
for doc in tqdm(dataset):
    image = task.doc_to_visual(doc)
    text = task.doc_to_text(doc)
    qid = task.doc_to_id(doc)
    pred = {
        "question_id": qid,
        "text": model.generate(image, text),
    }
    preds.append(pred)

# save the predictions to jsonl file
save_path = os.path.join("tmp", "predictions.jsonl")
os.makedirs(os.path.dirname(save_path), exist_ok=True)
with open(save_path, "w") as f:
    for pred in preds:
        f.write(json.dumps(pred, ensure_ascii=False) + "\n")
print(f"Predictions saved to {save_path}")

# evaluate the predictions
metrics, eval_results = task.compute_metrics(preds)
print(f"Metrics: {metrics}")
print(f"Evaluation result example: {eval_results[0]}")
