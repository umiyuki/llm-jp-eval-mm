import json
import os

import src as eval_mm
from tqdm import tqdm
import importlib
import argparse
import time

parser = argparse.ArgumentParser()
parser.add_argument("--class_path", type=str, default="llava")
args = parser.parse_args()

class_path = args.class_path

module = importlib.import_module(class_path)
model = module.VLM()
model_id = model.model_id


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

print("Evaluation start")
# evaluate the predictions
metrics, eval_results = task.compute_metrics(preds)

# save the predictions to jsonl file
model_name = model_id.replace("/", "-")

result_dir = "result"
os.makedirs(result_dir, exist_ok=True)
prediction_result_dir = os.path.join(result_dir, "prediction")
os.makedirs(prediction_result_dir, exist_ok=True)
evaluation_result_dir = os.path.join(result_dir, "evaluation")
os.makedirs(evaluation_result_dir, exist_ok=True)

unix_time = int(time.time())

prediction_result_file_path = os.path.join(
    prediction_result_dir, f"{model_name}-{unix_time}.jsonl"
)
with open(os.path.join(prediction_result_file_path), "w") as f:
    for pred, eval_result in zip(preds, eval_results):
        result = {
            "question_id": pred["question_id"],
            "text": pred["text"],
            "score": eval_result["score"],
            "score_gpt": eval_result["score_gpt"],
        }
        f.write(json.dumps(result, ensure_ascii=False) + "\n")
print(f"Prediction result saved to {prediction_result_file_path}")


eval_result_file_path = os.path.join(
    evaluation_result_dir, f"{model_name}-{unix_time}.jsonl"
)
with open(eval_result_file_path, "w") as f:
    f.write(json.dumps(metrics, ensure_ascii=False) + "\n")

print(f"Metrics: {metrics}")
print(f"Evaluation result example: {eval_results[0]}")
