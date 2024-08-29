import json
import os

import src as eval_mm
from tqdm import tqdm
import importlib
import argparse
import time

parser = argparse.ArgumentParser()
parser.add_argument("--class_path", type=str, default="llava")
parser.add_argument("--task_id", type=str, default="japanese-heron-bench")
parser.add_argument("--openai_model_id", type=str, default="gpt-4o-mini-2024-07-18")

args = parser.parse_args()

class_path = args.class_path
task_id = args.task_id
openai_model_id = args.openai_model_id

module = importlib.import_module(class_path)
model = module.VLM()
model_id = model.model_id


task = eval_mm.api.registry.get_task(task_id)
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
metrics, eval_results = task.compute_metrics(preds, model_id=openai_model_id)

# save the predictions to jsonl file
model_name = model_id.replace("/", "-")
result_dir = f"result/{task_id}"
os.makedirs(result_dir, exist_ok=True)
prediction_result_dir = os.path.join(result_dir, "prediction")
os.makedirs(prediction_result_dir, exist_ok=True)
evaluation_result_dir = os.path.join(result_dir, "evaluation")
os.makedirs(evaluation_result_dir, exist_ok=True)

unix_time = int(time.time())

prediction_result_file_path = os.path.join(
    prediction_result_dir, f"{model_name}-{unix_time}.jsonl"
)
results = task.format_result(preds, eval_results)
with open(os.path.join(prediction_result_file_path), "w") as f:
    for result in results:
        f.write(json.dumps(result, ensure_ascii=False) + "\n")
print(f"Prediction result saved to {prediction_result_file_path}")


eval_result_file_path = os.path.join(
    evaluation_result_dir, f"{model_name}-{unix_time}.jsonl"
)
with open(eval_result_file_path, "w") as f:
    f.write(json.dumps(metrics, ensure_ascii=False) + "\n")

print(f"Metrics: {metrics}")
print(f"Evaluation result example: {eval_results[0]}")
