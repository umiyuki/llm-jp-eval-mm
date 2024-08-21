from llm_jp_eval_mm.api.registry import get_task

# タスクの準備
cfg = {"dataset_id": "TaskA"}  # argparseするなど自由
task = get_task(cfg)

# データセットのロード
dataset = task.load_dataset()

# データセットを処理して結果を保存
with open("hoge.jsonl", "w") as f:
    for sample in dataset:
        pass
        # ここを書いてもらう
        # completion = model.generate(sample)
        # output = {"id": sample["id"], "output": completion}
        # f.write(json.dumps(output) + "\n")

# 保存した結果を読み込み
output = task.load_output("hoge.jsonl")

# メトリクスの計算
metrics = task.compute_metrics(output)
print(metrics)
