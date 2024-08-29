import glob
import pandas as pd
import json
import datasets
import pathlib

files = glob.glob("result/prediction/*.jsonl")
print(files)


model_ids = ["-".join(pathlib.Path(file).stem.split("-")[:-1]) for file in files]
print(model_ids)

dataset = datasets.load_dataset("Silviase/Japanese-Heron-Bench", split="train")


# question id, model_1's answer, model_2's answer, model_3's answer, ...
df = pd.DataFrame()

for example in dataset:
    if len(df) == 0:
        df["question_id"] = [example["question_id"]]
    if example["question_id"] not in df["question_id"].values:
        df = df._append({"question_id": example["question_id"]}, ignore_index=True)
    answer = example["answer"]
    answer_models = [
        "claude-3-opus-20240229",
        "gpt-4-0125-preview",
        "gpt-4-vision-preview",
    ]
    df.loc[df["question_id"] == example["question_id"], "category"] = example[
        "category"
    ]
    df.loc[df["question_id"] == example["question_id"], "context"] = example["context"]
    df.loc[df["question_id"] == example["question_id"], "input_text"] = example["text"]
    for model_id in answer_models:
        df.loc[df["question_id"] == example["question_id"], model_id] = answer[model_id]

for file in files:
    with open(file, "r") as f:
        model_id = "-".join(pathlib.Path(file).stem.split("-")[:-1])
        for line in f:
            data = json.loads(line)
            if len(df) == 0:
                df["question_id"] = [data["question_id"]]
            if data["question_id"] not in df["question_id"].values:
                df = df._append({"question_id": data["question_id"]}, ignore_index=True)
            df.loc[df["question_id"] == data["question_id"], model_id] = (
                data["text"] + "\n" + "score: " + str(data["score"])
            )


print(df.head())
df.to_excel("result/prediction.xlsx", index=False)


# box plot of the scores for each model
import matplotlib.pyplot as plt
import seaborn as sns

# metrics_for_model = {
#     "model_id": {
#         "detail": [],
#         "conv": [],
#         "complex": [],
#         "overall": [],
#     }
# }

metrics_for_model = {}
for file in files:
    with open(file, "r") as f:
        model_id = "-".join(pathlib.Path(file).stem.split("-")[:-1])
        metrics_for_model[model_id] = {
            "detail": [],
            "conv": [],
            "complex": [],
            "overall": [],
        }
        for example, line in zip(dataset, f):
            data = json.loads(line)
            metrics_for_model[model_id][example["category"]].append(data["score"])
            metrics_for_model[model_id]["overall"].append(data["score"])

# # plot the box plot
# fig, axes = plt.subplots(len(metrics_for_model), 1, figsize=(10, 10))
# for i, (model_id, metrics) in enumerate(metrics_for_model.items()):
#     sns.boxplot(data=metrics, ax=axes[i])
#     axes[i].set_title(model_id)
#     axes[i].set_ylabel("score")
# plt.tight_layout()
# plt.savefig("result/boxplot.png")


# プロット
plt.figure(figsize=(10, 6))
sns.set(style="whitegrid")

# 各モデルのスコアをプロット
colors = sns.color_palette("Set2", n_colors=len(metrics_for_model))
# box plot
df = pd.DataFrame()
for model_id, metrics in metrics_for_model.items():
    for category, scores in metrics.items():
        for score in scores:
            df = df._append(
                {"model_id": model_id, "category": category, "score": score},
                ignore_index=True,
            )
sns.boxplot(data=df, x="category", y="score", hue="model_id", palette=colors)

# ラベルとタイトル
plt.ylabel("Score")
plt.xlabel("Category")
# outer legend
plt.legend(title="Models", loc="upper left", bbox_to_anchor=(1, 1))
plt.tight_layout()
plt.savefig("result/stripplot.png")


# result/evaluation/*.jsonl to excel
files = glob.glob("result/evaluation/*.jsonl")
print(files)


model_ids = ["-".join(pathlib.Path(file).stem.split("-")[:-1]) for file in files]
print(model_ids)


# model_id, detail, conv, complex, overall
df = pd.DataFrame()
for i, file in enumerate(files):
    with open(file, "r") as f:
        model_id = "-".join(pathlib.Path(file).stem.split("-")[:-1])
        line = f.readline()
        data = json.loads(line)
        df.loc[i, "model_id"] = model_id
        df.loc[i, "detail"] = data["detail"]
        df.loc[i, "conv"] = data["conv"]
        df.loc[i, "complex"] = data["complex"]
        df.loc[i, "overall"] = data["overall"]


print(df.head())

df.to_excel("result/evaluation.xlsx", index=False)
