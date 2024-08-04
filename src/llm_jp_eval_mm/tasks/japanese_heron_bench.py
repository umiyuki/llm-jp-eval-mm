import os

from datasets import load_dataset
from dotenv import load_dotenv
from openai import AzureOpenAI
from tqdm import tqdm

from llm_jp_eval_mm.api.registry import register_task
from llm_jp_eval_mm.api.tasks import Task
from llm_jp_eval_mm.configs import config

from .utlis import RULES, ask_gpt4


def parse_score(review):
    try:
        score_pair = review.split("\n")[0]
        score_pair = score_pair.replace(",", " ")
        sp = score_pair.split(" ")
        if len(sp) == 2:
            return [float(sp[0]), float(sp[1])]
        else:
            print("error", review)
            return [-1, -1]
    except Exception as e:
        print(e)
        print("error", review)
        return [-1, -1]


@register_task("japanese-heron-bench")
class JapaneseHeronBench(Task):
    def __init__(
        self,
        config=None,
    ):
        super().__init__(config)

    def download(self) -> None:
        # TODO: Change the dataset name if needed
        self._dataset = load_dataset("Silviase/Japanese-Heron-Bench")

    @property
    def dataset(self):
        """Returns the dataset associated with this class."""
        return self._dataset["train"]

    def doc_to_text(self, doc):
        text = doc["text"]
        return f"{text}"

    def doc_to_visual(self, doc):
        return doc["image"]

    def process_pred(self, pred, doc):
        processed = {
            "question_id": doc["question_id"],
            "image_category": doc["image_category"],
            "prompt": doc["text"],
            "answer_id": "",
            "model_id": config.model.model_id,
            "text": pred,
        }
        return processed

    def process_results(self, docs, preds):
        """Process the results of the model.
        Args:
            doc: dataset instance
            preds: [pred]
        Return:
            a dictionary with key: { 'score' : score }
        """

        # TODO: Please load your OWN .env file
        load_dotenv("/home/silviase/llmjp/llm-jp-eval-multimodal/.env")

        client = AzureOpenAI(
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_version="2023-05-15",
            api_key=os.getenv("AZURE_OPENAI_KEY"),
        )

        results_verbose = []

        for idx, (doc, pred) in enumerate(tqdm(zip(docs, preds), total=len(docs), desc="Evaluation ...")):
            category = doc["category"]
            answer_1 = doc["answer"]["gpt-4-0125-preview"]
            answer_2 = pred
            role = RULES[category]["role"]
            prompt = RULES[category]["prompt"]
            content = (
                f'[Context]\n{doc["context"]}\n\n'
                f'[Question]\n{doc["text"]}\n\n'
                f"[{role} 1]\n{answer_1}\n\n[End of {role} 1]\n\n"
                f"[{role} 2]\n{answer_2}\n\n[End of {role} 2]\n\n"
                f"[System]\n{prompt}\n\n"
                f"If it is not relevant to the context, does not answer directly, or says the wrong thing, give it a low score.\n\n"
            )

            eval_result = ask_gpt4(client, content, max_tokens=1024)
            scores = parse_score(eval_result)
            results_verbose.append(
                {
                    "id": idx,
                    "category": category,
                    "answer_gpt": answer_1,
                    "answer_eval": answer_2,
                    "score_gpt": scores[0],
                    "score_eval": scores[1],
                }
            )

            # for checking, break after 10
            if idx == 10:
                break

        # average score for each category, and overall
        metrics = {}
        for category in RULES.keys():
            scores = [r["score_eval"] for r in results_verbose if r["category"] == category]
            if len(scores) == 0:
                continue
            avg_score = sum(scores) / len(scores)
            metrics[category] = avg_score

        metrics["overall"] = sum([r["score_eval"] for r in results_verbose]) / len(results_verbose)
        print(metrics)
        return metrics, results_verbose
