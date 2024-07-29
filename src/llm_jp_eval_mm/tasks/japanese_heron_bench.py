import asyncio
import os
from typing import Any, Dict

from datasets import load_dataset
from dotenv import load_dotenv
from openai import AzureOpenAI
from tqdm import tqdm

from llm_jp_eval_mm.api.registry import register_task
from llm_jp_eval_mm.api.tasks import Task

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
        # context = doc["context"]
        text = doc["text"]
        return f"{text}"

    def doc_to_visual(self, doc):
        return doc["image"]

    def process_results(self, docs, preds):
        """Process the results of the model.
        Args:
            doc: dataset instance
            results: [pred]
        Return:
            a dictionary with key: { 'score' : score }
        """

        # assert len(docs) == len(results), "Number of docs and results should be the same"
        load_dotenv("/home/silviase/llmjp/llm-jp-eval-multimodal/.env")
        client = AzureOpenAI(
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_version="2023-05-15",
            api_key=os.getenv("AZURE_OPENAI_KEY"),
        )

        results = []

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

            pred = ask_gpt4(client, content, max_tokens=1024)
            scores = parse_score(pred)
            results.append(
                {
                    "id": idx,
                    "pred": pred,
                    "answer_gpt": answer_1,
                    "answer_eval": answer_2,
                    "score_gpt": scores[0],
                    "score_eval": scores[1],
                }
            )

        return results
