from typing import Any, Dict

import pandas as pd
from datasets import load_dataset

from llm_jp_eval_mm.api.registry import register_task
from llm_jp_eval_mm.api.tasks import Task


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

    def process_results(self, doc, results):
        return results
