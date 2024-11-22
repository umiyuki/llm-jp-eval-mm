from datasets import Dataset, load_dataset
import re

from ..api.registry import register_task
from ..api.task import Task
from ..utils.azure_client import OpenAIChatAPI
from eval_mm.metrics import ScorerRegistry

# import neologdn FIXME: fix c++12 error when installing neologdn


@register_task("ja-multi-image-vqa")
class JAMultiImageVQA(Task):
    def __init__(
        self, max_dataset_len: int = None, judge_model: str = "gpt-4o-mini-2024-07-18"
    ):
        super().__init__()
        self.client = OpenAIChatAPI()
        self.judge_model = judge_model
        if max_dataset_len is not None:
            self.dataset = self._prepare_dataset().select(range(max_dataset_len))
        else:
            self.dataset = self._prepare_dataset()

    @staticmethod
    def _prepare_dataset() -> Dataset:
        ds = load_dataset("SakanaAI/JA-Multi-Image-VQA", split="test")
        ds = ds.rename_column("question", "input_text")
        ds = ds.map(lambda example, idx: {"question_id": idx}, with_indices=True)
        return ds

    def doc_to_text(self, doc):
        # delete redundant image tags
        text = re.sub(r"<image> ", "", doc["input_text"])
        return text

    def doc_to_visual(self, doc):
        print("CAUTION: This task provides MULTIPLE images.")
        return doc["images"]

    def doc_to_id(self, doc):
        return doc["question_id"]

    def doc_to_answer(self, doc):
        return doc["answer"]

    def calc_scores(self, preds: list, metric: str) -> list:
        """Calculate scores of each prediction based on the metric."""
        scorer = ScorerRegistry.get_scorer(metric)
        docs = self.dataset
        refs = [doc["answer"] for doc in docs]
        pred_texts = [pred["text"] for pred in preds]
        kwargs = {
            "docs": docs,
            "client": self.client,
            "judge_model": self.judge_model,
            "batch_size": 10,
        }
        return scorer.score(refs, pred_texts, **kwargs)

    def gather_scores(self, scores: list[dict], metric: str):
        scorer = ScorerRegistry.get_scorer(metric)
        kwargs = {"docs": self.dataset}
        return scorer.aggregate(scores, **kwargs)
