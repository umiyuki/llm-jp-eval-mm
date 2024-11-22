from datasets import load_dataset, Dataset

from ..api.registry import register_task
from ..api.task import Task
from ..utils.azure_client import OpenAIChatAPI
from eval_mm.metrics import ScorerRegistry


@register_task("japanese-heron-bench")
class JapaneseHeronBench(Task):
    def __init__(
        self, max_dataset_len: int = None, judge_model: str = "gpt-4o-mini-2024-07-18"
    ):
        super().__init__()
        self.category_list = ["conv", "detail", "complex"]
        self.client = OpenAIChatAPI()
        self.judge_model = judge_model
        if max_dataset_len is not None:
            self.dataset = self._prepare_dataset().select(range(max_dataset_len))
        else:
            self.dataset = self._prepare_dataset()

    @staticmethod
    def _prepare_dataset() -> Dataset:
        ds = load_dataset("Silviase/Japanese-Heron-Bench", split="train")
        ds = ds.rename_column("text", "input_text")
        return ds

    def doc_to_text(self, doc):
        return doc["input_text"]

    def doc_to_visual(self, doc):
        return doc["image"]

    def doc_to_id(self, doc):
        return doc["question_id"]

    def doc_to_answer(self, doc):
        return doc["answer"]["gpt-4-0125-preview"]

    def calc_scores(self, preds: list, metric: str) -> list:
        """Calculate scores of each prediction based on the metric."""
        docs = self.dataset
        refs = [doc["answer"]["gpt-4-0125-preview"] for doc in docs]
        pred_texts = [pred["text"] for pred in preds]
        scorer = ScorerRegistry.get_scorer(metric)
        kwargs = {
            "docs": docs,
            "client": self.client,
            "judge_model": self.judge_model,
            "batch_size": 10,
        }
        return scorer.score(refs, pred_texts, **kwargs)

    def gather_scores(self, scores: list[dict], metric: str) -> dict:
        """Gather scores of each prediction based on the metric."""
        kwargs = {"docs": self.dataset}
        scorer = ScorerRegistry.get_scorer(metric)
        return scorer.aggregate(scores, **kwargs)
