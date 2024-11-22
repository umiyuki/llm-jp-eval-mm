from datasets import load_dataset

from ..api.registry import register_task
from ..api.task import Task
from ..utils.azure_client import OpenAIChatAPI
from ..metrics.exact_match_scorer import ExactMatchScorer
from ..metrics.llm_as_a_judge_scorer import LlmAsaJudgeScorer
from ..metrics.rougel_scorer import RougeLScorer
from ..metrics.substring_match_scorer import SubstringMatchScorer
from ..metrics.heron_bench_scorer import HeronBenchScorer


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
    def _prepare_dataset():
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
        if metric == "llm_as_a_judge_heron_bench":
            return HeronBenchScorer.score(
                docs, pred_texts, self.client, self.judge_model
            )
        elif metric == "exact_match":
            return ExactMatchScorer.score(refs, pred_texts)
        elif metric == "llm_as_a_judge":
            return LlmAsaJudgeScorer.score(
                self.client,
                docs["input_text"],
                refs,
                pred_texts,
                10,
                self.judge_model,
            )
        elif metric == "rougel":
            return RougeLScorer.score(refs, pred_texts)
        elif metric == "substring_match":
            return SubstringMatchScorer.score(refs, pred_texts)
        else:
            raise ValueError(f"metric {metric} is not supported.")

    def gather_scores(self, scores: list[dict[str, int]], metric: str) -> dict:
        """Gather scores of each prediction based on the metric."""
        if metric == "llm_as_a_judge_heron_bench":
            return HeronBenchScorer.aggregate(self.dataset, scores)
        elif metric == "exact_match":
            return ExactMatchScorer.aggregate(scores)
        elif metric == "llm_as_a_judge":
            return LlmAsaJudgeScorer.aggregate(scores)
        elif metric == "rougel":
            return RougeLScorer.aggregate(scores)
        elif metric == "substring_match":
            return SubstringMatchScorer.aggregate(scores)
        else:
            raise ValueError(f"metric {metric} is not supported.")
