from datasets import Dataset, concatenate_datasets, load_dataset

from ..api.registry import register_task
from ..api.task import Task
from ..utils.azure_client import OpenAIChatAPI
from eval_mm.metrics import ScorerRegistry


@register_task("ja-vg-vqa-500")
class JaVGVQA500(Task):
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
        ds = load_dataset("SakanaAI/JA-VG-VQA-500", split="test")

        def flatten_sample(sample):
            dataset = {
                "image_id": [sample["image_id"] for _ in sample["qas"]],
                "image": [sample["image"] for _ in sample["qas"]],
                "qa_id": [qa["qa_id"] for qa in sample["qas"]],
                "question": [qa["question"] for qa in sample["qas"]],
                "answer": [qa["answer"] for qa in sample["qas"]],
            }
            return Dataset.from_dict(dataset)

        fragments = []
        for i, sample in enumerate(ds):
            data_fragment = flatten_sample(sample)
            fragments.append(data_fragment)

        ds = concatenate_datasets(fragments)
        ds = ds.rename_column("question", "input_text")
        ds = ds.rename_column("qa_id", "question_id")

        return ds

    def doc_to_text(self, doc):
        return doc["input_text"]

    def doc_to_visual(self, doc):
        return doc["image"]

    def doc_to_id(self, doc):
        return doc["question_id"]

    def doc_to_answer(self, doc):
        return doc["answer"]

    def calc_scores(self, preds: list, metric: str) -> list:
        """Calculate scores of each prediction based on the metric."""
        docs = self.dataset
        refs = [doc["answer"] for doc in docs]
        pred_texts = [pred["text"] for pred in preds]
        scorer = ScorerRegistry.get_scorer(metric)
        kwargs = {"docs": docs, "client": self.client, "batch_size": 10}
        return scorer.score(refs, pred_texts, **kwargs)

    def gather_scores(self, scores: list[dict], metric: str) -> dict:
        """Gather scores of each prediction based on the metric."""
        kwargs = {"docs": self.dataset}
        scorer = ScorerRegistry.get_scorer(metric)
        return scorer.aggregate(scores, **kwargs)
