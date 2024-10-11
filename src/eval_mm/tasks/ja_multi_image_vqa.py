from datasets import Dataset, load_dataset
import re

from ..api.registry import register_task
from ..api.task import Task
from ..utils.azure_client import OpenAIChatAPI
from ..utils.templates import qa_pointwise
from ..utils.metrics import llm_as_a_judge


@register_task("ja-multi-image-vqa")
class JAMultiImageVQA(Task):
    def __init__(self, config=None) -> None:
        super().__init__(config)
        self.client = OpenAIChatAPI()

    @property
    def dataset(self):
        """Returns the dataset associated with this class."""
        return self._dataset

    def prepare_task(self, config) -> None:
        self._dataset = self.prepare_dataset(config)

        # rename columns
        if self._dataset is not None:
            self._dataset = self._dataset.rename_column("question", "input_text")
        else:
            raise ValueError("Dataset is None, cannot rename column.")

        # add question_id
        self._dataset = self._dataset.map(
            lambda example, idx: {"question_id": idx}, with_indices=True
        )

    def prepare_dataset(self, config) -> Dataset:
        dataset = load_dataset("SakanaAI/JA-Multi-Image-VQA", split="test")
        return dataset

    def doc_to_text(self, doc):
        # delete redundant image tags
        text = re.sub(r"<image> ", "", doc["input_text"])
        return text

    def doc_to_visual(self, doc):
        print("CAUTION: This task provides MULTIPLE images.")
        return doc["images"]

    def doc_to_id(self, doc):
        return doc["question_id"]

    def process_pred(self, doc, pred):
        processed = doc
        processed["pred"] = pred
        return processed

    def evaluate(
        self, docs: list, preds: list, batch_size, model_name: str
    ) -> list[dict]:
        """Evaluate batch prediction.
        Args:
        doc : list of instance of the eval dataset
        pred : list of dict with keys: { 'question_id', 'text' }
        Returns:
        eval_results: list of dictionary with keys:
            { 'input_text', 'pred', 'qa_id','answer', 'score' }
        Reference:
        注: 評価方法はGPT-4oによるスコアリング方法を採用しました。各設問ごとに5点満点で評価するようGPT-4oに指示を出し、平均点をモデルのスコアとしています。値が高いほど複数画像に対する日本語での質疑応答能力が高いと言えます。
        """
        input_texts = [doc["input_text"] for doc in docs]
        answer_list = [doc["answer"] for doc in docs]
        pred_list = [pred["text"] for pred in preds]
        llm_as_a_judge_score_list = llm_as_a_judge(
            self.client,
            qa_pointwise,
            input_texts,
            answer_list,
            pred_list,
            batch_size,
            model_name,
        )
        eval_results = []
        for doc, pred, llm_as_a_judge_score in zip(
            docs, preds, llm_as_a_judge_score_list
        ):
            eval_result = {
                "input_text": doc["input_text"],
                "pred": pred["text"],
                "qa_id": doc["question_id"],
                "answer": doc["answer"],
                "score_llm_as_a_judge": llm_as_a_judge_score["score"],
            }
            eval_results.append(eval_result)
        return eval_results

    def compute_metrics(self, preds, model_id="gpt-4o-mini-2024-07-18", batch_size=10):
        """Process the results of the model.
        Args:
            jsonl_path: jsonl_path
            preds: [pred]
            model_id: openai api's model name (default: "gpt-4o-mini-2024-07-18")
        Return:
            metrics: a dictionary with key: { 'rouge-l' : rouge-l score }
            eval_results: a list of dictionaries with keys:
                { 'input_text', 'pred', 'qa_id','answer', 'score' }
        """
        eval_results = []
        docs = self.dataset

        eval_results = self.evaluate(docs, preds, batch_size, model_id)

        # average score for each category, and overall
        metrics = {
            "llm_as_a_judge": sum([doc["score_llm_as_a_judge"] for doc in eval_results])
            / len(eval_results),
            "openai_model_id": model_id,
        }

        return metrics, eval_results

    def format_result(self, preds: list[dict], eval_results: list[dict]) -> list[dict]:
        """Format the result of the model.
        Args:
            preds:
                list of dictionaries with keys:
                {
                    "question_id": str, "text": str,
                }
            eval_results:
                list of dictionaries with keys:
                {
                    "score", # etc.
                }
        Return:
            dictonaries with keys:
            {
                    "question_id": str,
                    "text": str,
                    "score": float,
            }
        """
        assert len(preds) == len(
            eval_results
        ), "Length of preds and eval_results must be equal."
        results = []
        for pred, eval_result in zip(preds, eval_results):
            result = {
                "question_id": pred["question_id"],
                "text": pred["text"],
                "score_llm_as_a_judge": eval_result["score_llm_as_a_judge"],
            }
            results.append(result)
        return results
