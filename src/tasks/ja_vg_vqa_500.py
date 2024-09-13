from datasets import Dataset, concatenate_datasets, load_dataset
from tqdm import tqdm

from ..api.registry import register_task
from ..api.task import Task
from ..utils.metrics import rouge_ja


@register_task("ja-vg-vqa-500")
class JaVGVQA500(Task):
    def __init__(self, config=None) -> None:
        super().__init__(config)

    @property
    def dataset(self):
        """Returns the dataset associated with this class."""
        return self._dataset

    def prepare_task(self, config) -> None:
        self._dataset = self.prepare_dataset(config)

        # rename columns
        if self._dataset is not None:
            self._dataset = self._dataset.rename_column("question", "input_text")
            self._dataset = self._dataset.rename_column("qa_id", "question_id")
        else:
            raise ValueError("Dataset is None, cannot rename column.")

    def prepare_dataset(self, config) -> Dataset:
        # データセットをロード
        dataset = load_dataset("SakanaAI/JA-VG-VQA-500", split="test")

        # flatten "qas" to q/a/id triplets
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
        for i, sample in enumerate(dataset):
            data_fragment = flatten_sample(sample)
            fragments.append(data_fragment)

        dataset = concatenate_datasets(fragments)

        return dataset

    def doc_to_text(self, doc):
        return doc["input_text"]

    def doc_to_visual(self, doc):
        return doc["image"]

    def doc_to_id(self, doc):
        return doc["question_id"]

    def process_pred(self, doc, pred):
        processed = doc
        processed["pred"] = pred
        return processed

    def evaluate(self, doc, pred):
        """Evaluate a single prediction.
        Args:
        doc : a instance of the eval dataset
        pred : a dict with keys: { 'question_id', 'text' }
        model_id : openai api's model name (default: "gpt-4o-mini-2024-07-18")
        Returns:
        eval_result: a dictionary with keys:
            { 'input_text', 'pred', 'qa_id','answer', 'score' }
        """

        assert doc["question_id"] == pred["question_id"]

        # TODO: Implement evaluation logic
        scores = rouge_ja(doc["answer"], pred["text"])

        eval_result = doc
        eval_result["score"] = scores["rougeL"]

        del doc["image"]
        return eval_result

    def compute_metrics(self, preds):
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

        for doc, pred in tqdm(
            zip(docs, preds), total=len(preds), desc="Evaluation ..."
        ):
            eval_result = self.evaluate(doc, pred)
            eval_results.append(eval_result)

        # average score for each category, and overall
        metrics = {
            "rougeL": sum([doc["score"] for doc in eval_results]) / len(eval_results),
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
                "score": eval_result["score"],
            }
            results.append(result)
        return results
