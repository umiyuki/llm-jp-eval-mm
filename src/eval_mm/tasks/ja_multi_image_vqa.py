from datasets import Dataset, load_dataset
from tqdm import tqdm
import re

from ..api.registry import register_task
from ..api.task import Task
from ..utils.azure_client import OpenAIChatAPI, batch_iter
from ..utils.templates import qa_pointwise


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

    def evaluate(self, docs: list, preds: list) -> list[dict]:
        """Evaluate batch prediction.
        Args:
        doc : list of instance of the eval dataset
        pred : list of dict with keys: { 'question_id', 'text' }
        Returns:
        eval_results: list of dictionary with keys:
            { 'input_text', 'pred', 'qa_id','answer', 'score' }
        Reference:
        注：評価方法はGPT-4oによるスコアリング方法を採用しました。各設問ごとに5点満点で評価するようGPT-4oに指示を出し、平均点をモデルのスコアとしています。値が高いほど複数画像に対する日本語での質疑応答能力が高いと言えます。
        """
        assert len(docs) == len(preds), "Length of docs and preds must be equal."
        assert all(
            [
                doc["question_id"] == pred["question_id"]
                for doc, pred in zip(docs, preds)
            ]
        ), "Question IDs must be the same."

        def build_message(template, doc, pred):
            content = template.format(
                input_text=doc["input_text"],
                pred=pred["text"],
                answer=doc["answer"],
            )
            message = [{"role": "user", "content": content}]
            return message

        messages = [
            build_message(qa_pointwise, doc, pred) for doc, pred in zip(docs, preds)
        ]
        completions = self.client.batch_generate_chat_response(
            messages,
            max_tokens=1024,
            temperature=0.0,
        )

        def parse_score(completion):
            # find Score: X
            score = re.search(r"Score: (\d)", completion)
            score = int(score.group(1)) if score else 1
            if score not in [1, 2, 3, 4, 5]:
                raise ValueError("Score Value Error.")

            return {"score": score, "rationale": completion}

        scores = [parse_score(completion) for completion in completions]

        eval_results = [doc for doc in docs]
        eval_results = []
        for doc, pred, score in zip(docs, preds, scores):
            eval_result = doc
            eval_result["pred"] = pred["text"]
            eval_result["score"] = score["score"]
            eval_result["rationale"] = score["rationale"]
            eval_results.append(eval_result)

        return eval_results

    def compute_metrics(self, preds, model_id="gpt-4o-mini-2024-07-18", batch_size=1):
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

        with tqdm(total=len(preds), desc="Evaluation ...") as pbar:
            for i, batch_idx in enumerate(batch_iter(range(len(preds)), batch_size)):
                doc_batch = [docs[idx] for idx in batch_idx]
                pred_batch = [preds[idx] for idx in batch_idx]
                eval_results_batch = self.evaluate(doc_batch, pred_batch)
                eval_results.extend(eval_results_batch)
                pbar.update(len(batch_idx))

        # average score for each category, and overall
        metrics = {
            "score": sum([doc["score"] for doc in eval_results]) / len(eval_results),
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
