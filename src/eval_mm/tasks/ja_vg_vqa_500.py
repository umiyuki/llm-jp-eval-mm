from datasets import Dataset, concatenate_datasets, load_dataset
from tqdm import tqdm

from ..api.registry import register_task
from ..api.task import Task
from ..utils.metrics import rouge_ja, llm_as_a_judge
from ..utils.azure_client import OpenAIChatAPI
from ..utils.templates import qa_pointwise


@register_task("ja-vg-vqa-500")
class JaVGVQA500(Task):
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

    def evaluate(self, docs, preds, batch_size, model_name) -> dict:
        """Evaluate batch prediction.
        Args:
        doc : list of instance of the eval dataset
        pred : list of dict with keys: { 'question_id', 'text' }
        Returns:
        eval_results: list of dictionary with keys:
            { 'input_text', 'pred', 'qa_id','answer', 'score' }
        """
        rouge_score_list = []
        from concurrent.futures import ProcessPoolExecutor

        with ProcessPoolExecutor() as executor:
            for doc, pred in tqdm(
                zip(docs, preds), total=len(docs), desc="Evaluating ROUGE"
            ):
                future = executor.submit(rouge_ja, [doc["answer"]], [pred["text"]])
                rouge_score_list.append(future)
            rouge_score_list = [future.result() for future in rouge_score_list]

        input_text_list = [doc["input_text"] for doc in docs]
        answer_list = [doc["answer"] for doc in docs]
        pred_list = [pred["text"] for pred in preds]
        llm_as_a_judge_score_list = llm_as_a_judge(
            self.client,
            qa_pointwise,
            input_text_list,
            answer_list,
            pred_list,
            batch_size,
            model_name,
        )
        eval_results = []
        for doc, pred, rouge_score, llm_as_a_judge_score in zip(
            docs, preds, rouge_score_list, llm_as_a_judge_score_list
        ):
            eval_result = {
                "input_text": doc["input_text"],
                "pred": pred["text"],
                "qa_id": doc["question_id"],
                "answer": doc["answer"],
                "score_rougeL": rouge_score["rougeL"],
                "score_llm_as_a_judge": llm_as_a_judge_score["score"],
            }
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

        eval_results = self.evaluate(docs, preds, batch_size, model_id)

        metrics = {
            "rougeL": sum([doc["score_rougeL"] for doc in eval_results])
            / len(eval_results),
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
                "score_rougeL": eval_result["score_rougeL"],
                "score_llm_as_a_judge": eval_result["score_llm_as_a_judge"],
            }
            results.append(result)
        return results
