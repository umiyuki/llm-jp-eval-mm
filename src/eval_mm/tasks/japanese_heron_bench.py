from datasets import load_dataset

from ..api.registry import register_task
from ..api.task import Task
from ..utils.azure_client import OpenAIChatAPI, batch_iter
import numpy as np

RULES: dict = {
    "coding": {
        "role": "Assistant",
        "prompt": "Your task is to evaluate the coding abilities of the above two assistants. They have been asked to implement a program to solve a given problem. Please review their code submissions, paying close attention to their problem-solving approach, code structure, readability, and the inclusion of helpful comments.\n\nPlease ensure that the assistants' submissions:\n\n1. Correctly implement the given problem statement.\n2. Contain accurate and efficient code.\n3. Include clear and concise comments that explain the code's logic and functionality.\n4. Adhere to proper coding standards and best practices.\n\nOnce you have carefully reviewed both submissions, provide detailed feedback on their strengths and weaknesses, along with any suggestions for improvement. You should first output a single line containing two scores on the scale of 1-10 (1: no code/no sense; 10: perfect) for Assistant 1 and 2, respectively. Then give extra comments starting from the next line.",
    },
    "math": {
        "role": "Assistant",
        "prompt": "We would like to request your feedback on the mathematical proficiency of two AI assistants regarding the given user question.\nFirstly, please solve the problem independently, without referring to the answers provided by Assistant 1 and Assistant 2.\nAfterward, please examine the problem-solving process of Assistant 1 and Assistant 2 step-by-step to ensure their correctness, identifying any incorrect steps if present. Your evaluation should take into account not only the answer but also the problem-solving steps.\nFinally, please output a Python tuple containing two numerical scores for Assistant 1 and Assistant 2, ranging from 1 to 10, respectively. If applicable, explain the reasons for any variations in their scores and determine which assistant performed better.",
    },
    "default": {
        "role": "Assistant",
        "prompt": "We would like to request your feedback on the performance of two AI assistants in response to the user question displayed above.\nPlease rate the helpfulness, relevance, accuracy, level of details of their responses. Each assistant receives an overall score on a scale of 1 to 10, where a higher score indicates better overall performance.\nPlease first output a single line containing only two values indicating the scores for Assistant 1 and 2, respectively. The two scores are separated by a space.\nIn the subsequent line, please provide a comprehensive explanation of your evaluation, avoiding any potential bias and ensuring that the order in which the responses were presented does not affect your judgment. Sentences are given in Japanese, and language is not relevant to score.",
    },
    "conv": {
        "role": "Assistant",
        "prompt": "We would like to request your feedback on the performance of two AI assistants in response to the user question displayed above. The user asks the question on observing an image. For your reference, the visual content in the image is represented with five descriptive sentences describing the same image and the bounding box coordinates of each object in the scene. These coordinates are in the form of bounding boxes, represented as (x1, y1, x2, y2) with floating numbers ranging from 0 to 1. These values correspond to the top left x, top left y, bottom right x, and bottom right y. \nPlease rate the helpfulness, relevance, accuracy, level of details of their responses. Each assistant receives an overall score on a scale of 1 to 10, where a higher score indicates better overall performance.\nPlease first output a single line containing only two values indicating the scores for Assistant 1 and 2, respectively. The two scores are separated by a space.\nIn the subsequent line, please provide a comprehensive explanation of your evaluation, avoiding any potential bias and ensuring that the order in which the responses were presented does not affect your judgment. Sentences are given in Japanese, and language is not relevant to score.",
    },
    "detail": {
        "role": "Assistant",
        "prompt": "We would like to request your feedback on the performance of two AI assistants in response to the user question displayed above. The user asks the question on observing an image. For your reference, the visual content in the image is represented with five descriptive sentences describing the same image and the bounding box coordinates of each object in the scene. These coordinates are in the form of bounding boxes, represented as (x1, y1, x2, y2) with floating numbers ranging from 0 to 1. These values correspond to the top left x, top left y, bottom right x, and bottom right y. \nPlease rate the helpfulness, relevance, accuracy, level of details of their responses. Each assistant receives an overall score on a scale of 1 to 10, where a higher score indicates better overall performance.\nPlease first output a single line containing only two values indicating the scores for Assistant 1 and 2, respectively. The two scores are separated by a space.\nIn the subsequent line, please provide a comprehensive explanation of your evaluation, avoiding any potential bias and ensuring that the order in which the responses were presented does not affect your judgment. Sentences are given in Japanese, and language is not relevant to score.",
    },
    "complex": {
        "role": "Assistant",
        "prompt": "We would like to request your feedback on the performance of two AI assistants in response to the user question displayed above. The user asks the question on observing an image. For your reference, the visual content in the image is represented with five descriptive sentences describing the same image and the bounding box coordinates of each object in the scene. These coordinates are in the form of bounding boxes, represented as (x1, y1, x2, y2) with floating numbers ranging from 0 to 1. These values correspond to the top left x, top left y, bottom right x, and bottom right y. \nPlease rate the helpfulness, relevance, accuracy, level of details of their responses. Each assistant receives an overall score on a scale of 1 to 10, where a higher score indicates better overall performance.\nPlease first output a single line containing only two values indicating the scores for Assistant 1 and 2, respectively. The two scores are separated by a space.\nIn the subsequent line, please provide a comprehensive explanation of your evaluation, avoiding any potential bias and ensuring that the order in which the responses were presented does not affect your judgment. Sentences are given in Japanese, and language is not relevant to score.",
    },
    "llava_bench_conv": {
        "role": "Assistant",
        "prompt": "We would like to request your feedback on the performance of two AI assistants in response to the user question displayed above. The user asks the question on observing an image. For your reference, the visual content in the image is represented with a few sentences describing the image. \nPlease rate the helpfulness, relevance, accuracy, level of details of their responses. Each assistant receives an overall score on a scale of 1 to 10, where a higher score indicates better overall performance.\nPlease first output a single line containing only two values indicating the scores for Assistant 1 and 2, respectively. The two scores are separated by a space.\nIn the subsequent line, please provide a comprehensive explanation of your evaluation, avoiding any potential bias and ensuring that the order in which the responses were presented does not affect your judgment.",
    },
    "llava_bench_detail": {
        "role": "Assistant",
        "prompt": "We would like to request your feedback on the performance of two AI assistants in response to the user question displayed above. The user asks the question on observing an image. For your reference, the visual content in the image is represented with a few sentences describing the image. \nPlease rate the helpfulness, relevance, accuracy, level of details of their responses. Each assistant receives an overall score on a scale of 1 to 10, where a higher score indicates better overall performance.\nPlease first output a single line containing only two values indicating the scores for Assistant 1 and 2, respectively. The two scores are separated by a space.\nIn the subsequent line, please provide a comprehensive explanation of your evaluation, avoiding any potential bias and ensuring that the order in which the responses were presented does not affect your judgment.",
    },
    "llava_bench_complex": {
        "role": "Assistant",
        "prompt": "We would like to request your feedback on the performance of two AI assistants in response to the user question displayed above. The user asks the question on observing an image. For your reference, the visual content in the image is represented with a few sentences describing the image. \nPlease rate the helpfulness, relevance, accuracy, level of details of their responses. Each assistant receives an overall score on a scale of 1 to 10, where a higher score indicates better overall performance.\nPlease first output a single line containing only two values indicating the scores for Assistant 1 and 2, respectively. The two scores are separated by a space.\nIn the subsequent line, please provide a comprehensive explanation of your evaluation, avoiding any potential bias and ensuring that the order in which the responses were presented does not affect your judgment.",
    },
}


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


def ask_gpt4_batch(
    content_list: str, max_tokens: int, async_client: OpenAIChatAPI, model_name: str
) -> list:
    message_list = [
        [
            {
                "role": "system",
                "content": "You are a helpful and precise assistant for checking the quality of the answer.",
            },
            {"role": "user", "content": content},
        ]
        for content in content_list
    ]
    completions = async_client.batch_generate_chat_response(
        message_list,
        max_tokens=max_tokens,
        temperature=0,
        seed=0,
        model_name=model_name,
    )
    return completions


@register_task("japanese-heron-bench")
class JapaneseHeronBench(Task):
    def __init__(self, config=None) -> None:
        super().__init__(config)
        self.category_list = ["conv", "detail", "complex"]
        self.rules: dict = {k: RULES[k] for k in self.category_list}
        self.client = OpenAIChatAPI()

    @property
    def dataset(self):
        """Returns the dataset associated with this class."""
        return self._dataset

    def prepare_task(self, config) -> None:
        self._dataset = load_dataset("Silviase/Japanese-Heron-Bench", split="train")

        # rename columns
        if self._dataset is not None:
            self._dataset = self._dataset.rename_column("text", "input_text")
        else:
            raise ValueError("Dataset is None, cannot rename column.")

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

    def evaluate(self, docs: list, preds: list, model_id: str) -> list:
        """Evaluate batch prediction.
        Args:
        docs : list of instance of the eval dataset
        preds : list of dict with keys: { 'question_id', 'text' }
        model_id : openai api's model name
        Returns:
        eval_results: list of dictionary with keys:
            { 'input_text', 'pred', 'context', 'category', 'answer', 'score', 'score_gpt' }
        """
        categories = [doc["category"] for doc in docs]
        answer_1s = [
            doc["answer"]["gpt-4-0125-preview"] for doc in docs
        ]  # reference answer
        answer_2s = [pred["text"] for pred in preds]  # predicted answer
        roles = [self.rules[category]["role"] for category in categories]
        prompts = [self.rules[category]["prompt"] for category in categories]

        def build_content(doc, answer_1, answer_2, role, prompt):
            return (
                f'[Context]\n{doc["context"]}\n\n'
                f'[Question]\n{doc["input_text"]}\n\n'
                f"[{role} 1]\n{answer_1}\n\n[End of {role} 1]\n\n"
                f"[{role} 2]\n{answer_2}\n\n[End of {role} 2]\n\n"
                f"[System]\n{prompt}\n\n"
                f"If it is not relevant to the context, does not answer directly, or says the wrong thing, give it a low score.\n\n"
            )

        contents = [
            build_content(doc, answer_1, answer_2, role, prompt)
            for doc, answer_1, answer_2, role, prompt in zip(
                docs, answer_1s, answer_2s, roles, prompts
            )
        ]
        completions = ask_gpt4_batch(contents, 1024, self.client, model_id)
        scores = [parse_score(completion) for completion in completions]
        eval_results = []
        for doc, score in zip(docs, scores):
            eval_result = doc
            eval_result["score"] = score[1]
            eval_result["score_gpt"] = score[0]
            eval_results.append(eval_result)

        return eval_results

    def compute_metrics(self, preds, model_id="gpt-4o-mini-2024-07-18", batch_size=10):
        """Process the results of the model.
        Args:
            jsonl_path: jsonl_path
            preds: [pred]
            model_id: openai api's model name (default: "gpt-4o-mini-2024-07-18")
            batch_size: batch size for evaluation
        Return:
            a dictionary with key: { 'score' : score }
        """
        eval_results = []
        docs = self.dataset

        for i, batch_idx in enumerate(batch_iter(range(len(preds)), batch_size)):
            doc_batch = [docs[idx] for idx in batch_idx]
            pred_batch = [preds[idx] for idx in batch_idx]
            eval_result_batch = self.evaluate(doc_batch, pred_batch, model_id)
            eval_results.extend(eval_result_batch)

        # average score for each category, and overall
        metrics = {}
        for category in self.category_list:
            scores = [r["score"] for r in eval_results if r["category"] == category]
            score_gpts = [
                r["score_gpt"] for r in eval_results if r["category"] == category
            ]
            if len(scores) == 0:
                continue
            avg_score = np.mean(scores)
            avs_score_rel = (
                100
                * np.mean(scores)
                / max(
                    0.01, np.mean(score_gpts)
                )  # divide by 0.01 when 0 division happens
            )
            metrics[category] = avg_score
            metrics[category + "_rel"] = avs_score_rel
        metrics["parse_error_count"] = sum(r["score"] == -1 for r in eval_results)

        metrics["overall"] = sum([r["score"] for r in eval_results]) / len(eval_results)
        metrics["overall_rel"] = sum(
            [metrics[category + "_rel"] for category in self.category_list]
        ) / len(self.category_list)
        metrics["openai_model_id"] = model_id

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
                    "score", "score_gpt", # etc.
                }
        Return:
            dictonaries with keys:
            {
                    "question_id": str,
                    "text": str,
                    "score": float,
                    "score_gpt": float,
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
                "score_gpt": eval_result["score_gpt"],
            }
            results.append(result)
        return results
