# Reference: https://github.com/EvolvingLMMs-Lab/lmms-eval/blob/main/lmms_eval/tasks/jmmmu/utils.py

from datasets import Dataset, load_dataset, concatenate_datasets

from ..api.registry import register_task
from ..api.task import Task

import ast
import re
from eval_mm.metrics import ScorerRegistry


MULTI_CHOICE_PROMPT = (
    "与えられた選択肢の中から最も適切な回答のアルファベットを直接記入してください。"
)
OPEN_ENDED_PROMPT = "質問に対する回答を単語や短いフレーズで記入してください。"


def replace_images_tokens(input_string):
    for i in range(1, 8):
        question_text = f"<image {i}>"
        query_text = "<image>"
        if question_text in input_string:
            input_string = input_string.replace(question_text, query_text)
    return input_string


def parse_options(options):
    option_letters = [chr(ord("A") + i) for i in range(len(options))]
    choices_str = "\n".join(
        [
            f"{option_letter}. {option}"
            for option_letter, option in zip(option_letters, options)
        ]
    )
    return choices_str


def construct_prompt(doc):
    question = doc["question"]
    question = question.replace("<image1>", "<image 1>")
    if doc["question_type"] == "multiple-choice":
        # Weirdly, data["options"] is a string in MMMU Huggingface dataset
        parsed_options = parse_options(ast.literal_eval(doc["options"]))
        # parsed_options already prepends a newline so no need to add space here
        question = f"{question}\n{parsed_options}\n\n{MULTI_CHOICE_PROMPT}"
    else:
        question = f"{question}\n\n{OPEN_ENDED_PROMPT}"
    return question


def jmmmu_doc_to_text(doc):
    question = construct_prompt(doc)
    question = replace_images_tokens(question)  # TODO: check if this is necessary
    return question


def jmmmu_doc_to_visual(doc):
    prompt = construct_prompt(doc)

    image_tokens = re.findall(r"<image \d+>", prompt)
    # Remove <> and  swap space as _
    image_tokens = sorted(
        list(
            set(
                [
                    image_token.strip("<>").replace(" ", "_")
                    for image_token in image_tokens
                ]
            )
        )
    )
    visual = [doc[image_token].convert("RGB") for image_token in image_tokens]
    return visual


@register_task("jmmmu")
class JMMMU(Task):
    def __init__(
        self, max_dataset_len: int = None, judge_model: str = "gpt-4o-mini-2024-07-18"
    ):
        super().__init__()
        self._dataset = None
        self.judge_model = judge_model
        if max_dataset_len is not None:
            self.dataset = self.prepare_dataset().select(range(max_dataset_len))
        else:
            self.dataset = self.prepare_dataset()

    @staticmethod
    def prepare_dataset() -> Dataset:
        SUBJECTS = [
            "Accounting",
            "Agriculture",
            "Architecture_and_Engineering",
            "Basic_Medical_Science",
            "Biology",
            "Chemistry",
            "Clinical_Medicine",
            "Computer_Science",
            "Design",
            "Diagnostics_and_Laboratory_Medicine",
            "Economics",
            "Electronics",
            "Energy_and_Power",
            "Finance",
            "Japanese_Art",
            "Japanese_Heritage",
            "Japanese_History",
            "Manage",
            "Marketing",
            "Materials",
            "Math",
            "Mechanical_Engineering",
            "Music",
            "Pharmacy",
            "Physics",
            "Psychology",
            "Public_Health",
            "World_History",
        ]
        dataset = None
        for subject in SUBJECTS:
            if dataset is None:
                dataset = load_dataset("JMMMU/JMMMU", name=subject, split="test")
            else:
                dataset = concatenate_datasets(
                    [dataset, load_dataset("JMMMU/JMMMU", name=subject, split="test")]
                )
        dataset = dataset.map(
            lambda x: {
                "input_text": jmmmu_doc_to_text(x),
                "question_id": x["id"],
                "answer": x["answer"],
            }
        )
        return dataset

    def doc_to_text(self, doc):
        return doc["input_text"]

    def doc_to_visual(self, doc):
        return jmmmu_doc_to_visual(doc)

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
