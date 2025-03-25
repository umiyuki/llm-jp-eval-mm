# Reference: https://github.com/EvolvingLMMs-Lab/lmms-eval/blob/main/lmms_eval/tasks/jmmmu/utils.py

from datasets import (
    DatasetDict,
    Dataset,
    load_dataset,
    concatenate_datasets,
    get_dataset_config_names,
)

from ..api.registry import register_task
from ..api.task import Task

import ast
import re
from PIL import Image
from loguru import logger

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

def filter_single_image_questions(dataset):
    def has_single_image(doc):
        prompt = construct_prompt(doc)
        image_tokens = re.findall(r"<image \d+>", prompt)
        unique_tokens = len(set(image_tokens))  # 重複を除いた画像トークンの数
        return unique_tokens <= 1  # 画像が1枚以下の場合にTrue

    return dataset.filter(has_single_image)

def filter_large_images(dataset):
    def is_large_enough(doc):
        visuals = jmmmu_doc_to_visual(doc)
        if not visuals:  # 画像がない場合は除外しない（必要に応じて調整）
            return True
        for img in visuals:
            width, height = img.size
            if min(width, height) < 28:  # 短辺が28未満ならFalse
                return False
        return True

    return dataset.filter(is_large_enough)

@register_task("jmmmu_limit")
class JMMMU_Limit(Task):
    @staticmethod
    def _prepare_dataset() -> Dataset:
        configs = get_dataset_config_names("JMMMU/JMMMU")
        logger.info(f"Configs: {configs}")
        if not configs:
            raise ValueError("No configurations found for JMMMU/JMMMU dataset.")

        filtered_datasets = []
        for subject in configs:
            logger.info(f"Loading dataset for subject: {subject}")
            dataset = load_dataset("JMMMU/JMMMU", name=subject, split="test")
            logger.info(f"Initial dataset size for {subject}: {len(dataset)}")

            dataset = filter_single_image_questions(dataset)
            logger.info(f"After filtering single images for {subject}: {len(dataset)}")

            dataset = filter_large_images(dataset)
            logger.info(f"After filtering large images for {subject}: {len(dataset)}")

            # サブセットから先頭10問を選択
            if len(dataset) > 0:
                num_samples = min(10, len(dataset))
                dataset = dataset.select(range(num_samples))
                filtered_datasets.append(dataset)
            else:
                logger.warning(f"No data remains for subject {subject} after filtering")

        if not filtered_datasets:
            logger.warning("No datasets remain after filtering and selection. Returning empty dataset.")
            return Dataset.from_dict({})

        combined_dataset = concatenate_datasets(filtered_datasets)
        logger.info(f"Final dataset size: {len(combined_dataset)}")

        # マッピング
        combined_dataset = combined_dataset.map(
            lambda x: {
                "input_text": jmmmu_doc_to_text(x),
                "question_id": x["id"],
                "answer": x["answer"],
                "question_type": x["question_type"],
                "options": x["options"],
            }
        )
        return combined_dataset

    @staticmethod
    def doc_to_text(doc) -> str:
        return doc["input_text"]

    @staticmethod
    def doc_to_visual(doc) -> list[Image.Image]:
        return jmmmu_doc_to_visual(doc)

    @staticmethod
    def doc_to_id(doc) -> str:
        return doc["question_id"]

    @staticmethod
    def doc_to_answer(doc) -> str:
        return doc["answer"]


def test_task():
    from eval_mm.api.task import TaskConfig

    task = JMMMU(TaskConfig())
    ds = task.dataset
    print(ds[0])
    assert isinstance(task.doc_to_text(ds[0]), str)
    assert isinstance(task.doc_to_visual(ds[0]), list)
    assert isinstance(task.doc_to_visual(ds[0])[0], Image.Image)
    assert isinstance(task.doc_to_id(ds[0]), str)
    assert isinstance(task.doc_to_answer(ds[0]), str)
