from datasets import Dataset, load_dataset, concatenate_datasets
from PIL import Image
from ..api.registry import register_task
from ..api.task import Task

Image.MAX_IMAGE_PIXELS = None

@register_task("jdocqa_limit")
class JDocQA_Limit(Task):
    @staticmethod
    def _prepare_dataset() -> Dataset:
        # umiyuki/JDocQA_SingleImage データセットを読み込み
        ds = load_dataset(
            "umiyuki/JDocQA_SingleImage_200",
            split="test",
            trust_remote_code=True,
        )

        return ds

    @staticmethod
    def doc_to_text(doc) -> str:
        return doc["input_text"]

    @staticmethod
    def doc_to_visual(doc) -> list[Image.Image]:
        # 既に画像が含まれているので、そのまま返す
        return [doc["image"]] if doc["image"] else []

    @staticmethod
    def doc_to_id(doc) -> int:
        return doc["question_id"]

    @staticmethod
    def doc_to_answer(doc) -> str:
        return doc["answer"]