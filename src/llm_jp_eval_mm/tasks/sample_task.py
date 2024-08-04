import os

from datasets import load_dataset
from dotenv import load_dotenv
from openai import AzureOpenAI
from tqdm import tqdm

from llm_jp_eval_mm.api.registry import register_task
from llm_jp_eval_mm.api.tasks import Task


@register_task("sample")
class SampleTask(Task):
    def __init__(
        self,
        config=None,
    ):
        super().__init__(config)

    @property
    def dataset(self):
        pass

    def doc_to_text(self, doc):
        return doc["text"]

    def doc_to_visual(self, doc):
        return doc["image"]

    def process_results(self, docs, preds):
        """Process the results of the model.
        Args:
            doc: dataset instance
            preds: [pred]
        Return:
            a dictionary with key: { 'score' : score }
        """

        return {"metric_sample": 0}
