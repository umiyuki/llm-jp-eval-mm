import abc

from dataclasses import dataclass
from eval_mm.utils.litellm_client import LLMChatAPI
from datasets import Dataset
from PIL import Image


@dataclass
class TaskConfig:
    max_dataset_len: int | None = None
    judge_model: str = "gpt-4o-mini-2024-07-18"
    batch_size_for_evaluation: int = 10
    rotate_choices: bool = False
    provider: str = "custom"
    api_base_judge: str | None = None


class Task(abc.ABC):
    """A task represents an entire task including its dataset, problems, answers, and evaluation methods.
    A `doc` can be any python object which represents one instance of evaluation.
    This is usually a dictionary e.g.
    {"question": ..., "answer": ...} or {"question": ..., question, answer)
    """

    def __init__(self, config: TaskConfig):
        self._dataset = None
        # judge_modelに基づいてproviderを設定
        if config.judge_model.startswith("gemini"):
            provider = "gemini"
        elif config.judge_model.startswith("gpt"):
            provider = "openai"
        elif config.judge_model.startswith("azure"):
            provider = "azure"
        else:
            provider = config.provider
        self.client = LLMChatAPI(provider=provider, api_base=config.api_base_judge)
        self.config = config

        if self.config.max_dataset_len is not None:
            self.dataset = self._prepare_dataset().select(
                range(self.config.max_dataset_len)
            )
        else:
            self.dataset = self._prepare_dataset()

    @abc.abstractmethod
    def _prepare_dataset(self) -> Dataset:
        """Prepares the dataset."""
        pass

    @abc.abstractmethod
    def doc_to_text(self, doc) -> str:
        """Converts a document to text."""
        pass

    @abc.abstractmethod
    def doc_to_visual(self, doc) -> list[Image.Image]:
        """Converts a document to visual."""
        pass

    @abc.abstractmethod
    def doc_to_id(self, doc) -> int:
        """Converts a document to id."""
        pass

    @abc.abstractmethod
    def doc_to_answer(self, doc) -> str:
        """Converts a document to answer."""
        pass
