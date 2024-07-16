import abc
import ast
import itertools
import json
import os
import random
import re
import shutil
import subprocess
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from glob import glob
from typing import Any, Dict, List, Optional, Union

import datasets
import numpy as np
from datasets import DownloadConfig, Image, Sequence
from huggingface_hub import snapshot_download
from PIL import ImageFile
from tqdm import tqdm

from llm_jp_eval_mm import utils

# HuggingfaceM4/NoCaps contains truncated image in test split
# Include this inside code block to avoid error
ImageFile.LOAD_TRUNCATED_IMAGES = True

ALL_OUTPUT_TYPES = [
    "loglikelihood",
    "multiple_choice",
    "generate_until",
]


@dataclass
class TaskConfig(dict):
    # task naming/registry
    # task: str = None
    # task_alias: str = None
    # group: Union[str, list] = None
    # group_alias: Union[str, list] = None
    # HF dataset options.
    # which dataset to use,
    # and what splits for what purpose
    # dataset_path: str = None
    # dataset_name: str = None
    # dataset_kwargs: dict = None
    # formatting / prompting options.
    # see docs/advanced_task_guide.md for more info
    # process_docs: Callable = None
    # doc_to_visual: Union[Callable, str] = None
    # doc_to_text: Union[Callable, str] = None
    # doc_to_target: Union[Callable, str] = None
    # doc_to_choice: Union[Callable, str, dict, list] = None
    # process_results: Union[Callable, str] = None
    # use_prompt: str = None
    # description: str = ""
    # target_delimiter: str = " "
    # fewshot_delimiter: str = "\n\n"
    # fewshot_config: dict = None
    # runtime configuration options
    # num_fewshot: int = None
    # scoring options
    # metric_list: list = None
    # output_type: str = "generate_until"
    # generation_kwargs: dict = None
    # repeats: int = 1
    # filter_list: Union[str, list] = None
    # should_decontaminate: bool = False
    # doc_to_decontamination_query: str = None

    # metadata: Union[str, list] = None  # by default, not used in the code. allows for users to pass arbitrary info to tasks

    # model_specific_prompt_kwargs: dict = None
    # model_specific_generation_kwargs: dict = None
    # model_specific_target_kwargs: dict = None

    def __post_init__(self) -> None:
        pass
        # if self.dataset_path and os.path.exists(os.path.dirname(self.dataset_path)):
        #     import inspect
        #     from importlib import import_module

        #     # self.dataset_path = inspect.getfile(import_module(self.dataset_path))

        # if self.generation_kwargs is not None:
        #     if self.output_type != "generate_until":
        #         eval_logger.warning(f"[{self.task}] passed `generation_kwargs`, but not using `output_type: generate_until`!")
        #         assert self.output_type != "generate_until"

        #     if "temperature" in self.generation_kwargs:
        #         self.generation_kwargs["temperature"] = float(self.generation_kwargs["temperature"])

        #     if "until" not in self.generation_kwargs:
        #         self.generation_kwargs["until"] = [self.fewshot_delimiter]
        # else:
        #     if self.output_type == "generate_until":
        #         # ensure that we greedily generate in absence of explicit arguments otherwise
        #         self.generation_kwargs = {
        #             "until": None if self.fewshot_delimiter is None else [self.fewshot_delimiter],
        #             "do_sample": False,
        #         }

        # # TODO: how to make TaskConfigs be de- and re-serializable, even when using the !function constructor?

    def __getitem__(self, item):
        return getattr(self, item)

    def __setitem__(self, item, value):
        return setattr(self, item, value)

    def to_dict(self):
        """dumps the current config as a dictionary object, as a printable format.
        null fields will not be printed.
        Used for dumping results alongside full task configuration

        :return: dict
            A printable dictionary version of the TaskConfig object.

        # TODO: should any default value in the TaskConfig not be printed?
        """
        cfg_dict = asdict(self)
        # remove values that are `None`
        for k, v in list(cfg_dict.items()):
            if v is None:
                cfg_dict.pop(k)
            elif isinstance(v, Callable):
                # TODO: this should handle Promptsource template objects as a separate case?
                cfg_dict[k] = str(v)
        return cfg_dict


class Task(abc.ABC):
    """A task represents an entire benchmark including its dataset, problems,
    answers, and evaluation methods. See BoolQ for a simple example implementation

    A `doc` can be any python object which represents one instance of evaluation.
    This is usually a dictionary e.g.
        {"question": ..., "answer": ...} or
        {"question": ..., question, answer)
    """

    # The name of the `Task` benchmark as denoted in the HuggingFace datasets Hub
    # or a path to a custom `datasets` loading script.
    DATASET_PATH: str = ""
    # The name of a subset within `DATASET_PATH`.
    DATASET_NAME: str = ""
    # OUTPUT_TYPE : str = ""

    def __init__(
        self,
        config=None,
    ) -> None:
        self._config = TaskConfig({**config}) if config else TaskConfig()
        self._dataset = None
        self.download()

    def download(self) -> None:
        pass

    @property
    def config(self):
        """Returns the TaskConfig associated with this class."""
        return self._config

    @property
    def dataset(self):
        """Returns the dataset associated with this class."""
        return self._dataset

    @abc.abstractmethod
    def doc_to_text(self, doc):
        """Converts a document to text."""
        pass

    @abc.abstractmethod
    def doc_to_visual(self, doc):
        """Converts a document to visual."""
        pass

    @abc.abstractmethod
    def process_results(self, doc, results):
        """
        Args:
            doc: a instance of the eval dataset
            results: [pred]
        Returns:
            a dictionary with key: metric name (in this case coco_bleu), value: metric value
        """
        pass
