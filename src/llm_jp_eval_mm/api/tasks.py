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
from typing import Any, List, Optional, Union

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
        data_dir=None,
        cache_dir=None,
        download_mode=None,
        config=None,
    ) -> None:
        """
        :param data_dir: str
            Stores the path to a local folder containing the `Task`'s data files.
            Use this to specify the path to manually downloaded data (usually when
            the dataset is not publicly accessible).
        :param cache_dir: str
            The directory to read/write the `Task` dataset. This follows the
            HuggingFace `datasets` API with the default cache directory located at:
                `~/.cache/huggingface/datasets`
            NOTE: You can change the cache location globally for a given process
            to another directory:
                `export HF_DATASETS_CACHE="/path/to/another/directory"`
        :param download_mode: datasets.DownloadMode
            How to treat pre-existing `Task` downloads and data.
            - `datasets.DownloadMode.REUSE_DATASET_IF_EXISTS`
                Reuse download and reuse dataset.
            - `datasets.DownloadMode.REUSE_CACHE_IF_EXISTS`
                Reuse download with fresh dataset.
            - `datasets.DownloadMode.FORCE_REDOWNLOAD`
                Fresh download and fresh dataset.
        """
        self.download(data_dir, cache_dir, download_mode)
        self._training_docs = None
        self._fewshot_docs = None
        self._instances: Optional[list[Any]] = None
        self._config = TaskConfig({**config}) if config else TaskConfig()

    def download(self, data_dir=None, cache_dir=None, download_mode=None) -> None:
        """Downloads and returns the task dataset.
        Override this method to download the dataset from a custom API.

        :param data_dir: str
            Stores the path to a local folder containing the `Task`'s data files.
            Use this to specify the path to manually downloaded data (usually when
            the dataset is not publicly accessible).
        :param cache_dir: str
            The directory to read/write the `Task` dataset. This follows the
            HuggingFace `datasets` API with the default cache directory located at:
                `~/.cache/huggingface/datasets`
            NOTE: You can change the cache location globally for a given process
            by setting the shell environment variable, `HF_DATASETS_CACHE`,
            to another directory:
                `export HF_DATASETS_CACHE="/path/to/another/directory"`
        :param download_mode: datasets.DownloadMode
            How to treat pre-existing `Task` downloads and data.
            - `datasets.DownloadMode.REUSE_DATASET_IF_EXISTS`
                Reuse download and reuse dataset.
            - `datasets.DownloadMode.REUSE_CACHE_IF_EXISTS`
                Reuse download with fresh dataset.
            - `datasets.DownloadMode.FORCE_REDOWNLOAD`
                Fresh download and fresh dataset.
        """
        self.dataset = datasets.load_dataset(
            path=self.DATASET_PATH,
            name=self.DATASET_NAME,
            data_dir=data_dir,
            cache_dir=cache_dir,
            download_mode=download_mode,
        )
        self.dataset_no_image = datasets.load_dataset(
            path=self.DATASET_PATH,
            name=self.DATASET_NAME,
            data_dir=data_dir,
            cache_dir=cache_dir,
            download_mode=download_mode,
        )
        for doc_name in self.dataset_no_image:
            remove_cols = []
            features = self.dataset_no_image[doc_name].features
            # If it is an Image instance or a Sequence of Image instance. Remove it
            for feature in features:
                if isinstance(features[feature], Image):
                    remove_cols.append(feature)
                elif isinstance(features[feature], Sequence) and isinstance(features[feature].feature, Image):
                    remove_cols.append(feature)
            for remove_col in remove_cols:
                self.dataset_no_image[doc_name] = self.dataset_no_image[doc_name].remove_columns(remove_col)

    @property
    def config(self):
        """Returns the TaskConfig associated with this class."""
        return self._config

    @abc.abstractmethod
    def has_training_docs(self):
        """Whether the task has a training set"""
        pass

    @abc.abstractmethod
    def has_validation_docs(self):
        """Whether the task has a validation set"""
        pass

    @abc.abstractmethod
    def has_test_docs(self):
        """Whether the task has a test set"""
        pass

    def training_docs(self):
        """
        :return: Iterable[obj]
            A iterable of any object, that doc_to_text can handle
        """
        return []

    def validation_docs(self):
        """
        :return: Iterable[obj]
            A iterable of any object, that doc_to_text can handle
        """
        return []

    def test_docs(self):
        """
        :return: Iterable[obj]
            A iterable of any object, that doc_to_text can handle
        """
        return []

    def fewshot_docs(self):
        """
        :return: Iterable[obj]
            A iterable of any object, that doc_to_text can handle
        """
        if self.has_training_docs():
            return self.training_docs()
        elif self.has_validation_docs():
            return self.validation_docs()
        else:
            return self.test_docs()

    def _process_doc(self, doc):
        """
        Override this to process (detokenize, strip, replace, etc.) individual
        documents. This can be used in a map over documents of a data split.
        E.g. `map(self._process_doc, self.dataset["validation"])`

        :return: dict
            The processed version of the specified `doc`.
        """
        return doc

    @property
    def instances(self):
        """After calling `task.build_all_requests()`, tasks
        maintain a list of the dataset instances which will be evaluated.
        """
        return self._instances

    def fewshot_examples(self, k, rnd):
        if self._training_docs is None:
            self._training_docs = list(self.training_docs())

        return rnd.sample(self._training_docs, k)

    def doc_to_decontamination_query(self, doc) -> None:
        print("Override doc_to_decontamination_query with document specific decontamination query.")
        assert False

    @abc.abstractmethod
    def doc_to_text(self, doc):
        pass

    @abc.abstractmethod
    def doc_to_target(self, doc):
        pass

    @abc.abstractmethod
    def construct_requests(self, doc_id, ctx, **kwargs):
        """Uses RequestFactory to construct Requests and returns an iterable of
        Requests which will be sent to the LMM.

        :param doc_id: int
            The index of a document within `self.test_docs()` or `self.validation_docs()`,
            whichever is the main split used.
        :param ctx: str
            The context string, generated by fewshot_context. This includes the natural
            language description, as well as the few shot examples, and the question
            part of the document for `doc`.
        :param repeats: int
        TODO: update this docstring
            The number of times each instance in a dataset is inferred on. Defaults to 1,
            can be increased for techniques like majority voting.
        """
        pass

    @abc.abstractmethod
    def process_results(self, doc, results):
        """Take a single document and the LMM results and evaluates, returning a
        dict where keys are the names of submetrics and values are the values of
        the metric for that one document

        :param doc:
            The document as returned from training_docs, validation_docs, or test_docs.
        :param results:
            The results of the requests created in construct_requests.
        """
        pass

    @abc.abstractmethod
    def aggregation(self):
        """
        :returns: {str: [metric_score] -> float}
            A dictionary where keys are the names of submetrics and values are
            functions that aggregate a list of metric scores
        """
        pass

    def dump_config(self) -> dict:
        """Returns a dictionary representing the task's config.

        :returns: str
            The fewshot context.
        """
        # TODO: this should only return the overrides applied to a non-YAML task's configuration.
        # (num_fewshot)
        return self.config.to_dict()
