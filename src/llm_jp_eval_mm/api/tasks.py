import abc
from collections.abc import Callable
from dataclasses import asdict, dataclass


@dataclass
class TaskConfig(dict):

    def __post_init__(self) -> None:
        pass

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
