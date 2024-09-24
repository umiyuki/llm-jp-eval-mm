import abc
from collections.abc import Callable
from dataclasses import asdict, dataclass


@dataclass
class TaskConfig(dict):
    def __getitem__(self, item):
        return getattr(self, item)

    def __setitem__(self, item, value):
        return setattr(self, item, value)

    def to_dict(self):
        """dumps the current config as a dictionary object, as a printable format.
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
    """A task represents an entire task including its dataset, problems, answers, and evaluation methods.
    A `doc` can be any python object which represents one instance of evaluation.
    This is usually a dictionary e.g.
    {"question": ..., "answer": ...} or {"question": ..., question, answer)
    """

    def __init__(self, config=None) -> None:
        self._config = TaskConfig({**config}) if config else TaskConfig()
        self._dataset = None
        self.prepare_task(config)

    @property
    def config(self):
        """Returns the TaskConfig associated with this class."""
        return self._config

    @property
    def dataset(self):
        """Returns the dataset associated with this class."""
        return self._dataset

    @abc.abstractmethod
    def prepare_task(self, config):
        """Prepares a document for evaluation."""
        pass

    @abc.abstractmethod
    def doc_to_text(self, doc):
        """Converts a document to text."""
        pass

    @abc.abstractmethod
    def doc_to_visual(self, doc):
        """Converts a document to visual."""
        pass

    @abc.abstractmethod
    def doc_to_id(self, doc):
        """Converts a document to id."""
        pass

    @abc.abstractmethod
    def evaluate(self, docs: list, preds: list) -> list[dict]:
        """Evaluate batch prediction."""
        pass

    @abc.abstractmethod
    def compute_metrics(self, preds):
        """
        Args:
            doc: a instance of the eval dataset
            results: [pred]
        Returns:
            metrics: a dictionary with key: metric name (in this case coco_bleu), value: metric value
            results_verbose: a dictionary with key: metric name, value: a dictionary with key: 'score' and 'verbose'
        """
        pass
