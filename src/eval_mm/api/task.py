import abc


class Task(abc.ABC):
    """A task represents an entire task including its dataset, problems, answers, and evaluation methods.
    A `doc` can be any python object which represents one instance of evaluation.
    This is usually a dictionary e.g.
    {"question": ..., "answer": ...} or {"question": ..., question, answer)
    """

    def __init__(self) -> None:
        self.dataset = None

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
    def doc_to_answer(self, doc):
        """Converts a document to answer."""
        pass
