from abc import ABC, abstractmethod


class Scorer(ABC):
    def __init__(self) -> None:
        pass

    @abstractmethod
    def score(refs: list[str], preds: list[str], **kwargs) -> list:
        raise NotImplementedError

    @abstractmethod
    def aggregate(scores: list, **kwargs) -> object:
        raise NotImplementedError
