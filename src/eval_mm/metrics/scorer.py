class Scorer:
    def __init__(self) -> None:
        pass

    @staticmethod
    def score(refs: list[str], preds: list[str], **kwargs) -> list:
        raise NotImplementedError

    @staticmethod
    def aggregate(scores: list, **kwargs) -> object:
        raise NotImplementedError
