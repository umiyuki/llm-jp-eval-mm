class Scorer:
    def __init__(self) -> None:
        pass

    @staticmethod
    def score(refs: list[str], preds: list[str]) -> list[int]:
        raise NotImplementedError

    @staticmethod
    def aggregate(scores: list) -> float:
        raise NotImplementedError
