from .scorer import Scorer


class ExactMatchScorer(Scorer):
    @staticmethod
    def score(refs: list[str], preds: list[str], **kwargs) -> list[int]:
        scores = [int(ref == pred) for ref, pred in zip(refs, preds)]
        return scores

    @staticmethod
    def aggregate(scores: list[int], **kwargs) -> float:
        return sum(scores) / len(scores)
