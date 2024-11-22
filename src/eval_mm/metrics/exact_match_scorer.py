from .scorer import Scorer


class ExactMatchScorer(Scorer):
    @staticmethod
    def score(refs: list[str], preds: list[str]) -> list[int]:
        scores = [int(ref == pred) for ref, pred in zip(refs, preds)]
        return scores

    @staticmethod
    def aggregate(scores: list[int]) -> float:
        return sum(scores) / len(scores)
