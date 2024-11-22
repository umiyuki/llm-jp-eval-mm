from .scorer import Scorer


class SubstringMatchScorer(Scorer):
    @staticmethod
    def score(refs: list[str], preds: list[str]) -> list[int]:
        scores = [int(ref in pred) for ref, pred in zip(refs, preds)]
        return scores

    @staticmethod
    def aggregate(scores: list[int]) -> float:
        return sum(scores) / len(scores)


def test_substring_match_scorer():
    refs = ["私は猫です。"]
    preds = ["私は猫です。"]
    scores = SubstringMatchScorer.score(refs, preds)
    assert scores == [1]
    refs = ["たかしが公園で遊んでいた。"]
    preds = ["たかしが公園にいたようだ。"]
    scores = SubstringMatchScorer.score(refs, preds)
    assert scores == [0]
    refs = ["私は猫です。", "私は犬です。"]
    preds = ["私は犬です。", "私は猫です。"]
    scores = SubstringMatchScorer.score(refs, preds)
    assert scores == [0, 0]
    refs = ["池のほとりです。"]
    preds = ["ここは湖の岸です。"]
    scores = SubstringMatchScorer.score(refs, preds)
    assert scores == [0]

    scores = SubstringMatchScorer.aggregate([1, 1, 1, 0])
    assert scores == 0.75

    scores = SubstringMatchScorer.aggregate([1, 1, 0, 0])
    assert scores == 0.5

    scores = SubstringMatchScorer.aggregate([1, 0, 0, 0])
    assert scores == 0.25

    scores = SubstringMatchScorer.aggregate([0, 0, 0, 0])
    assert scores == 0.0

    scores = SubstringMatchScorer.aggregate([1, 1, 1, 1])
    assert scores == 1.0
