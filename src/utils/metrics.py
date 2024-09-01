# TODO: Implement metrics for model evaluation. Refer below URL for more details:
# https://github.com/SakanaAI/evolutionary-model-merge/blob/main/evomerge/eval/metrics.py

import re

from rouge_score import rouge_scorer, scoring


class MecabTokenizer:
    def __init__(self) -> None:
        from fugashi import Tagger

        self.tagger = Tagger("-Owakati")

    def normalize_answer(self, text):
        """Lower case text, remove punctuation and extra whitespace, etc."""
        import emoji
        # import neologdn TODO: fix c++12 error when installing neologdn

        def white_space_fix(text):
            return " ".join(text.split())

        def remove_emoji(text):
            text = "".join(["" if emoji.is_emoji(c) else c for c in text])
            emoji_pattern = re.compile(
                "["
                "\U0001f600-\U0001f64f"  # emoticons
                "\U0001f300-\U0001f5ff"  # symbols & pictographs
                "\U0001f680-\U0001f6ff"  # transport & map symbols
                "\U0001f1e0-\U0001f1ff"  # flags (iOS)
                "\U00002702-\U000027b0"
                "]+",
                flags=re.UNICODE,
            )
            return emoji_pattern.sub(r"", text)

        text = remove_emoji(text)
        # see neologdn docs for details, but handles things like full/half width variation
        # text = neologdn.normalize(text) TODO: fix c++12 error when installing neologdn
        text = white_space_fix(text)
        return text

    def tokenize(self, text):
        return self.tagger.parse(self.normalize_answer(text)).split()


def rouge_ja(refs, preds):
    """This uses a MeCab tokenizer for Japanese text."""
    tokenizer = MecabTokenizer()
    rouge_types = ["rouge1", "rouge2", "rougeL"]
    # mecab-based rouge
    scorer = rouge_scorer.RougeScorer(
        rouge_types,
        tokenizer=tokenizer,
    )

    # Accumulate confidence intervals.
    aggregator = scoring.BootstrapAggregator()
    for ref, pred in zip(refs, preds):
        aggregator.add_scores(scorer.score(ref, pred))
    result = aggregator.aggregate()
    return {type: result[type].mid.fmeasure * 100 for type in rouge_types}


def test_rouge_ja():
    import pytest

    refs = ["私は猫です。"]
    preds = ["私は猫です。"]
    scores = rouge_ja(refs, preds)
    assert scores["rougeL"] == 100.0
    refs = ["たかしが公園で遊んでいた。"]
    preds = ["たかしが公園にいたようだ。"]
    scores = rouge_ja(refs, preds)
    assert pytest.approx(scores["rougeL"], 0.01) == 66.66

    refs = ["私は猫です。", "私は犬です。"]
    preds = ["私は犬です。", "私は猫です。"]
    scores = rouge_ja(refs, preds)
    assert pytest.approx(scores["rougeL"], 0.01) == 80.0

    refs = ["日本語T5モデルの公開"] * 3
    preds = ["T5モデルの日本語版を公開", "日本語T5をリリース", "Japanese T5を発表"]
    scores = rouge_ja(refs, preds)
    tokenizer = MecabTokenizer()
    refs = [tokenizer.tokenize(ref) for ref in refs]
    preds = [tokenizer.tokenize(pred) for pred in preds]

    def _lcs_table(ref, can):
        """Create 2-d LCS score table."""
        rows = len(ref)
        cols = len(can)
        lcs_table = [[0] * (cols + 1) for _ in range(rows + 1)]
        for i in range(1, rows + 1):
            for j in range(1, cols + 1):
                if ref[i - 1] == can[j - 1]:
                    lcs_table[i][j] = lcs_table[i - 1][j - 1] + 1
                else:
                    lcs_table[i][j] = max(lcs_table[i - 1][j], lcs_table[i][j - 1])
        return lcs_table

    def rouge_score(ref, preds):
        f_score = 0
        for pred in preds:
            lcs = _lcs_table(ref, pred)
            print(lcs[-1])
            precision = lcs[-1][-1] / len(pred)
            recall = lcs[-1][-1] / len(ref)
            f_score += 2 * precision * recall / (precision + recall)
            print(2 * precision * recall / (precision + recall))
        return (f_score * 100) / len(preds)

    rougeL_score = rouge_score(refs[0], preds)
    assert pytest.approx(scores["rougeL"], 0.01) == rougeL_score
