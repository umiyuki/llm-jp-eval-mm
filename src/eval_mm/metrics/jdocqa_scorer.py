from eval_mm.metrics.scorer import Scorer, AggregateOutput
from sacrebleu import sentence_bleu
from unicodedata import normalize

ANSWER_TYPE_MAP = {
    "yesno": 0,  # Yes/No questions
    "factoid": 1,  # Factoid questions
    "numerical": 2,  # Numerical questions
    "open-ended": 3,  # Open-ended questions
}

NUM_TO_ANSWER_TYPE = {v: k for k, v in ANSWER_TYPE_MAP.items()}


def jdocqa_normalize(text):
    text = (
        text.replace("です", "")
        .replace("。", "")
        .replace("、", "")
        .replace(" ", "")
        .strip()
    )
    text = normalize("NFKC", text)
    return text


def bleu_ja(refs, pred):
    """Calculate BLEU score for Japanese text. Score is normalized to [0, 1]."""
    bleu_score = sentence_bleu(
        hypothesis=pred,
        references=refs,
        smooth_method="exp",
        smooth_value=0.0,
        tokenize="ja-mecab",
        use_effective_order=False,
        lowercase=False,
    )
    return bleu_score.score / 100


class JDocQAScorer(Scorer):
    @staticmethod
    def score(refs: list[str], preds: list[str], **kwargs) -> list[int]:
        docs = kwargs["docs"]
        scores = []
        for doc, ref, pred in zip(docs, refs, preds):
            try:
                if doc["answer_type"] == ANSWER_TYPE_MAP["open-ended"]:
                    score = bleu_ja([ref], pred)
                elif doc["answer_type"] in [
                    ANSWER_TYPE_MAP["yesno"],
                    ANSWER_TYPE_MAP["factoid"],
                    ANSWER_TYPE_MAP["numerical"],
                ]:
                    ref = jdocqa_normalize(ref)
                    pred = jdocqa_normalize(pred)
                    score = 1 if ref in pred else 0
                else:
                    raise NotImplementedError("Bad answer type.")
                scores.append(score)
            except Exception as e:
                print(f"Error in scoring question_id {doc['question_id']}: {e}")
                scores.append(None)  # エラー時はNoneを追加
        return scores

    @staticmethod
    def aggregate(scores: list[int], **kwargs) -> AggregateOutput:
        docs = kwargs["docs"]
        metrics = {
            "yesno_exact": [],
            "factoid_exact": [],
            "numerical_exact": [],
            "open-ended_bleu": [],
        }
        # エラー（None）を除外
        valid_pairs = [(doc, score) for doc, score in zip(docs, scores) if score is not None]
        if not valid_pairs:
            return AggregateOutput(0, {k: 0 for k in metrics.keys()} | {"overall": 0})
        for doc, score in valid_pairs:
            answer_type = doc["answer_type"]
            if answer_type == ANSWER_TYPE_MAP["open-ended"]:
                metrics["open-ended_bleu"].append(score)
            else:
                metrics[f"{NUM_TO_ANSWER_TYPE[answer_type]}_exact"].append(score)
        for key, value in metrics.items():
            metrics[key] = sum(value) / len(value) if value else 0
        metrics["overall"] = sum(score for _, score in valid_pairs) / len(valid_pairs)
        return AggregateOutput(metrics["overall"], metrics)


def test_jdocqa_scorer():
    refs = ["私は猫です。"]
    preds = ["私は猫です。"]
    scores = JDocQAScorer.score(refs, preds, docs=[{"answer_type": 1}])
    assert scores == [1.0]
    output = JDocQAScorer.aggregate(scores, docs=[{"answer_type": 1}])
    assert output.overall_score == 1.0
    assert output.details == {
        "factoid_exact": 1.0,
        "yesno_exact": 0,
        "numerical_exact": 0,
        "open-ended_bleu": 0,
        "overall": 1.0,
    }


if __name__ == "__main__":
    from datasets import load_dataset

    ds = load_dataset("shunk031/JDocQA", split="test")
    ds = ds.select(range(10))

    ref = ds["answer"][0]
    pred = ds["answer"][0]
    print(ref)
    print(pred)
    print(bleu_ja([ref], pred))
    answer_types = ds["answer_type"]
    answers = ds["answer"]
    print("Original answers")
    for answer_type, answer in zip(answer_types, answers):
        print(NUM_TO_ANSWER_TYPE[answer_type], answer)

    print("JDocQA normalized answers")
    jdocqa_normalize_answers = [jdocqa_normalize(x) for x in ds["answer"]]
    for answer_type, answer in zip(answer_types, jdocqa_normalize_answers):
        print(NUM_TO_ANSWER_TYPE[answer_type], answer)

    scores = JDocQAScorer.score(refs=ds["answer"], preds=ds["answer"], docs=ds)
    print(scores)
    metrics = JDocQAScorer.aggregate(scores, docs=ds)
    print(metrics)
