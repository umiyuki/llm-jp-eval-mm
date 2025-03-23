import litellm
from eval_mm.utils.litellm_client import LLMChatAPI
from collections import defaultdict
import numpy as np
from eval_mm.metrics.scorer import Scorer, AggregateOutput
from loguru import logger

RULES: dict = {
    "coding": {
        "role": "Assistant",
        "prompt": "Your task is to evaluate the coding abilities of the above two assistants. They have been asked to implement a program to solve a given problem. Please review their code submissions, paying close attention to their problem-solving approach, code structure, readability, and the inclusion of helpful comments.\n\nPlease ensure that the assistants' submissions:\n\n1. Correctly implement the given problem statement.\n2. Contain accurate and efficient code.\n3. Include clear and concise comments that explain the code's logic and functionality.\n4. Adhere to proper coding standards and best practices.\n\nOnce you have carefully reviewed both submissions, provide detailed feedback on their strengths and weaknesses, along with any suggestions for improvement. You should first output a single line containing two scores on the scale of 1-10 (1: no code/no sense; 10: perfect) for Assistant 1 and 2, respectively. Then give extra comments starting from the next line.",
    },
    "math": {
        "role": "Assistant",
        "prompt": "We would like to request your feedback on the mathematical proficiency of two AI assistants regarding the given user question.\nFirstly, please solve the problem independently, without referring to the answers provided by Assistant 1 and Assistant 2.\nAfterward, please examine the problem-solving process of Assistant 1 and Assistant 2 step-by-step to ensure their correctness, identifying any incorrect steps if present. Your evaluation should take into account not only the answer but also the problem-solving steps.\nFinally, please output a Python tuple containing two numerical scores for Assistant 1 and Assistant 2, ranging from 1 to 10, respectively. If applicable, explain the reasons for any variations in their scores and determine which assistant performed better.",
    },
    "default": {
        "role": "Assistant",
        "prompt": "We would like to request your feedback on the performance of two AI assistants in response to the user question displayed above.\nPlease rate the helpfulness, relevance, accuracy, level of details of their responses. Each assistant receives an overall score on a scale of 1 to 10, where a higher score indicates better overall performance.\nPlease first output a single line containing only two values indicating the scores for Assistant 1 and 2, respectively. The two scores are separated by a space.\nIn the subsequent line, please provide a comprehensive explanation of your evaluation, avoiding any potential bias and ensuring that the order in which the responses were presented does not affect your judgment. Sentences are given in Japanese, and language is not relevant to score.",
    },
    "conv": {
        "role": "Assistant",
        "prompt": "We would like to request your feedback on the performance of two AI assistants in response to the user question displayed above. The user asks the question on observing an image. For your reference, the visual content in the image is represented with five descriptive sentences describing the same image and the bounding box coordinates of each object in the scene. These coordinates are in the form of bounding boxes, represented as (x1, y1, x2, y2) with floating numbers ranging from 0 to 1. These values correspond to the top left x, top left y, bottom right x, and bottom right y. \nPlease rate the helpfulness, relevance, accuracy, level of details of their responses. Each assistant receives an overall score on a scale of 1 to 10, where a higher score indicates better overall performance.\nPlease first output a single line containing only two values indicating the scores for Assistant 1 and 2, respectively. The two scores are separated by a space.\nIn the subsequent line, please provide a comprehensive explanation of your evaluation, avoiding any potential bias and ensuring that the order in which the responses were presented does not affect your judgment. Sentences are given in Japanese, and language is not relevant to score.",
    },
    "detail": {
        "role": "Assistant",
        "prompt": "We would like to request your feedback on the performance of two AI assistants in response to the user question displayed above. The user asks the question on observing an image. For your reference, the visual content in the image is represented with five descriptive sentences describing the same image and the bounding box coordinates of each object in the scene. These coordinates are in the form of bounding boxes, represented as (x1, y1, x2, y2) with floating numbers ranging from 0 to 1. These values correspond to the top left x, top left y, bottom right x, and bottom right y. \nPlease rate the helpfulness, relevance, accuracy, level of details of their responses. Each assistant receives an overall score on a scale of 1 to 10, where a higher score indicates better overall performance.\nPlease first output a single line containing only two values indicating the scores for Assistant 1 and 2, respectively. The two scores are separated by a space.\nIn the subsequent line, please provide a comprehensive explanation of your evaluation, avoiding any potential bias and ensuring that the order in which the responses were presented does not affect your judgment. Sentences are given in Japanese, and language is not relevant to score.",
    },
    "complex": {
        "role": "Assistant",
        "prompt": "We would like to request your feedback on the performance of two AI assistants in response to the user question displayed above. The user asks the question on observing an image. For your reference, the visual content in the image is represented with five descriptive sentences describing the same image and the bounding box coordinates of each object in the scene. These coordinates are in the form of bounding boxes, represented as (x1, y1, x2, y2) with floating numbers ranging from 0 to 1. These values correspond to the top left x, top left y, bottom right x, and bottom right y. \nPlease rate the helpfulness, relevance, accuracy, level of details of their responses. Each assistant receives an overall score on a scale of 1 to 10, where a higher score indicates better overall performance.\nPlease first output a single line containing only two values indicating the scores for Assistant 1 and 2, respectively. The two scores are separated by a space.\nIn the subsequent line, please provide a comprehensive explanation of your evaluation, avoiding any potential bias and ensuring that the order in which the responses were presented does not affect your judgment. Sentences are given in Japanese, and language is not relevant to score.",
    },
    "llava_bench_conv": {
        "role": "Assistant",
        "prompt": "We would like to request your feedback on the performance of two AI assistants in response to the user question displayed above. The user asks the question on observing an image. For your reference, the visual content in the image is represented with a few sentences describing the image. \nPlease rate the helpfulness, relevance, accuracy, level of details of their responses. Each assistant receives an overall score on a scale of 1 to 10, where a higher score indicates better overall performance.\nPlease first output a single line containing only two values indicating the scores for Assistant 1 and 2, respectively. The two scores are separated by a space.\nIn the subsequent line, please provide a comprehensive explanation of your evaluation, avoiding any potential bias and ensuring that the order in which the responses were presented does not affect your judgment.",
    },
    "llava_bench_detail": {
        "role": "Assistant",
        "prompt": "We would like to request your feedback on the performance of two AI assistants in response to the user question displayed above. The user asks the question on observing an image. For your reference, the visual content in the image is represented with a few sentences describing the image. \nPlease rate the helpfulness, relevance, accuracy, level of details of their responses. Each assistant receives an overall score on a scale of 1 to 10, where a higher score indicates better overall performance.\nPlease first output a single line containing only two values indicating the scores for Assistant 1 and 2, respectively. The two scores are separated by a space.\nIn the subsequent line, please provide a comprehensive explanation of your evaluation, avoiding any potential bias and ensuring that the order in which the responses were presented does not affect your judgment.",
    },
    "llava_bench_complex": {
        "role": "Assistant",
        "prompt": "We would like to request your feedback on the performance of two AI assistants in response to the user question displayed above. The user asks the question on observing an image. For your reference, the visual content in the image is represented with a few sentences describing the image. \nPlease rate the helpfulness, relevance, accuracy, level of details of their responses. Each assistant receives an overall score on a scale of 1 to 10, where a higher score indicates better overall performance.\nPlease first output a single line containing only two values indicating the scores for Assistant 1 and 2, respectively. The two scores are separated by a space.\nIn the subsequent line, please provide a comprehensive explanation of your evaluation, avoiding any potential bias and ensuring that the order in which the responses were presented does not affect your judgment.",
    },
}


def parse_score(review: str) -> dict[str, int]:
    try:
        score_pair = review.split("\n")[0]
        score_pair = score_pair.replace(",", " ")
        sp = score_pair.split(" ")
        if len(sp) == 2:
            return {"score": int(sp[1]), "score_gpt": int(sp[0])}
        else:
            logger.error(f"error: {review}")
            return {"score": -1, "score_gpt": -1}
    except Exception as e:
        logger.error(f"error: {e}")
        return {"score": -1, "score_gpt": -1}


def ask_gpt4_batch(
    content_list: str, max_tokens: int, async_client: LLMChatAPI, model_name: str
) -> list:
    message_list = [
        [
            {
                "role": "system",
                "content": "You are a helpful and precise assistant for checking the quality of the answer.",
            },
            {"role": "user", "content": content},
        ]
        for content in content_list
    ]
    completions = async_client.batch_generate_chat_response(
        message_list,
        max_tokens=max_tokens,
        temperature=0,
        seed=0,
        model_name=model_name,
    )
    return completions


def build_content(context, input_text, ref_answer, pred_answer, role, prompt):
    return (
        f"[Context]\n{context}\n\n"
        f"[Question]\n{input_text}\n\n"
        f"[{role} 1]\n{ref_answer}\n\n[End of {role} 1]\n\n"
        f"[{role} 2]\n{pred_answer}\n\n[End of {role} 2]\n\n"
        f"[System]\n{prompt}\n\n"
        f"If it is not relevant to the context, does not answer directly, or says the wrong thing, give it a low score.\n\n"
    )


class HeronBenchScorer(Scorer):
    @staticmethod
    def score(refs, preds: list[str], **kwargs) -> list[dict[str, int]]:
        #litellm._turn_on_debug()
        litellm.drop_params = True
        docs = kwargs["docs"]
        client = kwargs["client"]
        judge_model = kwargs["judge_model"]
        debug_limit = kwargs["debug_limit"]

        # デバッグ用にデータ数を制限
        if debug_limit is not None:
            print(f"Applying debug_limit: {debug_limit}")
            # docsがdatasets.Datasetの場合に対応
            if hasattr(docs, 'select'):
                docs = docs.select(range(min(debug_limit, len(docs))))
            else:
                docs = docs[:debug_limit]
            refs = refs[:debug_limit]
            preds = preds[:debug_limit]
        
        print(f"Type of docs: {type(docs)}")
        print(f"Number of docs after limit: {len(docs)}")
        print(f"Number of refs after limit: {len(refs)}")
        print(f"Number of preds after limit: {len(preds)}")
        
        # docsがdatasets.Datasetの場合、辞書形式に変換
        if hasattr(docs, '__getitem__') and not isinstance(docs, (list, dict)):
            docs = [dict(doc) for doc in docs]

        contents = []
        for doc, ref, pred in zip(docs, refs, preds):
            content = build_content(
                doc["context"],
                doc["input_text"],
                ref,
                pred,
                "Assistant",
                RULES[doc["category"]]["prompt"],
            )
            contents.append(content)
        completions = ask_gpt4_batch(contents, 1024, client, judge_model)
        scores = [parse_score(completion) for completion in completions]
        assert len(scores) == len(docs)
        return scores

    @staticmethod
    def aggregate(scores: list[dict[str, int]], **kwargs) -> AggregateOutput:
        docs = kwargs["docs"]
        category_list = ["conv", "detail", "complex"]
        heron_metrics = defaultdict(float)
        
        # エラー（-1）を除外した有効なスコアのみを抽出
        valid_scores = [score for score in scores if score["score"] != -1 and score["score_gpt"] != -1]
        valid_docs = [doc for score, doc in zip(scores, docs) if score["score"] != -1 and score["score_gpt"] != -1]
        
        # カテゴリごとの計算
        for category in category_list:
            score_owns = [
                score["score"]
                for score, doc in zip(scores, docs)
                if doc["category"] == category and score["score"] != -1
            ]
            score_gpts = [
                score["score_gpt"]
                for score, doc in zip(scores, docs)
                if doc["category"] == category and score["score_gpt"] != -1
            ]
            if len(score_owns) == 0:
                continue
            avg_score = np.mean(score_owns)
            avs_score_rel = (
                100
                * np.mean(score_owns)
                / max(0.01, np.mean(score_gpts))  # 0除算対策
            )
            heron_metrics[category] = avg_score
            heron_metrics[category + "_rel"] = avs_score_rel
        
        # エラーのカウント
        heron_metrics["parse_error_count"] = sum(
            score["score"] == -1 or score["score_gpt"] == -1 for score in scores
        )
        
        # 有効なスコアのみで全体スコアを計算
        if len(valid_scores) > 0:
            heron_metrics["overall"] = sum([score["score"] for score in valid_scores]) / len(valid_scores)
            heron_metrics["overall_rel"] = sum(
                [heron_metrics[category + "_rel"] for category in category_list if category + "_rel" in heron_metrics]
            ) / sum(1 for category in category_list if category + "_rel" in heron_metrics)
        else:
            heron_metrics["overall"] = 0.0
            heron_metrics["overall_rel"] = 0.0

        output = AggregateOutput(
            overall_score=heron_metrics["overall_rel"],
            details=heron_metrics,
        )
        return output


def test_heron_bench_scorer():
    from eval_mm.utils.azure_client import MockChatAPI

    refs = ["私は猫です。"]
    preds = ["私は犬です。"]
    docs = [{"context": "hoge", "input_text": "fuga", "category": "conv"}]
    scores = HeronBenchScorer.score(
        refs, preds, docs=docs, client=MockChatAPI(), judge_model="gpt-4o-2024-05-13"
    )
    assert scores == [{"score": -1, "score_gpt": -1}]
    output = HeronBenchScorer.aggregate(scores, docs=docs)
    assert output.overall_score == 0.0
    assert output.details == {
        "parse_error_count": 1,
        "overall": -1.0,
        "conv_rel": 0.0,
        "detail_rel": 0.0,
        "complex_rel": 0.0,
        "overall_rel": 0.0,
    }


# __main__部分の呼び出し例
if __name__ == "__main__":
    from datasets import load_dataset

    ds = load_dataset("Silviase/Japanese-Heron-Bench", split="train")
    ds = ds.rename_column("text", "input_text")
    # データセット全体を渡すが、score内で制限されるはず
    pred_texts = [
        "画像から判断すると、制限速度は50 km/hのようです。道路標識に「50」と表示されています。",
        "画像に写っている標識によると、現在地からニセコまでは12kmです。",
        "これはテスト回答です。",
        "これはテスト回答です。",
        "これはテスト回答です。",
    ]
    refs = [doc["answer"]["gpt-4-0125-preview"] for doc in ds]
    client = LLMChatAPI()
    judge_model = "gpt-4o-2024-05-13"
    # debug_limit=3を指定
    scores = HeronBenchScorer.score(
        refs, pred_texts, docs=ds, client=client, judge_model=judge_model, debug_limit=3
    )
    print("Scores:", scores)

    # aggregateでも制限されたdocsを渡す
    limited_docs = ds.select(range(3))  # Dataset形式のまま渡す
    output = HeronBenchScorer.aggregate(scores, docs=limited_docs)
    print("Output:", output)
