from eval_mm.metrics.scorer import Scorer
from tqdm import tqdm
import re


QA_POINTWISE = """# Instruction
You are an expert evaluator. Your task is to evaluate the quality of the responses generated by AI models.
We will provide you with the user prompt and an AI-generated responses.
You should first read the user prompt carefully for analyzing the task, and then evaluate the quality of the responses based on and rules provided in the Evaluation section below.

# Evaluation
## Metric Definition
You will be assessing question answering quality, which measures the overall quality of the answer to the question in the user prompt. Pay special attention to length constraints, such as in X words or in Y sentences. The instruction for performing a question-answering task is provided in the user prompt. The response should not contain information that is not present in the context (if it is provided).

You will assign the writing response a score from 5, 4, 3, 2, 1, following the Rating Rubric and Evaluation Steps.
Give step-by-step explanations for your scoring, and only choose scores from 5, 4, 3, 2, 1.

## Criteria Definition
Instruction following: The response demonstrates a clear understanding of the question answering task instructions, satisfying all of the instruction's requirements.
Groundedness: The response contains information included only in the context if the context is present in the user prompt. The response does not reference any outside information.
Completeness: The response completely answers the question with sufficient detail.
Fluent: The response is well-organized and easy to read.

## Rating Rubric
5: (Very good). The answer follows instructions, is grounded, complete, and fluent.
4: (Good). The answer follows instructions, is grounded, complete, but is not very fluent.
3: (Ok). The answer mostly follows instructions, is grounded, answers the question partially and is not very fluent.
2: (Bad). The answer does not follow the instructions very well, is incomplete or not fully grounded.
1: (Very bad). The answer does not follow the instructions, is wrong and not grounded.

## Evaluation Steps
STEP 1: Assess the response in aspects of instruction following, groundedness,completeness, and fluency according to the criteria.
STEP 2: Provide overall score based on the rubric in the format of `Score: X` where X is the score you assign to the response.

# Question, Reference Answer and AI-generated Response
## Question
{input_text}

## Reference Answer
{answer}

## AI-generated Response
{pred}"""


class LlmAsaJudgeScorer(Scorer):
    @staticmethod
    def score(
        client,
        questions: list,
        answers: list,
        preds: list,
        batch_size: int,
        model_name: str,
    ):
        template = QA_POINTWISE

        def build_message(template, question: str, answer: str, pred: str):
            content = template.format(input_text=question, pred=pred, answer=answer)
            message = [{"role": "user", "content": content}]
            return message

        messages = [
            build_message(template, question, answer, pred)
            for question, answer, pred in zip(questions, answers, preds)
        ]
        messages_list = [
            messages[i : i + batch_size] for i in range(0, len(messages), batch_size)
        ]
        completion = []
        for ms in tqdm(messages_list, desc="Evaluating LLM as a Judge"):
            completion.extend(
                client.batch_generate_chat_response(
                    ms, max_tokens=1024, temperature=0.0, seed=0, model_name=model_name
                )
            )

        def parse_score(completion: str):
            try:
                score = re.search(r"Score: (\d)", completion)
                score = int(score.group(1)) if score else 1
                if score not in [1, 2, 3, 4, 5]:
                    print(f"Invalid score: {score}")
                    return {"score": 1}
                return {"score": score}
            except Exception:
                print("parse_score error")
                return {"score": 1}

        scores = [parse_score(completion) for completion in completion]
        return scores

    def aggregate(scores: list) -> float:
        return sum([score["score"] for score in scores]) / len(scores)


def test_llm_as_a_judge_scorer():
    from eval_mm.utils.azure_client import OpenAIChatAPI

    client = OpenAIChatAPI()
    questions = ["What is the capital of Japan?", "What is the capital of France?"]
    answers = ["Tokyo", "Paris"]
    preds = ["Tokyo", "Paris"]
    batch_size = 1
    model_name = "gpt-4o-mini-2024-07-18"
    scores = LlmAsaJudgeScorer.score(
        client, questions, answers, preds, batch_size, model_name
    )
    assert scores == [{"score": 5}, {"score": 5}]
    scores = LlmAsaJudgeScorer.aggregate(scores)
    assert scores == 5.0


if __name__ == "__main__":
    from eval_mm.utils.azure_client import OpenAIChatAPI

    client = OpenAIChatAPI()
    questions = ["What is the capital of Japan?", "What is the capital of France?"]
    answers = ["Tokyo", "Paris"]
    preds = ["Tokyo", "Paris"]
    batch_size = 1
    model_name = "gpt-4o-mini-2024-07-18"
    scores = LlmAsaJudgeScorer.score(
        client, questions, answers, preds, batch_size, model_name
    )
    print(scores)
    print(LlmAsaJudgeScorer.aggregate(scores))
