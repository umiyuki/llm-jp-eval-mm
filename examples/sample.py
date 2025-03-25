import json
import os

import litellm
import eval_mm
from tqdm import tqdm
import argparse
import eval_mm.metrics
from utils import GenerationConfig
from model_table import get_class_from_model_id
from dataclasses import asdict
from loguru import logger
from examples.LiteLLM import LiteLLMVLM  # LiteLLMクラスのインポート


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="llava-hf/llava-1.5-7b-hf", help="Model ID to evaluate")
    parser.add_argument("--task_id", type=str, default="japanese-heron-bench", help="Task ID to evaluate")
    parser.add_argument("--judge_model", type=str, default="gpt-4o-mini-2024-07-18", help="Judge model for LLM-as-a-judge")
    parser.add_argument("--batch_size_for_evaluation", type=int, default=10, help="Batch size for evaluation")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing results")
    parser.add_argument("--result_dir", type=str, default="result", help="Directory to save results")
    parser.add_argument("--inference_only", action="store_true", help="Run inference only without evaluation")
    parser.add_argument("--max_new_tokens", type=int, default=256, help="Maximum new tokens to generate")
    parser.add_argument("--num_beams", type=int, default=1, help="Number of beams for beam search")
    parser.add_argument("--temperature", type=float, default=0.0, help="Temperature for sampling")
    parser.add_argument("--top_p", type=float, default=1.0, help="Top-p value for sampling")
    parser.add_argument("--do_sample", action="store_true", default=False, help="Enable sampling")
    parser.add_argument("--use_cache", action="store_true", default=True, help="Use cache for generation")
    parser.add_argument(
        "--max_dataset_len",
        type=int,
        default=None,
        help="Max data size for evaluation. If None, use all data. Else, use the first n data.",
    )
    parser.add_argument(
        "--metrics",
        type=str,
        default="llm_as_a_judge_heron_bench",
        help=f"Metrics to evaluate. You can specify multiple metrics separated by comma (e.g. --metrics exact_match,llm_as_a_judge). Available metrics: {list(eval_mm.metrics.ScorerRegistry._scorers.keys())}",
    )
    parser.add_argument(
        "--rotate_choices",
        action="store_true",
        help="If set, rotate the choices of MCQ options for evaluation.",
    )
    # LiteLLM用の引数を追加
    parser.add_argument("--provider", type=str, default="openai", help="Provider for LiteLLM (e.g., openai, azure, gemini, custom)")
    parser.add_argument("--api_base_eval", type=str, default=None, help="Custom API base URL for evaluation model")
    parser.add_argument("--api_base_judge", type=str, default=None, help="Custom API base URL for judge model")
    parser.add_argument("--api_key", type=str, default=None, help="Custom API key for LiteLLM")
    parser.add_argument("--debug_limit", type=int, default=None, help="Limit the number of samples for debugging")
    return parser.parse_args()


def validate_metrics(metrics: list[str]):
    for metric in metrics:
        if metric not in list(eval_mm.metrics.ScorerRegistry._scorers.keys()):
            raise ValueError(
                f"Invalid metric: {metric}. You can use the following metrics: {list(eval_mm.metrics.ScorerRegistry._scorers.keys())}"
            )


if __name__ == "__main__":
    #litellm._turn_on_debug()
    litellm.drop_params = True
    args = parse_args()
    metrics = args.metrics.split(",")
    validate_metrics(metrics)

    gen_kwargs = GenerationConfig(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        num_beams=args.num_beams,
        do_sample=args.do_sample,
        use_cache=args.use_cache,
    )

    task_config = eval_mm.api.task.TaskConfig(
        max_dataset_len=args.max_dataset_len,
        judge_model=args.judge_model,
        batch_size_for_evaluation=args.batch_size_for_evaluation,
        api_base_judge=args.api_base_judge,
        rotate_choices=args.rotate_choices,
    )
    task = eval_mm.api.registry.get_task_cls(args.task_id)(task_config)

    # debug_limitが指定されている場合、データセットを制限
    dataset = task.dataset
    if args.debug_limit is not None:
        logger.info(f"Applying debug_limit: {args.debug_limit}")
        dataset = dataset.select(range(min(args.debug_limit, len(dataset))))

    output_dir = os.path.join(args.result_dir, args.task_id, args.model_id)
    os.makedirs(output_dir, exist_ok=True)

    # モデルのロード
    model_class = get_class_from_model_id(args.model_id)
    print("model_class", model_class)
    if model_class is LiteLLMVLM:
        # LiteLLMの場合、追加の引数を渡してインスタンス化
        model = model_class(
            model_id=args.model_id,
            api_base=args.api_base_eval,
            api_key=args.api_key,
            provider=args.provider
        )
    else:
        # 既存のモデルはそのままインスタンス化
        model = model_class(args.model_id)

    # if prediction is already done, load the prediction
    prediction_path = os.path.join(output_dir, "prediction.jsonl")
    if os.path.exists(prediction_path) and not args.overwrite:
        with open(prediction_path, "r") as f:
            preds = [json.loads(line) for line in f]
        assert (
            len(preds) == len(dataset)
        ), f"Prediction result length is not equal to the dataset length. Prediction result length: {len(preds)}, Dataset length: {len(dataset)}"
        logger.info(f"Prediction result loaded from {prediction_path}")
    else:
        preds = []
        logger.info(dataset)
        for doc in tqdm(dataset):
            images = task.doc_to_visual(doc)
            text = task.doc_to_text(doc)
            if "<image>" in text:
                text = text.replace("<image>", "")
            qid = task.doc_to_id(doc)
            generated_text = model.generate(images, text, gen_kwargs)
            pred = {
                "question_id": qid,
                "text": generated_text,
            }
            preds.append(pred)
        with open(prediction_path, "w") as f:
            for pred in preds:
                f.write(json.dumps(pred, ensure_ascii=False) + "\n")

    if args.inference_only:
        logger.info("Inference only mode. Skip evaluation.")
        exit()
    logger.info("Evaluation start")
    # evaluate the predictions

    scores_for_each_metric = {}
    calculated_metrics = {}

    for metric in metrics:
        scorer = eval_mm.metrics.ScorerRegistry.get_scorer(metric)
        scores = scorer.score(
            [task.doc_to_answer(doc) for doc in dataset],
            [pred["text"] for pred in preds],
            docs=dataset,
            client=task.client,
            judge_model=task.config.judge_model,
            batch_size=task.config.batch_size_for_evaluation,
            debug_limit=args.debug_limit,
        )
        logger.info(scores)
        scores_for_each_metric[metric] = scores
        logger.info(f"Scores for {metric}: {scores_for_each_metric[metric]}")
        aggregate_output = scorer.aggregate(scores, docs=dataset)
        calculated_metrics[metric] = asdict(aggregate_output)
        logger.info(f"Aggregate score for {metric}: {aggregate_output}")

    with open(prediction_path, "w") as f:
        for i, pred in enumerate(preds):
            question_id = pred["question_id"]
            text = pred["text"]
            answer = task.doc_to_answer(dataset[i])
            input_text = task.doc_to_text(dataset[i])
            content = {
                "question_id": question_id,
                "text": text,
                "answer": answer,
                "input_text": input_text,
            }
            for metric in metrics:
                content[metric] = scores_for_each_metric[metric][i]
            f.write(json.dumps(content, ensure_ascii=False) + "\n")
    logger.info(f"Prediction result saved to {prediction_path}")

    evaluation_path = os.path.join(output_dir, "evaluation.jsonl")
    with open(evaluation_path, "w") as f:
        f.write(json.dumps(calculated_metrics, ensure_ascii=False) + "\n")
    logger.info(f"Evaluation result saved to {evaluation_path}")