import json
import os
import csv
from argparse import ArgumentParser

BENCHMARK_METRICS = {
    "japanese-heron-bench": {
        "llm_as_a_judge_heron_bench": [
            "conv",
            "detail",
            "complex",
            "overall",
            "conv_rel",
            "detail_rel",
            "complex_rel",
            "overall_rel",
        ],
    },
    "ja-vlm-bench-in-the-wild": [
        "rougel",
        "llm_as_a_judge",
    ],
    "ja-vg-vqa-500": [
        "rougel",
        "llm_as_a_judge",
    ],
    "jdocqa": {
        "jdocqa": [
            "yesno_exact",
            "factoid_exact",
            "numerical_exact",
            "open-ended_bleu",
        ],
    },
    "ja-multi-image-vqa": {
        "rougel",
        "llm_as_a_judge",
    },
    "jmmmu": {
        "jmmmu": [
            "Overall-Art and Psychology",
            "Design",
            "Music",
            "Psychology",
            "Overall-Business",
            "Accounting",
            "Economics",
            "Finance",
            "Manage",
            "Marketing",
            "Overall-Science",
            "Biology",
            "Chemistry",
            "Math",
            "Physics",
            "Overall-Health and Medicine",
            "Basic_Medical_Science",
            "Clinical_Medicine",
            "Diagnostics_and_Laboratory_Medicine",
            "Pharmacy",
            "Public_Health",
            "Overall-Tech and Engineering",
            "Agriculture",
            "Architecture_and_Engineering",
            "Computer_Science",
            "Electronics",
            "Energy_and_Power",
            "Materials",
            "Mechanical_Engineering",
            "Overall",
        ],
    },
}

def get_benchmark_metrics(benchmark_name: str):
    """Retrieve metrics for the given benchmark name."""
    return BENCHMARK_METRICS.get(benchmark_name)

def process_metrics(model_name: str, benchmark_name: str, metric_name: str, metric_scores: float|list|dict):
    """Process metrics and return them in a standardized format.
    """
    results = []
    if isinstance(metric_scores, float):
        results.append([model_name, benchmark_name, metric_name, metric_scores])
    elif isinstance(metric_scores, list):
        results.extend(
            [[model_name, benchmark_name, metric_name, score] for score in metric_scores]
        )
    elif isinstance(metric_scores, dict):
        results.extend(
            [[model_name, benchmark_name, name, value] for name, value in metric_scores.items()]
        )
    else:
        raise ValueError(f"Unsupported metric type for {benchmark_name}: {metric_name}")
    return results

def extract_results(result_dir: str):
    """
    Extracts evaluation results, filtering by specified metrics for each benchmark.
    """
    csv_data = []
    for benchmark_name in filter(
        lambda name: os.path.isdir(os.path.join(result_dir, name)),
        os.listdir(result_dir),
    ):
        benchmark_dir = os.path.join(result_dir, benchmark_name)
        evaluation_dir = os.path.join(benchmark_dir, "evaluation")

        for metrics_file in filter(lambda f: f.endswith(".jsonl"), os.listdir(evaluation_dir)):
            model_name = metrics_file[:-6]
            metrics_path = os.path.join(evaluation_dir, metrics_file)

            with open(metrics_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            metrics = get_benchmark_metrics(benchmark_name)
            if not metrics:
                continue

            for metric_name in metrics:
                metric_scores = data.get(metric_name)
                if metric_scores is not None:
                    csv_data.extend(process_metrics(model_name, benchmark_name, metric_name, metric_scores))
    return csv_data

def write_to_csv(csv_data: list, output_file: str):
    """Writes the extracted data to a CSV file."""
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Model Name", "Benchmark Name", "Metric Name", "Score"])
        writer.writerows(csv_data)

def get_args():
    parser = ArgumentParser(description="Extract evaluation results and write to a CSV file.")
    parser.add_argument("--result_dir", default="result", help="Directory containing evaluation results.")
    parser.add_argument("--output_csv", default="result/benchmark_results.csv", help="Output CSV file.")
    return parser.parse_args()
if __name__ == "__main__":
    args = get_args()
    try:
        csv_data = extract_results(args.result_dir)
        write_to_csv(csv_data, args.output_csv)
        print(f"Results written to {output_csv}")
    except Exception as e:
        print(f"An error occurred: {e}")
