import json
import os
import csv

benchmark_metrics = {
    "japanese-heron-bench": [
        "conv",
        "detail",
        "complex",
        "overall",
        "conv_rel",
        "detail_rel",
        "complex_rel",
        "overall_rel",
    ],
    "ja-vlm-bench-in-the-wild": [
        "rougeL",
    ],
    "jdocqa": ["yesno_exact", "factoid_exact", "numerical_exact", "open-ended_bleu"],
    "ja-multi-image-vqa": [
        "score",
    ],
}


def extract_results(result_dir):
    """
    Extracts evaluation results, filtering by specified metrics for each benchmark.
    """
    csv_data = []
    for benchmark_name in os.listdir(result_dir):
        benchmark_path = os.path.join(result_dir, benchmark_name)
        if os.path.isdir(benchmark_path):
            evaluation_path = os.path.join(benchmark_path, "evaluation")
            if os.path.exists(evaluation_path):
                allowed_metrics = benchmark_metrics.get(
                    benchmark_name, []
                )  # Get allowed metrics, or empty list if not found
                for filename in os.listdir(evaluation_path):
                    if filename.endswith(".jsonl"):
                        model_name = filename[:-6]
                        filepath = os.path.join(evaluation_path, filename)
                        try:
                            with open(filepath, "r", encoding="utf-8") as f:
                                data = json.load(f)
                                for metric_name, score in data.items():
                                    if (
                                        not allowed_metrics
                                        or metric_name in allowed_metrics
                                    ):  # filter metrics
                                        csv_data.append(
                                            [
                                                model_name,
                                                benchmark_name,
                                                metric_name,
                                                score,
                                            ]
                                        )
                        except json.JSONDecodeError as e:
                            print(f"Error decoding JSON in {filepath}: {e}")
                        except Exception as e:
                            print(
                                f"An unexpected error occurred while processing {filepath}: {e}"
                            )

    return csv_data


def write_to_csv(csv_data, output_file):
    """Writes the extracted data to a CSV file."""
    with open(output_file, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Model Name", "Benchmark Name", "Metric Name", "Score"])
        writer.writerows(csv_data)


if __name__ == "__main__":
    result_dir = "result"
    output_csv = "result/benchmark_results.csv"
    csv_data = extract_results(result_dir)
    write_to_csv(csv_data, output_csv)
    print(f"Results written to {output_csv}")
