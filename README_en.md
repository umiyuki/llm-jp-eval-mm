# LLM-jp-eval-mm
[![pypi](https://img.shields.io/pypi/v/eval-mm.svg)](https://pypi.python.org/pypi/eval-mm)

[ English | [**日本語**](./README.md) ]

This tool automatically evaluates Japanese multimodal large language models across multiple datasets. 
It provides the following features:

- Converts existing Japanese evaluation datasets into evaluation datasets for multimodal text generation tasks.
- Evaluates large language models and calculates metrics using inference results.

For details on the data format and a list of supported datasets, please refer to [DATASET.md](./DATASET.md).

## Table of Contents

- [LLM-jp-eval-mm](#llm-jp-eval-mm)
  - [Table of Contents](#table-of-contents)
  - [Environment Setup](#environment-setup)
  - [Evaluation Method](#evaluation-method)
    - [Running the Sample Code](#running-the-sample-code)
    - [Checking Evaluation Results](#checking-evaluation-results)
    - [Managing Evaluation Results with W&B](#managing-evaluation-results-with-wb)
  - [License](#license)
  - [Contribution](#contribution)

## Environment Setup

### Cloning from GitHub

eval-mm uses rye for virtual environment management.

1. Clone the repository and navigate to it:
```bash
git clone git@github.com:llm-jp/llm-jp-eval-mm.git
cd llm-jp-eval-mm
```

2. Set up the environment using rye:

Please refer to the [official doc](https://rye.astral.sh/guide/installation/) for installing rye.

```bash
cd llm-jp-eval-mm
rye sync
```

3. Create a `.env` file by referring to [.env.sample](./.env.sample) and set `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_KEY`.

This completes the environment setup.

### Installing with PyPI

1. If you want to use the eval-mm released on PyPI, you can include it in your virtual environment using the `pip` command:

```bash
pip install eval_mm
```

2. Create a `.env` file by referring to [.env.sample](./.env.sample) and set `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_KEY`.

This completes the environment setup.

## Evaluation Method

### Running the Evaluation

We provide a sample code `examples/sample.py` for running the evaluation.

Models included as `examples/{model_name}.py` are only supported for that specific inference method.

If you want to evaluate with a new inference method or a new model, you can do so by creating a similar file by referring to the existing `examples/{model_name}.py`.

`examples/evaluate.sh` is a script for running `examples/sample.py`. You can evaluate multiple benchmarks and models in batches by specifying them.

The command to run is as follows:

```bash
rye run bash examples/evaluate.sh
```

The current settings for `examples/evaluate.sh` are as follows:

- Evaluation model: `llava 1.5`
- Evaluation benchmark: `japanese-heron-bench`
- OpenAI model (used for LLM-as-a-judge): `gpt-4o-mini-2024-07-18`

You can get the same result by running `sample.py` directly.
In that case, please execute the following command:

```bash
rye run python3 examples/sample.py --class_path llava_1_5  --task_id japanese-heron-bench --openai_model_id gpt-4o-mini-2024-07-18
```

### Checking Evaluation Results

The evaluation score and output results are saved in 
`result/{task_id}/evaluation/{model_id}.jsonl`, `result/{task_id}/prediction/{model_id}.jsonl`.

To check the results for the japanese-heron-bench benchmark, run:

```python
rye run python3 scripts/japanese-heron-bench/record_output.py
```

This will generate:

- An Excel file listing the generation results of each model for each example.
- An Excel file listing the scores of each model.
- Graph images showing the score distribution of each model.


### Managing Evaluation Results with W&B

Currently, the evaluation results are only provided in `.jsonl` format, and integration with W&B is not planned.

## License

This tool is distributed under the [TODO: Required license].
Please refer to [DATASET.md](./DATASET.md) for the license of each evaluation dataset.

## Contribution

- Please report any issues or suggestions in the Issues section.
- Please send a Pull Request for any fixes or additions.

- Adding libraries
```
rye add <package_name>
```
- Formatting with ruff
```
rye run ruff format .
```

- How to release to PyPI
```
git tag -a v0.x.x -m "version 0.x.x"
git push origin --tags
```
