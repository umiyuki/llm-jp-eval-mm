# LLM-jp-eval-mm
[![pypi](https://img.shields.io/pypi/v/eval-mm.svg)](https://pypi.python.org/pypi/eval-mm)

[ [**English**](./README_en.md) | 日本語 ]

このツールは，複数のデータセットを横断して日本語マルチモーダル大規模言語モデルを自動評価するものです．
以下の機能を提供します．

- 既存の日本語評価データを利用し，マルチモーダルテキスト生成タスクの評価データセットに変換して提供
- 推論結果を用いた大規模言語モデルの評価・メトリクス計算

データフォーマットの詳細，サポートしているデータの一覧については，[DATASET.md](./DATASET.md)を参照ください．

## 目次

- [LLM-jp-eval-mm](#llm-jp-eval-mm)
  - [目次](#目次)
  - [環境構築](#環境構築)
  - [評価方法](#評価方法)
    - [サンプルコードの実行](#サンプルコードの実行)
    - [評価結果の確認](#評価結果の確認)
    - [評価結果をW\&Bで管理](#評価結果をwbで管理)
  - [サポートするタスク](#サポートするタスク)
  - [ライセンス](#ライセンス)
  - [Contribution](#contribution)

## 環境構築

### GitHubをCloneする場合

eval-mmは仮想環境の管理にryeを用いています．

1. リポジトリをクローンして移動する
```bash
git clone git@github.com:llm-jp/llm-jp-eval-mm.git
cd llm-jp-eval-mm
```

2. rye を用いて環境構築を行う

ryeは[official doc](https://rye.astral.sh/guide/installation/) を参考にインストールしてください．

```bash
cd llm-jp-eval-mm
rye sync
```

3. [.env.sample](./.env.sample)を参考にして, `.env`ファイルを作成し，`AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_KEY`を設定してください.

以上で環境構築は終了です.

### PyPIでインストールする

1. PyPIにリリースされているeval-mmを利用する場合，`pip`コマンドを用いて仮想環境に含めることができます．

```bash
pip install eval_mm
```

2. [.env.sample](./.env.sample)を参考にして, `.env`ファイルを作成し，`AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_KEY`を設定してください.

以上で環境構築は終了です.

## 評価方法

### 評価の実行

評価の実行のために，サンプルコード`examples/sample.py`を提供しています．

`examples/{モデル名}.py`として含まれているモデルは，その推論方法に限りサポートしています．

新たな推論方法・新たなモデルでの評価を実行したい場合，既存の`examples/{モデル名}.py`を参考に同様のファイルを作成することで，評価を実行することができます．

`examples/evaluate.sh`は，`examples/sample.py`を実行するためのスクリプトです．複数のベンチマーク・モデルを指定することで，一括で評価を行うことができます．

実行するためのコマンドは以下のとおりです．

```bash
rye run bash examples/evaluate.sh
```

`examples/evaluate.sh`の現在の設定は以下の通りです．

- 評価モデル：`llava 1.5`
- 評価ベンチマーク：`japanese-heron-bench`
- OpenAIモデル（LLM-as-a-judgeのために利用）：`gpt-4o-mini-2024-07-18`

`sample.py`を直接実行しても同じ結果が得られます．
その場合は以下のコマンドを実行してください．

```bash
rye run python3 examples/sample.py --class_path llava_1_5  --task_id japanese-heron-bench --openai_model_id gpt-4o-mini-2024-07-18
```

### 評価結果の確認

評価結果のスコアと出力結果は
`result/{task_id}/evaluation/{model_id}.jsonl`, `result/{task_id}/prediction/{model_id}.jsonl` に保存されます.

japanese-heron-benchベンチマークについての結果の確認については,
```python
rye run python3 scripts/japanese-heron-bench/record_output.py
```
を実行することで,
- 各exampleに対する各モデルの生成結果を載せたexcelファイル
- 各モデルのスコアを載せたexcelファイル
- 各モデルのスコア分布を示すグラフ画像
が生成されます.

### 評価結果をW&Bで管理

現在，評価結果は`.jsonl`で提供されるのみであり，W&Bとの連携はサポートする予定がありません．

## サポートするタスク

現在，以下のベンチマークタスクをサポートしている

- Japanese Heron Bench
- JA-VG-VQA500
- JA-VLM-Bench-In-the-Wild
- JA-Multi-Image-VQA
- JMMMU

## ライセンス

本ツールは [TODO: 必要なライセンス] の元に配布します．
各評価データセットのライセンスは[DATASET.md](./DATASET.md)を参照してください．

## Contribution

- 問題や提案があれば，Issue で報告してください．
- 修正や追加があれば，Pull Requestを送ってください．

- ライブラリの追加
```
rye add <package_name>
```
- ruffを用いたフォーマット
```
rye run ruff format .
```

- PyPIへのリリース方法
```
git tag -a v0.x.x -m "version 0.x.x"
git push origin --tags
```

