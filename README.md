# LLM-jp-eval-mm
[ [**English**](./README_en.md) | 日本語 ]

このツールは，複数のデータセットを横断して日本語マルチモーダル大規模言語モデルを自動評価するものです．
以下の機能を提供します．

- 既存の日本語評価データを利用し，テキスト生成タスクの評価データセットに変換
- 複数データセットを横断して大規模言語モデルの評価を実行

データフォーマットの詳細，サポートしているデータの一覧については，[DATASET.md](./DATASET.md)を参照ください．


## 目次

- [LLM-jp-eval-mm](#llm-jp-eval-mm)
  - [目次](#目次)
  - [環境構築](#環境構築)
  - [評価方法](#評価方法)
    - [サンプルコードの実行](#サンプルコードの実行)
    - [評価結果の確認](#評価結果の確認)
    - [評価結果をW\&Bで管理](#評価結果をwbで管理)
  - [ライセンス](#ライセンス)
  - [Contribution](#contribution)

## 環境構築

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

1. [.env.sample](./.env.sample)を参考にして, `.env`ファイルを作成し，`AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_KEY`を設定してください.

以上で環境構築は終了です.

## 評価方法

### サンプルコードの実行

現在のサンプルコードは`sample.py`です．
- 評価モデル：`EvoVLMv1`
- 評価ベンチマーク：`japanese-heron-bench`

```bash
rye run python3 examples/sample.py --class_path llava_1_5  --task_id japanese-heron-bench --openai_model_id gpt-4o-mini-2024-07-18
```

### 評価結果の確認

評価結果のスコアと出力結果は
`result/{task_id}/evaluation/{model_id}-{unixtime}.jsonl`, `result/{task_id}/prediction/{model_id}-{unixtime}.jsonl` に保存されます.

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

現在は行われていない．

## ライセンス

本ツールは [TODO: 必要なライセンス] の元に配布します．
各評価データセットのライセンスは[DATASET.md](./DATASET.md)を参照してください．

## Contribution

- 問題や提案があれば，Issue で報告してください
- 修正や追加があれば，Pull Requestを送ってください
    - コードのフォーマッターの管理に [pre-commit](https://pre-commit.com)を使用しています
        - `pre-commit run --all-files` を実行することでコードのフォーマットを自動で行い，修正が必要な箇所が表示されます．全ての項目を修正してからコミットしてください

- ライブラリの追加
```
rye add <package_name>
```
- ruffを用いたフォーマット
```
rye run ruff format .
```