# LLM-jp-eval-mm
[ [**English**](./README_en.md) | 日本語 ]

このツールは，複数のデータセットを横断して日本語マルチモーダル大規模言語モデルを自動評価するものです．
以下の機能を提供します．

- 既存の日本語評価データを利用し，テキスト生成タスクの評価データセットに変換
- 複数データセットを横断して大規模言語モデルの評価を実行
- 評価データのプロンプトと同形式のインストラクションデータ (jaster) を生成

データフォーマットの詳細，サポートしているデータの一覧，またjasterの詳細については，[DATASET.md](./DATASET.md)を参照ください．


## 目次

- [インストール](#インストール)
- [評価方法](#評価方法)
  - [データセットのダウンロードと前処理](#データセットのダウンロードと前処理)
  - [評価の実行](#評価の実行)
  - [評価結果の確認](#評価結果の確認)
- [ライセンス](#ライセンス)
- [Contribution](#Contribution)

## インストール

1. [poetry](https://python-poetry.org/docs/)（推奨） または pip を使用

- poetry の場合
    ```bash
    cd llm-jp-eval-mm
    poetry install
    ```


## 評価方法

### 評価の実行

<!-- 設定は config ファイルで管理し，[hydra](https://github.com/facebookresearch/hydra)を使用して読み込む． -->

設定は config ファイルで管理し，現在は[hydra]ではなく``CONFIG=``と直に指定する形をとっている．
hydraを可及的速やかに導入する予定である．


```bash
CONFIG=models/evovlm-jp-v1,tasks/japanese-heron-bench poetry run python src/llm_jp_eval_mm/evaluator.py
```

### 評価結果の確認

評価結果のスコアと出力結果は `src/llm_jp_eval_mm/{task_id}/{model_id}/eval.json` に保存される．
詳細な結果は`src/llm_jp_eval_mm/{task_id}/{model_id}/eval_verbose.jsonl`に保存される．

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
