# 対応データセット一覧

## Instruction Following

### Japanese-Heron-Bench

- 出処：https://huggingface.co/datasets/turing-motors/Japanese-Heron-Bench
- しかし，便宜上以下のURLのデータセットを利用している
    - https://huggingface.co/datasets/Silviase/Japanese-Heron-Bench
- ライセンス：画像により異なる．[LICENSE.md](https://huggingface.co/datasets/turing-motors/Japanese-Heron-Bench/blob/main/LICENCE.md)を参照されたし．

#### データフォーマット

```yaml
language:
  - ja
dataset_info:
  features:
    - name: question_id
      dtype: int64
    - name: image
      dtype: image
    - name: category
      dtype: string
    - name: image_category
      dtype: string
    - name: context
      dtype: string
    - name: text
      dtype: string
    - name: answer
      struct:
        - name: claude-3-opus-20240229
          dtype: string
        - name: gemini-1.0-pro-vision-latest
          dtype: string
        - name: gpt-4-0125-preview
          dtype: string
        - name: gpt-4-vision-preview
          dtype: string
  splits:
    - name: train
```


## 統一的なフォーマット

データセットは以下のキーを必ず含んで提供される.

- **ID**: str, データセット内でのID．
- **images**: List[PIL.Image], 画像データ. 基本的に1枚だが複数枚の場合もあるためリストで表現
- **text**: str, 指示文．
- **answer**: str, 回答．

よく利用されるキーは以下の通り．必ず含まれているわけでは無いが，共通するタスクではキーを統一すること．

- **choices**: List[str], MMEはImageが選択肢に含まれることもある．
