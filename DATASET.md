# 対応データセット一覧

## Instruction Following

### Japanese-Heron-Bench

- 出処：https://huggingface.co/datasets/turing-motors/Japanese-Heron-Bench
- しかし，便宜上以下のURLのデータセットを利用している
    - https://huggingface.co/datasets/Silviase/Japanese-Heron-Bench

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
    - name: input_text
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


#### citation

```bibtex
@misc{inoue2024heronbench,
      title={Heron-Bench: A Benchmark for Evaluating Vision Language Models in Japanese}, 
      author={Yuichi Inoue and Kento Sasaki and Yuma Ochi and Kazuki Fujii and Kotaro Tanahashi and Yu Yamaguchi},
      year={2024},
      eprint={2404.07824},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

## QA Dataset

### JA-VG-VQA-500

- 出処：https://huggingface.co/datasets/SakanaAI/JA-VG-VQA-500

#### データフォーマット

```yaml
dataset_info:
  features:
    - name: question_id
      dtype: int64
    - name: image
      dtype: image
    - name: context
      dtype: string
    - name: input_text
      dtype: string
    - name: answer
      dtype: string
  splits:
    - name: train
```

#### citation

```bibtex
@article{Krishna2016VisualGC,
  title   = {Visual Genome: Connecting Language and Vision Using Crowdsourced Dense Image Annotations},
  author. = {Ranjay Krishna and Yuke Zhu and Oliver Groth and Justin Johnson and Kenji Hata and Joshua Kravitz and Stephanie Chen and Yannis Kalantidis and Li-Jia Li and David A. Shamma and Michael S. Bernstein and Li Fei-Fei},
  journal = {International Journal of Computer Vision},
  year.   = {2017},
  volume. = {123},
  pages.  = {32-73},
  URL     = {https://doi.org/10.1007/s11263-016-0981-7},
  doi     = {10.1007/s11263-016-0981-7}
}
```

```bibtex
@InProceedings{C18-1163,
  author    = "Shimizu, Nobuyuki and Rong, Na and Miyazaki, Takashi",
  title     = "Visual Question Answering Dataset for Bilingual Image Understanding: A Study of Cross-Lingual Transfer Using Attention Maps",
  booktitle = "Proceedings of the 27th International Conference on Computational Linguistics",
  year      = "2018",
  publisher = "Association for Computational Linguistics",
  pages     = "1918--1928",
  location  = "Santa Fe, New Mexico, USA",
  url       = "http://aclweb.org/anthology/C18-1163"
}
```



### JDocQA

- 出処：https://github.com/mizuumi/JDocQA
- 便宜上以下のURLのデータセットを利用している
    - https://huggingface.co/datasets/shunk031/JDocQA


#### citation:

```bibtex
@inproceedings{onami2024jdocqa,
  title={JDocQA: Japanese Document Question Answering Dataset for Generative Language Models},
  author={Onami, Eri and Kurita, Shuhei and Miyanishi, Taiki and Watanabe, Taro},
  booktitle={Proceedings of the 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation (LREC-COLING 2024)},
  pages={9503--9514},
  year={2024}
}
```


## 統一的なフォーマット

データセットは以下のキーを必ず含んで提供される.

- **ID**: str, データセット内でのID．
- **images**: List[PIL.Image], 画像データ. 基本的に1枚だが複数枚の場合もあるためリストで表現
- **text**: str, 指示文．
- **answer**: str, 回答．

よく利用されるキーは以下の通り．必ず含まれているわけでは無いが，共通するタスクではキーを統一すること．

- **choices**: List[str], MMEはImageが選択肢に含まれることもある．

## Tips

- `src/tasks`内で取り出される時は，`doc_to_xxx()`関数を用いて取り出す．
