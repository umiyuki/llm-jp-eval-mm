#!/bin/bash

# Available models
models=(
    # "EvoVLM_JP_v1_7B" 
    # "InternVL2_8B"
    # "japanese_instructblip_alpha"
    # "japanese_stable_vlm" 
    # "Llama_3_2_11B_Vision_Instruct"
    "llava_1_5_7b_hf"
    # "llava_calm2_siglip"
    # "llava_v1_6_mistral_7b_hf"
    # "Qwen2_VL_7B_Instruct"
)

# Available tasks
tasks=(
    "japanese-heron-bench" 
    # "ja-vg-vqa-500" 
    # "ja-vlm-bench-in-the-wild" 
    # "ja-multi-image-vqa" 
    # "jmmmu"
)

# Execute evaluation
for model in "${models[@]}"; do
    for task in "${tasks[@]}"; do
        echo "Execute: rye run python examples/sample.py --class_path $model --task_id $task"
        rye run python examples/sample.py --class_path $model --task_id $task
    done
done

echo "Evaluation is done." 