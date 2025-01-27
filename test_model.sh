uv sync --group evovlm
uv run  --group evovlm examples/EvoVLM_JP_v1_7B.py

uv sync --group vilaja
uv run --group vilaja examples/VILA_ja.py

uv sync --group normal --group normal
uv run --group normal examples/japanese_instructblip_alpha.py
uv run --group normal examples/japanese_stable_vlm.py
uv run --group normal examples/llava_1_5.py
uv run --group normal examples/llava_v1_6_mistral_7b_hf.py
uv run --group normal examples/Llama_3_2_11B_Vision_Instruct.py
uv run --group normal examples/llava_calm2_siglip.py
uv run --group normal examples/Pangea_7B_hf.py
uv run --group normal examples/InternVL2.py
uv run --group normal examples/Qwen2_VL.py

uv sync --group pixtral
uv run  --group pixtral examples/Pixtral_12B_2409.py