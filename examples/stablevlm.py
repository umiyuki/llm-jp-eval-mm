import torch
from transformers import AutoTokenizer, AutoModelForVision2Seq, AutoImageProcessor
from PIL import Image
import requests

# helper function to format input prompts
TASK2INSTRUCTION = {
    "caption": "画像を詳細に述べてください。",
    "tag": "与えられた単語を使って、画像を詳細に述べてください。",
    "vqa": "与えられた画像を下に、質問に答えてください。",
}


def build_prompt(task="caption", input=None, sep="\n\n### "):
    assert (
        task in TASK2INSTRUCTION
    ), f"Please choose from {list(TASK2INSTRUCTION.keys())}"
    if task in ["tag", "vqa"]:
        assert input is not None, "Please fill in `input`!"
        if task == "tag" and isinstance(input, list):
            input = "、".join(input)
    else:
        assert input is None, f"`{task}` mode doesn't support to input questions"
    sys_msg = "以下は、タスクを説明する指示と、文脈のある入力の組み合わせです。要求を適切に満たす応答を書きなさい。"
    p = sys_msg
    roles = ["指示", "応答"]
    instruction = TASK2INSTRUCTION[task]
    msgs = [": \n" + instruction, ": \n"]
    if input:
        roles.insert(1, "入力")
        msgs.insert(1, ": \n" + input)
    for role, msg in zip(roles, msgs):
        p += sep + role + msg
    return p

class VLM:
    def __init__(self) -> None:
        self.model_id = "stabilityai/japanese-stable-vlm"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = AutoModelForVision2Seq.from_pretrained(
            self.model_id,
            trust_remote_code=True,
        )
        self.processor = AutoImageProcessor.from_pretrained(self.model_id)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.model.to(self.device)

    def generate(self, image, text: str):
        prompt = build_prompt(task="vqa", input=text)
        inputs = self.processor(images=image, return_tensors="pt")
        text_encoding = self.tokenizer(prompt, add_special_tokens=False, return_tensors="pt")
        inputs.update(text_encoding)

        # autoregressively complete prompt
        output = self.model.generate(**inputs.to(self.device, dtype=self.model.dtype), max_new_tokens=100)[0]

        generated_text = self.tokenizer.decode(output, skip_special_tokens=True).strip()
        return generated_text


if __name__ == "__main__":
    model = VLM()
    image_file = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(image_file, stream=True).raw)
    print(model.generate(image, "What is in the image?"))