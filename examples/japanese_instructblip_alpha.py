import requests
from PIL import Image
from transformers import LlamaTokenizer, AutoModelForVision2Seq, BlipImageProcessor
from typing import Union


# helper function to format input prompts
def build_prompt(prompt="", sep="\n\n### "):
    sys_msg = "以下は、タスクを説明する指示と、文脈のある入力の組み合わせです。要求を適切に満たす応答を書きなさい。"
    p = sys_msg
    roles = ["指示", "応答"]
    user_query = "与えられた画像について、詳細に述べてください。"
    msgs = [": \n" + user_query, ": "]
    if prompt:
        roles.insert(1, "入力")
        msgs.insert(1, ": \n" + prompt)
    for role, msg in zip(roles, msgs):
        p += sep + role + msg
    return p


class VLM:
    model_id = "stabilityai/japanese-instructblip-alpha"

    def __init__(self) -> None:
        self.model = AutoModelForVision2Seq.from_pretrained(
            self.model_id,
            trust_remote_code=True,
        ).to("cuda")
        self.processor = BlipImageProcessor.from_pretrained(self.model_id)
        self.tokenizer = LlamaTokenizer.from_pretrained(
            "novelai/nerdstash-tokenizer-v1", additional_special_tokens=["▁▁"]
        )

    def generate(
        self,
        images: Union[Image.Image, list[Image.Image]],
        text: str,
        max_new_tokens: int = 256,
    ):
        prompt = build_prompt(prompt=text)
        inputs = self.processor(images=images, return_tensors="pt")
        text_encoding = self.tokenizer(
            prompt, add_special_tokens=False, return_tensors="pt"
        )
        text_encoding["qformer_input_ids"] = text_encoding["input_ids"].clone()
        text_encoding["qformer_attention_mask"] = text_encoding[
            "attention_mask"
        ].clone()
        inputs.update(text_encoding)

        # autoregressively complete prompt
        output = self.model.generate(
            **inputs.to(self.model.device),
            num_beams=5,
            max_new_tokens=max_new_tokens,
            min_length=1,
        )
        # TODO: white space return problem some times
        response = self.tokenizer.batch_decode(output, skip_special_tokens=True)
        generated_text = response[0].strip()
        return generated_text


if __name__ == "__main__":
    model = VLM()
    image_file = "https://images.unsplash.com/photo-1582538885592-e70a5d7ab3d3?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=1770&q=80"
    image = Image.open(requests.get(image_file, stream=True).raw).convert("RGB")
    print(model.generate(image, "これはなんですか?"))

    multi_images = [image for _ in range(3)]
    print(model.generate(multi_images, "What is the difference between these images?"))
