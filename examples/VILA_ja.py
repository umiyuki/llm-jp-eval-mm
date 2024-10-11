import torch
from PIL import Image
import requests
from llava.constants import IMAGE_TOKEN_INDEX
from llava.conversation import conv_templates
from llava.mm_utils import (
    get_model_name_from_path,
    process_images,
    tokenizer_image_token,
)
from llava.model.builder import load_pretrained_model


def load_images(image_files):
    out = []
    for image_file in image_files:
        image = Image.open(image_file).convert("RGB")
        out.append(image)
    return out


class VLM:
    model_id = "llm-jp/VILA-ja"

    def __init__(self) -> None:
        model_checkpoint_path = "/model/sasagawa/VILA-ja/checkpoints/llm-jp-3-13b-instruct_siglip_mlp2xgelu_step-2_20241004/"
        model_name = get_model_name_from_path(model_checkpoint_path)
        print(model_name)
        self.tokenizer, self.model, self.image_processor, self.context_len = (
            load_pretrained_model(model_checkpoint_path, model_name)
        )

    def generate(self, image, text: str, max_new_tokens: int = 256):
        qs = text
        if "<image>" not in text:
            if isinstance(image, list):
                qs = "<image>\n" * len(image) + text
            else:
                qs = "<image>\n" + text
        print(qs)
        print(image)
        conv_mode = "llmjp_v3"
        conv = conv_templates[conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        if isinstance(image, list):
            images = image
        else:
            images = [image]
        images_tensor = process_images(
            images, self.image_processor, self.model.config
        ).to(self.model.device, dtype=torch.float16)
        input_ids = (
            tokenizer_image_token(
                prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
            )
            .unsqueeze(0)
            .cuda()
        )

        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=[
                    images_tensor,
                ],
                do_sample=False,
                num_beams=1,
                max_new_tokens=max_new_tokens,
                use_cache=True,
            )

        outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
        return outputs


if __name__ == "__main__":
    model = VLM()
    image_file = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(image_file, stream=True).raw)
    print(model.generate(image, "What is in the image?"))
    print(model.generate([image, image], "What is in the image?"))
