from PIL import Image
import requests
from transformers import AutoProcessor, LlavaForConditionalGeneration
import torch


class VLM:
    model_id = "cyberagent/llava-calm2-siglip"

    def __init__(self) -> None:
        self.model = LlavaForConditionalGeneration.from_pretrained(
            self.model_id,
            torch_dtype=torch.bfloat16,
        ).to("cuda")
        self.processor = AutoProcessor.from_pretrained(self.model_id)

    def generate(self, image, text: str, max_new_tokens: int = 256):
        prefix = None
        if "<image>" in text:
            prompt = "USER: " + text + "\nASSISTANT: "
        else:
            if isinstance(image, list):
                num_images = len(image)
                prefix = "<image> " * num_images
            else:
                prefix = "<image> "
            prompt = "USER: " + prefix + text + "\nASSISTANT: "

        inputs = (
            self.processor(text=prompt, images=image, return_tensors="pt")
            .to(self.model.device)
            .to(self.model.dtype)
        )
        output_ids = self.model.generate(
            **inputs,
            max_length=max_new_tokens,
            do_sample=True,
            temperature=0.2,
        )
        generate_ids = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(inputs.input_ids, output_ids)
        ]

        output = self.processor.tokenizer.decode(
            generate_ids[0][:-1], clean_up_tokenization_spaces=False
        )

        return output


if __name__ == "__main__":
    model = VLM()
    image_file = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(image_file, stream=True).raw)
    print(model.generate(image, "What is in the image?"))
    print(
        model.generate(
            [image, image], "What is the difference between these two images?"
        )
    )
