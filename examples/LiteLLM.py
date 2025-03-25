import base64
from io import BytesIO
from PIL import Image
from litellm import completion
from base_vlm import BaseVLM
from utils import GenerationConfig
import os


def encode_image_to_base64(image: Image.Image) -> str:
    """画像をBase64文字列に変換"""
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str


class LiteLLMVLM(BaseVLM):
    def __init__(
        self,
        model_id: str,
        api_base: str = None,
        api_key: str = None,
        provider: str = "openai"
    ) -> None:
        """
        LiteLLMを介したVLMモデルの初期化。
        
        Args:
            model_id: モデル識別子 (例: "gpt-4o-2024-05-13", "gemini-1.5-pro")
            api_base: カスタムAPIエンドポイント（オプション）
            api_key: APIキー（オプション、デフォルトは環境変数から取得）
            provider: 使用するプロバイダ ("openai", "azure", "gemini", etc.)
        """
        self.model_id = model_id
        self.provider = provider.lower()
        self.api_base = api_base
        
        # APIキーの設定
        if api_key:
            self.api_key = api_key
        elif self.provider == "openai":
            self.api_key = os.getenv("OPENAI_API_KEY")
        elif self.provider == "azure":
            self.api_key = os.getenv("AZURE_OPENAI_KEY")
            self.api_base = os.getenv("AZURE_OPENAI_ENDPOINT")
        elif self.provider == "gemini":
            self.api_key = os.getenv("GOOGLE_API_KEY")
        else:
            self.api_key = os.getenv("CUSTOM_API_KEY")
        
        if not self.api_key:
            raise ValueError(f"API key not found for provider {self.provider}. Set it via argument or environment variable.")

    def generate(
        self,
        images: list[Image.Image],
        prompt: str,
        gen_kwargs: GenerationConfig = GenerationConfig(),
    ) -> str:
        """
        画像とプロンプトを受け取り、LiteLLM経由で生成結果を返す。
        """
        # 画像をBase64に変換し、メッセージに追加
        content = []
        for img in images:
            base64_img = encode_image_to_base64(img)
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{base64_img}"}
            })
        content.append({"type": "text", "text": prompt})

        messages = [
            {
                "role": "user",
                "content": content
            }
        ]

        # providerがopenaiの場合、modelを"openai/モデルID"形式に変更
        model = self.model_id
        if self.provider == "openai":
            model = f"openai/{self.model_id}"

        # LiteLLMを介して推論
        try:
            response = completion(
                model=model,
                messages=messages,
                model_response_format={"type": "text"},
                mock_response=False,
                api_base=self.api_base,
                api_key=self.api_key,
                max_tokens=gen_kwargs.max_new_tokens,
                temperature=gen_kwargs.temperature,
                top_p=gen_kwargs.top_p,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error during inference: {e}")
            return "Error"

if __name__ == "__main__":
    # テスト用コード
    vlm = LiteLLMVLM(model_id="google/gemma-3-12b-it", provider="openai")
    vlm.test_vlm()