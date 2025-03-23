import os
from litellm import completion, RateLimitError
from typing import Optional
import logging
from backoff import on_exception, expo, full_jitter
import litellm
from tqdm import tqdm
import time

logger = logging.getLogger(__name__)

# 429エラーとその他の例外に対して指数バックオフでリトライ
@on_exception(expo, Exception, max_tries=10, max_time=60, jitter=full_jitter)
def call_llm(
    model_name: str,
    messages_list: list[dict[str, str]],
    api_base: Optional[str] = None,
    api_key: Optional[str] = None,
    **kwargs
) -> dict:
    """
    Calls an LLM via LiteLLM with retry logic.
    """
    print("call_llm" + str(messages_list) + str(model_name) + str(api_base) + str(api_key) + str(kwargs))
    try:
        response = litellm.completion(
            model=model_name,
            messages=messages_list,
            api_base=api_base,
            api_key=api_key,
            **kwargs
        )
        print("response: " + str(response.choices[0].message.content))
        return response
    except Exception as e:
        print(f"Error in call_llm: {e}")
        raise  # 例外を再スローしてリトライ処理をトリガー

class LLMChatAPI:
    """
    Wrapper class for LLM API client using LiteLLM.
    Supports OpenAI-compatible endpoints and various providers.
    """

    def __init__(self, provider: str = "custom", api_base: Optional[str] = None) -> None:
        self.provider = provider.lower()
        self.api_base = api_base 
        # APIキーの設定
        if self.provider == "azure":
            self.api_key = os.getenv("AZURE_OPENAI_KEY")
            self.api_base = os.getenv("AZURE_OPENAI_ENDPOINT")
            if not self.api_key or not self.api_base:
                raise ValueError("AZURE_OPENAI_KEY and AZURE_OPENAI_ENDPOINT must be set.")
        elif self.provider == "openai":
            self.api_key = os.getenv("OPENAI_API_KEY")
            if not self.api_key:
                raise ValueError("OPENAI_API_KEY must be set.")
        elif self.provider == "custom":
            self.api_key = os.getenv("CUSTOM_API_KEY")
            if not self.api_key or not self.api_base:
                raise ValueError("CUSTOM_API_KEY and API_BASE_URL must be set.")
        elif self.provider == "gemini":
            self.api_key = os.getenv("GOOGLE_API_KEY")
            if not self.api_key:
                raise ValueError("GOOGLE_API_KEY must be set.")
        else:
            raise ValueError(f"Unsupported provider: {provider}")

    def _sync_batch_run_llm(
        self,
        messages_list: list[list[dict[str, str]]],
        stop_sequences: Optional[str | list[str]] = None,
        max_new_tokens: Optional[int] = None,
        model_name: Optional[str] = None,
        **kwargs,
    ) -> list[Optional[dict]]:
        """
        Send multiple chat requests to the LLM API sequentially.
        """
        if stop_sequences is not None:
            if "stop" in kwargs:
                raise ValueError("Specify either `stop_sequences` or `stop`, not both.")
            kwargs["stop"] = stop_sequences

        if max_new_tokens is not None:
            if "max_tokens" in kwargs:
                raise ValueError("Specify either `max_new_tokens` or `max_tokens`, not both.")
            kwargs["max_tokens"] = max_new_tokens

        print("messages_list: " + str(messages_list) + "\nmodel_name: " + str(model_name) + " provider: " + str(self.provider) + "api_base: " + str(self.api_base) + "api_key: " + str(self.api_key) + "kwargs: " + str(kwargs))
        
        results = []
        for ms in messages_list:
            try:
                result = call_llm(
                    model_name=model_name,
                    messages_list=ms,
                    api_base=self.api_base,
                    api_key=self.api_key,
                    **kwargs
                )
                results.append(result)
            except Exception as e:
                logger.exception(f"Error in processing message: {e}")
                results.append(None)

        return results

    def batch_generate_chat_response(
        self,
        chat_messages_list: list[list[dict[str, str]]],
        model_name: Optional[str] = None,
        **kwargs,
    ) -> list[str]:
        """
        Generate chat responses for a batch of messages.
        """
        api_responses = self._sync_batch_run_llm(
            chat_messages_list,
            model_name=model_name,
            **kwargs
        )
        model_outputs = []
        for res in api_responses:
            if res is None:
                model_outputs.append("")
                continue
            model_output = res.choices[0].message.content
            model_outputs.append(model_output)
            logger.info(f"Model output: {model_output}")
            logger.info(f"Usage: {res.usage}")
        return model_outputs

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(provider={self.provider}, api_base={self.api_base})"

if __name__ == "__main__":
    # テストコード
    client = LLMChatAPI(provider="openai", api_base="http://127.0.0.1:5001/v1")  # OpenAIプロバイダーを使用
    messages_list = [
        [{"role": "system", "content": "こんにちは"}],
        [{"role": "user", "content": "今日の天気は？"}],
    ]
    responses = client.batch_generate_chat_response(
        messages_list, model_name="openai/gpt-4-turbo-preview"  # 実際のOpenAIモデル名を使用
    )
    print(responses)