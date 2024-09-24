# Reference: https://github.com/ryokan0123/yans-2024-hackathon-baseline

import os
from openai import AsyncAzureOpenAI, AsyncOpenAI
import asyncio
from typing import Awaitable, Callable, TypeVar, Iterable, Iterator
import openai
import logging
from itertools import islice

logger = logging.getLogger(__name__)

T = TypeVar("T")


def batch_iter(iterable: Iterable[T], batch_size: int) -> Iterator[list[T]]:
    """
    Yield batches of elements from an iterable.
    Example:
    >>> list(batch_iter(range(10), 3))
    [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]
    """
    it = iter(iterable)
    while True:
        batch = list(islice(it, batch_size))
        if not batch:
            break
        yield batch


async def _retry_on_error(
    openai_call: Callable[[], Awaitable[T]],
    max_num_trials: int = 5,
    first_wait_time: int = 10,
) -> Awaitable[T] | None:
    """
    API送信時にエラーが発生した場合にリトライするための関数です。
    """
    for i in range(max_num_trials):
        try:
            return await openai_call()
        except openai.APIError as e:  # noqa: PERF203
            if i == max_num_trials - 1:
                raise
            logger.warning(f"We got an error: {e}")
            wait_time_seconds = first_wait_time * (2**i)
            logger.warning(f"Wait for {wait_time_seconds} seconds...")
            await asyncio.sleep(wait_time_seconds)
    return None


class OpenAIChatAPI:
    """
    OpenAI APIのラッパーです。
    `batch_generate_chat_response`メソッドを使用して、複数のチャットリクエストを並列で送信できます。

    Args:
        model: 使用するモデルの名前。
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini-2024-07-18",  # ハッカソン中は gpt-4o-mini 以外使用禁止です
    ) -> None:
        self.model = model

        if (
            os.getenv("AZURE_OPENAI_KEY") is not None
            and os.getenv("AZURE_OPENAI_ENDPOINT") is not None
        ):
            self._client = AsyncAzureOpenAI(
                api_key=os.getenv("AZURE_OPENAI_KEY"),
                api_version="2023-05-15",
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            )
        elif os.getenv("OPENAI_API_KEY") is not None:
            self._client = AsyncOpenAI()
        else:
            raise ValueError("API Key not found.")

    async def _async_batch_run_chatgpt(
        self,
        messages_list: list[list[dict[str, str]]],
        stop_sequences: str | list[str] | None = None,
        max_new_tokens: int | None = None,
        **kwargs,
    ) -> list[str]:
        """Send multiple chat requests to the OpenAI in parallel."""
        if stop_sequences is not None:
            if "stop" in kwargs:
                msg = (
                    "You specified both `stop_sequences` and `stop` in generation kwargs. "
                    "However, `stop_sequences` will be normalized into `stop`. "
                    "Please specify only one of them."
                )
                raise ValueError(msg)
            kwargs["stop"] = stop_sequences

        if max_new_tokens is not None:
            if "max_tokens" in kwargs:
                msg = (
                    "You specified both `max_new_tokens` and `max_tokens` in generation kwargs. "
                    "However, `max_new_tokens` will be normalized into `max_tokens`. "
                    "Please specify only one of them."
                )
                raise ValueError(msg)
            kwargs["max_tokens"] = max_new_tokens

        tasks = [
            _retry_on_error(
                # Define an anonymous function with a lambda expression and pass it,
                # and call it inside the _retry_on_error function
                openai_call=lambda x=ms: self._client.chat.completions.create(
                    model=self.model,
                    messages=x,
                    **kwargs,
                ),
            )
            for ms in messages_list
        ]
        return await asyncio.gather(*tasks)

    def batch_generate_chat_response(
        self,
        chat_messages_list: list[list[dict[str, str]]],
        **kwargs,
    ) -> list[str]:
        api_responses = asyncio.run(
            self._async_batch_run_chatgpt(chat_messages_list, **kwargs),
        )
        model_outputs: list[str] = []
        for res in api_responses:
            model_output = res.choices[0].message.content
            model_outputs.append(model_output)

            logger.info(f"モデル出力: {model_output}")
            logger.info(res.usage)
        return model_outputs

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model={self.model})"


async_client = OpenAIChatAPI()

if __name__ == "__main__":
    # テストコード
    messages_list = [
        [{"role": "system", "content": "こんにちは"}],
        [{"role": "user", "content": "今日の天気は？"}],
    ]
    responses = async_client.batch_generate_chat_response(messages_list)
    print(responses)
    dataset = [messages_list[0] for _ in range(10)]
    # progress bar
    from tqdm import tqdm

    with tqdm(total=len(dataset)) as pbar:
        for i, items in tqdm(enumerate(batch_iter(dataset, batch_size=4))):
            print(items)
            responses = async_client.batch_generate_chat_response(items)
            print(responses)
            pbar.update(len(items))
