import time
from dataclasses import dataclass
from typing import Optional, Sequence, List
from multiprocessing.pool import ThreadPool

from openai import Client, OpenAIError
import copy


@dataclass
class OpenAIDecodingArguments:
    max_tokens: int = 4096
    temperature: float = 0.3
    top_p: float = 1.0
    n: int = 1
    stream: bool = False
    stop: Optional[Sequence[str]] = None
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    api_key: str = None


def openai_completion(
        messages: List[dict],
        decoding_args: OpenAIDecodingArguments,
        model_name: str,
        sleep_time: float
):
    decoding_args = copy.deepcopy(decoding_args)
    client = Client(api_key=decoding_args.__dict__.pop("api_key"))
    assert decoding_args.n == 1
    while True:
        try:
            completions = client.chat.completions.create(
                messages=messages,
                model=model_name,
                response_format={"type": "json_object"},
                **decoding_args.__dict__
            )
            break
        except OpenAIError as e:
            if "Please reduce" in str(e):
                decoding_args.max_tokens = int(decoding_args.max_tokens * 0.95)
            elif "rate" in str(e).lower():
                time.sleep(sleep_time)
            elif " takes 1 positional" in str(e).lower():
                print("OpenAIError: ", e)
            else:
                raise e
        except TypeError as e:
            print(e)
    return completions.choices[0].message.content


def openai_batch_completion(
        batch,
        decoding_args: OpenAIDecodingArguments,
        model_name="gpt-4-1106-preview",
        sleep_time=2,
):
    completions = []
    with ThreadPool(len(batch)) as pool:
        results = pool.starmap(openai_completion, [
            (messages, decoding_args, model_name, sleep_time) for messages in batch
        ])
        for result in results:
            completions.append(result)
    return completions
