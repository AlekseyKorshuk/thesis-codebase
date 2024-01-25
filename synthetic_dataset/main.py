import json
import os
import random
from functools import partial
from typing import Any, Dict

import click
import tqdm
from datasets import Dataset, load_dataset

from utils import utils
from utils.openai_utils import openai_completion, OpenAIDecodingArguments
from prompts import v1 as prompts

# Constants
OPENAI_KEYS = [os.getenv("OPENAI_API_KEY", "")]
SLEEP_TIME = 1

assert len(OPENAI_KEYS) > 0


@click.command()
@click.option("--config_path", type=str)
@click.option("--push_current", type=bool, is_flag=True)
def generate_samples(config_path: str, push_current: bool):
    """Entry point for the CLI."""
    config = utils.load_yaml(config_path)
    config["push_current"] = push_current
    resulting_dataset = run_sample_generation(config)
    push_to_hub(resulting_dataset, config)


def run_sample_generation(config: Dict[str, Any]):
    """Runs the sample generation process based on the configuration."""
    ds = load_dataset(config["seed_dataset_path"], split="train")
    process_function = partial(process_sample, config=config)
    os.makedirs(config["output_path"], exist_ok=True)
    ds = ds.map(process_function, num_proc=config["num_cpus"])
    ds = ds.filter(lambda example: example["better_response"] is not None and example["better_response"] != "")
    return ds


def process_sample(sample, config):
    sample_hash = utils.hash_sample(sample)
    output_file_path = os.path.join(config["output_path"], f"{sample_hash}.json")
    if os.path.exists(output_file_path):
        with open(output_file_path) as json_file:
            json_data = json.load(json_file)
        return postprocess_sample(json_data)

    if config.get("push_current", False):
        return {
            "better_response": "",
            "worse_response": "",
        }
    json_data = generate_completion(
        sample["instruction"],
        sample["response"],
        config
    )
    with open(output_file_path, "w") as outfile:
        json.dump(json_data, outfile)
    return postprocess_sample(json_data)


def generate_completion(instruction, response, config):
    user_content = {"instruction": instruction, "response": response}
    messages = [
        {"role": "system", "content": prompts.system_prompt},
        {"role": "user", "content": json.dumps(user_content, indent=1)}
    ]
    decoding_args = OpenAIDecodingArguments(**config["openai_generation_params"], api_key=random.choice(OPENAI_KEYS))
    json_response = None
    max_retries = 5
    while json_response is None or json_response.get("better_response", "") == "":
        openai_content = openai_completion(messages, decoding_args, config["model_name"], SLEEP_TIME)
        json_response = json.loads(openai_content)
        max_retries -= 1
        if max_retries == 0:
            print("Max retries reached, returning empty response")
            return {}
    return json_response


def push_to_hub(dataset, config):
    # dataset.push_to_hub(config["dataset_path"], private=True)

    sft_dataset = _get_sft_dataset(dataset, config)
    sft_dataset.push_to_hub(config["dataset_path"] + "-sft", private=True)

    dpo_dataset = _get_dpo_dataset(dataset, config)
    dpo_dataset.push_to_hub(config["dataset_path"] + "-dpo", private=True)


def _get_sft_dataset(dataset, config):
    system_counter = 0

    def _process(sample):
        nonlocal system_counter
        system_counter = (system_counter + 1) % len(prompts.chatml_system_prompts)
        new_sample = {
            "conversations": [
                {"from": "system", "value": prompts.chatml_system_prompts[system_counter]},
                {"from": "human", "value": sample["instruction"]},
                {"from": "gpt", "value": sample["better_response"]},
            ]
        }
        return new_sample

    sft_dataset = dataset.map(_process, num_proc=config["num_cpus"], desc="Collecting SFT dataset")
    sft_dataset = sft_dataset.remove_columns(list(dataset[0].keys()))
    return sft_dataset


def _get_dpo_dataset(dataset, config):
    data = []
    for i, sample in tqdm.tqdm(enumerate(dataset), total=len(dataset), desc="Collecting DPO dataset"):
        rejected_response = sample[
            "worse_response" if i % 2 == 0 else "response"
        ]
        data.extend(
            [
                {
                    "system": prompts.chatml_system_prompts[i % len(prompts.chatml_system_prompts)],
                    "question": sample["instruction"],
                    "chosen": sample["better_response"],
                    "rejected": rejected_response
                },
            ]
        )
    dpo_dataset = Dataset.from_list(data)
    return dpo_dataset


def postprocess_sample(sample):
    return {
        "better_response": sample["better_response"],
        "worse_response": sample["worse_response"],
    }


if __name__ == "__main__":
    generate_samples()
