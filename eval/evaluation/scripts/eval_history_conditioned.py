import json
import random
import re
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer, HfArgumentParser

from huggingface_hub import create_branch, list_repo_commits, repo_exists
from math_equivalence import evaluate_answer
from src import get_gpu_count_for_vllm, rev2step
from src.chat_templates import LLAMA_3_PREFILL_TEMPLATE
from template import build_conv
from vllm import LLM, SamplingParams


"""Usage:

PYTHONPATH=$(pwd) python evaluation/scripts/eval_history_conditioned.py \
    --model_path hf-cmu-collab/Llama-3.1-8B-Instruct-sft_suffix-v01.00 \
    --model_revision main \
    --column_name backtrack_with_shared_prefix \
    --given_backtrack true \
    --dataset_name hf-cmu-collab/metaMATH_eval_history_conditioned \
    --dataset_split MATH500 \
    --env backtrack \
    --num_samples 10 \
    --temperature 0.7 \
    --max_tokens 4096
"""


def split_at_backtrack(text, given_backtrack):
    # Pattern to match "Backtrack to Line " followed by numbers
    if "Backtrack to Line" in text:
        pattern = r"(Backtrack to Line \d+\n)"
    else:
        pattern = r"(Backtrack to beginning\n)"

    # Find the match
    match = re.search(pattern, text)

    if match:
        # Get the start and end positions of the match
        start = match.start()
        end = match.end()

        # Split into three parts
        if given_backtrack:
            prefix = text[:end]  # Everything up to and including "Backtrack to Line X"/"Backtrack to Beginning"
            suffix = text[end:]  # Everything after
        else:
            prefix = text[:start]  # Everything up to "Backtrack to Line X"/"Backtrack to Beginning"
            suffix = text[start:]  # Everything after

        return prefix, suffix
    else:
        return "", ""


@dataclass
class CustomConfig:
    model_path: str = "hf-cmu-collab/Llama-3.1-8B-Instruct-sft_suffix-v00.00"
    model_revision: str = "main"
    dataset_name: str = "hf-cmu-collab/metaMATH_eval_history_conditioned"
    dataset_split: str = "MATH500"
    env: str = "backtrack"
    column_name: str = "without_full_rollout"
    given_backtrack: bool = True
    randomize: bool = False
    dataset_start: int = None
    dataset_end: int = None
    num_samples: int = 10
    temperature: float = 0.7
    max_tokens: int = 4096
    output_path: str = "output.json"
    hub_dataset_id: str = None
    push_to_hub: bool = False


if __name__ == "__main__":

    parser = HfArgumentParser((CustomConfig))
    config = parser.parse_args_into_dataclasses()[0]
    tokenizer = AutoTokenizer.from_pretrained(config.model_path, revision=config.model_revision)
    tokenizer.chat_template = LLAMA_3_PREFILL_TEMPLATE
    tp = get_gpu_count_for_vllm(config.model_path, num_gpus=torch.cuda.device_count())

    llm = LLM(
        model=config.model_path, revision=config.model_revision, tensor_parallel_size=tp, enable_prefix_caching=True
    )
    sampling_params = SamplingParams(
        temperature=config.temperature, n=config.num_samples, max_tokens=config.max_tokens, logprobs=5
    )

    dataset = load_dataset(config.dataset_name, split=config.dataset_split, trust_remote_code=True)
    if config.randomize:
        random.shuffle(dataset)
    if config.dataset_start is not None and config.dataset_end is not None:
        dataset = dataset.select(range(config.dataset_start, config.dataset_end))

    problems = dataset["problem"]
    solutions = dataset["answer"]
    prefixes = dataset[config.column_name]

    prompts = []
    records = {}
    new_prefixes = []
    for i in range(len(problems)):
        prefix, suffix = split_at_backtrack(prefixes[i], config.given_backtrack)
        records[str(i)] = {"problem": problems[i], "answer": solutions[i], "prefix": prefix, "suffix": suffix}
        prefixes[i] = prefix

    convs = [build_conv(config.env, q) for q in problems]
    for i in range(len(convs)):
        convs[i].append({"role": "assistant", "content": prefixes[i]})
        # records[str(i)]['conv'] = convs[i]

    # outputs = llm.chat(convs, sampling_params=sampling_params)

    tokenized_convs = [
        tokenizer.apply_chat_template(
            messages, tokenize=False, return_tensors="pt", continue_final_message=True, add_generation_prompt=False
        )
        for messages in convs
    ]
    outputs = llm.generate(tokenized_convs, sampling_params)

    for i in range(len(outputs)):
        output = outputs[i]
        ref_answer = records[str(i)]["answer"]
        responses = []
        rewards = []
        cumulative_logprobs = []
        for j in range(len(output.outputs)):
            gen_answer = output.outputs[j].text
            cumulative_logprob = output.outputs[j].cumulative_logprob
            cumulative_logprobs.append(cumulative_logprob)
            reward = evaluate_answer(ref_answer, gen_answer)
            responses.append(gen_answer)
            rewards.append(reward)
        records[str(i)]["responses"] = responses
        records[str(i)]["cumulative_logprobs"] = cumulative_logprobs
        records[str(i)]["rewards"] = rewards
        records[str(i)]["max_reward"] = max(rewards)
        records[str(i)]["mean_reward"] = np.mean(rewards)

    if config.push_to_hub:
        df = pd.DataFrame.from_dict(records, orient="index")
        records_ds = Dataset.from_pandas(df, preserve_index=False)
        step = rev2step(config.model_path, config.model_revision)
        dataset_name = config.dataset_name.replace("/", "__")
        if config.hub_dataset_id is None:
            hub_dataset_id = f"{config.model_path}-evals"
        else:
            hub_dataset_id = config.hub_dataset_id
        if step is not None:
            config_name = f"{config.dataset_split}--step-{step}--pass@{config.num_samples}"
        else:
            config_name = f"{config.dataset_split}--rev-{config.model_revision}--pass@{config.num_samples}"

        # Since concurrent pushes can get rejected by the Hub, we make several attempts to push the dataset with try/except
        for _ in range(20):
            try:
                # Create branch from the repo's initial commit.
                # This is needed to avoid branching from a commit on main that already has data
                if repo_exists(hub_dataset_id, repo_type="dataset"):
                    initial_commit = list_repo_commits(hub_dataset_id, repo_type="dataset")[-1]
                    create_branch(
                        repo_id=hub_dataset_id,
                        branch=config.model_revision,
                        revision=initial_commit.commit_id,
                        exist_ok=True,
                        repo_type="dataset",
                    )
                url = records_ds.push_to_hub(
                    hub_dataset_id,
                    config_name=config_name,
                    split="test",
                    private=True,
                    commit_message=f"Add {config_name}",
                )
                break
            except Exception as e:
                print(f"Error pushing dataset to the Hub: {e}")
                time.sleep(5)
        print(f"Pushed dataset to {url}")

    else:
        with open(config.output_path, "w") as f:
            json.dump(records, f, indent=4)
