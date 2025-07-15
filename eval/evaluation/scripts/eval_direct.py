import json
import random
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
from template import build_conv
from vllm import LLM, SamplingParams


"""Usage:

PYTHONPATH=$(pwd) python evaluation/scripts/eval_direct.py \
    --model_path meta-llama/Llama-3.1-8B-Instruct \
    --model_revision main \
    --dataset_name hf-cmu-collab/metaMATH_eval_direct \
    --dataset_split AIME \
    --env MATH-1-shot \
    --num_samples 10 \
    --temperature 0.7 \
    --max_tokens 4096 \
    --hub_dataset_id hf-cmu-collab/Llama-3.1-8B-Instruct-evals \
    --push_to_hub
"""


@dataclass
class CustomConfig:
    model_path: str = "hf-cmu-collab/Llama-3.1-8B-Instruct-sft_conversations-v00.00"
    model_revision: str = "main"
    dataset_name: str = "hf-cmu-collab/metaMATH_eval_direct"
    dataset_split: str = "MATH500"
    env: str = "backtrack"
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
    prompts = []
    records = {}
    for i in range(len(problems)):
        records[str(i)] = {"problem": problems[i], "answer": solutions[i]}

    convs = [build_conv(config.env, q) for q in problems]
    tokenized_convs = [
        tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, return_tensors="pt")
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
