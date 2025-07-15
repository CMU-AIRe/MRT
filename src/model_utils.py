import os
import re
from pathlib import Path

from transformers import AutoConfig
from transformers.trainer_utils import get_last_checkpoint

from huggingface_hub import list_repo_commits


def get_checkpoint(training_args) -> Path | None:
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
    return last_checkpoint


def get_gpu_count_for_vllm(model_path: str, revision: str = "main", num_gpus: int = 8) -> int:
    """vLLM enforces a constraint that the number of attention heads must be divisible by the number of GPUs and 64 must be divisible by the number of GPUs. This function calculates the number of GPUs to use for decoding based on the number of attention heads in the model."""
    config = AutoConfig.from_pretrained(model_path, revision=revision, trust_remote_code=True)
    # Get number of attention heads
    num_heads = config.num_attention_heads
    # Reduce num_gpus so that num_heads is divisible by num_gpus and 64 is divisible by num_gpus
    while num_heads % num_gpus != 0 or 64 % num_gpus != 0:
        print(f"Reducing num_gpus from {num_gpus} to {num_gpus - 1} to make num_heads divisible by num_gpus")
        num_gpus -= 1
    return num_gpus


def rev2step(model_id: str, revision: str) -> str | None:
    """Extracts the commit title and checks if 'step' is in the title"""
    commits = list_repo_commits(model_id)
    for commit in commits:
        if commit.commit_id == revision:
            match = re.search(r"(step\s\d+)", commit.title)
            if match:
                return match.group().split(" ")[-1]
    return None
