
# Environment

```
conda create -n trl python=3.12.7
conda activate trl
pip install datasets trl wandb deepspeed==0.16.2
```

## Usage

From the root of the repo, run either:

* `./finetune/bash/sft_conversations.sh`
* `./finetune/bash/sft_suffix.sh`

Alternatively, you can use the YAML configs provided in the `experiments` folder:

```shell
HF_HUB_ENABLE_HF_TRANSFER=1 ACCELERATE_LOG_LEVEL=info accelerate launch --config_file finetune/experiments/accelerate_configs/fsdp.yaml finetune/scripts/{task}.py finetune/{model_name}/{task}/config_{version}.yaml
```

Here `{task}` refers to the type of training you wish to run. Currently, the following tasks are supported:

* `sft_conversations`: Uses `messages` as input and computes loss on assistant's response.
* `sft_suffix`: Takes `messages`, `prefix`, and `suffix` as input and computes loss only on the `suffix`. Simulates scenario where another model generates the first rollout and Trains model to generate improved responses based on history.

`{model_name}` refers to the choice of a recipe in the `experiments` directory, while `{version}` refers to a particular set of hyperparameters. For example, to fine-tune Llama-3.2-1B-Instruct you can run

```shell
# sft_conversations
ACCELERATE_LOG_LEVEL=info accelerate launch --config_file finetune/experiments/accelerate_configs/zero3.yaml finetune/scripts/sft_conversations.py --config finetune/experiments/Llama-3.2-1B-Instruct/sft_conversations/config_v00.00.yaml

# sft_suffix
ACCELERATE_LOG_LEVEL=info accelerate launch --config_file finetune/experiments/accelerate_configs/zero3.yaml finetune/scripts/sft_suffix.py --config finetune/experiments/Llama-3.2-1B-Instruct/sft_suffix/config_v00.00.yaml
```

> [!NOTE] 
> The config `{version}` uses a major.minor convention, where increments before the period indicate different datasets, while increments after the period indicate different training hyperparameters for the _same_ dataset. For example, `v00.00` and `v00.01` train on the same dataset, while `v00.00` and `v01.00` refer to training on different datasets.

By default, these scripts will push each model to the  `hf-cmu-collab` org i.e. `hf-cmu-collab/{model_name}-{task}-{version}`. You can override the parameters in each YAML config by appending them to the command as follows:

```shell
# Change batch size, number of epochs etc
ACCELERATE_LOG_LEVEL=info accelerate launch --config_file finetune/experiments/accelerate_configs/zero3.yaml finetune/scripts/sft_conversations.py --config finetune/experiments/Llama-3.2-1B-Instruct/sft_conversations/config_v00.00.yaml --per_device_train_batch_size=42 --num_train_epochs=5
```

## Launching jobs on a Slurm cluster

If you have access to a Slurm cluster, we provide a `experiments/launch.slurm` script that will automatically queue training jobs for you. Here's how you can use it:

```shell
sbatch --job-name=hf-cmu-{task} --nodes=1 recipes/launch.slurm {task} {model_name} {version} {accelerator}
```

Here `{model_name}` and `{task}` are defined as above, while `{version}` refers to the YAML config suffix (e.g. `v00.00`) and `{accelerator}` refers to the choice of 🤗 Accelerate config in `experiments/accelerate_configs`. If you wish to override the default config parameters, you can provide them by appending a space-separated string like `'--arg1=value1 --arg2=value2'. Here's a concrete example to run SFT on 1 node of 8 GPUs:

```shell
# Launch on Slurm and override default hyperparameters
sbatch --job-name=hf-cmu-sft-conversations --nodes=1 finetune/launch.slurm sft_conversations Llama-3.2-1B-Instruct v00.00 zero3 '--per_device_train_batch_size=42 --num_train_epochs=5'
```

You can scale the number of nodes by increasing the `--nodes` flag.

**⚠️ Note:** the configuration in `experiments/launch.slurm` is optimised for the Hugging Face Compute Cluster and may require tweaking to be adapted to your own compute nodes.