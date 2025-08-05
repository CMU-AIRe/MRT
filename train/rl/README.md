# Termination-Reward Baseline for RL in Open-R1 and VERL

This README provide guidelines for modifying **[Hugging Face Open-R1](https://github.com/huggingface/open-r1)** or **[VERL (Volcengine RL)](https://github.com/volcengine/verl)** to use **MRT** instead of the standard **group-based baseline** in reinforcement learning (RL) fine-tuning for language models.

## Overview

Standard GRPO-based or PPO-based RL methods in Open-R1 and VERL compute **advantages** using a group baseline, typically the mean reward across sampled rollouts in a group:

$$
A(s, a) = R(s, a) - R_{group}
$$

where $R(s, a)$ is the reward for a given rollout, and $R_{group}$ is the average reward across rollouts for the same prompt.

In **MRT**, we replace the group-based baseline with a **termination reward baseline**:

$$
A(s, a) = R(s, a) - R_{termination}
$$

where `R_termination` is the reward assigned to the termination action. This allows the model to optimize relative to a stable, per-sequence baseline rather than cross-rollout averages.

## Implementation

### 1. Clone Open-R1 or VERL

```bash
git clone https://github.com/huggingface/open-r1.git
# or
git clone https://github.com/volcengine/verl.git
```

### 2. Modify the Reward Function

Locate the reward computation script:

Open-R1: `open_r1/reward/reward_fn.py` (or your custom reward provider)

VERL: `verl/rewards/base_reward.py` (extend or modify an existing reward class)

Replace or wrap the existing compute_rewards() function:

```
def compute_rewards(samples, **kwargs):
    # Original rollout rewards
    rollout_rewards = get_original_rewards(samples)

    # Compute termination reward
    termination_rewards = get_termination_rewards(samples)

    # Use termination reward as baseline
    advantages = [r - term for r, term in zip(rollout_rewards, termination_rewards)]

    return rollout_rewards, advantages
```

Ensure that advantages is returned or passed to the trainer instead of recomputing group-based baselines.

### 3. Disable Group-Based Baseline in Advantage Computation

In the trainer:

Open-R1: Edit `open_r1/trainer/ppo_trainer.py` (look for compute_advantages).

VERL: Edit `verl/algorithms/grpo.py or ppo.py`.

```
adv = rewards - rewards.mean()  # group-based baseline
```

with:

```
adv = rewards - termination_rewards
```

Make sure termination_rewards is correctly batched and broadcasted.

### 4. Training

Run RL as usual, specifying your modified reward function:

```
# Open-R1 example
python train_rl.py --reward_fn termination_baseline_reward

# VERL example
python examples/grpo_trainer/run_termination_baseline.sh
```
