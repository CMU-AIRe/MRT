In this example, we will run the regret comparison in the paper between the three variants of DeepScaleR-1.5B-Preview (Base, GRPO, MRT).

### Generate pass at k for different models
1. `bash scripts/regret/batch_pass_at_k.sh`
    1.1. For each model that you want to compare, add it to this file (see example)
2. Step through `break_into_meta_steps_batched.ipynb`, with the appropriate `output_dir`, `max_steps`, and `group_size`.
    2.1 `max_steps` is the number of episodes at which we stop breaking the response into further episodes
    2.2 `group_size` controls how many episodes are grouped together
3. `bash scripts/regret/batch_pass_at_k_given_prefix.sh`

### Plot
1. Step through `regret.ipynb` with the appropriate `output_dir` and `folders`.

