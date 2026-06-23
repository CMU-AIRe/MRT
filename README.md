# Optimizing Test-Time Compute via Meta Reinforcement Finetuning

Code for *Optimizing Test-Time Compute via Meta Reinforcement Finetuning* (MRT). MRT adds
a dense **progress** reward — the change in the likelihood of eventual success contributed
by each reasoning episode — on top of outcome-reward RL, training LLMs to make steady
progress and use test-time compute more efficiently.

**v0.1** ships the full RL **training + evaluation** code so the open-ended result
(Table 1, DeepSeek-R1-Distill-Qwen-1.5B) can be reproduced end to end, including a new
on-policy variant.

## ✨ MRT-online (new in v0.1)

The paper applies the progress bonus over an **off-policy stale prefix** as a single
end-of-trace scalar, and leaves as an open problem: *how to do branched rollouts from a
meta-prover policy efficiently.* **MRT-online** does exactly that — it generates the
thinking trace **on-policy**, segments episodes **online**, forks short forced-termination
branches at **each** episode boundary, and assigns a **per-episode dense progress reward**
(cheap via SGLang radix prefix caching).

Empirically, MRT-online beats outcome-reward GRPO (gain +0.62 vs +0.48) but trails the
offline single-scalar MRT (+1.20) — useful evidence that the off-policy scalar form is a
good default. Code: [`train/rl/mrt_plugin/rollout_mrt_online.py`](train/rl/mrt_plugin/rollout_mrt_online.py);
details in [`train/rl/REPRODUCTION.md`](train/rl/REPRODUCTION.md).

## Reproducing the open-ended RL result

Everything needed is in **[`train/rl/`](train/rl/)** — see
**[`train/rl/REPRODUCTION.md`](train/rl/REPRODUCTION.md)** for the full recipe (method,
assumptions, run commands, infra notes). Headline (5-benchmark avg pass@1 **gain over the
base model**, 248 training steps, eval n=64):

| | GRPO | MRT | MRT-online | paper (GRPO / MRT) |
|---|---:|---:|---:|---|
| gain over base | +0.48 | **+1.20** | +0.62 | +1.1 / +2.2 |
| ratio vs GRPO | 1.0× | **2.50×** | 1.29× | 2–3× |

The MRT/GRPO ratio (2.50×) reproduces the paper's headline 2–3×. The training code is
built on [`miles`](https://github.com/radixark/miles) (Megatron-LM + SGLang), vendored at
[`train/rl/miles/`](train/rl/miles/) in a separate commit with the MRT changes added on
top — so `git diff` against that commit is exactly the MRT delta.

## Repository layout

| path | what |
|---|---|
| [`train/rl/`](train/rl/) | **v0.1 RL training + eval** (GRPO, MRT, MRT-online) on vendored miles |
| [`train/sft/`](train/sft/) | STaR/SFT training (backtracking setting) |
| [`eval/`](eval/) | evaluation scripts for the SFT / backtracking setting |
| [`analysis/`](analysis/) | R1 scaling-curve, regret, and extrapolation analyses (paper §5, §8.5.1, Appendices D–F) |
| [`src/`](src/) | shared utilities |

## Citation

If you use our work or codebase in your research, please cite our paper:

```bibtex
@misc{qu2025optimizingtesttimecomputemeta,
      title={Optimizing Test-Time Compute via Meta Reinforcement Fine-Tuning},
      author={Yuxiao Qu and Matthew Y. R. Yang and Amrith Setlur and Lewis Tunstall and Edward Emanuel Beeching and Ruslan Salakhutdinov and Aviral Kumar},
      year={2025},
      eprint={2503.07572},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2503.07572},
}
```

## Acknowledgements & licensing

This repo is MIT-licensed. `train/rl/miles/` vendors
[radixark/miles](https://github.com/radixark/miles) (Apache-2.0); its license is preserved
at `train/rl/miles/LICENSE` and noted in [`NOTICE`](NOTICE). MRT builds on TRL and Open-R1;
we thank those teams.
