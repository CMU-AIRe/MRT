# Optimizing Test-Time Compute via Meta Reinforcement Finetuning

This repository contains the code for our paper titled "Optimizing Test-Time Compute via Meta Reinforcement Finetuning." In this work, we introduce a novel approach to optimizing test-time compute through meta reinforcement learning, aiming to balance the efficiency and discovery capabilities of Large Language Models (LLMs). Below is a list of code components that we plan to release for reproducing the results and methodologies discussed in our paper.

## Codebase Components

### TODO List

1. **Efficiency Analysis Code on R1-Series Model**

   - [ ] Implement and release the code used for conducting efficiency analysis on the R1-series model. This will include scripts to measure token efficiency and accuracy across different test-time compute budgets.

2. **Training Code for the STaR Variant of MRT**

   - [ ] Provide the training scripts for our Meta Reinforcement fine-Tuning (MRT) with the STaR variant. This includes the adaptations made to the standard STaR approach to integrate our meta RL objectives.

3. **Training Code for the RL Variant of MRT**

   - [ ] Release the training framework for the RL-based variant of our MRT approach. This includes detailed configuration files and training parameters to replicate our reinforcement learning experiments.

4. **Maj and pass@k Evaluation Code**

   - [ ] Develop and release the evaluation scripts for the majority vote (Maj) and pass@k metrics. These scripts will be crucial for assessing the model's performance under different configurations and test scenarios.

5. **Extrapolation/Exploitation Evaluation with Budget Forcing**
   - [ ] Release scripts for performing evaluations that test the model's capability to extrapolate and exploit under forced budget constraints. This will help in understanding how the model behaves with extended or restricted token budgets beyond the training conditions.

### Additional Notes

- The code will be made available as it is finalized and thoroughly tested to ensure reproducibility and reliability.
- Each component will include detailed instructions on how to set up the environment, run the scripts, and interpret the results.
- Please stay tuned for updates as we prepare the codebase for public release.

## Citation

If you use our work or codebase in your research, please cite our paper:

```bibtex
@misc{,
      title={Optimizing Test-Time Compute via Meta Reinforcement Finetuning},
      author={Yuxiao Qu*, Matthew Y. R. Yang*, Amrith Setlur, Lewis Tunstall, Edward Emanuel Beeching, Ruslan Salakhutdinov, Aviral Kumar},
      year={2025},
      eprint={},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={},
}
```
