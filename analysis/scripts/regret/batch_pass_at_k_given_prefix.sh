folder=("base" "mrt" "grpo")
model=("agentica-org/DeepScaleR-1.5B-Preview" "hf-cmu-collab/DeepScaleR-1.5B-Preview_alpha_Iteration1_MRT_RL_new" "hf-cmu-collab/DeepScaleR-1.5B-Preview_without_info_gain_Iteration1")

for ((i = 0; i < ${#folder[@]}; i++)); do
    for batch_number in 0 1 2; do
        sbatch /home/myang4/mrt-analysis/scripts/regret/pass_at_k_given_prefix.sh ${folder[$i]} ${model[$i]} $batch_number
    done
done
