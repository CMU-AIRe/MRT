#!/bin/bash
#SBATCH --gres=gpu:L40S:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time 48:00:00
#SBATCH --partition=general

folder=$1
output_path=/home/myang4/mrt-analysis/outputs/regret/$folder/pass_at_k_given_prefix
logs_path=/home/myang4/mrt-analysis/logs/regret/$folder/pass_at_k_given_prefix
prefix_file=/home/myang4/mrt-analysis/outputs/regret/$folder/pass_at_k/prefixes.pkl

model=$2
batch_number=$3

export NCCL_P2P_DISABLE=1

source ~/miniconda3/etc/profile.d/conda.sh
conda activate mrt

cd /home/myang4/mrt-analysis

python code/pass_at_k_given_prefix.py \
    --input_dataset_name hf-cmu-collab/omni_math_filtered \
    --dataset_split test \
    --dataset_start $((batch_number * 5)) \
    --dataset_end $(((batch_number + 1) * 5)) \
    --output_path $output_path \
    -K 8 \
    --model $model \
    --temperature 0.6 \
    --top_p 0.95 \
    --max_tokens 4096 \
    --prefix_file $prefix_file > $logs_path/pass_at_k_given_prefix_$batch_number.log 2>&1
