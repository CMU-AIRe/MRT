#!/bin/bash
#SBATCH --gres=gpu:L40S:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time 48:00:00
#SBATCH --partition=general

# ray stop --force && ray start --head

folder=$1
output_path=/home/myang4/mrt-analysis/outputs/regret/$folder/pass_at_k
logs_path=/home/myang4/mrt-analysis/logs/regret/$folder/pass_at_k

model=$2
batch_number=$3

export NCCL_P2P_DISABLE=1

source ~/miniconda3/etc/profile.d/conda.sh
conda activate mrt

cd /home/myang4/mrt-analysis

python code/pass_at_k_extrapolate.py \
    --input_dataset_name hf-cmu-collab/math_reasoning_benchmark \
    --dataset_split AIME2025 \
    --dataset_start $((batch_number * 5)) \
    --dataset_end $(((batch_number + 1) * 5)) \
    --output_path $output_path \
    -K 16 \
    -W 8 \
    -R 8 \
    --model $model \
    --temperature 0.6 \
    --top_p 0.95 \
    --max_tokens 16384 > $logs_path/pass_at_k_$batch_number.log 2>&1
