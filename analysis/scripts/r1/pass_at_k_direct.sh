#!/bin/bash
#SBATCH --gres=gpu:L40S:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time 48:00:00
#SBATCH --partition=general

batch_number=$1
start=$(($batch_number * 10))
end=$(($start + 10))

output_path=/home/myang4/mrt-analysis/outputs/r1/pass_at_k_direct
logs_path=/home/myang4/mrt-analysis/logs/r1/pass_at_k_direct
model=/data/group_data/rl/myang4/models--Qwen--Qwen2.5-32B-Instruct/snapshots/5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd
# replace ^ with "Qwen/Qwen2.5-32B-Instruct"

export NCCL_P2P_DISABLE=1

source ~/miniconda3/etc/profile.d/conda.sh
conda activate mrt

cd /home/myang4/mrt-analysis

python code/pass_at_k_direct.py \
    --input_dataset_name hf-cmu-collab/omni_math_filtered \
    --dataset_split test \
    --dataset_start $start \
    --dataset_end $end \
    --output_path $output_path \
    -K 32 \
    --model $model \
    --temperature 0.7 \
    --top_p 1 \
    --max_tokens 8192 \
    --tensor_parallel_size 2 \
    --batch_size 1 > $logs_path/pass_at_k_direct_$batch_number.log 2>&1

