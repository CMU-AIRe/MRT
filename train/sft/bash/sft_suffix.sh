task=metaMATH_t0_n1_direct_0113
folder=backtrack
hub_model_id=hf-cmu-collab/Llama-3.1-8B-Instruct-$task

accelerate launch --config_file=finetune/experiments/accelerate_configs/fsdp.yaml --num_processes 8 finetune/scripts/sft_suffix.py \
    --model_name_or_path meta-llama/Llama-3.1-8B-Instruct \
    --dataset_name hf-cmu-collab/${task} \
    --learning_rate 1.0e-6 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 4 \
    --gradient_checkpointing \
    --max_seq_length 2048 \
    --dataset_num_proc 12 \
    --logging_steps 5 \
    --eval_strategy steps \
    --eval_steps 100 \
    --output_dir ${folder}/${task} \
    --report_to wandb \
    --push_to_hub \
    --hub_model_id ${hub_model_id} \
    --hub_private_repo true
