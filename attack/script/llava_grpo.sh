#!/bin/bash
#SBATCH --job-name=llava_7b_grpo_no_lora           
#SBATCH --output=llava_7b_grpo_no_lora.log           
#SBATCH --error=llava_7b_grpo_no_lora.log                
#SBATCH --time=150:00:00                       
#SBATCH --cpus-per-task=4                    
#SBATCH --ntasks=1                           
#SBATCH --mem-per-cpu=16G                    
#SBATCH --gres=gpu:4

module load conda
conda activate grpo

export CUDA_LAUNCH_BLOCKING=1 \
CUDA_VISIBLE_DEVICES=0,1,2,3, \



swift rlhf \
    --rlhf_type grpo \
    --model ./output/new_prom/checkpoint-3000 \
    --model_type llava1_5_hf \
    --reward_funcs token_contribution \
    --external_plugins ./ms-swift/examples/train/grpo/plugin/plugin.py \
    --train_type full \
    --target_modules all-linear \
    --torch_dtype bfloat16 \
    --dataset './data/lexica_dataset/train.json' \
    --max_completion_length 1024 \
    --num_train_epochs 10 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --learning_rate 1e-5 \
    --gradient_accumulation_steps 1 \
    --eval_steps 200000 \
    --save_steps 1000 \
    --save_total_limit 25 \
    --logging_steps 1 \
    --max_length 2048 \
    --output_dir ./output \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --dataset_num_proc 4 \
    --num_generations 8 \
    --temperature 0.9 \
    --log_completions true
