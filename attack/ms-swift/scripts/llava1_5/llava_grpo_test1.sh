#!/bin/bash
#SBATCH --job-name=llava_7b_grpo_2gpu          
#SBATCH --output=llava_7b_grpo_2gpu_80_full_lora.log           
#SBATCH --error=llava_7b_grpo_2gpu_80_full_error_lora.log               
#SBATCH --time=150:00:00                       
#SBATCH --cpus-per-task=4                    
#SBATCH --ntasks=1                           
#SBATCH --mem-per-cpu=16G                    
#SBATCH --gres=gpu:2

module load conda
conda activate grpo

export CUDA_LAUNCH_BLOCKING=1 \
export CUDA_VISIBLE_DEVICES=0,1, \
export NPROC_PER_NODE=1 \

swift rlhf \
    --rlhf_type grpo \
    --model /fred/oz337/zdeng/MS-SWIFT/output/v14-20250221-155443/checkpoint-12294 \
    --model_type llava1_5_hf \
    --reward_funcs CLIP \
    --external_plugins /fred/oz337/zdeng/ms-swift/examples/train/grpo/plugin/plugin.py \
    --train_type lora \
    --lora_rank 8 \
    --lora_alpha 64 \
    --target_modules all-linear \
    --torch_dtype bfloat16 \
    --dataset '/fred/oz337/zdeng/prompt_stealing_ours/data/lexica_dataset/train888_newprom.json' \
    --max_completion_length 1024 \
    --num_train_epochs 10 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4  \
    --learning_rate 1e-5 \
    --gradient_accumulation_steps 1 \
    --eval_steps 200000 \
    --save_steps 100 \
    --save_total_limit 25 \
    --logging_steps 1 \
    --max_length 2048 \
    --output_dir /fred/oz339/zdeng/output/lora \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --dataset_num_proc 4 \
    --num_generations 4 \
    --temperature 0.9 \
    --log_completions true
