#!/bin/bash
#SBATCH --job-name=sft_one          
#SBATCH --output=sft_one.log           
#SBATCH --error=sft_one.log                
#SBATCH --time=150:00:00                       
#SBATCH --cpus-per-task=4                    
#SBATCH --ntasks=1                           
#SBATCH --mem-per-cpu=16G                    
#SBATCH --gres=gpu:1    


module load conda
conda activate grpo

export CUDA_VISIBLE_DEVICES=0, 

export CUDA_LAUNCH_BLOCKING=1 

export TORCH_USE_CUDA_DSA=1

swift sft \
    --model /fred/oz339/zdeng/output/new_prom/v0-20250417-115129/checkpoint-3000 \
    --model_type llava1_5_hf \
    --train_type full \
    --dataset '/fred/oz337/zdeng/prompt_stealing_ours/data/lexica_dataset/train888_newprom.json' \
    --split_dataset_ratio 0.0001 \
    --torch_dtype bfloat16 \
    --num_train_epochs 10 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
    --learning_rate 1e-5 \
    --gradient_accumulation_steps 1 \
    --eval_steps 100000 \
    --save_steps 1000 \
    --save_total_limit 100 \
    --logging_steps 5 \
    --max_length 1024 \
    --output_dir /fred/oz339/zdeng/output/new_prom \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --model_author swift \
    --model_name swift-robot
