#!/bin/bash
#SBATCH --job-name=sft_special           
#SBATCH --output=sft_special.log           
#SBATCH --error=sft_special.log                
#SBATCH --time=50:00:00                       
#SBATCH --cpus-per-task=4                    
#SBATCH --ntasks=1                           
#SBATCH --mem-per-cpu=16G                    
#SBATCH --gres=gpu:1    


module load conda
conda activate grpo

CUDA_VISIBLE_DEVICES=0, \


swift sft \
    --model /fred/oz339/zdeng/output/sft/v4-20250304-185927/checkpoint-5000 \
    --model_type llava1_5_hf \
    --train_type full \
    --dataset '/fred/oz337/zdeng/prompt_stealing_ours/data/lexica_dataset/test.json' \
    --split_dataset_ratio 0.0001 \
    --torch_dtype bfloat16 \
    --num_train_epochs 4 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --learning_rate 1e-5 \
    --gradient_accumulation_steps 1 \
    --eval_steps 100000 \
    --save_steps 1000 \
    --save_total_limit 5 \
    --logging_steps 5 \
    --max_length 512 \
    --output_dir /fred/oz339/zdeng/output/sftspecial \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --model_author swift \
    --model_name swift-robot
