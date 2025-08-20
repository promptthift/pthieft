#!/bin/bash
#SBATCH --job-name=NewPrompt          
#SBATCH --output=NewPrompt.log           
#SBATCH --error=NewPrompt.log                
#SBATCH --time=100:00:00                       
#SBATCH --cpus-per-task=4                    
#SBATCH --ntasks=1                           
#SBATCH --mem-per-cpu=16G                    
#SBATCH --gres=gpu:2    


module load conda
conda activate grpo

CUDA_VISIBLE_DEVICES=0,1, \

CUDA_LAUNCH_BLOCKING=1 \

swift sft \
    --model /fred/oz337/zdeng/ms-swift/llava-1.5-7b-hf/models/swift/llava-1.5-7b-hf \
    --model_type llava1_5_hf \
    --train_type full \
    --dataset '/fred/oz337/zdeng/prompt_stealing_ours/data/lexica_dataset/train999.json' \
    --split_dataset_ratio 0.0001 \
    --torch_dtype bfloat16 \
    --num_train_epochs 10 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 4 \
    --learning_rate 1e-5 \
    --gradient_accumulation_steps 1 \
    --eval_steps 100000 \
    --save_steps 1000 \
    --save_total_limit 100 \
    --logging_steps 5 \
    --max_length 256 \
    --output_dir /fred/oz339/zdeng/output/sft_subject \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --model_author swift \
    --model_name swift-robot
