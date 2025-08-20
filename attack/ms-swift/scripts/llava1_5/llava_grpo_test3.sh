#!/bin/bash
#SBATCH --job-name=llava_7b_grpo_4gpu          
#SBATCH --output=llava_7b_grpo_4gpu_80_full_batch2.log           
#SBATCH --error=llava_7b_grpo_4gpu_80_full_batch2.log               
#SBATCH --time=150:00:00                       
#SBATCH --cpus-per-task=4                    
#SBATCH --ntasks=1                           
#SBATCH --mem-per-cpu=16G                    
#SBATCH --gres=gpu:4

module load conda
conda activate grpo

export CUDA_LAUNCH_BLOCKING=1 \
export CUDA_VISIBLE_DEVICES=0,1,2,3 \
export NPROC_PER_NODE=4 \

swift rlhf \
    --rlhf_type grpo \
    --model /fred/oz337/zdeng/MS-SWIFT/output/v14-20250221-155443/checkpoint-12294 \
    --model_type llava1_5_hf \
    --reward_funcs token_contribution \
    --external_plugins /fred/oz337/zdeng/ms-swift/examples/train/grpo/plugin/plugin.py \
    --train_type full \
    --target_modules all-linear \
    --torch_dtype bfloat16 \
    --dataset '/fred/oz337/zdeng/prompt_stealing_ours/data/lexica_dataset/train5_modified.json' \
    --max_completion_length 512 \
    --num_train_epochs 10 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2  \
    --learning_rate 1e-5 \
    --gradient_accumulation_steps 1 \
    --eval_steps 200000 \
    --save_steps 100 \
    --save_total_limit 25 \
    --logging_steps 1 \
    --max_length 2048 \
    --output_dir /fred/oz339/zdeng/output/new \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --dataset_num_proc 4 \
    --num_generations 8 \
    --temperature 0.9 \
    --log_completions true
