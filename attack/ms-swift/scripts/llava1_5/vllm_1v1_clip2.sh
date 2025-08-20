#!/bin/bash
#SBATCH --job-name=cliptest          
#SBATCH --output=cliptest.log           
#SBATCH --error=cliptest.log               
#SBATCH --time=160:00:00                       
#SBATCH --cpus-per-task=4                    
#SBATCH --ntasks=1                           
#SBATCH --mem-per-cpu=16G                    
#SBATCH --gres=gpu:2

module load conda
conda activate grpo

export CUDA_VISIBLE_DEVICES=0,1 

export NPROC_PER_NODE=1 

swift rlhf \
    --rlhf_type grpo \
    --model /fred/oz339/zdeng/output/new_promp_grpo_vllm_1v1_clip2/v0-20250419-215549/checkpoint-11000-merged \
    --model_type llava1_5_hf \
    --reward_funcs CLIP \
    --external_plugins /fred/oz337/zdeng/ms-swift/examples/train/grpo/plugin/plugin.py \
    --train_type lora \
    --lora_rank 32 \
    --lora_alpha 64 \
    --use_vllm true \
    --vllm_device auto \
    --vllm_gpu_memory_utilization 0.7 \
    --vllm_max_model_len 4096 \
    --target_modules all-linear \
    --torch_dtype bfloat16 \
    --dataset '/fred/oz337/zdeng/prompt_stealing_ours/data/lexica_dataset/train888_newprom1.json' \
    --max_completion_length 256 \
    --num_train_epochs 10 \
    --per_device_train_batch_size 40 \
    --per_device_eval_batch_size 40  \
    --learning_rate 1e-5 \
    --gradient_accumulation_steps 1 \
    --eval_steps 200000 \
    --save_steps 1000 \
    --save_total_limit 100 \
    --split_dataset_ratio 0.00001 \
    --logging_steps 1 \
    --max_length 1024 \
    --output_dir /fred/oz339/zdeng/output/new_promp_grpo_vllm_1v1_clip2 \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --dataset_num_proc 4 \
    --num_generations 10 \
    --temperature 1.0 \
    --log_completions true \
    --deepspeed zero3 \


