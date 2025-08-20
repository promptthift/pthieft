#!/bin/bash
#SBATCH --job-name=jacBLEcon          
#SBATCH --output=jacBLEcon.log           
#SBATCH --error=jacBLEcon.log               
#SBATCH --time=160:00:00                       
#SBATCH --cpus-per-task=4                    
#SBATCH --ntasks=1                           
#SBATCH --mem-per-cpu=32G                    
#SBATCH --gres=gpu:2

module load conda
conda activate grpo

export CUDA_VISIBLE_DEVICES=0,1 \
export NPROC_PER_NODE=1 \

swift rlhf \
    --rlhf_type grpo \
    --model /fred/oz339/zdeng/output/sftspecial/v1-20250304-225705/checkpoint-12296 \
    --model_type llava1_5_hf \
    --reward_funcs JacBLEUContri \
    --external_plugins /fred/oz337/zdeng/ms-swift/examples/train/grpo/plugin/plugin.py \
    --train_type lora \
    --lora_rank 16 \
    --lora_alpha 64 \
    --use_vllm true \
    --vllm_device auto \
    --vllm_gpu_memory_utilization 0.7 \
    --vllm_max_model_len 4096 \
    --target_modules all-linear \
    --torch_dtype bfloat16 \
    --dataset '/fred/oz337/zdeng/prompt_stealing_ours/data/lexica_dataset/train6.json' \
    --max_completion_length 256 \
    --num_train_epochs 10 \
    --per_device_train_batch_size 10 \
    --per_device_eval_batch_size 10  \
    --learning_rate 1e-5 \
    --gradient_accumulation_steps 1 \
    --eval_steps 200000 \
    --save_steps 1000 \
    --save_total_limit 100 \
    --split_dataset_ratio 0.00001 \
    --logging_steps 1 \
    --max_length 512 \
    --output_dir /fred/oz339/zdeng/output/lora/JAC_contri \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --dataset_num_proc 4 \
    --num_generations 10 \
    --temperature 1.0 \
    --log_completions true \
    --deepspeed zero3 \


