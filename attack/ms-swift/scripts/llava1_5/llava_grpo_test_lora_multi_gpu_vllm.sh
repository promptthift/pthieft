#!/bin/bash
#SBATCH --job-name=NewMetrixs        
#SBATCH --output=NewMetrixs.log           
#SBATCH --error=NewMetrixs.log               
#SBATCH --time=160:00:00                       
#SBATCH --cpus-per-task=4                    
#SBATCH --ntasks=1                           
#SBATCH --mem-per-cpu=32G                    
#SBATCH --gres=gpu:4

module load conda
conda activate grpo

export CUDA_VISIBLE_DEVICES=0,1,2,3 \
export NPROC_PER_NODE=3 \
export CUDA_LAUNCH_BLOCKING=1 \

swift rlhf \
    --rlhf_type grpo \
    --model /fred/oz339/zdeng/output/sft/v2-20250228-134920/checkpoint-12294 \
    --model_type llava1_5_hf \
    --reward_funcs JacBLEUContri \
    --external_plugins /fred/oz337/zdeng/ms-swift/examples/train/grpo/plugin/plugin.py \
    --use_vllm true \
    --vllm_device auto \
    --vllm_gpu_memory_utilization 0.6 \
    --vllm_max_model_len 4096 \
    --train_type full \
    --target_modules all-linear \
    --torch_dtype bfloat16 \
    --dataset '/fred/oz337/zdeng/prompt_stealing_ours/data/lexica_dataset/train6.json' \
    --split_dataset_ratio 0.00001 \
    --max_completion_length 256 \
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
    --output_dir /fred/oz339/zdeng/output/lora/BLUE_JAC_TOKEN \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --dataset_num_proc 4 \
    --num_generations 6 \
    --temperature 0.9 \
    --deepspeed zero3 \
    --log_completions true
