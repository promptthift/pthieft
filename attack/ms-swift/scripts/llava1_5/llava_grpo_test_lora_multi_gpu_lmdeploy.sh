#!/bin/bash
#SBATCH --job-name=4gpu_lmdeploy          
#SBATCH --output=4gpu_lmdeploy.log           
#SBATCH --error=4gpu_lmdeploy.log               
#SBATCH --time=150:00:00                       
#SBATCH --cpus-per-task=4                    
#SBATCH --ntasks=1                           
#SBATCH --mem-per-cpu=32G                    
#SBATCH --gres=gpu:4

module load conda
conda activate grpo

export CUDA_VISIBLE_DEVICES=0,1,2,3 \
export NPROC_PER_NODE=2 \

swift rlhf \
    --rlhf_type grpo \
    --model /fred/oz337/zdeng/MS-SWIFT/output/v14-20250221-155443/checkpoint-12294 \
    --model_type llava1_5_hf \
    --reward_funcs token_contribution \
    --external_plugins /fred/oz337/zdeng/ms-swift/examples/train/grpo/plugin/plugin.py \
    --use_lmdeploy true \
    --train_type full \
    --target_modules all-linear \
    --torch_dtype bfloat16 \
    --dataset '/fred/oz337/zdeng/prompt_stealing_ours/data/lexica_dataset/train6.json' \
    --max_completion_length 256 \
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
    --output_dir /fred/oz339/zdeng/output/lora/mixed_4gpu_vllm \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --dataset_num_proc 4 \
    --num_generations 4 \
    --temperature 1.0 \
    --top_p 0.9 \
    --top_k 5 \
    --async_generate true \
    --deepspeed zero3 \
    --log_completions true \
    --num_iterations 2 \
    --num_infer_workers 2\
