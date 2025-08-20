    # --lora_rank 8 \
    # --lora_alpha 64 \


#!/bin/bash
#SBATCH --job-name=llava_7b_grpo_2gpu_vllm          
#SBATCH --output=llava_7b_grpo_2gpu_80_full_lora_gpu_ablation.log           
#SBATCH --error=llava_7b_grpo_2gpu_80_full_lora_gpu_ablation.log               
#SBATCH --time=150:00:00                       
#SBATCH --cpus-per-task=4                    
#SBATCH --ntasks=1                           
#SBATCH --mem-per-cpu=16G                    
#SBATCH --gres=gpu:1

module load conda
conda activate grpo

# Since `output/vx-xxx/checkpoint-xxx` is trained by swift and contains an `args.json` file,
# there is no need to explicitly set `--model`, `--system`, etc., as they will be automatically read.
swift export \
    --adapters /fred/oz339/zdeng/output/lora/BLUE/v1-20250729-161921/checkpoint-17000 \
    --merge_lora true