#!/bin/bash
    # --validation_data_path /fred/oz337/zdeng/prompt_stealing_ours/data/lexica_dataset/test.json \
        # --bits 16 \
        #  openai/clip-vit-large-patch14-336
        # llava_llama_2
# Set the prompt and model versions directly in the command
# HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 \

# nohup srun --time=4:0:0 --cpus-per-task=2 --ntasks=1 --mem-per-cpu=8G --gres=gpu:1 \
export CUDA_VISIBLE_DEVICES=0
source ~/anaconda3/etc/profile.d/conda.sh

conda activate llava

python ./LLaVA/llava/eval/model_vqa.py \
    --model-path mark941101/llava-2-7b-chat-task-full-2 \
    --question-file data/qs_adv_wb.json \
    --image-folder data/lexica_adv_wb_eps0.05 \
    --answers-file \
    output/PS_results/answer-file-our_wb.jsonl > inference.log 2>&1 &

python ./LLaVA/llava/eval/model_vqa.py \
    --model-path mark941101/llava-2-7b-chat-task-full-2 \
    --question-file data/qs_adv_bb.json \
    --image-folder data/lexica_adv_bb_eps0.05 \
    --answers-file \
    output/PS_results/answer-file-our_bb.jsonl > inference.log 2>&1 &