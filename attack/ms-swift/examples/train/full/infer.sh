# If you are using the validation set for inference, add the parameter `--load_data_args true`.
CUDA_VISIBLE_DEVICES=0 \
swift infer \
    --model /fred/oz339/zdeng/output/lora/BLUE/v1-20250729-161921/checkpoint-17000-merged \
    --stream true \
    --temperature 0 \
    --val_dataset /fred/oz337/zdeng/prompt_stealing_ours/data/lexica_dataset/test_sample.json \
    --max_new_tokens 1024 \
    --temperature 0.2 
