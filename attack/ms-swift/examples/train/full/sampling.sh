# If you are using the validation set for inference, add the parameter `--load_data_args true`.
# /fred/oz339/zdeng/output/lora/vllm1v1/v0-20250226-164822/checkpoint-82400-merged 
#  /fred/oz339/zdeng/output/lora/BLUE/v1-20250729-161921/checkpoint-10000-merged
CUDA_VISIBLE_DEVICES=0 \

swift sample \
    --model /fred/oz339/zdeng/output/lora/vllm1v1_jac/v2-20250227-072806/checkpoint-58000-merged \
    --sampler_engine pt \
    --num_return_sequences 10 \
    --dataset /fred/oz337/zdeng/prompt_stealing_ours/data/lexica_dataset/test_sample.json \