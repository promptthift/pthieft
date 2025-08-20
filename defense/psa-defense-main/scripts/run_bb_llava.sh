export CUDA_VISIBLE_DEVICES=0
source ~/anaconda3/etc/profile.d/conda.sh

conda activate stable-diffusion
python gen_PromptStealer.py --wb=0 --attack="llava"
python train_wm_detect_clip.py \
    --log_name="lexica_llava_wmbb" --resume_fc=1 \
    --wb=0 --attack="llava"