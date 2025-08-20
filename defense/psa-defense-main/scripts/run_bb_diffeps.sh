export CUDA_VISIBLE_DEVICES=0
source ~/anaconda3/etc/profile.d/conda.sh

export diffeps=(0.01 0.03 0.10 0.15)
for eps in "${diffeps[@]}"; do
    conda activate stable-diffusion
    python gen_wm_target.py --wb=0 --eps=$eps --num_samples=100

    conda activate psa
    python gen_adv_bb.py --eps=$eps --num_samples=100
    python eval_adv.py --wb=0 --eps=$eps --num_samples=100

    conda activate stable-diffusion
    python gen_PromptStealer.py --wb=0 --eps=$eps --num_samples=100
    python train_wm_detect_clip.py --wb=0 --log_name="lexica_wmbb" --eps=$eps --num_samples=100
done