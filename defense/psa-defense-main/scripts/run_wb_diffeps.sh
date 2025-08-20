export CUDA_VISIBLE_DEVICES=1
source ~/anaconda3/etc/profile.d/conda.sh

export diffeps=(0.01 0.03 0.10 0.15)
for eps in "${diffeps[@]}"; do
    conda activate psa
    python gen_adv_wb.py --eps=$eps --num_samples=100
    python eval_adv.py --wb=1 --eps=$eps --num_samples=100

    conda activate stable-diffusion
    python gen_wm_target.py --wb=1 --eps=$eps --num_samples=100
    python train_wm_detect_clip.py --log_name="lexica_wmwb" --wb=1 --eps=$eps --num_samples=100
done