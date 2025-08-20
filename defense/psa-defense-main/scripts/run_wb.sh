export CUDA_VISIBLE_DEVICES=1
source ~/anaconda3/etc/profile.d/conda.sh

export diffeps=(0.01 0.03 0.10 0.15)

conda activate psa
# python gen_adv_wb.py --eps=0.05  --num_samples=100 # --eps=0.05
python eval_adv.py --wb=1 --eps=0.05 --num_samples=100 # --eps=0.05
# python eval_token.py --wb=1 --eps=0.05 --num_samples=100 # --eps=0.05

# conda activate stable-diffusion
python gen_wm_target.py --wb=1 --eps=0.05 --num_samples=100 # --eps=0.05

python train_wm_detect_clip.py --log_name="lexica_wmwb" --wb=1 --eps=0.05 --num_samples=100 # --eps=0.05