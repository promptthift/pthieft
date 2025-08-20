export CUDA_VISIBLE_DEVICES=0
source ~/anaconda3/etc/profile.d/conda.sh

export diffeps=(0.01 0.03 0.10 0.15)

conda activate stable-diffusion
# python gen_wm_target.py --wb=0 --eps=0.05 --num_samples=100 # 1000

conda activate psa
# python gen_adv_bb.py --eps=0.05 --num_samples=100 # 1000 --eps=0
python eval_adv.py --wb=0 --eps=0.05 --num_samples=100 # 1000 --eps=0

# conda activate stable-diffusion
python gen_wm_target.py --wb=0 --eps=0.05 --num_samples=100 --bb_gen_target=1

python gen_PromptStealer.py --wb=0
python train_wm_detect_clip.py --log_name="lexica_wmbb" --wb=0 --eps=0.05 --num_samples=100