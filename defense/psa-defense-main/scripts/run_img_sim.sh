export CUDA_VISIBLE_DEVICES=0
source ~/anaconda3/etc/profile.d/conda.sh

conda activate stable-diffusion
python evaluate/eval_img_sim.py
python evaluate/eval_img_sim.py --wb=0
python evaluate/eval_img_sim.py --wb=1