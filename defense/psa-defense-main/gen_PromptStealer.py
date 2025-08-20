import torch
import os
import pandas as pd
import ast
from diffusers import StableDiffusionPipeline
from eval_token import get_freq
from eval_PromptStealer import get_dataset
from utils import build_prompt_with_saved_cap, artists

import argparse

def main(args):
    pipeline = StableDiffusionPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4",
        torch_dtype=torch.float16,
        cache_dir="data/stable-diffusion"
    )
    pipeline = pipeline.to('cuda')
    pipeline.safety_checker = None

    num_samples = args.num_samples
    num_gens = 10
    testset = get_dataset()
    num_samples = len(testset) if len(testset) < num_samples else num_samples

    csv_path = 'output/PS_results/prompt_stealer_results.csv'
    if args.wb is not None and args.attack == 'psa':
        csv_path = f'output/PS_results/eps{args.eps}/prompt_stealer_results_wb.csv' if args.wb else \
                    f'output/PS_results/eps{args.eps}/prompt_stealer_results_bb.csv'
    elif args.wb is not None and args.attack == 'llava':
        csv_path = f'output/PS_results/eps{args.eps}/llava_results_wb.csv' if args.wb else \
                    f'output/PS_results/eps{args.eps}/llava_results_bb.csv'
    df = pd.read_csv(csv_path)

    save_path = 'output/lexica_psa'
    if args.wb is not None and args.attack == 'psa':
        save_path = f'output/eps{args.eps}/lexica_psa_wb' if args.wb else \
                    f'output/eps{args.eps}/lexica_psa_bb'
    elif args.wb is not None and args.attack == 'llava':
        save_path = f'output/eps{args.eps}/lexica_llava_wb' if args.wb else \
                    f'output/eps{args.eps}/lexica_llava_bb'
    os.makedirs(save_path, exist_ok=True)

    for i, row in df.iterrows():
        inferred_prompt = row['inferred_prompt']
        with torch.no_grad():
            images = pipeline(inferred_prompt, num_images_per_prompt=num_gens).images
        
        for idx, image in enumerate(images):
            image.save(os.path.join(save_path, f'{str(i).zfill(4)}_{str(idx).zfill(2)}.png'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--wb', type=int, default=None)
    parser.add_argument('--eps', type=float, default=0.05)
    parser.add_argument('--num_samples', type=int, default=1000)
    parser.add_argument('--attack', type=str, default='psa')
    args = parser.parse_args()
    main(args)