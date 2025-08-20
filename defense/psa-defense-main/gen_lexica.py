import torch
import os
import pandas as pd
import ast
from diffusers import StableDiffusionPipeline
from eval_token import get_freq
from eval_PromptStealer import get_dataset
from utils import build_prompt_with_saved_cap, artists
import datasets
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
    testset = datasets.load_dataset("vera365/lexica_dataset", split='test', cache_dir="./data/lexica_dataset/")
    num_samples = len(testset) if len(testset) < num_samples else num_samples

    save_path = 'data/lexica_sd_v1-4'
    os.makedirs(save_path, exist_ok=True)

    for i in range(num_samples):
        sample = testset[i]
        prompt = build_prompt_with_saved_cap(sample['subject'], sample['modifier10'], artists)
        with torch.no_grad():
            image = pipeline(prompt).images[0]

        image.save(os.path.join(save_path, f'{str(i).zfill(4)}.png'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_samples', type=int, default=1000)
    args = parser.parse_args()
    main(args)