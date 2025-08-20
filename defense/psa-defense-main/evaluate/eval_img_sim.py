import argparse
import pandas as pd
import ast
import torch
from diffusers import StableDiffusionPipeline
from transformers import CLIPProcessor, CLIPModel
import torch.nn.functional as F
import numpy as np
import os
import logging

def get_logger(log_name):
    os.makedirs('./output/logger', exist_ok=True)
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(f'./output/logger/{log_name}.txt', mode='a')
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger

def main(args):
    logger = get_logger(args.log_name)
    pipeline = StableDiffusionPipeline.from_pretrained(
        # "CompVis/stable-diffusion-v1-4",
        "stabilityai/stable-diffusion-2-1-base",
        torch_dtype=torch.float16,
        cache_dir="data/stable-diffusion"
    )
    pipeline.safety_checker = None
    pipeline = pipeline.to('cuda')
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", cache_dir="data/clip")
    clip_model = clip_model.to('cuda')
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", cache_dir="data/clip")

    threshold = args.threshold
    csv_path = 'output/PS_results/prompt_stealer_results.csv'
    if args.wb == 1:
        csv_path = 'output/PS_results/prompt_stealer_results_wb.csv'
    elif args.wb == 0:
        csv_path = 'output/PS_results/prompt_stealer_results_bb.csv'
    df = pd.read_csv(csv_path)
    df['target_modifiers'] = df['target_modifiers'].apply(ast.literal_eval)
    df['pred_modifiers'] = df['pred_modifiers'].apply(ast.literal_eval)
    df['semantic_sim'] = df['semantic_sim'].astype(float)
    df['modifier_sim'] = df['modifier_sim'].astype(float)

    feat_sim_total, pixel_sim_total, count = 0, 0, 0
    for idx, row in df.iterrows():
        if idx >= args.num_samples:
            break
        target_modifiers = row['target_modifiers']
        target_subject = row['prompt'].split(',')[0]
        target_prompt = row['prompt']
        pred_modifiers = row['pred_modifiers']
        inferred_prompt = row['inferred_prompt']
        with torch.no_grad():
            images = pipeline([target_prompt, inferred_prompt]).images
            inputs = processor(images=images, return_tensors="pt", padding=True)
            inputs = {key: value.to('cuda') for key, value in inputs.items()}
            features = clip_model.get_image_features(**inputs)

        feat_sim_total += F.cosine_similarity(features[0].unsqueeze(0), features[1].unsqueeze(0))
        array1 = np.array(images[0]).astype(np.float32).flatten()
        array2 = np.array(images[1]).astype(np.float32).flatten()
        tensor1 = torch.tensor(array1)
        tensor2 = torch.tensor(array2)
        pixel_sim_total += F.mse_loss(tensor1, tensor2)
        count += 1

    feat_sim = feat_sim_total / count
    pixel_sim = pixel_sim_total / count

    logger.info(f'--wb {args.wb}, feat_sim {feat_sim}, pixel_sim {pixel_sim}')
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_samples', type=int, default=1000)
    parser.add_argument('--threshold', type=int, default=1000)
    parser.add_argument('--wb', type=int, default=None)
    parser.add_argument('--log_name', type=str, default='test')
    args = parser.parse_args()

    main(args)