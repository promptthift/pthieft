import pandas as pd
import ast
from diffusers import StableDiffusionPipeline
import torch
import os

def main():
    pipeline = StableDiffusionPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4",
        torch_dtype=torch.float16,
        cache_dir="data/stable-diffusion"
    )
    pipeline = pipeline.to('cuda')

    csv_path = 'output/PS_results/prompt_stealer_results.csv'
    df = pd.read_csv(csv_path)
    df['target_modifiers'] = df['target_modifiers'].apply(ast.literal_eval)
    df['pred_modifiers'] = df['pred_modifiers'].apply(ast.literal_eval)
    df['semantic_sim'] = df['semantic_sim'].astype(float)
    df['modifier_sim'] = df['modifier_sim'].astype(float)

    num_samples = 100
    out_dir = 'output/images/adv_vis'
    os.makedirs(out_dir, exist_ok=True)
    for idx, row in df.iterrows():
        if idx > num_samples:
            break

        prompt = row['prompt']
        inferred_prompt = row['inferred_prompt']
        with torch.no_grad():
            img = pipeline(prompt).images[0]
            inferred_img = pipeline(inferred_prompt).images[0]
        img.save(os.path.join(out_dir, f'{str(idx).zfill(3)}_orig.jpg'))
        inferred_img.save(os.path.join(out_dir, f'{str(idx).zfill(3)}_infer.jpg'))

if __name__ == "__main__":
    main()