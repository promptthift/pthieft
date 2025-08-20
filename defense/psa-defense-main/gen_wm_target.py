import torch
import os
import argparse
import pandas as pd
import ast
from diffusers import StableDiffusionPipeline
from eval_token import get_freq
from eval_PromptStealer import get_dataset
from utils import build_prompt_with_saved_cap, artists
from copy import deepcopy

import open_clip
import random
from torchvision import transforms

def active_tokens(clip_model, tokenizer, preprocess, image,
                  highfreq_list, modifier10_target, num_gens):

    sim_list = []
    # target_len = min(len(modifier10_target), 5)
    # target_len = max(target_len, 1)
    target_len = 10
    for modifier in highfreq_list:
        # if modifier in modifier10_target:
        #     sim_list.append(0)
        # else:
            with torch.no_grad():
                image_features = clip_model.encode_image(preprocess(image).unsqueeze(0).to('cuda'), normalize=True)
                text_inputs = tokenizer([modifier]).to('cuda')
                text_features = clip_model.encode_text(text_inputs, normalize=True)
                similarity = (image_features @ text_features.T).cpu().numpy()[0][0]
            sim_list.append(similarity)

    top_indices = sorted(range(len(sim_list)), key=lambda i: sim_list[i], reverse=True)[:target_len*3] # *5]
    highfreq_top = [highfreq_list[i] for i in top_indices]
    highfreq_selected = []
    for _ in range(num_gens):
        highfreq_selected.append(random.sample(highfreq_top, target_len))

    return highfreq_selected

def main(args):
    pipeline = StableDiffusionPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4",
        torch_dtype=torch.float16,
        cache_dir="data/stable-diffusion"
    )
    pipeline = pipeline.to('cuda')
    pipeline.safety_checker = None

    # clip_model, _, preprocess = open_clip.create_model_and_transforms(
    #     'ViT-H-14',
    #     pretrained='laion2b_s32b_b79k',
    #     cache_dir='data/clip',
    #     device='cuda'
    # )
    # tokenizer = open_clip.get_tokenizer('ViT-H-14')
    # preprocess.transforms.insert(0, transforms.ToPILImage())

    threshold = 1000
    num_gens = 10
    num_sample = args.num_samples
    freq_dict = get_freq()
    testset = get_dataset()
    num_gens = len(testset) if len(testset) < num_gens else num_gens

    if args.wb or args.bb_gen_target:
        csv_path = f'output/PS_results/eps{args.eps}/prompt_stealer_results_wb.csv' if args.wb else \
                    f'output/PS_results/eps{args.eps}/prompt_stealer_results_bb.csv'
        df = pd.read_csv(csv_path)
        df['pred_modifiers'] = df['pred_modifiers'].apply(ast.literal_eval)

    csv_path = f'output/PS_results/prompt_stealer_results.csv'
    df_nodefense = pd.read_csv(csv_path)
    df_nodefense['pred_modifiers'] = df_nodefense['pred_modifiers'].apply(ast.literal_eval)

    save_path = f'output/eps{args.eps}/lexica_wm_target_wb' if args.wb else \
                f'output/eps{args.eps}/lexica_wm_target_bb'
    os.makedirs(save_path, exist_ok=True)

    # highfreq_dict = [modifier for modifier in freq_dict.keys() if freq_dict[modifier] > threshold]

    for i in range(num_sample):
        image, prompt, modifier10_vector, id = testset[i]
        subject = prompt.split(',')[0]
        image = image.to('cuda')

        modifier10 = testset.getCategoryListByArray(modifier10_vector)

        if args.wb:
            # original prompt without protection
            prompt = build_prompt_with_saved_cap(
                subject, 
                [key for key in df_nodefense['pred_modifiers'][i].keys()],
                artists
            )

            # build target list
            modifier10_target = [key for key in df['pred_modifiers'][i].keys()]
            prompt_target = build_prompt_with_saved_cap(subject, modifier10_target, artists)
            # generate
            with torch.no_grad():
                images_target = pipeline(prompt_target, num_images_per_prompt=2 * num_gens).images
                images_origin = pipeline(prompt, num_images_per_prompt=2 * num_gens).images
            # save
            for idx, image in enumerate(images_target):
                image.save(os.path.join(save_path, f'{str(i).zfill(4)}_target_{str(idx).zfill(2)}.png'))
            for idx, image in enumerate(images_origin):
                image.save(os.path.join(save_path, f'{str(i).zfill(4)}_origin_{str(idx).zfill(2)}.png'))
        elif not args.wb and not args.bb_gen_target:
            # original prompt without protection
            prompt = build_prompt_with_saved_cap(
                subject, 
                [key for key in df_nodefense['pred_modifiers'][i].keys()],
                artists
            )

            # build
            modifier10_recons = []
            for modifier in modifier10:
                if freq_dict[modifier] > threshold:
                    modifier10_recons.append(modifier)
            prompt_recons = build_prompt_with_saved_cap(subject, modifier10_recons, artists)
            # modifier10_target_list = active_tokens(clip_model, tokenizer, preprocess, image, highfreq_dict, modifier10_target, num_gens)

            # prompt_target = []
            # for modifier10_target in modifier10_target_list:
            #     prompt_target.append(build_prompt_with_saved_cap(subject, modifier10_target, artists))
            # generate
            with torch.no_grad():
                images_target = pipeline(prompt_recons, num_images_per_prompt=num_gens).images
                images_origin = pipeline(prompt, num_images_per_prompt=2 * num_gens).images
                # images_nodefense = pipeline(prompt_nodefense, num_images_per_prompt=num_gens).images
            # save
            for idx, image in enumerate(images_target):
                image.save(os.path.join(save_path, f'{str(i).zfill(4)}_target_{str(idx).zfill(2)}.png'))
            for idx, image in enumerate(images_origin):
                image.save(os.path.join(save_path, f'{str(i).zfill(4)}_origin_{str(idx).zfill(2)}.png'))
            # for idx, image in enumerate(images_nodefense):
            #     image.save(os.path.join(save_path, f'{str(i).zfill(4)}_origin_{str(idx + num_gens).zfill(2)}.png'))
        elif not args.wb and args.bb_gen_target:
            # build target list
            modifier10_target = [key for key in df['pred_modifiers'][i].keys()]
            prompt_target = build_prompt_with_saved_cap(subject, modifier10_target, artists)
            # generate
            with torch.no_grad():
                images_target = pipeline(prompt_target, num_images_per_prompt=num_gens).images
            # save
            for idx, image in enumerate(images_target):
                image.save(os.path.join(save_path, f'{str(i).zfill(4)}_target_{str(idx + num_gens).zfill(2)}.png'))



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--wb', type=int, default=1)
    parser.add_argument('--eps', type=float, default=0.05)
    parser.add_argument('--num_samples', type=int, default=1000)
    parser.add_argument('--bb_gen_target', type=int, default=0)
    
    args = parser.parse_args()
    main(args)