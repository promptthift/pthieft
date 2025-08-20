import datasets
import torch
from diffusers import StableDiffusionPipeline
from utils import build_prompt_with_saved_cap, artists
from transformers import CLIPProcessor, CLIPModel
import numpy as np
import json
import os

# MahD
def cal_mahalanobis_distance(features_origin, features_target):
    mu_origin = torch.mean(features_origin, dim=0)
    mu_target = torch.mean(features_target, dim=0)
    cov_origin = torch.cov(features_origin.T)
    cov_target = torch.cov(features_target.T)
    cov_avg = (cov_origin + cov_target) / 2
    cov_avg_inv = torch.linalg.pinv(cov_avg)
    delta_mu = mu_origin - mu_target
    mahalanobis_distance = torch.sqrt(torch.dot(delta_mu, torch.matmul(cov_avg_inv, delta_mu)))
    return mahalanobis_distance.item()

if __name__ == '__main__':
    testset = datasets.load_dataset('vera365/lexica_dataset', split='test', cache_dir='data/lexica_dataset')
    num_images = 10
    num_samples = 1000
    seed = 42
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

    dist_dict_list = {}
    for i in range(num_samples):
        sample = testset[i]
        subject = sample['subject']
        modifier10 = sample['modifier10']
        prompt_origin = sample['prompt']
        image_origin = sample['image']

        # origin
        dist_dict = {}
        with torch.no_grad():
            generator = torch.manual_seed(seed)
            images = pipeline(prompt_origin, num_images_per_prompt=num_images, generator=generator).images
            inputs = processor(images=images, return_tensors="pt", padding=True)
            inputs = {key: value.to('cuda') for key, value in inputs.items()}
            features_origin = clip_model.get_image_features(**inputs)

        # none
        with torch.no_grad():
            generator = torch.manual_seed(seed)
            images = pipeline(subject, num_images_per_prompt=num_images, generator=generator).images
            inputs = processor(images=images, return_tensors="pt", padding=True)
            inputs = {key: value.to('cuda') for key, value in inputs.items()}
            features_target = clip_model.get_image_features(**inputs)
        mahalanobis_distance = cal_mahalanobis_distance(features_origin, features_target)
        dist_dict['None'] = mahalanobis_distance
        # print(f"None: {mahalanobis_distance}")

        for modifier in modifier10:
            modifier_list = [m for m in modifier10 if not m == modifier]
            prompt_target = build_prompt_with_saved_cap(subject, modifier_list, artists)
            features_target = []

            with torch.no_grad():
                generator = torch.manual_seed(seed)
                images = pipeline(prompt_target, num_images_per_prompt=num_images, generator=generator).images
                inputs = processor(images=images, return_tensors="pt", padding=True)
                inputs = {key: value.to('cuda') for key, value in inputs.items()}
                features_target = clip_model.get_image_features(**inputs)
            mahalanobis_distance = cal_mahalanobis_distance(features_origin, features_target)
            dist_dict[modifier] = mahalanobis_distance
            # print(f"{modifier}: {mahalanobis_distance}")

        dist_dict_list[i] = dist_dict

    save_path = 'output/stats'
    os.makedirs(save_path, exist_ok=True)
    dict_path = os.path.join(save_path, 'MahD_dict.csv')
    with open(dict_path, 'w') as f:
        json.dump(dist_dict_list, f)