import datasets
import torch
from diffusers import StableDiffusionPipeline
from utils import build_prompt_with_saved_cap, artists
from transformers import CLIPProcessor, CLIPModel
import numpy as np
import json
import os
from tqdm import tqdm
from eval_token import get_freq
from cal_tokens_kernelshap import sample_subsets, marginal_contribution, shap_kernel, LinearRegression
import random

def cal_sim(processor, clip_model, modifier, images):
    inputs = processor(text=['a photo of ' + modifier], images=images, return_tensors="pt", padding=True)
    inputs = inputs.to('cuda')
    with torch.no_grad():
        outputs = clip_model(**inputs)

    image_features = outputs.image_embeds
    text_features = outputs.text_embeds
    # image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
    # text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
    # similarity = torch.matmul(image_features, text_features.T)
    # average_similarity = similarity.mean().item()
    # return average_similarity

    return ((image_features - text_features) ** 2).mean().item()

def cal_spearman(orig_dict_list, dist_dict_list):
    
    total, count = 0, 0
    for i in orig_dict_list.keys():
        orig_dict = orig_dict_list[str(i)]
        dist_dict = dist_dict_list[str(i)]
        orig_dict = {key: abs(orig_dict[key]) for key in orig_dict.keys()}
        if len(orig_dict) <= 2:
            continue

        keys = list(orig_dict.keys())
        if 'None' in keys:
            keys.remove('None')
        values1 = torch.tensor([orig_dict[key] for key in keys])
        values2 = torch.tensor([dist_dict[key] for key in keys])
        rank1 = torch.argsort(torch.argsort(values1))
        rank2 = torch.argsort(torch.argsort(values2))
        n = len(rank1)
        diff_squared = (rank1 - rank2).float().pow(2).sum()
        spearman_corr = 1 - (6 * diff_squared) / (n * (n**2 - 1))
        count += 1
        total += spearman_corr.item()
        print(f'{len(orig_dict)} {spearman_corr.item()}')
    total /= count

    return total


def kernel_shap(pipeline, clip_model, processor, subject, modifier10):
    
    M = len(modifier10)  # 特征数
    nsamples = 2 * M
    # subsets = balanced_sampling(M, nsamples)
    subsets = sample_subsets(M, nsamples)
    shap_values = []
    weights = []
    contributions = []

    # 遍历每个子集
    for subset in subsets:
        contribution = marginal_contribution(
            pipeline,
            clip_model,
            processor,
            subject,
            modifier10,
            subset
        )
        weight = shap_kernel(M, np.sum(subset))  # 核函数权重
        contributions.append(contribution)
        weights.append(weight)
    
    # 构建线性回归
    contributions = np.array(contributions)
    weights = np.array(weights)
    subsets = np.array(subsets)

    # 线性回归求解 Shapley 值
    reg = LinearRegression()
    reg.fit(subsets, contributions, sample_weight=weights)
    shap_values = reg.coef_  # Shapley 值即为回归系数
    
    return shap_values

if __name__ == '__main__':
    trainset = datasets.load_dataset('vera365/lexica_dataset', split='train', cache_dir='data/lexica_dataset')
    testset = datasets.load_dataset('vera365/lexica_dataset', split='test', cache_dir='data/lexica_dataset')
    fullset = datasets.concatenate_datasets([trainset, testset])
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
    save_path = 'output/stats'
    os.makedirs(save_path, exist_ok=True)

    freq_dict = get_freq()
    dict_path = os.path.join(save_path, 'eff_kernelshap_dict.csv')
    if os.path.exists(dict_path):
        with open(dict_path, 'r') as f:
            shap_dict = json.load(f)
    else:
        shap_dict = {k: None for k in freq_dict.keys()}
    for idx, modifier in tqdm(enumerate(freq_dict.keys())):
        if shap_dict[modifier] is not None:
            # print(f"Found {idx}! skip")
            continue
        for sample in fullset:
            subject = sample['subject']
            modifier10 = sample['modifier10']
            if modifier in modifier10:
                if len(modifier10) > 50:
                    modifier10 = [i for i in modifier10 if i != modifier]
                    modifier10 = random.sample(modifier10, 9)
                    modifier10.append(modifier)
                print(f'calculating {modifier}...')
                shapley_value = kernel_shap(pipeline, clip_model, processor, subject, modifier10)
                for m, v in zip(modifier10, shapley_value):
                    shap_dict[m] = v
                    print(f'{m} {v}')
                break

        with open(dict_path, 'w') as f:
            json.dump(shap_dict, f)
