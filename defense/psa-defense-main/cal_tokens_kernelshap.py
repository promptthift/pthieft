import datasets
import torch
from diffusers import StableDiffusionPipeline
from utils import build_prompt_with_saved_cap, artists
from transformers import CLIPProcessor, CLIPModel
import numpy as np
import json
import os
from sklearn.linear_model import LinearRegression
import math
from tqdm import tqdm

def shap_kernel(M, subset_size):
    if subset_size == 0 or subset_size == M:
        return 1e-5 # 0
    return (M - 1) / (np.math.comb(M, subset_size) * subset_size * (M - subset_size))

def sample_subsets(M, nsamples):
    subsets = []
    for _ in range(nsamples):
        subset = np.random.choice([0, 1], size=M, p=[0.5, 0.5])  # 随机选择特征是否包含
        subsets.append(subset)
    return np.array(subsets)

def marginal_contribution(
    pipeline,
    clip_model,
    processor,
    subject,
    modifier10,
    subset
):
    # 有特征时的模型预测
    included = [a for a, b in zip(modifier10, subset == 0) if b]
    prompt_included = build_prompt_with_saved_cap(subject, included, artists)
    
    # 无特征时的模型预测
    excluded = [a for a, b in zip(modifier10, subset == 1) if b]
    prompt_excluded = build_prompt_with_saved_cap(subject, excluded, artists)

    with torch.no_grad():
        images = pipeline([prompt_included, prompt_excluded]).images
        inputs = processor(images=images, return_tensors="pt", padding=True)
        inputs = {key: value.to('cuda') for key, value in inputs.items()}
        features = clip_model.get_image_features(**inputs)
    
    return ((features[0] - features[1]) ** 2).mean().item()

def balanced_sampling(n_features, n_samples):
    """
    生成特征采样矩阵，确保每个特征在 n_samples 中被采样到 n_samples / 2 次，
    并且每次采样中至少有一个特征被采样到。
    
    参数：
        n_features (int): 特征数量。
        n_samples (int): 采样次数，必须为偶数。
    
    返回：
        numpy.ndarray: 大小为 (n_samples, n_features) 的采样矩阵，
        其中 1 表示特征被采样，0 表示未被采样。
    """
    if n_samples % 2 != 0:
        raise ValueError("n_samples 必须是偶数，以确保均匀采样。")
    
    # 初始化采样矩阵
    samples = np.zeros((n_samples, n_features), dtype=int)
    
    # 每个特征被采样的目标次数
    target_count = n_samples // 2
    
    # 初始化每个特征的采样次数
    feature_counts = np.zeros(n_features, dtype=int)
    
    for i in range(n_samples):
        # 可采样特征：当前采样次数未达到目标次数的特征
        available_features = np.where(feature_counts < target_count)[0]
        
        # 确保至少采样一个特征
        print(len(available_features))
        n_to_sample = np.random.randint(1, len(available_features) + 1)
        
        # 从可采样特征中随机选择 n_to_sample 个特征
        sampled_features = np.random.choice(available_features, size=n_to_sample, replace=False)
        
        # 更新采样矩阵和采样计数
        samples[i, sampled_features] = 1
        feature_counts[sampled_features] += 1
    
    # 验证所有特征采样次数是否符合要求
    print(samples)
    if not np.all(feature_counts == target_count):
        raise ValueError("采样分布未满足每个特征被采样 n_samples / 2 次的条件。")
    
    return samples

def kernel_shap(pipeline, clip_model, processor, sample):
    subject = sample['subject']
    modifier10 = sample['modifier10']
    prompt_origin = sample['prompt']
    image_origin = sample['image']
    
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
    testset = datasets.load_dataset('vera365/lexica_dataset', split='test', cache_dir='data/lexica_dataset')
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
    for i in tqdm(range(num_samples)):
        sample = testset[i]

        shapley_value = kernel_shap(pipeline, clip_model, processor, sample)
        dist_dict = {}
        for m, v in zip(sample['modifier10'], shapley_value):
            dist_dict[m] = v
        dist_dict_list[i] = dist_dict
    
    save_path = 'output/stats'
    os.makedirs(save_path, exist_ok=True)
    dict_path = os.path.join(save_path, 'kernelshap_dict.csv')
    with open(dict_path, 'w') as f:
        json.dump(dist_dict_list, f)