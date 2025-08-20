import pandas as pd
import torch
import os
import ast
import json
import argparse
from tqdm import tqdm
# from eval_PromptStealer import get_dataset
from datasets import load_dataset, concatenate_datasets
import matplotlib.pyplot as plt
import numpy as np
from utils import build_prompt_with_saved_cap, get_text_single_crop_similarity, artists

def get_modifier_similarity(target_modifiers, pred_modifiers):
    target_modifiers = set(target_modifiers)
    pred_modifiers = set(pred_modifiers)
    intersection = target_modifiers & pred_modifiers
    # union = target_modifiers | pred_modifiers
    # return len(intersection) / len(union) if len(union) > 0 else 0.0
    return len(intersection) if len(target_modifiers) > 0 else 0.0

def get_freq():
    save_path = 'output/stats'
    os.makedirs(save_path, exist_ok=True)
    dict_path = os.path.join(save_path, 'freq_dict.csv')

    if os.path.exists(dict_path):
        with open(dict_path, 'r') as f:
            freq_dict = json.load(f)
    else:
        trainset = load_dataset('vera365/lexica_dataset', split='train', cache_dir="./data/lexica_dataset/")
        testset  = load_dataset('vera365/lexica_dataset', split='test', cache_dir="./data/lexica_dataset/")
        fullset = concatenate_datasets([trainset, testset])

        # print(testset[0]['image']) # PIL.JpegImagePlugin.JpegImageFile
        freq_dict = {}
        for sample in tqdm(fullset):
            modifier_list = sample['modifier10']
            for modifier in modifier_list:
                if modifier in freq_dict.keys(): 
                      freq_dict[modifier] += 1
                else: freq_dict[modifier] = 1

        # save
        with open(dict_path, 'w') as f:
            json.dump(freq_dict, f)

    return freq_dict

def plot_freq(freq_dict: dict):
    save_path = 'output/fig'
    os.makedirs(save_path, exist_ok=True)
    fig_path = os.path.join(save_path, 'modifier.png')

    freq_sort = sorted(freq_dict.values(), reverse=True)
    plt.figure(figsize=(10, 6))
    plt.plot(freq_sort) # , label='Values')
    plt.xlabel('Modifier Appearance Times')
    plt.ylabel('Count')
    # plt.title('Plot of Sorted Values in Dictionary')
    # plt.legend()
    plt.grid(True)
    # plt.show()
    plt.savefig(fig_path)
    return

def main(args):
    threshold = args.threshold
    num_samples = args.num_samples
    eps = args.eps

    csv_path = 'output/PS_results/prompt_stealer_results.csv'
    if args.wb == 1:
        csv_path = f'output/PS_results/eps{eps}/prompt_stealer_results_wb.csv'
    elif args.wb == 0:
        csv_path = f'output/PS_results/eps{eps}/prompt_stealer_results_bb.csv'
    df = pd.read_csv(csv_path)
    df['target_modifiers'] = df['target_modifiers'].apply(ast.literal_eval)
    df['pred_modifiers'] = df['pred_modifiers'].apply(ast.literal_eval)
    df['semantic_sim'] = df['semantic_sim'].astype(float)
    df['modifier_sim'] = df['modifier_sim'].astype(float)

    freq_dict = get_freq()
    plot_freq(freq_dict)
    print("Successfully fetched the frequency dict of modifiers!")

    stats = [[0, 0, 0], [0, 0, 0]]
    for idx, row in df.iterrows():
        target_modifiers = row['target_modifiers']
        target_subject = row['prompt'].split(',')[0]
        pred_modifiers = row['pred_modifiers']
        inferred_prompt = row['inferred_prompt']

        sim_normal = get_modifier_similarity(target_modifiers, pred_modifiers) / len(target_modifiers)
        target_prompt = build_prompt_with_saved_cap(target_subject, target_modifiers, artists)
        sem_normal = get_text_single_crop_similarity(target_prompt, inferred_prompt)
        stats[0][0] += sim_normal
        stats[1][0] += sem_normal

        target_highfreq, target_lowfreq = [], []
        for m in target_modifiers:
            if freq_dict[m] > threshold:
                target_highfreq.append(m)
            else:
                target_lowfreq.append(m)

        sim_highfreq = get_modifier_similarity(target_highfreq, pred_modifiers) / len(target_modifiers)
        target_prompt = build_prompt_with_saved_cap(target_subject, target_highfreq, artists)
        sem_highfreq = get_text_single_crop_similarity(target_prompt, inferred_prompt)
        stats[0][1] += sim_highfreq
        stats[1][1] += sem_highfreq

        sim_lowfreq = get_modifier_similarity(target_lowfreq, pred_modifiers) / len(target_modifiers)
        target_prompt = build_prompt_with_saved_cap(target_subject, target_lowfreq, artists)
        sem_lowfreq = get_text_single_crop_similarity(target_prompt, inferred_prompt)
        stats[0][2] += sim_lowfreq
        stats[1][2] += sem_lowfreq

        if idx >= num_samples:
            break

    stats = [[i / num_samples for i in stats[0]], [i / num_samples for i in stats[1]]]
    print(f'sim_normal: {stats[0][0]}; sim_highfreq: {stats[0][1]}; sim_lowfreq: {stats[0][2]}')
    print(f'sem_normal: {stats[1][0]}; sem_highfreq: {stats[1][1]}; sem_lowfreq: {stats[1][2]}')

    # calculate dict freq
    count_highfreq, count_lowfreq = 0, 0
    for m in freq_dict.keys():
        freq = freq_dict[m]
        if freq > threshold:
            count_highfreq += 1
        else:
            count_lowfreq += 1
    print(f'threshold: {threshold}; total num_tokens: {count_highfreq + count_lowfreq}; high_freq: {count_highfreq}; low_freq: {count_lowfreq}')
    print()
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--threshold', type=int, default=1000)
    parser.add_argument('--wb', type=int, default=None)
    parser.add_argument('--eps', type=float, default=0.05)
    parser.add_argument('--num_samples', type=int, default=1000)
    args = parser.parse_args()

    main(args)