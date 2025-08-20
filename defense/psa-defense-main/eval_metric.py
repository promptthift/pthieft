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

def main(args):
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

    freq_dict = get_freq()
    # plot_freq(freq_dict)
    print("Successfully fetched the frequency dict of modifiers!")

    with open("output/stats/MahD_dict.csv", 'r') as f:
        dist_dict_list = json.load(f)

    score_total, score_highfreq, score_lowfreq = 0, 0, 0
    for idx, row in df.iterrows():
        if args.wb is None and idx >= 1000:
            break

        target_modifiers = row['target_modifiers']
        target_subject = row['prompt'].split(',')[0]
        pred_modifiers = row['pred_modifiers']
        inferred_prompt = row['inferred_prompt']

        dist_dict = dist_dict_list[str(idx)]
        values = dist_dict.values()
        min_val = min(values)
        max_val = max(values)
        normalized_dict = {k: (v - min_val) / (max_val - min_val) for k, v in dist_dict.items()}

        for modifier in dist_dict.keys():
            if not modifier == 'None':
                exist = 1 if modifier in pred_modifiers else 0
                if freq_dict[modifier] > threshold:
                    score_highfreq += exist * normalized_dict[modifier]
                else:
                    score_lowfreq += exist * normalized_dict[modifier]
                score_total += exist * normalized_dict[modifier]

    print(f'Score total: {score_total / 1000}; Score high: {score_highfreq / 1000}; Score low: {score_lowfreq / 1000}')
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--threshold', type=int, default=1000)
    parser.add_argument('--wb', type=int, default=None)
    args = parser.parse_args()

    main(args)