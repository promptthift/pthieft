from datasets import load_dataset, load_from_disk, DatasetDict, Dataset, concatenate_datasets
raw_dataset = load_dataset("vera365/lexica_dataset",cache_dir= "./.cache")

import requests
import os
train_dir = "./lexica_dataset/train"
test_dir = "./lexica_dataset/test"
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)
def save_images(data_split, folder):
    for item in data_split:
        image = item['image']
        image_id = item['id']
        image_path = os.path.join(folder, f"{image_id}.jpg")
        image.save(image_path, format='JPEG')
save_images(raw_dataset['train'], train_dir)
save_images(raw_dataset['test'], test_dir)
