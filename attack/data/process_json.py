from datasets import load_dataset, load_from_disk, DatasetDict, Dataset, concatenate_datasets
raw_dataset = load_dataset("vera365/lexica_dataset",cache_dir= "./.cache")

import requests
import os

def create_json_file(data_split, filename, types = 'test'):
    data_list = []
    i = 0
    for item in data_split:
        data_list.append({
            "promptid": item['promptid'],
            "image": f"./lexica_dataset/{types}/{item['id']}.jpg",
            "width":item['width'],
            "height": item['height'],
            "conversations": [
                {
                    "from": "human",
                    "value": ("<image>\nWrite an image prompt for this image with details about the surrounding. Include color and details about all of variables. If the image involves an artistic style or inspiration from an artist, explicitly mention the artist name.")
                },
                {
                    "from": "gpt",
                    "value": item['prompt']
                },
            ]
        })
        i +=1
    
    with open(os.path.join(output_dir, filename), 'w', encoding='utf-8') as f:
        json.dump(data_list, f, ensure_ascii=False, indent=2)


output_dir = "./lexica_dataset"
os.makedirs(output_dir, exist_ok=True)
import json

file_path = "train.json"

with open(file_path, 'r', encoding='utf-8') as file:
    data = json.load(file)
create_json_file(raw_dataset['train'], "train.json")
