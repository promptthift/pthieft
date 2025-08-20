import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import clip
from transformers import CLIPProcessor, CLIPVisionModel
from diffusers import StableDiffusionPipeline
import numpy as np
from tqdm import tqdm
import os
import io
import copy
from transformers import pipeline
import pandas as pd

cache_dir = "./.cache"
device = "cuda" if torch.cuda.is_available() else "cpu"

class CSD_CLIP(nn.Module):
    """backbone + projection head"""
    def __init__(self, name='vit_large', content_proj_head='default'):
        super(CSD_CLIP, self).__init__()
        self.content_proj_head = content_proj_head
        if name == 'vit_large':
            clipmodel, _ = clip.load("ViT-L/14", download_root=cache_dir)
            self.backbone = clipmodel.visual
            self.embedding_dim = 1024
        elif name == 'vit_base':
            clipmodel, _ = clip.load("ViT-B/16", download_root=cache_dir)
            self.backbone = clipmodel.visual
            self.embedding_dim = 768 
            self.feat_dim = 512
        else:
            raise Exception('This model is not implemented')

        self.last_layer_style = copy.deepcopy(self.backbone.proj)
        self.last_layer_content = copy.deepcopy(self.backbone.proj)

        self.backbone.proj = None

    @property
    def dtype(self):
        return self.backbone.conv1.weight.dtype
    
    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, input_data):
        feature = self.backbone(input_data)

        style_output = feature @ self.last_layer_style
        style_output = F.normalize(style_output, dim=1, p=2)

        content_output = feature @ self.last_layer_content
        content_output = F.normalize(content_output, dim=1, p=2)
        
        return feature, content_output, style_output


def get_model_processor(name, device=device):
    if name == "CLIP":
        model = CLIPVisionModel.from_pretrained("openai/clip-vit-large-patch14",download_root=cache_dir)
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14",download_root=cache_dir)
        model = model.to(device)
    elif name == "CSD":
        model = CSD_CLIP("vit_large")
        model.load_state_dict(torch.load('./csd_clip_model_pytorch.pt', 
                                         map_location=device, 
                                         weights_only=True))
        model.eval()
        model = model.to(device)
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14",download_root=cache_dir)
    return model, processor


def process_batch(image_paths, model, processor, model_name):
    images = [Image.open(img_path).convert("RGB") for img_path in image_paths]
    inputs = processor(images=images, return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(model.dtype) for k, v in inputs.items()}

    if model_name == "CLIP":
        with torch.no_grad():
            outputs = model(inputs['pixel_values'].to(model.device))
        
        return outputs.last_hidden_state[:, 0, :].cpu().numpy()
    elif model_name == "CSD":
        with torch.no_grad():
            _, content_output, style_output = model(inputs['pixel_values'].to(model.device))
        
        return content_output.cpu().numpy(), style_output.cpu().numpy()
    else:
        raise Exception(f'Model {model_name} not implemented')


def to_jpeg(image):
    buffered = io.BytesIO()
    image.save(buffered, format='JPEG')
    return buffered.getvalue() 


def calculate_jaccard_similarity(str1, str2):
    set1 = set(str1.lower().split())
    set2 = set(str2.lower().split())
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union != 0 else 0


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    cache_dir = "./.cache"
    os.makedirs(cache_dir, exist_ok=True)
    
    csd_model, csd_processor = get_model_processor("CSD", device)
    clip_model, preprocess = clip.load("ViT-L/14", device=device, download_root=cache_dir)
    
    sd_model = StableDiffusionPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4",
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        cache_dir=cache_dir
    ).to(device)
    
    json_path = "./resampling_results.jsonl"
    data = []
    with open(json_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                if line.strip():
                    json_obj = json.loads(line.strip())
                    data.append(json_obj)
            except json.JSONDecodeError as e:
                continue
            except Exception as e:
                continue
    

    
    results = []
    
    for idx, row in tqdm(enumerate(data), total=len(data)):
        try:

            generated_caption = row['response']
            ground_truth = row['labels']
            image_path = row['images'][0]['path']
            

            image = Image.open(image_path).convert('RGB')
            

            with torch.no_grad():
                text_input1 = clip.tokenize([generated_caption], context_length=77, truncate=True).to(device)
                text_input2 = clip.tokenize([ground_truth], context_length=77, truncate=True).to(device)
                text_features1 = clip_model.encode_text(text_input1)
                text_features2 = clip_model.encode_text(text_input2)
                clip_similarity = torch.cosine_similarity(text_features1, text_features2).item()
            

            jaccard_score = calculate_jaccard_similarity(generated_caption, ground_truth)
            

            generated_image = sd_model(generated_caption, num_inference_steps=50).images[0]


            with torch.no_grad():

                image_input1 = preprocess(image).unsqueeze(0).to(device)
                image_input2 = preprocess(generated_image).unsqueeze(0).to(device)
                

                image_features1 = clip_model.encode_image(image_input1)
                image_features2 = clip_model.encode_image(image_input2)
                

                clip_image_similarity = torch.cosine_similarity(image_features1, image_features2).item()
                

                inputs1 = csd_processor(images=image, return_tensors="pt")
                inputs1 = {k: v.to(csd_model.dtype) for k, v in inputs1.items()}
                _, _, style_features1 = csd_model(inputs1['pixel_values'].to(device))
                
                inputs2 = csd_processor(images=generated_image, return_tensors="pt")
                inputs2 = {k: v.to(csd_model.dtype) for k, v in inputs2.items()}
                _, _, style_features2 = csd_model(inputs2['pixel_values'].to(device))
                
                csd_style_similarity = torch.cosine_similarity(style_features1, style_features2).item()
            
            result = {
                'generated_caption': generated_caption,
                'ground_truth': ground_truth,
                'clip_text_similarity': clip_similarity,
                'jaccard_score': jaccard_score,
                'clip_image_similarity': clip_image_similarity,
                'csd_style_similarity': csd_style_similarity
            }
            results.append(result)
        except Exception as e:
            continue
    
    output_path = './ours_evaluation_results_test.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=4)
    

if __name__ == "__main__":
    main()
