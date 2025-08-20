import json
from collections import Counter
from typing import List, Dict, Tuple
import torch
import torch.nn.functional as F
from diffusers import StableDiffusionPipeline
from PIL import Image
import clip
from transformers import CLIPProcessor, CLIPModel
import numpy as np
from scipy.spatial.distance import cosine
import io
import copy
import numpy as np
import torch
import torch.nn as nn

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
        model.load_state_dict(torch.load('/fred/oz339/zdeng/.cache/CSD/csd_clip_model_pytorch.pt', 
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


def load_jsonl(file_path: str) -> Dict[str, List[str]]:
    data = {}
    with open(file_path, 'r') as f:
        for line in f:
            entry = json.loads(line)
            sample_id = entry['id']
            message_content = entry['messages'][1]['content'].strip().lower()
            
            if sample_id not in data:
                data[sample_id] = []
            data[sample_id].append(message_content)
    return data

def extract_subject_modifiers(text: str) -> Tuple[str, List[str]]:
    parts = text.split(',')
    subject = parts[0].strip()
    # 使用set去重，然后转回list
    modifiers = list(set(mod.strip() for mod in parts[1:]))
    return subject, modifiers

def get_frequency_stats(texts: List[str]) -> Tuple[Dict[str, int], Dict[str, int]]:
    subject_counter = Counter()
    modifier_counter = Counter()
    
    for text in texts:
        subject, modifiers = extract_subject_modifiers(text)
        subject_counter[subject] += 1
        modifier_counter.update(modifiers)
    
    return dict(subject_counter), dict(modifier_counter)

def compute_clip_score(image1, image2, clip_model, preprocess):
    # 使用preprocess处理图片
    image1_processed = preprocess(image1).unsqueeze(0).to(device)
    image2_processed = preprocess(image2).unsqueeze(0).to(device)
    
    # 计算特征
    with torch.no_grad():
        image1_features = clip_model.encode_image(image1_processed)
        image2_features = clip_model.encode_image(image2_processed)
    
    # 计算余弦相似度
    similarity = F.cosine_similarity(image1_features, image2_features)
    return similarity[0].item()

def filter_modifiers_by_semantics(modifiers: List[str], clip_model, threshold: float) -> List[str]:
    unique_modifiers = []
    for modifier in modifiers:
        is_unique = True
        for unique_mod in unique_modifiers:
            # 使用CLIP计算文本相似度
            if semantic_similarity(modifier, unique_mod, clip_model) > threshold:
                is_unique = False
                break
        if is_unique:
            unique_modifiers.append(modifier)
    return unique_modifiers

def semantic_similarity(text1: str, text2: str, clip_model) -> float:
    # 使用CLIP的文本编码器计算相似度
    text_features1 = clip_model.encode_text(clip.tokenize(text1))
    text_features2 = clip_model.encode_text(clip.tokenize(text2))
    return 1 - cosine(text_features1, text_features2)

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sd_model = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4",
                                                       torch_dtype=torch.float16 if device =="cuda" else torch.float32,
                                                       cache_dir=cache_dir,
                                                       safety_checker=None,
                                                       requires_safety_checker=False).to(device)

    clip_model, preprocess = clip.load("ViT-L/14", device=device, download_root=cache_dir)

    file_path = "./sample_output/2025-04-23-14-41-41.jsonl"
    data = load_jsonl(file_path)

    results = {}
    total_sd_calls = 0  
    
    for sample_id, messages in data.items():
        print(f"Processing sample {sample_id}")
        sd_calls = 0  
        
        with open(file_path, 'r') as f:
            for line in f:
                entry = json.loads(line)
                if entry['id'] == sample_id:
                    image_path = entry['images'][0]
                    break
        original_image = Image.open(image_path).convert("RGB")
        
        subject_freq, modifier_freq = get_frequency_stats(messages)

        best_subject = None
        best_score = -float('inf')
        for subject in subject_freq:
            generated_image = sd_model(subject, num_inference_steps=50, safety_checker=None).images[0]
            sd_calls += 1  
            score = compute_clip_score(original_image, generated_image, clip_model, preprocess)
            if score > best_score:
                best_score = score
                best_subject = subject

        half_sample_count = len(messages) / 2
        common_modifiers = []
        remaining_modifiers = []
        
        for modifier, freq in modifier_freq.items():
            if freq >= half_sample_count:
                common_modifiers.append(modifier)
            else:
                remaining_modifiers.append(modifier)
        
        base_prompt = best_subject
        if common_modifiers:
            base_prompt += ", " + ", ".join(common_modifiers)
        
        base_image = sd_model(base_prompt, num_inference_steps=50, safety_checker=None).images[0]
        sd_calls += 1  # 计数
        base_score = compute_clip_score(original_image, base_image, clip_model, preprocess)
        
        final_modifiers = common_modifiers.copy()
        for modifier in remaining_modifiers:
            test_prompt = f"{base_prompt}, {modifier}"
            test_image = sd_model(test_prompt, num_inference_steps=50, safety_checker=None).images[0]
            sd_calls += 1  # 计数
            test_score = compute_clip_score(original_image, test_image, clip_model, preprocess)
            
            if test_score > base_score:
                final_modifiers.append(modifier)
                base_score = test_score
                base_prompt = test_prompt

        results[sample_id] = {
            "image_path": image_path,
            "prompt": base_prompt,  
            "final_score": base_score,
            "sd_calls": sd_calls
        }
        
        total_sd_calls += sd_calls


    with open("/fred/oz337/zdeng/sample_output/resampling_results2.json", "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()