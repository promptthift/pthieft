import torch
import json
import os
from torch import optim, nn
from eval_token import get_freq
from eval_PromptStealer import PromptStealer, get_dataset
from src.ml_decoder.loss_functions.losses import AsymmetricLoss
from PIL import Image

from eval_PromptStealer import ml_decoder_args
from src.ml_decoder.models import create_model
import torchvision.transforms as transforms

import open_clip
import datasets
import torchvision.transforms as T

class AdvGenerator(PromptStealer):
    def __init__(self, modifier_detector_path, device="cuda"):
        print("\n\nPromptStealer init...")
        self.device = device
        self.blip_image_eval_size = 384
        self.load_modifier_detector(modifier_detector_path)
        self.eval()

        self.eps = 0.05

    def load_modifier_detector(self, _):
        print("\nLoading modifier detector...")
        args = ml_decoder_args()
        args.load_pretrain = True

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, _, preprocess = open_clip.create_model_and_transforms(
            'ViT-H-14',
            pretrained='laion2b_s32b_b79k',
            cache_dir='data/clip',
            device=device
        )
        print(preprocess)
        self.modifier_detector = model

        self.modifier_detector_transform = preprocess
        self.modifier_detector_threshold = 0.6

    def get_feature(self, x):
        return self.modifier_detector.encode_image(x)

    def gen_one(self, x, x_target):
        # params
        iters = 500 # 100
        alpha = 1.0 / 255

        criterion = nn.MSELoss()
        criterion = criterion.to(self.device)
        x = self.modifier_detector_transform(x).unsqueeze(0).to('cuda')
        x_adv = x.clone().detach()
        optimizer = optim.Adam([x_adv], lr=0.1)

        with torch.no_grad():
            x_target = self.modifier_detector_transform(x_target).unsqueeze(0).to('cuda')
            f_target = self.get_feature(x_target)

        for i in range(iters):
            x_adv.requires_grad_(True)

            f_adv = self.get_feature(x_adv)
            loss = criterion(f_adv, f_target)
            optimizer.zero_grad()

            loss.backward()
            # Updating parameters
            adv_images = x_adv - alpha * x_adv.grad.sign()
            eta = torch.clamp(adv_images - x, min=-self.eps, max=+self.eps)
            x_adv = torch.clamp(x + eta, min=x.min(), max=x.max()).detach_()

            # Print
            if (i + 1) % 100 == 0:
                print(f'Epoch {i + 1}, Loss {loss.item()}')

        x_adv.data = torch.clamp(x_adv, min=x_adv.min(), max=x_adv.max())
        return x_adv.cpu()
    
    def eval(self):
        self.modifier_detector.eval()

def main():
    num_samples = 10 # 1000
    # freq_dict = get_freq()
    testset = datasets.load_dataset("vera365/lexica_dataset", split='test', cache_dir="./data/lexica_dataset/")
    modifier_detector_path = "output/PS_ckpt/modifier_detector.ckpt"
    adv_generator = AdvGenerator(modifier_detector_path)
    to_pil = T.ToPILImage()
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)

    num_samples = len(testset) if len(testset) < num_samples else num_samples

    save_path = f'data/lexica_adv_bb_eps{adv_generator.eps}'
    target_path = 'data/lexica_target'
    os.makedirs(save_path, exist_ok=True)
    metadata_file = os.path.join(save_path, "metadata.jsonl")
    with open(metadata_file, "w") as metadata_out:
        for i in range(num_samples):
            sample = testset[i]
            image = sample['image']
            modifier10_vector = sample['modifier10_vector']
            modifier10 = sample['modifier10']
            prompt = sample['prompt']
            id = sample['id']

            file_name = str(i).zfill(5) + '.png'

            # load target image
            image_target = Image.open(os.path.join(target_path, file_name))

            image_adv = adv_generator.gen_one(image, image_target)

            # image_adv = adv_generator.modifier_detector_transform(image).unsqueeze(0)
            image_adv = image_adv * std + mean
            image_adv = image_adv.clamp(0, 1)
            image_adv = to_pil(image_adv[0]).convert("RGB")
            image_adv.save(os.path.join(save_path, file_name))
            # image.save(os.path.join(save_path, file_name))
            metadata_out.write(json.dumps({
                "file_name": file_name,
                "prompt": prompt,
                "id": id,
                "subject": prompt.split(',')[0],
                "modifier10": modifier10,
                "modifier10_vector": modifier10_vector # .tolist()
            }) + "\n")
            print(f'Finish generating image {i}.')

    return

if __name__ == "__main__":
    main()