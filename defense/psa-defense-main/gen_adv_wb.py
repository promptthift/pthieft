import torch
import json
import os
import pandas as pd
import ast
from torch import optim, nn
from eval_token import get_freq
from data.lexica_dataset import LexicaDataset
from eval_PromptStealer import PromptStealer, get_dataset
from src.ml_decoder.loss_functions.losses import AsymmetricLoss
from PIL import Image
import argparse
from src.ml_decoder.models import create_model
import torchvision.transforms as transforms

class AdvGenerator:
    def __init__(self, modifier_detector_path, device="cuda"):
        print("\n\nPromptStealer init...")
        self.device = device
        self.blip_image_eval_size = 384
        self.load_modifier_detector(modifier_detector_path)
        self.eval()
        self.eps = self.args.eps

    def forward_pass(self, x):
        x.requires_grad_(True)
        output = self.modifier_detector(x.unsqueeze(0)).float()
        return output

    def gen_one(self, x, target, target_exclude):
        # params
        iters = 100
        alpha = 1.0 / 255

        criterion = AsymmetricLoss(gamma_neg=4, gamma_pos=0, clip=0.05, disable_torch_grad_focal_loss=True)
        criterion = criterion.to(self.device)
        mse = nn.MSELoss()
        x_adv = x.clone().detach()
        optimizer = optim.Adam([x_adv], lr=0.1)

        with torch.no_grad():
            output_orig = self.forward_pass(x_adv)
        # target_exclude = [i for i in range(output_orig.shape[1]) if i not in target]

        for i in range(iters):
            x_adv.requires_grad_(True)

            output = self.modifier_detector(x_adv.unsqueeze(0)).float()
            assym_loss = criterion(output[:, target], torch.zeros_like(output[:, target]).to('cuda'))
            if len(target_exclude) > 0:
                mse_loss = mse(output[:, target_exclude], output_orig[:, target_exclude])
                loss = assym_loss + 0.1 * mse_loss
                print(f'{mse_loss} {assym_loss}')
            else:
                loss = assym_loss
                print(f'NA {assym_loss}')
            # print(f'{mse_loss} {assym_loss}')
            optimizer.zero_grad()

            loss.backward()
            # Updating parameters
            adv_images = x_adv - alpha * x_adv.grad.sign()
            eta = torch.clamp(adv_images - x, min=-self.eps/2, max=+self.eps/2)
            x_adv = torch.clamp(x + eta, min=0., max=1.).detach_()

            # Print
            if (i + 1) % 100 == 0:
                print(f'Epoch {i + 1}, Loss {loss.item()}')

        x_adv.data = torch.clamp(x_adv, min=0., max=1.)
        return x_adv
    
    def load_modifier_detector(self, path):
        print("\nLoading modifier detector...")
        self.args = ml_decoder_args()
        self.modifier_detector = create_model(self.args).to(self.device)
        ckpt = torch.load(path, map_location='cpu')
        if 'model' in ckpt:
            self.modifier_detector.load_state_dict(ckpt['model'], strict=True)
        else:
            self.modifier_detector.load_state_dict(ckpt, strict=True)
        print(f'Resume from checkpoint: {path}')

        self.modifier_detector_transform = transforms.Compose([
                                    transforms.Resize((448, 448)),
                                    transforms.ToTensor(),
                                ])
        self.modifier_detector_threshold = 0.6

    def eval(self):
        self.modifier_detector.eval()

def ml_decoder_args():
    parser = argparse.ArgumentParser(description='PyTorch ML Decoder Training')
    parser.add_argument('--dataset', type=str, default='coco')
    parser.add_argument('--data_path', type=str, default='/home/MSCOCO_2014/')
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--model-name', default='tresnet_l')
    parser.add_argument('--model-path', default='https://miil-public-eu.oss-eu-central-1.aliyuncs.com/model-zoo/ML_Decoder/tresnet_l_pretrain_ml_decoder.pth', type=str)
    parser.add_argument('--num-classes', default=7672, type=int)
    parser.add_argument('--epoch', default=40, type=int)
    parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                        help='number of data loading workers')
    parser.add_argument('--image-size', default=448, type=int,
                        metavar='N', help='input image size (default: 448)')
    parser.add_argument('--batch-size', default=56, type=int,
                        metavar='N', help='mini-batch size')

    # ML-Decoder
    parser.add_argument('--use-ml-decoder', default=1, type=int)
    parser.add_argument('--num-of-groups', default=-1, type=int)  # full-decoding
    parser.add_argument('--decoder-embedding', default=768, type=int)
    parser.add_argument('--zsl', default=0, type=int)

    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--load_pretrain', action='store_true')
    parser.add_argument('--save_pred', action='store_true')
    parser.add_argument('--save_path', type=str, default='output/test/')
    parser.add_argument('--start-epoch', type=int, default=0)
    parser.add_argument('--resume', type=str, default='')

    parser.add_argument('--eps', type=float, default=0.05)
    parser.add_argument('--num_samples', type=int, default=1000)
    args = parser.parse_args()
    return args

def main():
    threshold = 1000
    freq_dict = get_freq()
    testset = get_dataset()
    modifier_detector_path = "output/PS_ckpt/modifier_detector.ckpt"
    adv_generator = AdvGenerator(modifier_detector_path)
    args = adv_generator.args
    num_samples = args.num_samples
    num_samples = len(testset) if len(testset) < num_samples else num_samples

    csv_path = 'output/PS_results/prompt_stealer_results.csv'
    df = pd.read_csv(csv_path)
    df['pred_modifiers'] = df['pred_modifiers'].apply(ast.literal_eval)

    save_path = f'data/lexica_adv_wb_eps{adv_generator.eps}'
    os.makedirs(save_path, exist_ok=True)
    metadata_file = os.path.join(save_path, "metadata.jsonl")
    with open(metadata_file, "w") as metadata_out:
        for i in range(num_samples):
            image, prompt, modifier10_vector, id = testset[i]
            image = image.to('cuda')
            file_name = str(i).zfill(5) + '.png'
            modifier10 = testset.getCategoryListByArray(modifier10_vector)

            # build target list (low frequency token) or protect list
            target_list, highfreq_dict = [], {}
            pred_modifiers = df['pred_modifiers'][i].keys()
            for modifier in pred_modifiers:
                if modifier in modifier10 and freq_dict[modifier] <= threshold:
                    target_list.append(modifier)
                elif modifier in modifier10:
                    highfreq_dict[modifier] = freq_dict[modifier]

            if len(target_list) == 0 and len(highfreq_dict) > 0:
                target_list.append(min(highfreq_dict, key=highfreq_dict.get))

            target_list = testset.getLabelVector(target_list)
            full_list = testset.getLabelVector(pred_modifiers)
            target = [i for i in range(len(target_list)) if target_list[i] > 0]
            target_exclude = [i for i in range(len(target_list)) \
                              if target_list[i] == 0 and full_list[i] > 0]


            image_adv = adv_generator.gen_one(image, target, target_exclude)

            Image.fromarray(
                (image_adv * 255 + 0.).clamp(0, 255).to(torch.uint8).permute(1, 2, 0).cpu().numpy()
            ).save(os.path.join(save_path, file_name))
            # image_adv.save(os.path.join(save_path, file_name))
            metadata_out.write(json.dumps({
                "file_name": file_name,
                "prompt": prompt,
                "id": id,
                "subject": prompt.split(',')[0],
                "modifier10": modifier10,
                "modifier10_vector": modifier10_vector.tolist()
            }) + "\n")
            print(f'Finish generating image {i}.')

    return

if __name__ == "__main__":
    main()