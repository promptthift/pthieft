import torch
import json
import os
from torch import optim, nn
from eval_token import get_freq
from eval_PromptStealer import PromptStealer, get_dataset
from src.ml_decoder.loss_functions.losses import AsymmetricLoss
from PIL import Image

# from eval_PromptStealer import ml_decoder_args
from src.ml_decoder.models import create_model
import torchvision.transforms as transforms
import argparse
from src.ml_decoder.models import create_model
import torchvision.transforms as transforms

class AdvGenerator: #(PromptStealer):
    def __init__(self, modifier_detector_path, device="cuda"):
        print("\n\nPromptStealer init...")
        self.device = device
        self.blip_image_eval_size = 384
        self.load_modifier_detector(modifier_detector_path)
        self.eval()
        image_size = 448
        self.transform_val = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            # normalize, # no need, toTensor does normalization
        ])
        self.eps = self.args.eps

    def load_modifier_detector(self, _):
        print("\nLoading modifier detector...")
        self.args = ml_decoder_args()
        self.args.load_pretrain = True
        self.modifier_detector = create_model(self.args).to(self.device)

        self.modifier_detector_transform = transforms.Compose([
                                    transforms.Resize((448, 448)),
                                    transforms.ToTensor(),
                                ])
        self.modifier_detector_threshold = 0.6

    def get_feature(self, x):
        return self.modifier_detector.body(x)

    def gen_one(self, x, x_target):
        # params
        iters = 500 # 100
        alpha = 1.0 / 255

        criterion = nn.MSELoss()
        criterion = criterion.to(self.device)
        x_adv = x.clone().detach()
        optimizer = optim.Adam([x_adv], lr=0.1)

        with torch.no_grad():
            f_target = self.get_feature(self.transform_val(x_target).unsqueeze(0).to(self.device))

        for i in range(iters):
            x_adv.requires_grad_(True)

            f_adv = self.get_feature(x_adv.unsqueeze(0))
            loss = criterion(f_adv, f_target) * 1e3
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

        x_adv = x.clone().detach()
        return x_adv
    
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
    # num_samples = 1000
    # freq_dict = get_freq()
    testset = get_dataset()
    modifier_detector_path = "output/PS_ckpt/modifier_detector.ckpt"
    adv_generator = AdvGenerator(modifier_detector_path)
    args = adv_generator.args
    num_samples = args.num_samples
    num_samples = len(testset) if len(testset) < num_samples else num_samples

    save_path = f'data/lexica_adv_bb_eps{adv_generator.eps}'
    target_path = f'output/eps{adv_generator.eps}/lexica_wm_target_bb'
    os.makedirs(save_path, exist_ok=True)
    metadata_file = os.path.join(save_path, "metadata.jsonl")
    with open(metadata_file, "w") as metadata_out:
        for i in range(num_samples):
            image, prompt, modifier10_vector, id = testset[i]
            image = image.to('cuda')
            modifier10 = testset.getCategoryListByArray(modifier10_vector)

            # load target image
            file_name = f'{str(i).zfill(4)}_target_00.png'
            image_target = Image.open(os.path.join(target_path, file_name))

            image_adv = adv_generator.gen_one(image, image_target)

            file_name = str(i).zfill(5) + '.png'
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