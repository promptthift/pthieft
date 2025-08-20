"""
Code for training subject generator, adapted from BLIP (https://github.com/salesforce/BLIP).
"""
import argparse
import os
import ruamel.yaml as yaml
import numpy as np
import pandas as pd
import random
import time
import datetime
import json
from pathlib import Path
import torch
import torch.backends.cudnn as cudnn
PROJECT_PATH = os.getcwd()
print( "Current Path:", os.getcwd())

from src.BLIP_finetune.models.blip import blip_decoder
import src.BLIP_finetune.utils as utils
from src.BLIP_finetune.utils import cosine_lr_schedule
from src.BLIP_finetune.data import create_dataset, create_sampler, create_loader

device = "cuda" if torch.cuda.device_count() >= 1 else "cpu"
print("Device:", device)

@torch.no_grad()
def evaluate(data_loader, model, config, stop_num=None):
    model.eval() 
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Evaluate:'
    print_freq = 200

    result = []
    for images, subjects, _, indices in metric_logger.log_every(data_loader, print_freq, header): 
        images = images.to(device)       
        generated_subjects = model.generate(images, sample=False, num_beams=config['num_beams'], max_length=config['max_length'], 
                                  min_length=config['min_length'])
        
        for id, subject, generated_subject in zip(indices, subjects, generated_subjects):
            result.append({"id": id, "subject": subject, "generated_subject": generated_subject})
        if stop_num is not None and len(result) >= stop_num:
            break
    df = pd.DataFrame(result)
    return df

def train(model, data_loader, optimizer, epoch, device):
    model.train()  
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('loss', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    header = 'Train Epoch: [{}]'.format(epoch)
    print_freq = 50

    for i, (images, subjects, _, _) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        images = images.to(device) # (batch_size, 3, 384, 384)  
        
        loss = model(images, subjects)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()    
        
        metric_logger.update(loss=loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())     
    return {k: "{:.3f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}  

def main(args, config):
    args.distributed = False
    
    device = torch.device(args.device)

    # fix the seed
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    print("Load dataset: ", args.dataset)
    train_dataset, val_dataset, test_dataset = create_dataset(args.dataset, config)

    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()            
        samplers = create_sampler([train_dataset, val_dataset, test_dataset], [True, False, False], num_tasks, global_rank)         
    else:
        samplers = [None, None, None]

    train_loader, _, test_loader = create_loader([train_dataset, val_dataset, test_dataset],samplers,
                                                        batch_size=[config['batch_size']]*3,num_workers=[0,0,0],
                                                            is_trains=[True, False, False], collate_fns=[None,None,None])

    print("Creating model")
    model = blip_decoder(pretrained=config['pretrained'], image_size=config['image_size'], vit=config['vit'], 
                           vit_grad_ckpt=config['vit_grad_ckpt'], vit_ckpt_layer=config['vit_ckpt_layer'], 
                           prompt=config['prompt'], med_config=config['med_config'])

    model = model.to(device)   
    
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module
    
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=config['init_lr'], weight_decay=config['weight_decay'])

    if args.resume:
        print("Resume from checkpoint:", args.resume)
        checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        if 'optimizer' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            epoch = checkpoint['epoch']
            args.start_epoch = epoch+1
            config = checkpoint['config']
            print("Resume from epoch {}".format(epoch))

    model = model.to(device)

    print("Start training")
    start_time = time.time()    
    for epoch in range(args.start_epoch, config['max_epoch']):
        if not args.evaluate:        
            if args.distributed:
                train_loader.sampler.set_epoch(epoch)
                
            cosine_lr_schedule(optimizer, epoch, config['max_epoch'], config['init_lr'], config['min_lr'])
                
            train_stats = train(model, train_loader, optimizer, epoch, device) 
        
        pred_df = evaluate(test_loader, model, config, stop_num=args.stop_num)
        pred_df.to_csv(os.path.join(args.output_dir, f'pred_subject.csv'), index=False)

        if utils.is_main_process():            
            save_obj = {
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'config': config,
                'epoch': epoch,
            }

            torch.save(save_obj, os.path.join(args.output_dir, f'subject_generator.pth')) 

            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                            'epoch': epoch,
                        }
            with open(os.path.join(args.output_dir, "log.txt"),"a") as f:
                f.write(json.dumps(log_stats) + "\n")
                
        if args.evaluate: 
            break

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str)) 


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./src/BLIP_finetune/configs/lexica_subject.yaml')
    parser.add_argument('--dataset', default='lexica')
    parser.add_argument('--return_text', default='subject')
    parser.add_argument('--output_dir', default='output/PS_ckpt')    
    parser.add_argument('--stop_num', default=200, type=int)    
    parser.add_argument('--resume', default=None, type=str)
    parser.add_argument('--start-epoch', type=int, default=0)
    parser.add_argument('--evaluate', action='store_true')    
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')    
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', default=True, type=bool)
    args = parser.parse_args()

    # PromptStealer-Subject
    # args.resume = "output/PS_ckpt/subject_generator.pth"

    args.output_dir = 'output/PS_ckpt'
    args.dataset = 'lexica'
    args.return_text = 'subject'
    args.stop_num = None # 200 # just for debugging, set to None for full evaluation

    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
    config['dataset'] = args.dataset
    config['return_text'] = args.return_text
    print(config['max_epoch'])

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))    
    
    main(args, config)