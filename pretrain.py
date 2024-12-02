import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from argparse import ArgumentParser
import os
from glob import glob
from tqdm import tqdm
from grad_cache.functional import cached, cat_input_tensor

from src.mslandcover.data.datasets import PreTrainDataset
from src.mslandcover.data import transforms
from src.mslandcover.models import HRNetSegmentationModel
from src.mslandcover.config import HRNET_BASE_CONFIG
from src.mslandcover import utils
import time

def parse_arguments():
    parser = ArgumentParser()
    
    parser.add_argument(
        '--pretrain_scheme',
        type=str,
        default='hsv_simclr',
        choices=['hsv', 'simclr', 'hsv_simclr'],
        help='The pretraining scheme to use. One of ["hsv", "simclr", "hsv_simclr"].',
    )
    
    parser.add_argument(
        '--pretrain_data_dir', 
        type=str,  
        default='./data/splits/train/input/',
        help='Path to the directory containing the pretraining data.',
    )
    
    parser.add_argument(
        '--pretrain_val_data_dir', 
        type=str,  
        default='./data/splits/val/input/',
        help='Path to the directory containing the pretraining validation data.',
    )
    
    parser.add_argument(
        '--batch_size', 
        type=int, 
        default=8, # 4096 in original SimCLR implementation
        help='The batch size to use for pretraining.',
    )
    
    parser.add_argument(
        '--log_dir', 
        type=str, 
        default='./logs/pretrain/',
        help='The directory to save logs and checkpoints.',
    )
    
    parser.add_argument(
        '--output_dir', 
        type=str, 
        default='./weights/pretrain/',
        help='The directory to save the final model weights.',
    )
    
    parser.add_argument(
        '--imagenet_weights',
        type=str,
        default='./weights/hrnetv2_w48_imagenet_pretrained.pth',
        help='Path to the pretrained ImageNet weights for the backbone.',
    )
    
    parser.add_argument(
        '--num_epochs',
        type=int,
        default=100,
        help='The number of epochs to train for.',
    )
    
    parser.add_argument(
        '--grad_cache_steps',
        type=int,
        default=2,
        help='The number of steps to accumulate gradients before performing an optimizer step.',
    )
    
    return parser.parse_args()



def main():
    
    args = parse_arguments()
    
    print(f"Pretraining scheme: {args.pretrain_scheme}")
    print(f'Configuration: {args}')
    
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    
    device = utils.get_torch_device()
    print(f'Using device: {device}')
    
    pretrain_data_glob = os.path.join(args.pretrain_data_dir, '*.tif')
    print('Searching for pretraining data with glob:', pretrain_data_glob)
    pretrain_data_paths = glob(pretrain_data_glob)
    pretrain_val_data_paths = glob(os.path.join(args.pretrain_val_data_dir, '*.tif'))
    
    print(f'Found {len(pretrain_data_paths)} pretraining images.')
    print(f'Found {len(pretrain_val_data_paths)} pretraining validation images.')
    
    if True:

        transform = transforms.SimCLRDataAugmentation() if 'simclr' in args.pretrain_scheme \
            else transforms.StandardDataAugmentations()
        return_hsv = 'hsv' in args.pretrain_scheme
        n_views = 2 if 'simclr' in args.pretrain_scheme else 1
        
        train_dataset = PreTrainDataset(
            data_paths=pretrain_data_paths,
            transform=transform,
            n_views=n_views,
            return_hsv=return_hsv,
            device=device,
        )
        val_dataset = PreTrainDataset(
            data_paths=pretrain_val_data_paths,
            n_views=n_views,
            mean=train_dataset.mean,
            std=train_dataset.std,
            transform=None,
            return_hsv=return_hsv,
            device=device,
        )
        
        # save the mean and std for the training dataset
        torch.save(train_dataset.mean, os.path.join(args.output_dir, 'mean.pth')) 
        torch.save(train_dataset.std, os.path.join(args.output_dir, 'std.pth'))
        
        # take into account grad cache/acculumation steps when setting the batch size
        loader_batch_size = args.batch_size // args.grad_cache_steps
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=loader_batch_size, 
            shuffle=True, 
            drop_last=True,
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=loader_batch_size, 
            shuffle=False, 
        )
    
    model = HRNetSegmentationModel(
        config=HRNET_BASE_CONFIG,
        img_decoder_head='hsv' in args.pretrain_scheme,
        aux_simclr_head='simclr' in args.pretrain_scheme,
    )
    imagenet_weights = torch.load(args.imagenet_weights, weights_only=True)
    model.load_encoder_weights(imagenet_weights)
    
    if 'simclr' in args.pretrain_scheme:
        contrastive_loss_fn = utils.NTXentLoss()
        
    if 'hsv' in args.pretrain_scheme:
        reconstruction_loss_fn = nn.MSELoss()
    
    # define optimizer and scheduler - the specifics are taken from the original SimCLR implementation
    optimizer = utils.LARS(
        params=model.parameters(),
        lr=0.3*args.batch_size/256, # per original SimCLR implementation
        weight_decay=1e-6,
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer=optimizer,
        schedulers=[
            torch.optim.lr_scheduler.LambdaLR( # linear warmup for 10 epochs
                optimizer=optimizer,
                lr_lambda=lambda epoch: min(1, epoch / 10),
            ),
            torch.optim.lr_scheduler.CosineAnnealingLR( # cosine annealing with no restarts
                optimizer=optimizer,
                T_max=args.num_epochs - 10, # 90 epochs after warmup
            ),
        ],
        milestones=[10], # step after warmup
    )
    
    model.to(device)
    
    history_dict = {
        'train_loss': [],
        'val_loss': [],
        'learning_rate': [],
    }
    
    for epoch in range(args.num_epochs):
        
        
        if 'simclr' in args.pretrain_scheme:
            cache = {}
            closure = {}
            
            for view in range(train_dataset.n_views):
                cache[f'z_{view}'], closure[f'z_{view}'] = [], []
        
        for phase in ['train', 'val']:
            
            if phase == 'train':
                model.train()
                loader = tqdm(
                    train_loader, 
                    desc=f'Epoch {epoch+1} Training', 
                    total=len(train_loader),
                    unit='batch',
                )
            else:
                model.eval()
                loader = tqdm(
                    val_loader, 
                    desc=f'Epoch {epoch+1} Training', 
                    total=len(val_loader),
                    unit='batch',
                )
            
            running_loss = 0.0
            for step, batch in enumerate(loader):
                
                optimizer.zero_grad(set_to_none=True)
                
                if args.pretrain_scheme == 'simclr':
                    
                    for view in range(len(batch)):
                        X = batch[view]
                        z, cz = utils.cached_model_call(model, X)
                        cache[f'z_{view}'].append(z)
                        closure[f'z_{view}'].append(cz)
                    
                elif args.pretrain_scheme == 'hsv':
                    
                    X, y = batch
                    y_hat = model(X)
                    reconstruction_loss = reconstruction_loss_fn(y_hat, y)
                    if phase == 'train':
                        reconstruction_loss.backward() # reconstruction loss does not require gradient caching, instead
                    
                
                elif args.pretrain_scheme == 'hsv_simclr':
                    
                    for view in range(len(batch)):
                        X, y = batch[view]
                        y_hat, z, cz = utils.cached_model_call(model, X) # do not need to cache y_hat
                            
                        cache[f'z_{view}'].append(z)
                        closure[f'z_{view}'].append(cz)
                        
                        reconstruction_loss = reconstruction_loss_fn(y_hat, y)
                        if phase == 'train':
                            reconstruction_loss.backward()

                
                if (step + 1) % args.grad_cache_steps == 0:
                    
                    total_loss = torch.tensor(0.0, device=device)
                    
                    if 'simclr' in args.pretrain_scheme:
                        print(len(cache['z_0']), len(cache['z_1']))
                        total_loss += utils.cached_contrastive_loss_call(cache['z_0'], cache['z_1'])
                        total_loss.backward()
                        
                        for view in range(train_dataset.n_views):
                            for closure, z in zip(closure[f'z_{view}'], cache[f'z_{view}']):
                                closure(z)
                        
                    if 'hsv' in args.pretrain_scheme:
                        total_loss += reconstruction_loss
                            
                    if phase == 'train':
                        optimizer.step()
                        optimizer.zero_grad(set_to_none=True)
                
                    running_loss += total_loss.item()
            
            if phase == 'train':
                epoch_loss = running_loss / len(loader) * train_loader.batch_size
            
            epoch_loss = running_loss / len(loader)
            history_dict[f'{phase}_loss'].append(epoch_loss)
            history_dict['learning_rate'].append(optimizer.param_groups[0]['lr'])
            
            scheduler.step()
            print(f'Epoch {epoch+1} {phase} loss: {epoch_loss}')
            
    


if __name__ == '__main__':
    main()