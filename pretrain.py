import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from argparse import ArgumentParser
import os
from glob import glob
from tqdm import tqdm

from src.mslandcover.data.datasets import PreTrainDataset
from src.mslandcover.data import transforms
from src.mslandcover.models import HRNetSegmentationModel
from src.mslandcover.config import HRNET_BASE_CONFIG
from src.mslandcover import utils

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
        default=16, # 4096 in original SimCLR implementation
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
    
    return parser.parse_args()



def main():
    
    args = parse_arguments()
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if args.pretrain_scheme in ['simclr', 'hsv_simclr']:
        return_hsv = True
        transform = transforms.SimCLRDataAugmentation()
    else:
        return_hsv = False
        transform = transforms.SimCLRDataAugmentation()
    
    pretrain_data_glob = os.path.join(args.pretrain_data_dir, '*.tif')
    print('Searching for pretraining data with glob:', pretrain_data_glob)
    pretrain_data_paths = glob(pretrain_data_glob)
    pretrain_val_data_paths = glob(os.path.join(args.pretrain_val_data_dir, '*.tif'))
    
    print(f'Found {len(pretrain_data_paths)} pretraining images.')
    print(f'Found {len(pretrain_val_data_paths)} pretraining validation images.')
    
    if True:
        train_dataset = PreTrainDataset(
            data_paths=pretrain_data_paths,
            transform=transform,
            return_hsv=return_hsv,
            device=device,
        )
        val_dataset = PreTrainDataset(
            data_paths=pretrain_val_data_paths,
            mean=train_dataset.mean,
            std=train_dataset.std,
            transform=None,
            return_hsv=return_hsv,
            device=device,
        )
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=args.batch_size, 
            shuffle=True, 
            drop_last=True,
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=args.batch_size, 
            shuffle=False, 
        )
    
    model = HRNetSegmentationModel(
        config=HRNET_BASE_CONFIG,
        img_decoder_head=True if args.pretrain_scheme in ['hsv', 'hsv_simclr'] else False,
        aux_simclr_head=True if args.pretrain_scheme in ['simclr', 'hsv_simclr'] else False,
    )
    imagenet_weights = torch.load(args.imagenet_weights, weights_only=True)
    model.load_encoder_weights(imagenet_weights)
    
    # define loss functions for the different pretraining schemes
    if args.pretrain_scheme == 'hsv': # reconstruction loss
        loss_fn = nn.MSELoss()
    elif args.pretrain_scheme == 'simclr': # contrastive loss
        loss_fn = utils.nt_xent_loss
    elif args.pretrain_scheme == 'hsv_simclr': # contrastive loss + reconstruction loss
        contrastive_loss = utils.nt_xent_loss
        reconstruction_loss = nn.MSELoss()
        def loss_fn(
            z_1: torch.Tensor, z_2: torch.Tensor, # embeddings from the projection head
            y_pred_1: torch.Tensor, y_pred_2: torch.Tensor, # predictions from the image decoder head
            y_true_1: torch.Tensor, y_true_2: torch.Tensor, # target images
        ) -> torch.Tensor:
            return contrastive_loss(z_1, z_2) + \
                0.5 * (reconstruction_loss(y_pred_1, y_true_1) + reconstruction_loss(y_pred_2, y_true_2))
    
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
                    train_loader, 
                    desc=f'Epoch {epoch+1} Training', 
                    total=len(train_loader),
                    unit='batch',
                )
            
            running_loss = 0.0
            for i, batch in enumerate(loader):
                
                optimizer.zero_grad(set_to_none=True)
                
                if args.pretrain_scheme in ['simclr', 'hsv_simclr']:
                    img, hsv, meta = batch
                    img, hsv = img.to(device), hsv.to(device)
                    img_logits, hsv_logits = model(img, hsv)
                    loss = contrastive_loss(img_logits, hsv_logits)
                else:
                    img, meta = batch
                    img = img.to(device)
                    img_recon, img_logits = model(img)
                    loss = reconstruction_loss(img_recon, img)
                
                if phase == 'train':
                    loss.backward()
                    optimizer.step()
                    scheduler.step()
                
                running_loss += loss.item()
            
            epoch_loss = running_loss / len(loader)
            history_dict[f'{phase}_loss'].append(epoch_loss)
            history_dict['learning_rate'].append(optimizer.param_groups[0]['lr'])
            
            print(f'Epoch {epoch+1} {phase} loss: {epoch_loss}')
    
    


if __name__ == '__main__':
    main()