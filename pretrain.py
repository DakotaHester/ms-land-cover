from multiprocessing import get_context
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from calflops import calculate_flops
import numpy as np
from argparse import ArgumentParser
import os
from glob import glob
from tqdm import tqdm
import pandas as pd
from time import time

from src.mslandcover.data.datasets import PreTrainDataset
from src.mslandcover.data import transforms
from src.mslandcover.models import HRNetSegmentationModel
from src.mslandcover import config
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
        '--model',
        type=str,
        default='hrnet_w48',
        choices=['hrnet_w48', 'hrnet_w18'],
    )
    
    parser.add_argument(
        '--pretrain_hdf5_path',
        type=str,
        default='/scratch/dhester/mslc/pretrain.hdf5',
        help='Path to the pretraining HDF5 dataset.',
    )
    
    parser.add_argument(
        '--pretrain_hdf5_group',
        type=str,
        default='pretrain',
        help='The group in the HDF5 dataset to use for pretraining.',
    )
    
    parser.add_argument(
        '--pretrain_val_hdf5_group',
        type=str,
        default='pretrain_val',
        help='The group in the HDF5 dataset to use for pretraining validation.',
    )    
    
    parser.add_argument(
        '--pretrain_data_dir', 
        type=str,  
        default='/scratch/dhester/mslc/pretrain/',
        help='Path to the directory containing the pretraining data.',
    )
    
    parser.add_argument(
        '--pretrain_val_data_dir', 
        type=str,  
        default='/scratch/dhester/mslc/pretrain_val/',
        help='Path to the directory containing the pretraining validation data.',
    )
    
    parser.add_argument(
        '--num_epochs',
        type=int,
        default=1000,
        help='The number of epochs to train for.',
    )
    
    parser.add_argument(
        '--early_stopping_patience',
        type=int,
        default=50,
        help='The number of epochs to wait for validation loss improvement before stopping training.',
    )
    
    parser.add_argument(
        '--full_batch_size', 
        type=int, 
        default=2048, # 4096 in original SimCLR implementation
        help='The batch size to use for pretraining.',
    )
    
    parser.add_argument(
        '--mini_batch_size',
        type=int,
        default=64,
        help='The mini-batch size to use for gradient accumulation and/or caching.',
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
        '--seed',
        type=int,
        default=1701,
        help='The random seed to use for reproducibility.',
    )
    
    parser.add_argument(
        '--num_workers',
        type=int,
        default=16,
        help='The number of workers to use for data loading.',
    )
    
    parser.add_argument(
        '--image_size',
        type=int,
        default=256,
        help='The size of the input.'
    )
    
    parser.add_argument(
        '--visualize_augmentations_dir',
        type=str,
        default='./paper/images/augmentations/' if False else None,
        help='The directory to save augmented images for visualization. If \
            provided, the script will only visualize the augmentations and not \
            train the model.',
    )
    
    return parser.parse_args()



def main():
    
    args = parse_arguments()
    
    print(f'{utils.get_datetime()} Configuration:')
    for k, v in vars(args).items():
        print(f'{k}: {v}')
    print('-'*20)
    
    torch.random.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    log_dir = os.path.join(args.log_dir, args.pretrain_scheme, args.model)
    out_dir = os.path.join(args.output_dir, args.pretrain_scheme, args.model)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)
    
    device = utils.get_torch_device()
    print(f'{utils.get_datetime()} Using device: {device}')
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True
        torch.cuda.manual_seed_all(args.seed)
    
    # pretrain_data_paths = glob(os.path.join(args.pretrain_data_dir, '*.tif'))
    # print(f'{utils.get_datetime()} Found {len(pretrain_data_paths)} pretraining images in {args.pretrain_data_dir}.')
    
    # pretrain_val_data_paths = glob(os.path.join(args.pretrain_val_data_dir, '*.tif'))
    # print(f'{utils.get_datetime()} Found {len(pretrain_val_data_paths)} pretraining validation images in {args.pretrain_val_data_dir}.')
    
    transform = transforms.SimCLRDataAugmentation(size=args.image_size) if 'simclr' in args.pretrain_scheme \
        else transforms.StandardDataAugmentations()
    return_hsv = 'hsv' in args.pretrain_scheme
    n_views = 2 if 'simclr' in args.pretrain_scheme else 1
    
    mean_path = os.path.join(args.output_dir, 'mean.pth')
    std_path = os.path.join(args.output_dir, 'std.pth')
    
    mean = torch.load(mean_path, weights_only=True) if os.path.exists(mean_path) else None
    std = torch.load(std_path, weights_only=True) if os.path.exists(std_path) else None
    
    train_dataset = PreTrainDataset(
        hdf5_path=args.pretrain_hdf5_path,
        hdf5_group=args.pretrain_hdf5_group,
        transform=transform,
        n_views=n_views,
        mean=mean,
        std=std,
        return_hsv=return_hsv,
        # device=device,
    )
    val_dataset = PreTrainDataset(
        hdf5_path=args.pretrain_hdf5_path,
        hdf5_group=args.pretrain_val_hdf5_group,
        n_views=n_views,
        mean=train_dataset.mean,
        std=train_dataset.std,
        transform=None,
        return_hsv=return_hsv,
        # device=device,
    )
    print(f'{utils.get_datetime()} Training dataset mean: {train_dataset.mean}')
    print(f'{utils.get_datetime()} Training dataset std: {train_dataset.std}')
    
    # save the mean and std for the training dataset
    if mean is None:  torch.save(train_dataset.mean, mean_path)
    if std is None: torch.save(train_dataset.std, std_path) 
    
    # take into account grad cache/acculumation steps when setting the batch size
    grad_accum_steps = args.full_batch_size // args.mini_batch_size
    print(f'{utils.get_datetime()} Full batch size: {args.full_batch_size}, Mini batch size: {args.mini_batch_size}, Grad accumulation steps: {grad_accum_steps}')
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.mini_batch_size, 
        shuffle=True, 
        drop_last=True,
        pin_memory=True,
        num_workers=args.num_workers,
        prefetch_factor=2,
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.mini_batch_size, 
        shuffle=False,
        drop_last=False,
        pin_memory=True,
        num_workers=args.num_workers,
        prefetch_factor=2,
    )
    
    if args.visualize_augmentations_dir is not None:
        transforms.visualize_transforms(
            args.visualize_augmentations_dir,
            n_views,
            return_hsv,
            train_dataset,
            glob(os.path.join(args.pretrain_data_dir, '*.tif')),
            args.pretrain_scheme,
            train_dataset.mean,
            train_dataset.std,
        )
        return

    model_config = config.HRNET_W48_CONFIG if args.model == 'hrnet_w48' else config.HRNET_W18_CONFIG
    model = HRNetSegmentationModel(
        config=model_config,
        img_decoder_head='hsv' in args.pretrain_scheme,
        aux_simclr_head='simclr' in args.pretrain_scheme,
    )
    imagenet_weights = torch.load(config.WEIGHTS_PATH[args.model], weights_only=True)
    model.load_encoder_weights(imagenet_weights)
    model.to(device)
    
    flops, macs, params = calculate_flops(
        model=model,
        input_shape=(1, 3, 256, 256),
        output_as_string=False,
        print_results=False,
        print_detailed=False,
    )
    print(f'{utils.get_datetime()} Model {args.model} Computational Complexity:')
    print(f'{utils.get_datetime()} FLOPs: {flops}, MACs: {macs}, Parameters: {params}')
    with open(os.path.join(log_dir, 'flops_macs_params.txt'), 'w') as f:
        f.write(f'FLOPs: {flops}, MACs: {macs}, Parameters: {params}')
    
    # define optimizer and scheduler - the specifics are taken from the original SimCLR implementation
    optimizer = utils.LARS(
        params=model.parameters(),
        lr=0.3*args.full_batch_size/256, # per original SimCLR implementation
        weight_decay=1e-6,
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer=optimizer,
        schedulers=[
            torch.optim.lr_scheduler.LambdaLR( # linear warmup for 10 epochs
                optimizer=optimizer,
                lr_lambda=lambda epoch: min(1, (epoch+1) / 10),
            ),
            torch.optim.lr_scheduler.CosineAnnealingLR( # cosine annealing with no restarts
                optimizer=optimizer,
                T_max=args.num_epochs - 10, # 90 epochs after warmup
            ),
        ],
        milestones=[10], # step after warmup
    )
    
    history_dict = {
        'train_loss': [],
        'val_loss': [],
        'learning_rate': [],
    }
    profiler = utils.ProfilerHistory(device)
    profiler.update(epoch=-1, phase='init', time=0)
    
    best_val_loss = np.inf
    best_epoch = -1
    print(f'{utils.get_datetime()} Starting training...')
    for epoch in range(args.num_epochs):
        
        optimizer.zero_grad(set_to_none=True) # just to be super safe
        
        for phase in ['train', 'val']:
            phase_start_time = time()
            
            batch_losses = []
            
            if phase == 'train':
                torch.set_grad_enabled(True)
                model.train()
                loader = tqdm(
                    train_loader, 
                    desc=f'Epoch {epoch+1}/{args.num_epochs} Training', 
                    total=len(train_loader),
                    unit='batch',
                    position=0,
                    leave=False,
                )
            
            else:
                torch.set_grad_enabled(False)
                model.eval()
                loader = tqdm(
                    val_loader, 
                    desc=f'Epoch {epoch+1}/{args.num_epochs} Validation', 
                    total=len(val_loader),
                    unit='batch',
                    # position=1,
                    leave=False,
                )

            # next_batch = [_.to(device, non_blocking=True) for _ in next_batch]
            
            if 'simclr' in args.pretrain_scheme:
                cache = {}
                closure = {}
            
                for view in range(train_dataset.n_views):
                    cache[f'z_{view}'], closure[f'z_{view}'] = [], []
            
            if 'hsv' in args.pretrain_scheme:
                reconstruction_loss = torch.tensor(0.0, device=device)
            
            for step, batch in enumerate(loader):
                
                optimizer.zero_grad(set_to_none=True)
                
                if args.pretrain_scheme == 'simclr':
                    
                    for view in range(len(batch)):
                        X = batch[view]
                        X = X.to(device, non_blocking=True)
                        z, cz = utils.cached_model_call(model, X)
                        
                        cache[f'z_{view}'].append(z)
                        closure[f'z_{view}'].append(cz)
                        del X
                    
                elif args.pretrain_scheme == 'hsv':
                    
                    X, y = batch
                    X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
                    y_hat = model(X)
                    y_hat = F.sigmoid(y_hat) # sigmoid as y is in [0, 1]
                    
                    reconstruction_loss = F.mse_loss(y, y_hat, reduction='sum')
                    if phase == 'train':
                        reconstruction_loss.backward() # reconstruction loss does not require gradient caching
                    del X, y, y_hat
                
                elif args.pretrain_scheme == 'hsv_simclr':
                    
                    for view in range(len(batch)):
                        X, y = batch[view]
                        X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
                        y_hat, z, cz = utils.cached_model_call(model, X) # do not need to cache y_hat
                        y_hat = F.sigmoid(y_hat) # sigmoid as y is in [0, 1] 
                            
                        cache[f'z_{view}'].append(z)
                        closure[f'z_{view}'].append(cz)
                        
                        reconstruction_loss = F.mse_loss(y, y_hat, reduction='sum') * (1 / train_dataset.n_views)
                        if phase == 'train':
                            reconstruction_loss.backward()
                        del X, y, y_hat

                
                if (step + 1) % grad_accum_steps == 0:
                    
                    total_loss = torch.tensor(0.0, device=device)
                    
                    if 'simclr' in args.pretrain_scheme:
                        total_loss += utils.cached_contrastive_loss_call(cache['z_0'], cache['z_1'])
                        total_loss.backward()
                        
                        for view in range(train_dataset.n_views):
                            for closure_fn, z in zip(closure[f'z_{view}'], cache[f'z_{view}']):
                                closure_fn(z)
                            cache[f'z_{view}'], closure[f'z_{view}'] = [], []
                        
                    if 'hsv' in args.pretrain_scheme:
                        total_loss += reconstruction_loss
                            
                    if phase == 'train':
                        optimizer.step()
                    optimizer.zero_grad(set_to_none=True) # just to be super safe (again)
                
                    batch_losses.append(total_loss.item())
                    epoch_loss = np.sum(batch_losses) / ((step * args.mini_batch_size) + len(batch)) 
                    loader.set_postfix(loss=f'{epoch_loss:.5f}')
                    del total_loss
            
            history_dict[f'{phase}_loss'].append(epoch_loss)
            
            if phase == 'val':
                if epoch_loss < best_val_loss:
                    best_val_loss = epoch_loss
                    best_epoch = epoch
                    torch.save(model.state_dict(), os.path.join(out_dir, f'{args.pretrain_scheme}.pth'))
                    with open(os.path.join(log_dir, 'best_epoch.txt'), 'w') as f:
                        f.write(str(best_epoch)) # just in case

            # profiling information
            profiler.update(epoch=epoch, phase=phase, time=time()-phase_start_time)
            profiler.save(os.path.join(log_dir, 'profiler.csv'))
            
        lr = optimizer.param_groups[0]['lr']
        history_dict['learning_rate'].append(lr)
        scheduler.step()
        
        history_df = pd.DataFrame(history_dict).set_index(pd.Index(range(epoch+1)))
        history_df.to_csv(os.path.join(log_dir, 'history.csv'), index=True)
        
        if epoch - best_epoch > args.early_stopping_patience:
            print(f'{utils.get_datetime()} No improvement in validation loss for {args.early_stopping_patience} epochs. Stopping early.')
            break


if __name__ == '__main__':
    main()