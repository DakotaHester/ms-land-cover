from multiprocessing import get_context
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from calflops import calculate_flops
import numpy as np
from argparse import ArgumentParser
import os
from glob import glob
from tqdm import tqdm
import pandas as pd
from time import time
import json

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
        default=100,
        help='The number of epochs to train for.',
    )
    
    parser.add_argument(
        '--early_stopping_patience',
        type=int,
        default=10,
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
        '--weights_dir', 
        type=str, 
        default='./weights/',
        help='The directory from which model weights will be loaded and saved.' + \
            'The directory should have the following structure: ' + \
                'output_dir/model_name/imagenet.pth'
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
        '--contrastive_loss_weight',
        type=float,
        default=1.0,
        help='The weight to apply to the contrastive loss under pretraining schema `hsv_simclr`.',
    )
    
    parser.add_argument(
        '--reconstruction_loss_weight',
        type=float,
        default=0.5,
        help='The weight to apply to the reconstruction loss under pretraining schema `hsv_simclr`.',
    )
    
    parser.add_argument(
        '--visualize_augmentations_dir',
        type=str,
        default='./paper/images/augmentations/' if False else None,
        help='The directory to save augmented images for visualization. If \
            provided, the script will only visualize the augmentations and not \
            train the model.',
    )
    
    parser.add_argument(
        '--debug',
        default=False,
        action='store_true',
        help='Run the script in debug mode - reduce amount of training data used.',
    )
    
    return parser.parse_args()



def main():
    
    args = parse_arguments()
    
    torch.random.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    log_dir = os.path.join(args.log_dir, args.model, args.pretrain_scheme)
    out_dir = os.path.join(args.weights_dir, args.model)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)
    
    logger = utils.Logger(os.path.join(log_dir, 'log.txt'))
    
    logger.log(f'Configuration:')
    for k, v in vars(args).items():
        logger.log(f'{k}: {v}', prepend_timestamp=False)
    logger.log('='*20, prepend_timestamp=False)
    
    device = utils.get_torch_device()
    logger.log(f'Using device: {device}')
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True
        torch.cuda.manual_seed_all(args.seed)
    
    transform = transforms.SimCLRDataAugmentation(size=args.image_size) if 'simclr' in args.pretrain_scheme \
        else transforms.StandardDataAugmentations()
    return_hsv = 'hsv' in args.pretrain_scheme
    n_views = 2 if 'simclr' in args.pretrain_scheme else 1
    
    mean_path = os.path.join(args.weights_dir, 'pretrain_mean.pth')
    std_path = os.path.join(args.weights_dir, 'pretrain_std.pth')
    
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
    )
    val_dataset = PreTrainDataset(
        hdf5_path=args.pretrain_hdf5_path,
        hdf5_group=args.pretrain_val_hdf5_group,
        n_views=n_views,
        mean=train_dataset.mean,
        std=train_dataset.std,
        transform=None,
        return_hsv=return_hsv,
    )
    
    if args.debug:
        train_dataset.ids_list = train_dataset.ids_list[:10000]
        val_dataset.ids_list = val_dataset.ids_list[:5000]
    
    logger.log(f'Training dataset size: {len(train_dataset)}')
    logger.log(f'Validation dataset size: {len(val_dataset)}')
    
    logger.log(f'Training dataset mean: {train_dataset.mean}')
    logger.log(f'Training dataset std: {train_dataset.std}')
    
    # save the mean and std for the training dataset
    if mean is None:  torch.save(train_dataset.mean, mean_path)
    if std is None: torch.save(train_dataset.std, std_path) 
    
    # take into account grad cache/acculumation steps when setting the batch size
    grad_accum_steps = args.full_batch_size // args.mini_batch_size
    logger.log(f'Full batch size: {args.full_batch_size}, Mini batch size: {args.mini_batch_size}, Grad accumulation steps: {grad_accum_steps}')
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.mini_batch_size, 
        shuffle=True, 
        drop_last=True,
        pin_memory=True,
        num_workers=args.num_workers,
        prefetch_factor=4,
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.mini_batch_size, 
        shuffle=False,
        drop_last=False,
        pin_memory=True,
        num_workers=args.num_workers,
        prefetch_factor=4,
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
    imagenet_weights = torch.load(os.path.join(out_dir, 'imagenet.pth'), weights_only=True)
    model.load_encoder_weights(imagenet_weights)
    model.to(device)
    
    flops, macs, _ = calculate_flops(
        model=model,
        input_shape=(1, 3, 256, 256),
        output_as_string=False,
        print_results=False,
        print_detailed=False,
    )
    params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.log(f'Model GFLOPs: {flops*1e-9:,.2f} | GMACs: {macs*1e-9:,.2f} | Total Parameters: {params:,} | Trainable Parameters: {trainable_params:,}')
    with open(os.path.join(log_dir, 'model_complexity.json'), 'w') as f:
        json.dump({
            'flops': flops,
            'macs': macs,
            'total_parameters': params,
            'trainable_parameters': trainable_params,
        }, f, indent=4)
    
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
    scaler = GradScaler()
    reconstruction_loss_weight = args.reconstruction_loss_weight if args.pretrain_scheme == 'hsv_simclr' else 1.0
    contrastive_loss_weight = args.contrastive_loss_weight if args.pretrain_scheme == 'hsv_simclr' else 1.0
    
    history_dict = {
        'learning_rate': [],
    }
    for phase in ['train', 'val']:
        history_dict[f'{phase}_total_loss'] = []
        history_dict[f'{phase}_reconstruction_loss'] = []
        history_dict[f'{phase}_contrastive_loss'] = []
    
    profiler = utils.ProfilerHistory(device)
    profiler.update(epoch=-1, phase='init', time=0)
    
    best_val_loss = np.inf
    best_epoch = -1
    logger.log(f'Starting training...')
    for epoch in range(args.num_epochs):
        
        lr = optimizer.param_groups[0]['lr']
        history_dict['learning_rate'].append(lr)
        
        for phase in ['train', 'val']:
            phase_start_time = time()
            tqdm_postfix = {'lr': f'{lr:.2E}',}
            
            if phase == 'train':
                torch.set_grad_enabled(True)
                optimizer.zero_grad() # just in case
                model.train()
                loader = tqdm(
                    train_loader, 
                    desc=f'Epoch {epoch+1}/{args.num_epochs} Training', 
                    total=len(train_loader),
                    unit='batch',
                    postfix=tqdm_postfix,
                )
            
            else:
                torch.set_grad_enabled(False)
                model.eval()
                loader = tqdm(
                    val_loader, 
                    desc=f'Epoch {epoch+1}/{args.num_epochs} Validation', 
                    total=len(val_loader),
                    unit='batch',
                    postfix=tqdm_postfix,
                )

            if 'simclr' in args.pretrain_scheme:
                cache, closure = utils.init_grad_cache_closure_dicts(train_dataset.n_views)
                contrastive_loss_values = []
            
            if 'hsv' in args.pretrain_scheme:
                reconstruction_losses = []
                reconstruction_loss_values = []
            
            for step, batch in enumerate(loader):
                            
                with autocast(device.type):
                    
                    if args.pretrain_scheme == 'simclr':
                        for view in range(len(batch)):
                            X = batch[view]
                            X = X.to(device)
                            z, cz = utils.cached_model_call(model, X)
                            
                            cache[f'z_{view}'].append(z)
                            closure[f'z_{view}'].append(cz)
                        
                    elif args.pretrain_scheme == 'hsv':
                        X, y = batch
                        X, y = X.to(device), y.to(device)
                        y_hat = model(X)
                        y_hat = F.sigmoid(y_hat) # sigmoid as y is in [0, 1]
                        
                        reconstruction_losses.append(utils.normalized_mse_loss(y_hat, y))
                    
                    elif args.pretrain_scheme == 'hsv_simclr':
                        for view in range(len(batch)):
                            X, y = batch[view]
                            X, y = X.to(device), y.to(device)
                            y_hat, z, cz = utils.cached_model_call(model, X) # do not need to cache y_hat
                            y_hat = F.sigmoid(y_hat) # sigmoid as y is in [0, 1] 
                            
                            reconstruction_losses.append(utils.normalized_mse_loss(y_hat, y))
                            cache[f'z_{view}'].append(z)
                            closure[f'z_{view}'].append(cz)
                        
                
                if (step + 1) % grad_accum_steps == 0:
                                                            
                    if 'simclr' in args.pretrain_scheme:
                        contrastive_loss = utils.cached_contrastive_loss_call(cache['z_0'], cache['z_1'])
                        contrastive_loss_values.append(contrastive_loss.item())
                        epoch_contrastive_loss = np.sum(contrastive_loss_values) / ((step * args.mini_batch_size) + len(batch)) 
                        tqdm_postfix['NT-Xent Loss'] = f'{epoch_contrastive_loss:.2f}'
                    
                    if 'hsv' in args.pretrain_scheme:
                        reconstruction_loss = torch.stack(reconstruction_losses).sum(dim=0)
                        reconstruction_loss_values.append(reconstruction_loss.item())
                        epoch_reconstruction_loss = np.sum(reconstruction_loss_values) / ((step * args.mini_batch_size) + len(batch)) 
                        tqdm_postfix['MSE Loss'] = f'{epoch_reconstruction_loss:.2f}'
                    
                    if args.pretrain_scheme == 'hsv_simclr':
                        total_loss = (reconstruction_loss_weight * reconstruction_loss) + (contrastive_loss_weight * contrastive_loss)
                        epoch_loss = (reconstruction_loss_weight * epoch_reconstruction_loss) + (contrastive_loss_weight * epoch_contrastive_loss)
                        tqdm_postfix['Total Loss'] = f'{epoch_loss:.2f}'
                    
                    else:
                        total_loss = contrastive_loss if 'simclr' in args.pretrain_scheme else reconstruction_loss
                        epoch_loss = epoch_contrastive_loss if 'simclr' in args.pretrain_scheme else epoch_reconstruction_loss
                    
                    if phase == 'train':
                        scaler.scale(total_loss).backward()
                        if 'simclr' in args.pretrain_scheme:
                            utils.call_closures(cache, closure)
                        
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad(set_to_none=True)
                    
                    if 'simclr' in args.pretrain_scheme:
                        cache, closure = utils.init_grad_cache_closure_dicts(train_dataset.n_views)
                    
                    if 'hsv' in args.pretrain_scheme:
                        reconstruction_losses = []
                    
                    loader.set_postfix(tqdm_postfix)
                                
            history_dict[f'{phase}_total_loss'].append(epoch_loss)            
            if args.pretrain_scheme == 'hsv_simclr':
                history_dict[f'{phase}_reconstruction_loss'].append(epoch_reconstruction_loss)
                history_dict[f'{phase}_contrastive_loss'].append(epoch_contrastive_loss)
            
            if phase == 'val':
                if epoch_loss < best_val_loss:
                    logger.log(f'Validation loss improved from {best_val_loss:.2f} at epoch {best_epoch} to {epoch_loss:.2f} during epoch {epoch}. Saving model...')
                    best_val_loss = epoch_loss
                    best_epoch = epoch
                    torch.save(model.state_dict(), os.path.join(out_dir, f'{args.pretrain_scheme}.pth'))
                    torch.save(optimizer.state_dict(), os.path.join(log_dir, 'optimizer.pth'))
                    torch.save(scheduler.state_dict(), os.path.join(log_dir, 'scheduler.pth'))
                    torch.save(scaler.state_dict(), os.path.join(log_dir, 'scaler.pth'))
                    with open(os.path.join(log_dir, 'best_epoch.txt'), 'w') as f:
                        f.write(str(best_epoch)) # just in case

            profiler.update(epoch=epoch, phase=phase, time=time()-phase_start_time)
            profiler.save(os.path.join(log_dir, 'profiler.csv'))
            
        scheduler.step()
        
        history_df = pd.DataFrame(history_dict).set_index(pd.Index(range(epoch+1)))
        history_df.to_csv(os.path.join(log_dir, 'history.csv'), index=True)
        
        if epoch - best_epoch > args.early_stopping_patience:
            logger.log(f'No improvement in validation loss for {args.early_stopping_patience} epochs. Stopping early.')
            break



if __name__ == '__main__':
    main()