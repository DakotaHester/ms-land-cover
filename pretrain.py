from multiprocessing import get_context
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import SequentialLR, ReduceLROnPlateau, CosineAnnealingLR, LambdaLR
from torch.optim import Adam
from calflops import calculate_flops
import numpy as np
from argparse import ArgumentParser
import os
from glob import glob
from tqdm import tqdm
import pandas as pd
from time import time
import json
import math

from src.mslandcover.data.datasets import PreTrainDataset
from src.mslandcover.data import transforms
from src.mslandcover.models import HRNetSegmentationModel
from src.mslandcover.optim import LARS, PCGradAMP, UncertainLossWeighter
from src.mslandcover import config
from src.mslandcover import utils

def parse_arguments():
    parser = ArgumentParser()
    
    parser.add_argument(
        '--pretrain_scheme',
        type=str,
        default='dae_simclr',
        choices=['hsv', 'simclr', 'hsv_simclr', 'dae', 'dae_simclr'],
        help='The pretraining scheme to use. One of ["hsv", "simclr", "hsv_simclr", "dae", "dae_simclr"].',
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
        default=15,
        help='The number of epochs to wait for validation loss improvement before stopping training.',
    )
    
    parser.add_argument(
        '--reduce_lr_patience',
        type=int,
        default=5,
        help='The number of epochs to wait for validation loss improvement before reducing the learning rate.',
    )
    
    parser.add_argument(
        '--learning_rate_factor',
        type=float,
        default=.0001,
        help='The factor by which to reduce the learning rate after loading the imagenet weights.',
    )
    
    parser.add_argument(
        '--full_batch_size', 
        type=int, 
        default=1024, # 4096 in original SimCLR implementation
        help='The batch size to use for pretraining.',
    )
    
    parser.add_argument(
        '--mini_batch_size',
        type=int,
        default=16,
        help='The mini-batch size to use for gradient caching.',
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
        '--use_imagenet_weights',
        default=False,
        action='store_true',
        help='Use the pretrained weights from the ImageNet classification task.',
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
    
    parser.add_argument(
        '--use_amp',
        default=True,
        action='store_true',
        help='Use Automatic Mixed Precision (AMP) for training.',
    )
    
    parser.add_argument(
        '--use_pcgrad',
        default=True,
        action='store_true',
        help='Use Projected Conflicting Gradients (PCGrad) for training.',
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
    if not args.use_imagenet_weights:
        log_dir += '_randinit'
    out_dir = os.path.join(args.weights_dir, args.model)
    if not args.use_imagenet_weights:
        out_dir += '_randinit'
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)
    
    logger = utils.Logger(os.path.join(log_dir, 'log.txt'))
    
    logger.log(f'Configuration:')
    for k, v in vars(args).items():
        logger.log(f'{k}: {v}', prepend_timestamp=False)
    logger.log('='*20, prepend_timestamp=False)
    
    is_contrastive = 'simclr' in args.pretrain_scheme
    is_reconstruction = 'hsv' in args.pretrain_scheme or 'dae' in args.pretrain_scheme
    is_multitask = is_contrastive and is_reconstruction
    
    device = utils.get_torch_device()
    logger.log(f'Using device: {device}')
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True
        torch.cuda.manual_seed_all(args.seed)
    
    transform = transforms.SimCLRDataAugmentation(size=args.image_size) if is_contrastive \
        else transforms.StandardDataAugmentations()
    return_hsv = 'hsv' in args.pretrain_scheme
    noisy_input = 'dae' in args.pretrain_scheme
    n_views = 2 if is_contrastive else 1
    
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
        noisy_input=noisy_input,
    )
    val_dataset = PreTrainDataset(
        hdf5_path=args.pretrain_hdf5_path,
        hdf5_group=args.pretrain_val_hdf5_group,
        n_views=n_views,
        mean=train_dataset.mean,
        std=train_dataset.std,
        transform=None,
        return_hsv=return_hsv,
        noisy_input=noisy_input,
    )
    
    if args.debug:
        train_dataset.ids_list = train_dataset.ids_list[:(512)*4]
        val_dataset.ids_list = val_dataset.ids_list[:512]
        
        args.full_batch_size = 512
        args.mini_batch_size = 16
    
    logger.log(f'Training dataset size: {len(train_dataset)}')
    logger.log(f'Validation dataset size: {len(val_dataset)}')
    
    logger.log(f'Training dataset mean: {train_dataset.mean}')
    logger.log(f'Training dataset std: {train_dataset.std}')
    
    # save the mean and std for the training dataset
    if mean is None:  torch.save(train_dataset.mean, mean_path)
    if std is None: torch.save(train_dataset.std, std_path) 
    
    # take into account grad cache steps when setting the batch size
    grad_accum_steps = args.full_batch_size // args.mini_batch_size
    logger.log(f'Full batch size: {args.full_batch_size}, Mini batch size: {args.mini_batch_size}, Grad cache steps: {grad_accum_steps}')
    
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
            noisy_input,
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
        img_decoder_head=is_reconstruction,
        aux_simclr_head=is_contrastive,
        img_decoder_activation='sigmoid' if 'hsv' in args.pretrain_scheme else 'none',
    )
    if args.use_imagenet_weights:
        imagenet_weights = torch.load(os.path.join(out_dir, 'imagenet.pth'), weights_only=True)
        model.load_encoder_weights(imagenet_weights)
        args.learning_rate_factor *= 0.01 # reduce the learning rate as model is pretrained
    model.to(device)
    
    flops, macs, _ = calculate_flops(
        model=model,
        input_shape=(1, 3, 256, 256),
        output_as_string=False,
        print_results=False,
        print_detailed=False,
    )
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.log(f'Model GFLOPs: {flops*1e-9:,.2f} | GMACs: {macs*1e-9:,.2f} | Total Parameters: {n_params:,} | Trainable Parameters: {n_trainable_params:,}')
    with open(os.path.join(log_dir, 'model_complexity.json'), 'w') as f:
        json.dump({
            'flops': flops,
            'macs': macs,
            'total_parameters': n_params,
            'trainable_parameters': n_trainable_params,
        }, f, indent=4)
    
    # define optimizer and scheduler - the specifics are taken from the original SimCLR implementation
    learning_rate = 0.3 * args.full_batch_size / 256 # per original SimCLR implementation
    learning_rate *= args.learning_rate_factor # reduce the learning rate as model is pretrained
    
    # note: uncertainty based loss weighting usedo for multi-task learning, params 
    # need to be included in the optimizer
    params = list(model.parameters())
    if is_multitask:
        loss_weighter = UncertainLossWeighter(
            num_tasks=2,
        ).to(device)
        params.extend(list(loss_weighter.parameters()))
    
    # optimizer = LARS(
    #     params=params,
    #     lr=learning_rate, # per original SimCLR implementation
    #     weight_decay=1e-6,
    # )
    
    optimizer = Adam(
        params=params,
        lr=learning_rate,
        # weight_decay=1e-6,
    )
    
    warmup_epochs = 10
    scheduler = SequentialLR(
        optimizer=optimizer,
        schedulers=[
            LambdaLR( # linear warmup for 10 epochs
                optimizer=optimizer,
                lr_lambda=lambda epoch: min(1, (epoch+1) / warmup_epochs),
            ),
            CosineAnnealingLR( # cosine annealing with no restarts
                optimizer=optimizer,
                T_max=args.num_epochs-warmup_epochs,
            ),
        ],
        milestones=[warmup_epochs], # step after warmup
    )
    lr_reducer = ReduceLROnPlateau(
        optimizer=optimizer,
        patience=args.reduce_lr_patience,
    )
    if args.use_amp:
        scaler = GradScaler() # AMP for gradient scaling
    
    args.use_pcgrad = args.use_pcgrad and is_multitask 
    # Gradient Surgery (projected conflicting gradients) - only used for multi-task learning
    if args.use_pcgrad:
        grad_optimizer = PCGradAMP(
            num_tasks=2,
            optimizer=optimizer,
            scaler=scaler if args.use_amp else None,
        )
    
    cache_contents = []
    if is_contrastive:
        cache_contents.append('z')
        if is_multitask:
            cache_contents.extend(['y', 'y_hat'])
    
    history_dict = {
        'learning_rate': [],
    }
    for phase in ['train', 'val']:
        history_dict[f'{phase}_total_loss'] = []
        if is_multitask:
            history_dict[f'{phase}_reconstruction_loss'] = []
            history_dict[f'{phase}_contrastive_loss'] = []
    
    profiler = utils.ProfilerHistory(device)
    profiler.update(epoch=-1, phase='init', step=0, time=0)
    
    best_val_loss = np.inf
    best_epoch = -1
    logger.log(f'Starting training...')
    for epoch in range(args.num_epochs):
        
        lr = optimizer.param_groups[0]['lr']
        history_dict['learning_rate'].append(lr)
        
        for phase in ['train', 'val']:
            phase_start_time = time()
            tqdm_postfix = {'lr': f'{lr:.2e}',}
            
            if phase == 'train':
                torch.set_grad_enabled(True)
                optimizer.zero_grad() # just in case
                model.train()
                loader = train_loader 
                pbar = tqdm(
                    desc=f'Epoch {epoch+1}/{args.num_epochs} Training', 
                    total=math.ceil(len(train_loader) / grad_accum_steps),
                    unit='batch',
                    postfix=tqdm_postfix,
                )
            
            else:
                torch.set_grad_enabled(False)
                model.eval()
                loader = val_loader
                pbar = tqdm(
                    desc=f'Epoch {epoch+1}/{args.num_epochs} Validation', 
                    total=math.ceil(len(val_loader) / grad_accum_steps),
                    unit='batch',
                    postfix=tqdm_postfix,
                )
            
            if is_contrastive:
                cache, closures = utils.init_grad_cache_closure_dicts(n_views, cache_contents)
                contrastive_loss_values = []
            
            if is_reconstruction:
                reconstruction_loss_values = []
            
            if is_multitask:
                total_loss_values = []
            
            for step, batch in enumerate(loader):
                            
                with autocast(device.type, enabled=args.use_amp):
                    
                    if is_contrastive and not is_multitask:
                        for view in range(n_views):
                            X = batch[view]
                            X = X.to(device)
                            z, closure = utils.cached_model_call(model, X)
                            
                            cache[view]['z'].append(z)
                            closures[view].append(closure)
                        
                    elif is_reconstruction and not is_multitask:
                        X, y = batch
                        X, y = X.to(device), y.to(device)
                        
                        y_hat = model(X)
                        reconstruction_loss = F.mse_loss(y_hat, y, reduction='sum')
                        reconstruction_loss_values.append(reconstruction_loss.item())
                        
                        if phase == 'train':
                            if args.use_amp:
                                scaler.scale(reconstruction_loss).backward()
                            else:
                                reconstruction_loss.backward()
                        
                    elif is_multitask:
                        for view in range(n_views):
                            X, y = batch[view]
                            X, y = X.to(device), y.to(device)
                            y_hat, z, closure = utils.cached_model_call(model, X) 
                            
                            cache[view]['y'].append(y)
                            cache[view]['y_hat'].append(y_hat)
                            cache[view]['z'].append(z)
                            closures[view].append(closure)
                        
                if (step + 1) % grad_accum_steps == 0:
                                                            
                    if is_contrastive:
                        contrastive_loss = utils.cached_contrastive_loss_call(cache[0]['z'], cache[1]['z'])
                        contrastive_loss_values.append(contrastive_loss.item())
                        epoch_contrastive_loss = np.sum(contrastive_loss_values) / ((step * args.mini_batch_size) + len(batch)) 
                        tqdm_postfix['NT-Xent Loss'] = f'{epoch_contrastive_loss:.2e}'
                    
                    if is_reconstruction:
                        if is_multitask:
                            reconstruction_loss = torch.tensor(0.0, device=device)
                            for view in range(n_views):
                                reconstruction_loss += utils.cached_mse_loss_call(cache[view]['y_hat'], cache[view]['y'])
                            reconstruction_loss_values.append(reconstruction_loss.item())
                        epoch_reconstruction_loss = np.sum(reconstruction_loss_values) / ((step * args.mini_batch_size) + len(batch)) 
                        tqdm_postfix['MSE Loss'] = f'{epoch_reconstruction_loss:.2e}'
                    
                    if is_multitask:
                        loss = [contrastive_loss, reconstruction_loss]
                        loss = loss_weighter(loss)
                        total_loss_values.append(sum([l.item() for l in loss]))
                        epoch_loss = np.sum(total_loss_values) / ((step * args.mini_batch_size) + len(batch))
                        tqdm_postfix['Total Loss'] = f'{epoch_loss:.2e}'
                    
                    else:
                        loss = contrastive_loss if is_contrastive else reconstruction_loss
                        epoch_loss = epoch_contrastive_loss if is_contrastive else epoch_reconstruction_loss
                    
                    if phase == 'train':
                        if args.use_pcgrad:
                            grad_optimizer.backward(loss) 
                            utils.call_closures(cache, closures) 
                            grad_optimizer.step()
                            
                        elif args.use_amp:
                            if not is_reconstruction: # backward pass already done for reconstruction loss
                                scaler.scale(loss).backward()
                            if is_contrastive: 
                                utils.call_closures(cache, closures)
                            scaler.step(optimizer)
                            scaler.update()
                            
                        else:
                            if not is_reconstruction: # backward pass already done for reconstruction loss
                                loss.backward()
                            if is_contrastive: 
                                utils.call_closures(cache, closures)
                            optimizer.step()
                            
                        optimizer.zero_grad()
                    
                    cache, closures = utils.init_grad_cache_closure_dicts(n_views, cache_contents)
                    
                    pbar.set_postfix(tqdm_postfix)
                    pbar.update(1)

                profiler.update(epoch=epoch, phase=phase, step=step, time=time()-phase_start_time)
                                
            history_dict[f'{phase}_total_loss'].append(epoch_loss)            
            if is_multitask:
                history_dict[f'{phase}_reconstruction_loss'].append(epoch_reconstruction_loss)
                history_dict[f'{phase}_contrastive_loss'].append(epoch_contrastive_loss)
            
            profiler.save(os.path.join(log_dir, 'profiler.csv'))
            
        if epoch_loss < best_val_loss:
            logger.log(f'Validation loss improved from {best_val_loss:.5f} at epoch {best_epoch} to {epoch_loss:.5f} during epoch {epoch}. Saving model...')
            best_val_loss = epoch_loss
            best_epoch = epoch
            torch.save(model.state_dict(), os.path.join(out_dir, f'{args.pretrain_scheme}.pth'))
            torch.save(optimizer.state_dict(), os.path.join(log_dir, 'optimizer.pth'))
            torch.save(scheduler.state_dict(), os.path.join(log_dir, 'scheduler.pth'))
            if args.use_amp:
                torch.save(scaler.state_dict(), os.path.join(log_dir, 'scaler.pth'))
            if args.use_pcgrad:
                torch.save(grad_optimizer.state_dict(), os.path.join(log_dir, 'grad_optimizer.pth'))
            # torch.save(scaler.state_dict(), os.path.join(log_dir, 'scaler.pth'))
            with open(os.path.join(log_dir, 'best_epoch.txt'), 'w') as f:
                    f.write(str(best_epoch)) # just in case
        
        history_df = pd.DataFrame(history_dict).set_index(pd.Index(range(epoch+1)))
        history_df.to_csv(os.path.join(log_dir, 'history.csv'), index=True)
                
        if epoch - best_epoch > args.early_stopping_patience:
            logger.log(f'No improvement in validation loss for {args.early_stopping_patience} epochs. Stopping early.')
            break
        
        scheduler.step()
        if epoch >= warmup_epochs: # wait until after warmup to call ReduceLROnPlateau
            old_lr = optimizer.param_groups[0]['lr']
            lr_reducer.step(history_dict['val_total_loss'][-1])
            new_lr = optimizer.param_groups[0]['lr']
            if old_lr != new_lr:
                logger.log(f'No improvement in validation loss for {args.reduce_lr_patience} epochs. Reducing learning rate from {lr:.2e} to {new_lr:.2e}.')



if __name__ == '__main__':
    main()