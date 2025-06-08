import math
from time import time
from typing import OrderedDict
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision.models import resnet152, ResNet152_Weights, resnet101, ResNet101_Weights
import numpy as np
import os
from glob import glob
from calflops import calculate_flops
from tqdm import tqdm
import json

from mslandcover.utils import ProfilerHistory, Logger, get_torch_device, load_pth
from mslandcover.data.datasets import PreTrainDataset
from mslandcover.data import transforms
from mslandcover.models import DeepLabV3Plus, ProjectionHead, ResNetBackbone
from mslandcover.loss import nt_xent_loss
from mslandcover.metrics import psnr, ssim
from argparse import ArgumentParser

def parse_arguments():
    parser = ArgumentParser()
    
    parser.add_argument(
        '--pretrain_scheme',
        type=str,
        default='simclr',
        choices=['dae', 'simclr', 'hires_simclr'],
        help='The pretraining scheme to use.',
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default='resnet101',
        choices=['resnet101', 'resnet152', 'deeplabv3p'],
    )
    
    parser.add_argument(
        '--pretrain_data_dir', 
        type=str,  
        default='/scratch/dhester/mslc_data_v2/pretrain/',
        help='Path to the directory containing the pretraining data.',
    )
    
    parser.add_argument(
        '--pretrain_val_data_dir', 
        type=str,  
        default='/scratch/dhester/mslc_data_v2/pretrain_val/',
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
        default=5,
        help='The number of epochs to wait for validation loss improvement before stopping training.',
    )
    
    parser.add_argument(
        '--reduce_lr_patience',
        type=int,
        default=0,
        help='The number of epochs to wait for validation loss improvement before reducing the learning rate.',
    )
    
    parser.add_argument(
        '--init_lr',
        type=float,
        default=1e-5, # NOTE: TYPICALLY SET TO 1e-6, setting to 1e-7 for uresnetd
        help='The initial learning rate to use for training.',
    )
    
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.05,
        help='The temperature to use for the NT-Xent loss.',
    )
    
    parser.add_argument(
        '--full_batch_size', 
        type=int, 
        default=128, # 4096 in original SimCLR implementation
        help='The batch size to use for pretraining.',
    )
    
    parser.add_argument(
        '--mini_batch_size',
        type=int,
        default=128,
        help='The mini-batch size to use for gradient accumulation. Only relevant if using pretraining scheme "dae".',
    )
    
    parser.add_argument(
        '--frozen_encoder',
        default=False,
        action='store_true',
        help='Freeze the encoder during training.',
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
        '--encoder_weights',
        type=str,
        default=None,
        help='Path to the encoder weights to use for training.',
    )
    
    parser.add_argument(
        '--rand_init',
        default=False,
        action='store_true',
        help='Use random initialization for the model instead of Imagenet weights.',
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
        default=8,
        help='The number of workers to use for data loading.',
    )
    
    parser.add_argument(
        '--image_size',
        type=int,
        default=256,
        help='The size of the input.'
    )
    
    parser.add_argument(
        '--n_bands',
        type=int,
        default=4,
        help='The number of bands in the input data. If 3, then color infrared composites are used (NIR, Red, Green)',
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
    
    parser.add_argument(
        '--load_checkpoint',
        default=False,
        action='store_true',
        help='Load a checkpoint from the log directory and resume training.',
    )
    
    parser.add_argument(
        '--preload_data',
        default=False,
        action='store_true',
        help='Preload the entire dataset into memory. NOTE: This will use a lot of memory.',
    )
    
    return parser.parse_args()




def main():
    
    args = parse_arguments()
    
    torch.random.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    log_dir = os.path.join(args.log_dir, args.model, args.pretrain_scheme)
    if args.rand_init:
        log_dir += '_randinit'
    if args.frozen_encoder:
        log_dir += '_frozenencoder'
    out_dir = os.path.join(args.weights_dir, args.model)
    if args.rand_init:
        out_dir += '_randinit'
    if args.frozen_encoder:
        out_dir += '_frozenencoder'
        
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)
    
    logger = Logger(os.path.join(log_dir, 'log.txt'))
    
    logger.log(f'Configuration:')
    for k, v in vars(args).items():
        logger.log(f'{k}: {v}', prepend_timestamp=False)
    logger.log('='*20, prepend_timestamp=False)
    
    is_contrastive = 'simclr' in args.pretrain_scheme
    is_reconstruction = 'dae' in args.pretrain_scheme 
    
    if is_contrastive and args.mini_batch_size != args.full_batch_size:
        raise ValueError('Contrastive learning requires the mini-batch size to be equal to the full batch size.')
    
    device = get_torch_device()
    logger.log(f'Using device: {device}')
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True
        torch.cuda.manual_seed_all(args.seed)
    
    if is_contrastive:
        if args.pretrain_scheme == 'hires_simclr':
            transform = transforms.HiResDataAugmentation(size=args.image_size, s=1.0)
        else:
            transform = transforms.SimCLRDataAugmentation(size=args.image_size, s=1.0)
    else:
        transform = transforms.StandardDataAugmentations(size=args.image_size, use_color_transforms=False)

    noisy_input = args.pretrain_scheme in ['dae', 'hires_simclr']
    n_views = 2 if is_contrastive else 1
    
    if args.n_bands == 4:
        mean_path = os.path.join('weights', 'pretrain_mean_4.pt')
        std_path = os.path.join('weights', 'pretrain_std_4.pt')
    else:
        mean_path = os.path.join('weights', 'pretrain_mean.pth')
        std_path = os.path.join('weights', 'pretrain_std.pth')
    
    mean = load_pth(mean_path) if os.path.exists(mean_path) else None
    std = load_pth(std_path) if os.path.exists(std_path) else None
    
    train_dataset = PreTrainDataset(
        # hdf5_path=args.pretrain_hdf5_path,
        # hdf5_group=args.pretrain_hdf5_group,
        data_paths=glob(os.path.join(args.pretrain_data_dir, '*.tif')),
        transform=transform,
        n_views=n_views,
        mean=mean,
        std=std,
        noisy_input=noisy_input,
        noise_std=1.0,
        # noise_pct=0.5,
        preload=args.preload_data,
        n_bands=args.n_bands,
    )
    val_dataset = PreTrainDataset(
        # hdf5_path=args.pretrain_hdf5_path,
        # hdf5_group=args.pretrain_val_hdf5_group,
        data_paths=glob(os.path.join(args.pretrain_val_data_dir, '*.tif')),
        n_views=n_views,
        mean=train_dataset.mean,
        std=train_dataset.std,
        transform=transforms.ResizeTransform(size=args.image_size),
        noisy_input=noisy_input,
        noise_std=1.0,
        # noise_pct=0.5,
        preload=args.preload_data,
        n_bands=args.n_bands,
    )
    
    if args.debug:
        train_dataset.ids_list = train_dataset.ids_list[:(512)*4]
        val_dataset.ids_list = val_dataset.ids_list[:512]
        
        args.full_batch_size = 128
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
            False,
            False,
            False,
            noisy_input,
            train_dataset,
            glob(os.path.join(args.pretrain_data_dir, '*.tif')),
            args.pretrain_scheme,
            train_dataset.mean,
            train_dataset.std,
        )
        return

    if args.model == 'resnet152':
        encoder = resnet152(weights=ResNet152_Weights.IMAGENET1K_V2 if not args.rand_init else None)
        if args.n_bands == 4:
            encoder.conv1 = nn.Conv2d(4, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        encoder.fc = nn.Identity()
        model = nn.Sequential(OrderedDict([
            ('encoder', encoder),
            ('projection_head', ProjectionHead(in_channels=2048))
        ]))
    
    elif args.model == 'resnet101':
        encoder = resnet101(weights=ResNet101_Weights.IMAGENET1K_V2 if not args.rand_init else None)
        if args.n_bands == 4:
            encoder.conv1 = nn.Conv2d(4, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        encoder.fc = nn.Identity()
        model = nn.Sequential(OrderedDict([
            ('encoder', encoder),
            ('projection_head', ProjectionHead(in_channels=2048))
        ]))
     
    elif args.model == 'deeplabv3p':
        raise NotImplementedError('DeepLabV3+ not implemented yet.')
        # encoder = ResNetBackbone()
        # model = DeepLabV3Plus(
        #     backbone=ResNetBackbone(pretrained=args.encoder_weights if args.encoder_weights is not None else not args.rand_init, in_channels=args.n_bands),
        #     num_classes=args.n_bands,
        # )
        
    else:
        raise ValueError(f'Invalid model: {args.model}')
    
    model.to(device)
    
    if args.frozen_encoder and args.model not in ['deeplabv3p']:
        for encoder_block in model.encoder_blocks:
            for param in encoder_block.parameters():
                param.requires_grad = False
    elif args.frozen_encoder:
        for param in model.backbone.parameters():
            param.requires_grad = False
        # if not loading encoder weights, then unfreeze the backbone.initial.
        if args.encoder_weights is None:
            logger.log('NOTE! Unfreezing backbone.initial parameters to account for discrepancies in channels.')
            for param in model.backbone.initial[0].parameters():
                param.requires_grad = True
    
    flops, macs, _ = calculate_flops(
        model=model,
        input_shape=(1, args.n_bands, args.image_size, args.image_size),
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
        

    optimizer = torch.optim.AdamW(params=model.parameters(), lr=args.init_lr)
    
    reduce_lr_on_plateau = ReduceLROnPlateau(
        optimizer=optimizer,
        patience=args.reduce_lr_patience,
        factor=0.1,
        eps=0,
    )
    
    
    history_dict = {
        'learning_rate': [],
    }
    for phase in ['train', 'val']:
        history_dict[f'{phase}_loss'] = []
        if is_reconstruction:
            history_dict[f'{phase}_psnr'] = []
            history_dict[f'{phase}_ssim'] = []
    
    profiler = ProfilerHistory(device)
    profiler.update(epoch=-1, phase='init', step=0, time=0)
    
    starting_epoch = 0 # epoch 0 is a "dry run" to get a baseline loss
    best_val_loss = np.inf
    best_epoch = -1
    
    if args.load_checkpoint:
        checkpoint = torch.load(os.path.join(log_dir, 'checkpoint.pth'), weights_only=False)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        reduce_lr_on_plateau.load_state_dict(checkpoint['reduce_lr_on_plateau'])
        
        starting_epoch = checkpoint['epoch']  + 1 # start from the next epoch (checkpoints are saved at the end of the epoch)
        best_val_loss = checkpoint['best_val_loss']
        best_epoch = checkpoint['best_epoch']
        
        history_dict = checkpoint['history']
        profiler.profiler_history_dict = checkpoint['profiler']
        
        logger.log(f'Loaded checkpoint from epoch {starting_epoch - 1}.')
    
    logger.log(f'Starting training at epoch {starting_epoch}...')
    for epoch in range(starting_epoch, args.num_epochs+1):
        
        lr = optimizer.param_groups[0]['lr']
        history_dict['learning_rate'].append(lr)
        
        for phase in ['train', 'val']:
            phase_start_time = time()
            tqdm_postfix = {'LR': f'{lr:.0e}',}
            
            if phase == 'train':
                torch.set_grad_enabled(epoch != 0) # disable backpropagation for the first epoch to get a baseline loss
                optimizer.zero_grad() # just in case
                model.train()
                loader = train_loader 
            
            else:
                torch.set_grad_enabled(False)
                model.eval()
                loader = val_loader
            
            total_steps = math.ceil(len(loader) // grad_accum_steps) if phase == 'train' else len(loader) // grad_accum_steps
            with tqdm(
                desc=f'Epoch {epoch}/{args.num_epochs} {phase.capitalize()}', 
                total=total_steps,
                unit='batch',
                postfix=tqdm_postfix,
            ) as pbar:
                
                loss_values = []
                if is_reconstruction:
                    psnr_values = []
                    ssim_values = []
                
                # A note on extensive profiling:
                # Trying to track the memory usage of each method as extensively as possible
                for step, batch in enumerate(loader):
                    if is_contrastive:
                        # load batch
                        X_0, X_1 = batch
                        profiler.update(epoch=epoch, phase=phase, step=step, time=time()-phase_start_time)
                        
                        # send to GPU
                        X_0, X_1 = X_0[0].to(device), X_1[0].to(device)
                        profiler.update(epoch=epoch, phase=phase, step=step, time=time()-phase_start_time)
                        
                        # first pass through model
                        z_0 = model(X_0)
                        profiler.update(epoch=epoch, phase=phase, step=step, time=time()-phase_start_time)
                        
                        # second pass through model
                        z_1 = model(X_1)
                        profiler.update(epoch=epoch, phase=phase, step=step, time=time()-phase_start_time)
                        
                        # calculate loss
                        loss = nt_xent_loss(z_0, z_1, temperature=args.temperature, reduction='sum')
                        profiler.update(epoch=epoch, phase=phase, step=step, time=time()-phase_start_time)
                        
                        # metrics
                        loss_values.append(loss.item())
                        epoch_loss = np.sum(loss_values) / ((step * args.mini_batch_size) + len(batch))
                        tqdm_postfix['Loss'] = f'{epoch_loss:.2e}'
                        
                        # no grad accumulation for contrastive loss - can go ahead and update the model
                        if phase == 'train' and epoch != 0:
                            # backpropagation
                            loss.backward()
                            profiler.update(epoch=epoch, phase=phase, step=step, time=time()-phase_start_time)
                            
                            # parameter update
                            optimizer.step()
                            profiler.update(epoch=epoch, phase=phase, step=step, time=time()-phase_start_time)
                            
                            # reset gradients
                            optimizer.zero_grad()
                            profiler.update(epoch=epoch, phase=phase, step=step, time=time()-phase_start_time)
                        
                        pbar.set_postfix(tqdm_postfix)
                        pbar.update(1)
                    
                    else:
                        # load batch
                        X, y = batch
                        profiler.update(epoch=epoch, phase=phase, step=step, time=time()-phase_start_time)
                        
                        # send to GPU
                        X, y = X.to(device), y.to(device)
                        profiler.update(epoch=epoch, phase=phase, step=step, time=time()-phase_start_time)
                        
                        # forward pass through model
                        y_hat = model(X)
                        profiler.update(epoch=epoch, phase=phase, step=step, time=time()-phase_start_time)
                        
                        # reconstruction_loss = F.mse_loss(y_hat, y, reduction='sum')
                        # NOTE: L1/MAE loss is used instead of MSE loss 
                        # https://research.nvidia.com/sites/default/files/pubs/2017-03_Loss-Functions-for/NN_ImgProc.pdf
                        # https://openaccess.thecvf.com/content/WACV2022/papers/Mustafa_Training_a_Task-Specific_Image_Reconstruction_Loss_WACV_2022_paper.pdf
                        loss = F.l1_loss(y_hat, y, reduction='sum') 
                        profiler.update(epoch=epoch, phase=phase, step=step, time=time()-phase_start_time)
                        
                        loss_values.append(loss.item())
                        ssim_values.append(ssim(y_hat, y, reduction='sum').item())
                        psnr_values.append(psnr(y_hat, y, reduction='sum').item())
                        
                        epoch_loss = np.sum(loss_values) / ((step * args.mini_batch_size) + len(batch))
                        epoch_ssim = np.sum(ssim_values) / ((step * args.mini_batch_size) + len(batch))
                        epoch_psnr = np.sum(psnr_values) / ((step * args.mini_batch_size) + len(batch))
                        tqdm_postfix['Loss'] = f'{epoch_loss:.2e}'
                        tqdm_postfix['PSNR'] = f'{epoch_psnr:.2f}'
                        tqdm_postfix['SSIM'] = f'{epoch_ssim:.2f}'
                        
                        pbar.set_postfix(tqdm_postfix)
                        
                        if phase == 'train' and epoch != 0:
                            # backpropagation
                            loss.backward()
                            profiler.update(epoch=epoch, phase=phase, step=step, time=time()-phase_start_time)
                            
                        if (step + 1) % grad_accum_steps == 0:
                            if phase == 'train' and epoch != 0:
                                # parameter update
                                optimizer.step()
                                profiler.update(epoch=epoch, phase=phase, step=step, time=time()-phase_start_time)
                            
                                # reset gradients
                                optimizer.zero_grad()
                                profiler.update(epoch=epoch, phase=phase, step=step, time=time()-phase_start_time)
                        
                            pbar.update(1)

                    profiler.update(epoch=epoch, phase=phase, step=step, time=time()-phase_start_time)
                                   
                history_dict[f'{phase}_loss'].append(epoch_loss)            
                if is_reconstruction:
                    history_dict[f'{phase}_psnr'].append(epoch_psnr)
                    history_dict[f'{phase}_ssim'].append(epoch_ssim)
                
                profiler.save(os.path.join(log_dir, 'profiler.csv'))
                
        if epoch_loss < best_val_loss:
            logger.log(f'Validation loss improved from {best_val_loss:.5f} at epoch {best_epoch} to {epoch_loss:.5f} during epoch {epoch}. Saving model...')
            best_val_loss = epoch_loss
            best_epoch = epoch
            torch.save(model.state_dict(), os.path.join(out_dir, f'{args.pretrain_scheme}.pth'))
        
        reduce_lr_on_plateau.step(epoch_loss)
        checkpoint = {
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'reduce_lr_on_plateau': reduce_lr_on_plateau.state_dict(),
            'history': history_dict,
            'profiler': profiler.profiler_history_dict,
            'best_epoch': best_epoch,
            'best_val_loss': best_val_loss,
        }
        torch.save(checkpoint, os.path.join(log_dir, 'checkpoint.pth'))
        with open(os.path.join(log_dir, 'best_epoch.txt'), 'w') as f:
            f.write(str(best_epoch)) # just in case
        
        num_epochs_total = len(history_dict['learning_rate'])
        history_df = pd.DataFrame(history_dict).set_index(pd.Index(range(num_epochs_total)))
        history_df.to_csv(os.path.join(log_dir, 'history.csv'), index=True)
                
        if epoch - best_epoch > args.early_stopping_patience:
            logger.log(f'No improvement in validation loss for {args.early_stopping_patience} epochs. Stopping early.')
            break
    
    logger.log(f'Best validation loss: {best_val_loss:.5f} at epoch {best_epoch}.')
    logger.log(f'Finished training at epoch {epoch}.')
    
    # write a `finished.txt` file to the log directory so that we know the training is finished
    with open(os.path.join(log_dir, 'finished.txt'), 'w') as f:
        f.write(f'Finished training at epoch {epoch}.\n')
        f.write(f'Best validation loss: {best_val_loss:.5f} at epoch {best_epoch}.\n')

if __name__ == '__main__':
    main()
