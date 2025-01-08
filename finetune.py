import argparse
from glob import glob
import os
from time import time
import numpy as np
import pandas as pd
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from src.mslandcover.utils import Logger, get_torch_device, ProfilerHistory, load_pth
from src.mslandcover.config import HRNET_W18_CONFIG, HRNET_W48_CONFIG
from src.mslandcover.data.datasets import FineTuneDataset
from src.mslandcover.data.transforms import StandardDataAugmentations
from src.mslandcover.models import HRNetSegmentationModel
from src.mslandcover.loss import FocalLoss
from src.mslandcover import metrics


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        '--model',
        type=str,
        default='hrnet_w18',
        choices=['hrnet_w18', 'hrnet_w48'],
        help='The model to use for training',
    )
    
    parser.add_argument(
        '--weights',
        type=str,
        default='imagenet',
        choices=['none', 'imagenet', 'ae', 'dae', 'hsv', 'dae_hsv', 'simclr', 'ae_simclr', 'dae_simclr', 'hsv_simclr', 'dae_hsv_simclr'],
        help='The weights to use for the model',
    )
    
    parser.add_argument(
        '--weights_dir',
        type=str,
        default='./weights/',
        help='The directory containing the weights',
    )
    
    parser.add_argument(
        '--train_dir',
        type=str,
        default='./data/splits/train',
        help='The directory containing the training data',
    )
    
    parser.add_argument(
        '--val_dir',
        type=str,
        default='./data/splits/val',
        help='The directory containing the validation data',
    )
    
    parser.add_argument(
        '--test_dir',
        type=str,
        default='./data/splits/test',
        help='The directory containing the test data',
    )
    
    parser.add_argument(
        '--n_layers_unfrozen',
        type=int,
        default=1,
        help='The number of layers to unfreeze for training',
    )
    
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='The batch size to use for training',
    )
    
    parser.add_argument(
        '--lr',
        type=float,
        default=1e-3,
        help='The learning rate to use for training',
    )
    
    parser.add_argument(
        '--num_epochs',
        type=int,
        default=100,
        help='The number of epochs to train the model',
    )
    
    parser.add_argument(
        '--early_stopping_patience',
        type=int,
        default=15,
        help='The number of epochs to wait before early stopping',
    )
    
    parser.add_argument(
        '--reduce_lr_patience',
        type=int,
        default=5,
        help='The number of epochs to wait before reducing the learning rate',
    )
    
    parser.add_argument(
        '--log_dir',
        type=str,
        default='./logs/finetune',
        help='The directory to save logs',
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./weights/finetuned',
        help='The directory to save weights',
    )
    
    parser.add_argument(
        '--load_data_from_disk',
        action='store_true',
        help='Load data from disk instead of memory',
    )
    
    parser.add_argument(
        '--num_workers',
        type=int,
        default=8,
        help='The number of workers to use for data loading',
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=1701,
        help='The random seed to use for training',
    )
    
    parser.add_argument(
        '--load_checkpoint',
        action='store_true',
        help='Load a checkpoint from the log directory and resume training',
    )
    
    args = parser.parse_args()
    
    if args.n_layers_unfrozen < 1:
        parser.error('--n-layers-unfrozen must be greater than or equal to 1')
    
    if args.lr <= 0:
        parser.error('--lr must be greater than 0')
        
    if args.num_epochs < 1:
        parser.error('--n-epochs must be greater than or equal to 1')
        
    if args.early_stopping_patience < 1:
        parser.error('--early-stopping-patience must be greater than or equal to 1')
        
    if args.reduce_lr_patience < 1:
        parser.error('--lr-reduce-patience must be greater than or equal to 1')
    
    return args



def main() -> None:
    
    args = parse_arguments()
    
    torch.random.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    log_dir = os.path.join(args.log_dir, args.model, args.weights, str(args.n_layers_unfrozen))
    out_dir = os.path.join(args.output_dir, args.model, args.weights, str(args.n_layers_unfrozen))
    
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)
    
    logger = Logger(os.path.join(log_dir, 'log.txt'))
    logger.log(f'Configuration:')
    for k, v in vars(args).items():
        logger.log(f'{k}: {v}', prepend_timestamp=False)
    logger.log('='*20, prepend_timestamp=False)
    
    device = get_torch_device()
    logger.log(f'Using device: {device}')
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deteministic = True
        torch.cuda.manual_seed_all(args.seed)
    
    mean_path = os.path.join(args.weights_dir, 'pretrain_mean.pth')
    std_path = os.path.join(args.weights_dir, 'pretrain_std.pth')
    
    mean = load_pth(mean_path) if os.path.exists(mean_path) else None
    std = load_pth(std_path) if os.path.exists(std_path) else None
    
    train_dataset = FineTuneDataset(
        data_paths=glob(os.path.join(args.train_dir, 'input', '*.tif')),
        target_paths=glob(os.path.join(args.train_dir, 'target', '*.tif')),
        mean=mean,
        std=std,
        transform=StandardDataAugmentations(),
        preload=not args.load_data_from_disk,
    )
    val_dataset = FineTuneDataset(
        data_paths=glob(os.path.join(args.val_dir, 'input', '*.tif')),
        target_paths=glob(os.path.join(args.val_dir, 'target', '*.tif')),
        mean=mean,
        std=std,
        transform=None,
        preload=not args.load_data_from_disk,
    )
    
    logger.log(f'Training dataset: {len(train_dataset)} samples')
    logger.log(f'Validation dataset: {len(val_dataset)} samples')
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    
    model_config = HRNET_W48_CONFIG if args.model == 'hrnet_w48' else HRNET_W18_CONFIG
    model = HRNetSegmentationModel(
        config=model_config,
        img_decoder_head=True,
        aux_simclr_head=False,
        img_decoder_activation='softmax',
        num_classes=7,
    ).to(device)
    
    trainable_stages = [
        'decoder',
        'encoder.stage4.2',
        'encoder.stage4.1',
        'encoder.stage4.0',
        'encoder.transition3',
        'encoder.stage3.3',
        'encoder.stage3.2',
        'encoder.stage3.1',
        'encoder.stage3.0',
        'encoder.transition2',
        'encoder.stage2.0',
        'encoder.transition1',
        'encoder.layer1',
        'encoder' # full encoder
    ][:args.n_layers_unfrozen]
    
    total_params = 0
    for param in model.parameters():
        total_params += param.numel()
        param.requires_grad = False
    
    trainable_params = 0
    for param in model.named_parameters():
        for stage in trainable_stages:
            if param[0].startswith(stage):
                trainable_params += param.numel()
                param[1].requires_grad = True
    
    logger.log(f'Total parameters: {total_params}')
    logger.log(f'Trainable parameters: {trainable_params}')
    logger.log(f'Trainable stages: {trainable_stages}')
    
    optimizer = Adam(
        params=model.parameters(),
        lr=args.lr,
    )
    
    scheduler = ReduceLROnPlateau(
        optimizer=optimizer,
        patience=args.reduce_lr_patience,
    )
    
    criterion = FocalLoss()
    
    metric_fns = [
        metrics.accuracy,
        metrics.f1_score,
        metrics.precision_score,
        metrics.recall_score,
        metrics.macro_f1_score,
        metrics.macro_precision_score,
        metrics.macro_recall_score,
        metrics.kappa_score,
    ]
    
    history_dict = {
        'learning_rate': [],
    }
    
    for phase in ['train', 'val']:
        history_dict[f'{phase}_loss'] = []
        for metric_fn in metric_fns:
            history_dict[f'{phase}_{metric_fn.__name__}'] = []
    
    starting_epoch = 0
    best_epoch = -1
    best_val_loss = float('inf')
    profiler = ProfilerHistory(device)
    profiler.update(-1, 'init', 0, 0)
    
    if args.load_checkpoint:
        checkpoint_path = os.path.join(log_dir, 'checkpoint.pth')
        if os.path.exists(checkpoint_path):
            checkpoint = load_pth(checkpoint_path)
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            starting_epoch = checkpoint['epoch'] + 1
            best_epoch = checkpoint['best_epoch']
            best_val_loss = checkpoint['best_val_loss']
            history_dict = checkpoint['history_dict']
            profiler.profiler_history_dict = checkpoint['profiler_dict']
            logger.log(f'Loaded checkpoint from {checkpoint_path}')
        else:
            logger.log(f'No checkpoint found at {checkpoint_path}')
    
    logger.log(f'Starting training from epoch {starting_epoch}...')
    
    for epoch in range(starting_epoch, args.num_epochs):
        
        lr = optimizer.param_groups[0]['lr']
        history_dict['learning_rate'].append(lr)
        
        for phase in ['train', 'val']:
            
            phase_start_time = time()
            phase_stats = {'loss': []}
            for metric_fn in metric_fns:
                phase_stats[metric_fn.__name__] = []
            
            if phase == 'train':
                torch.set_grad_enabled(True)
                optimizer.zero_grad()
                model.train()
                loader = train_loader

            else:
                torch.set_grad_enabled(False)
                model.eval()
                loader = val_loader
            
            with tqdm(
                loader, 
                desc=f'Epoch {epoch+1}/{args.num_epochs} {phase.capitalize()}', 
                postfix={'lr': lr}, 
                unit='batch'
            ) as tloader:
                for step, (X, y) in enumerate(tloader):
                    X, y = X.to(device), y.to(device)
                    y_hat = model(X)
                    loss = criterion(y_hat, y)
                    
                    phase_stats['loss'].append(loss.item())
                    for metric_fn in metric_fns:
                        phase_stats[metric_fn.__name__].append(metric_fn(y_hat, y) * len(X)) # multiple by samples seen to get true average later
                    
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        optimizer.zero_grad()
                    
                    running_metrics = {
                        'loss': sum(phase_stats['loss']) / ((step * loader.batch_size) + len(X)),
                    }
                    for metric_fn in metric_fns:
                        running_metrics[metric_fn.__name__] = sum(phase_stats[metric_fn.__name__]) / ((step * loader.batch_size) + len(X))
                    
                    tqdm_postfix = {
                        'lr': lr,
                        'loss': running_metrics['loss'],
                        'f1': running_metrics['f1_score'],
                        'macro_f1': running_metrics['macro_f1_score'],
                    }
                    tloader.set_postfix(tqdm_postfix)
                    profiler.update(epoch, phase, step, time() - phase_start_time)
                
            history_dict[f'{phase}_loss'].append(running_metrics['loss'])
            for metric_fn in metric_fns:
                history_dict[f'{phase}_{metric_fn.__name__}'].append(running_metrics[metric_fn.__name__])
        
        epoch_loss = history_dict['val_loss'][-1]
        if epoch_loss < best_val_loss:
            logger.log(f'Validation loss improved from {best_val_loss:.5f} at epoch {best_epoch} to {epoch_loss:.5f} during epoch {epoch}. Saving model...')
            best_val_loss = epoch_loss
            best_epoch = epoch
            torch.save(
                model.state_dict(),
                os.path.join(out_dir, 'best_model.pth'),
            )
        
        scheduler.step(epoch_loss)
        checkpoint = {
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'epoch': epoch,
            'best_epoch': best_epoch,
            'best_val_loss': best_val_loss,
            'history_dict': history_dict,
            'profiler_dict': profiler.profiler_history_dict,
        }
        torch.save(checkpoint, os.path.join(log_dir, 'checkpoint.pth'))
        
        history_df = pd.DataFrame(history_dict).set_index(pd.Index(range(epoch+1)))
        history_df.to_csv(os.path.join(log_dir, 'history.csv'), index=True)
        profiler.save(os.path.join(log_dir, 'profiler.csv'))
        
        if epoch - best_epoch > args.early_stopping_patience:
            logger.log(f'No improvement in validation loss for {args.early_stopping_patience} epochs. Stopping early.')
            break
    
    logger.log(f'Finished training after {epoch+1} epochs.')
    
    test_dataset = FineTuneDataset(
        data_paths=glob(os.path.join(args.test_dir, 'input', '*.tif')),
        target_paths=glob(os.path.join(args.test_dir, 'target', '*.tif')),
        mean=mean,
        std=std,
        transform=None,
        preload=not args.load_data_from_disk,
    )
    logger.log(f'Test dataset: {len(test_dataset)} samples')

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    
    phase_metrics = {'loss': []}
    for metric_fn in metric_fns:
        phase_metrics[metric_fn.__name__] = []
    test_metrics = {}
    
    model.load_state_dict(load_pth(os.path.join(out_dir, 'best_model.pth')))
    with tqdm(test_loader, desc='Testing', unit='batch') as tloader:
        model.eval()
        for i, (X, y) in enumerate(tloader):
            X, y = X.to(device), y.to(device)
            y_hat = model(X)
            
            loss = criterion(y_hat, y)
            phase_metrics['loss'].append(loss.item())
            test_metrics['loss'] = sum(phase_metrics['loss']) / ((i * test_loader.batch_size) + len(X))
            
            for metric_fn in metric_fns:
                phase_metrics[metric_fn.__name__].append(metric_fn(y_hat, y) * len(X))
                test_metrics[metric_fn.__name__] = sum(phase_metrics[metric_fn.__name__]) / ((i * test_loader.batch_size) + len(X))
            
            tloader.set_postfix({
                'loss': test_metrics['loss'],
                'f1': test_metrics['f1_score'],
                'macro_f1': test_metrics['macro_f1_score'],
            })
    
    logger.log(f'Test loss: {test_metrics["loss"]:.5f}')
    for metric_fn in metric_fns:
        logger.log(f'Test {metric_fn.__name__}: {test_metrics[metric_fn.__name__]:.5f}')
    
    test_metrics_df = pd.DataFrame(test_metrics, index=[0])
    test_metrics_df.to_csv(os.path.join(log_dir, 'test_metrics.csv'), index=False)

if __name__ == '__main__':
    main()