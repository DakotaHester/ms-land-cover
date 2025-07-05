import argparse
from glob import glob
import math
import os
from time import time
import numpy as np
import pandas as pd
import torch
from torch.amp import autocast, GradScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from mslandcover.utils import Logger, get_torch_device, ProfilerHistory, load_pth
from mslandcover.data.datasets import FineTuneDataset
from mslandcover.data.transforms import StandardDataAugmentations 
from mslandcover.loss import FocalLoss
from mslandcover.models import DeepLabV3Plus, ResNetBackbone, UNet, AttentionUNet, ResNetBackboneUNet, SimpleLinearProbingResNet, MultiScaleLinearProbingResNet
from mslandcover import metrics


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        '--model',
        type=str,
        choices=['deeplabv3plus', 'unet', 'attention_unet', 'linear_probe', 'multiscale_linear_probe'],
        default='linear_probe',
        help='The model to use for training',
    )
    
    parser.add_argument(
        '--encoder_weights',
        type=str,
        default='imagenet',
        help='The path to the encoder weights to load for the full model. If `imagenet`, will load ImageNet weights.',
    )
    
    parser.add_argument(
        '--model_weights',
        type=str,
        default=None,
        help='The path to the model weights to load for the full model.',
    )
    
    parser.add_argument(
        '--split_dir',
        type=str,
        default='./data/splits',
        help='The directory containing the splits and a splits.csv file with the train/val splits',
    )
    
    parser.add_argument(
        '--n_train_samples',
        type=int,
        default=250,
        choices=[250, 500, 750],
        help="Number of samples to use for training. Choose from 250, 500, or 750 samples. This is used to limit the number of samples in the dataset for faster training.",
    )
    
    parser.add_argument(
        '--fold',
        type=int,
        default=1,
        choices=[1, 2, 3, 4, 5, 6],
        help='The fold to use for training. This is used to select the train/val splits from the splits.csv file.',
    )
    
    parser.add_argument(
        '--n_bands',
        type=int,
        default=3,
        choices=[3, 4],
        help='The number of bands in the input data. 3 for Color Infrared (CIR) and 4 for VisNIR data.',
    )
    
    # parser.add_argument(
    #     '--train_dir',
    #     type=str,
    #     default='./data/splits/train',
    #     help='The directory containing the training data',
    # )
    
    # parser.add_argument(
    #     '--val_dir',
    #     type=str,
    #     default='./data/splits/val',
    #     help='The directory containing the validation data',
    # )
    
    # parser.add_argument(
    #     '--test_dir',
    #     type=str,
    #     default='./data/splits/test',
    #     help='The directory containing the test data',
    # )
    
    parser.add_argument(
        '--freeze_encoder',
        action='store_true',
        help='Freeze the encoder weights',
    )
    
    parser.add_argument(
        '--freeze_decoder',
        action='store_true',
        help='Freeze the decoder weights',
    )
    
    parser.add_argument(
        '--mini_batch_size',
        type=int,
        default=16,
        help='The mini-batch size to use for training (for gradient accumulation)',
    )
    
    parser.add_argument(
        '--full_batch_size',
        type=int,
        default=16,
        help='The effective batch size to use for training',
    )
    
    parser.add_argument(
        '--lr',
        type=float,
        default=1e-5,
        help='The learning rate to use for training',
    )
    
    parser.add_argument(
        '--num_epochs',
        type=int,
        default=1000,
        help='The number of epochs to train the model',
    )
    
    parser.add_argument(
        '--early_stopping_patience',
        type=int,
        default=25,
        help='The number of epochs to wait before early stopping',
    )
    
    parser.add_argument(
        '--reduce_lr_patience',
        type=int,
        default=3,
        help='The number of epochs to wait before reducing the learning rate',
    )
    
    parser.add_argument(
        '--log_dir',
        type=str,
        default='./logs/finetune_unet',
        help='The directory to save logs',
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./weights/finetuned_unet',
        help='The directory to save weights',
    )
    
    parser.add_argument(
        '--preload',
        action='store_true',
        help='Load data into memory before training.',
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
    
    parser.add_argument(
        '--minimum_class_proportion',
        type=float,
        default=0.0, # set to 0 to disable oversampling
        help='Minimum proportion of a class in the dataset for it to be considered for oversampling',
    )
    
    parser.add_argument(
        '--oversample_factor',
        type=int,
        default=2,
        help='Number of times to duplicate samples that contain underrepresented classes',
    )
    
    parser.add_argument(
        '--minimum_oversample_ratio_factor',
        type=float,
        default=2.0,
        help='Factor to multiply the minimum class proportion by to determine the minimum oversample ratio',
    )
    
    parser.add_argument(
        '--alpha_power',
        type=float,
        default=0.0,
        help='The inverse power to raise the class distribution to for class weighting in the focal loss (i.e., 2.0 ~ sqrt(1 / class_distribution) to balance the loss for each class)',
    )
    
    parser.add_argument(
        '--focal_gamma',
        type=float,
        default=2.0,
        help='The gamma parameter for the focal loss',
    )
    
    parser.add_argument(
        '--warmup_epochs',
        type=int,
        default=10,
        help='Number of epochs for linear learning rate warmup.'
    )
    
    args, unkown = parser.parse_known_args()
    if len(unkown) > 0:
        for arg in unkown:
            print(f'Unknown argument: {arg}')
    
    if args.lr <= 0:
        parser.error('--lr must be greater than 0')
        
    if args.num_epochs < 1:
        parser.error('--n-epochs must be greater than or equal to 1')
        
    if args.early_stopping_patience < 1:
        parser.error('--early_stopping_patience must be greater than or equal to 1')
        
    if args.reduce_lr_patience < 1:
        parser.error('--lr_reduce_patience must be greater than or equal to 1')
    
    args.grad_accumulation_steps = args.full_batch_size // args.mini_batch_size
    
    return args



def main() -> None:
    
    args = parse_arguments()
    
    torch.random.manual_seed(args.seed)
    np.random.seed(args.seed)

    log_dir = os.path.join(args.log_dir)
    out_dir = os.path.join(args.output_dir)
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
    
    if args.n_bands == 3:
        mean_path = './weights/pretrain_mean.pth'
        std_path = './weights/pretrain_std.pth'
    else:
        mean_path = './weights/pretrain_mean_4.pt'
        std_path = './weights/pretrain_std_4.pt'
    mean = load_pth(mean_path) if os.path.exists(mean_path) else None
    std = load_pth(std_path) if os.path.exists(std_path) else None
    
    splits_df = pd.read_csv(os.path.join(args.split_dir, 'splits.csv'))
    splits_df = splits_df.loc[splits_df['n_train'] == args.n_train_samples]
    splits_df = splits_df.loc[splits_df['fold'] == args.fold]
    train_splits = [split for split in splits_df.columns if splits_df[split].iloc[0] == 'train']
    val_splits = [split for split in splits_df.columns if splits_df[split].iloc[0] == 'val']
    
    logger.log(f"Train splits: {train_splits}")
    logger.log(f"Validation splits: {val_splits}")
    
    train_dataset = FineTuneDataset(
        data_paths=[file for split in train_splits for file in glob(os.path.join(args.split_dir, split, 'input', '*.tif'))],
        target_paths=[file for split in train_splits for file in glob(os.path.join(args.split_dir, split, 'target', '*.tif'))],
        n_bands=args.n_bands,
        mean=mean,
        std=std,
        noise_std=0.25,
        transform=StandardDataAugmentations(s=1.0),
        preload=args.preload,
        n_threads=args.num_workers,
    )
    
    val_dataset = FineTuneDataset(
        data_paths=[file for split in val_splits for file in glob(os.path.join(args.split_dir, split, 'input', '*.tif'))],
        target_paths=[file for split in val_splits for file in glob(os.path.join(args.split_dir, split, 'target', '*.tif'))],
        n_bands=args.n_bands,
        mean=mean,
        std=std,
        noise_std=0.0,
        transform=None,
        preload=args.preload,
        n_threads=args.num_workers,
    )
    
    logger.log(f'Training dataset: {len(train_dataset)} samples')
    logger.log(f'Validation dataset: {len(val_dataset)} samples')
    
    class_dist = train_dataset.get_class_distribution()
    num_classes = len(class_dist)
    
    oversample_classes = []
    minimum_oversample_ratios = []
    for i, prob in enumerate(class_dist):
        if prob < args.minimum_class_proportion:
            oversample_classes.append(i)
            minimum_oversample_ratios.append(args.minimum_oversample_ratio_factor * prob)
    logger.log(f'Class distribution: {class_dist}')
    
    if len(oversample_classes) > 0:
        train_dataset.oversample_classes(oversample_classes, oversample_factor=args.oversample_factor, minimum_ratio=minimum_oversample_ratios)
        logger.log(f'Oversampled classes: {oversample_classes}')
        logger.log(f'New class distribution: {train_dataset.get_class_distribution()}')
        logger.log(f'New N_train: {len(train_dataset)}')
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.mini_batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers if args.num_workers > 1 else 0,
        pin_memory=True if args.num_workers > 1 else False,
        # prefetch_factor=4 if args.num_workers > 1 else 0,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.mini_batch_size,
        shuffle=False,
        num_workers=args.num_workers if args.num_workers > 1 else 0,
        pin_memory=True if args.num_workers > 1 else False,
        # prefetch_factor=4 if args.num_workers > 1 else 0,
    )
    
    # class weighting is the inverse of the class distribution raised to the power of 1 over the alpha power
    if args.alpha_power == 0:
        alpha = torch.ones(num_classes)
    else:
        alpha = class_dist ** (-1 / args.alpha_power)
    logger.log(f'Class weights: {alpha}')
    criterion = FocalLoss(alpha=alpha, gamma=args.focal_gamma, reduction='sum').to(device)
    
    num_classes = 8
    
    if args.model == 'deeplabv3plus':
        
        backbone = ResNetBackbone(
            in_channels=args.n_bands,
            pretrained=args.encoder_weights == 'imagenet',
        )
        
        if args.encoder_weights not in [None, 'none', 'imagenet']:
            if os.path.exists(args.encoder_weights):
                logger.log(f'Loading encoder weights from {args.encoder_weights}')
                encoder_weights = load_pth(args.encoder_weights, map_location=device)
                encoder_weights = adjust_backbone_weights(encoder_weights)
                backbone.load_state_dict(encoder_weights, strict=True)
            else:
                raise FileNotFoundError(f'Encoder weights not found at {args.encoder_weights}')
        
        model = DeepLabV3Plus(
            backbone=backbone,
            num_classes=num_classes,
        )
        model = model.to(device)
    
    elif args.model in ['unet', 'attention_unet', 'multiscale_linear_probe', 'linear_probe']:
        
        backbone = ResNetBackboneUNet(
            in_channels=args.n_bands,
            pretrained=args.encoder_weights == 'imagenet',
        )
        
        if args.encoder_weights not in [None, 'none', 'imagenet']:
            if os.path.exists(args.encoder_weights):
                logger.log(f'Loading encoder weights from {args.encoder_weights}')
                encoder_weights = load_pth(args.encoder_weights, map_location=device)
                encoder_weights = adjust_backbone_weights(encoder_weights)
                backbone.load_state_dict(encoder_weights, strict=True)
            else:
                raise FileNotFoundError(f'Encoder weights not found at {args.encoder_weights}')
        
        if args.model == 'unet':
            model = UNet(
                backbone=backbone,
                num_classes=num_classes,
            )
            
        elif args.model == 'attention_unet':
            model = AttentionUNet(
                backbone=backbone,
                num_classes=num_classes,
            )
            
        elif args.model == 'multiscale_linear_probe':
            model = MultiScaleLinearProbingResNet(
                backbone=backbone,
                num_classes=num_classes,
            )
        
        elif args.model == 'linear_probe':
            model = SimpleLinearProbingResNet(
                backbone=backbone,
                num_classes=num_classes,
            )
        
        model = model.to(device)
        
    
    else:
        raise ValueError(f'Unknown model: {args.model}')
    
    if args.freeze_encoder:
        logger.log('Freezing encoder weights')
        for param in model.backbone.parameters():
            param.requires_grad = False
        
        # if using 4 bands and imagenet weights, we need to unfreeze the first layer that has been replaced with a 4-channel input layer
        if args.n_bands == 4 and args.encoder_weights == 'imagenet':
            model.backbone.initial[0].weight.requires_grad = True
                
    if args.freeze_decoder:
        raise NotImplementedError('Freezing decoder weights is not implemented for DeepLabV3Plus')
        
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logger.log(f'Total parameters: {total_params}')
    logger.log(f'Trainable parameters: {trainable_params}')

    optimizer = AdamW(
        params=model.parameters(),
        lr=args.lr,  # This will be overridden by warmup if epoch < warmup_epochs
    )
    
    # Initialize mixed precision scaler
    scaler = GradScaler()
    
    scheduler = ReduceLROnPlateau(
        optimizer=optimizer,
        patience=args.reduce_lr_patience,
        eps=0,
    )
    
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
            criterion.load_state_dict(checkpoint['criterion'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scaler.load_state_dict(checkpoint['scaler'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            starting_epoch = checkpoint['epoch'] + 1
            best_epoch = checkpoint['best_epoch']
            best_val_loss = checkpoint['best_val_loss']
            history_dict = checkpoint['history_dict']
            profiler.profiler_history_dict = checkpoint['profiler_dict']
            logger.log(f'Loaded checkpoint from {checkpoint_path} at epoch {starting_epoch}')
        else:
            logger.log(f'No checkpoint found at {checkpoint_path}')
    
    logger.log(f'Starting training from epoch {starting_epoch}...')
    
    def get_warmup_lr(epoch, base_lr, warmup_epochs):
        if epoch >= warmup_epochs:
            return base_lr
        return base_lr * (epoch) / warmup_epochs
    
    for epoch in range(starting_epoch, args.num_epochs+1):
        # --- Linear LR Warmup ---
        if epoch <= args.warmup_epochs:
            warmup_lr = get_warmup_lr(epoch, args.lr, args.warmup_epochs)
            for param_group in optimizer.param_groups:
                param_group['lr'] = warmup_lr
        
        lr = optimizer.param_groups[0]['lr']
        history_dict['learning_rate'].append(lr)
        
        for phase in ['train', 'val']:
            
            phase_start_time = time()
            phase_stats = {'loss': []}
            for metric_fn in metric_fns:
                phase_stats[metric_fn.__name__] = []
            
            if phase == 'train':
                torch.set_grad_enabled(epoch != 0)
                optimizer.zero_grad(set_to_none=True)
                model.train()
                loader = train_loader

            else:
                torch.set_grad_enabled(False)
                model.eval()
                loader = val_loader
            
            total_steps = len(loader) / args.grad_accumulation_steps
            total_steps = math.ceil(total_steps) if phase == 'val' else math.floor(total_steps) # ceil val steps to ensure all samples are evaluated 
            
            with tqdm(
                total=total_steps,
                desc=f'Epoch {epoch}/{args.num_epochs} {phase.capitalize()}', 
                postfix={'lr': f'{lr:.0e}'}, 
                unit='batch'
            ) as pbar:
                
                for step, (X, y) in enumerate(loader):
                    
                    X, y = X.to(device), y.to(device)
                    
                    # Mixed precision forward pass
                    with autocast(device_type=device.type):
                        y_hat = model(X)
                        loss = criterion(y_hat, y)
                    
                    if torch.is_grad_enabled():
                        # Mixed precision backward pass
                        scaler.scale(loss).backward()
                
                    phase_stats['loss'].append(loss.detach().cpu().item())
                    for metric_fn in metric_fns:
                        y_true_flat = y.cpu().numpy().flatten()
                        y_pred_flat = torch.argmax(y_hat, dim=1).cpu().numpy().flatten()
                        phase_stats[metric_fn.__name__].append(metric_fn(y_true_flat, y_pred_flat) * len(X)) # multiple by samples seen to get true average later

                    running_metrics = {
                        'loss': sum(phase_stats['loss']) / ((step * loader.batch_size) + len(X)),
                    }
                    for metric_fn in metric_fns:
                        running_metrics[metric_fn.__name__] = sum(phase_stats[metric_fn.__name__]) / ((step * loader.batch_size) + len(X))
                    
                    if (step + 1) % args.grad_accumulation_steps == 0 :
                        if torch.is_grad_enabled():
                            # Mixed precision optimizer step
                            scaler.step(optimizer)
                            scaler.update()
                            optimizer.zero_grad(set_to_none=True)
                        pbar.update(1)
                    
                    tqdm_postfix = {
                        'lr': f"{lr:.0e}",
                        'loss': f"{running_metrics['loss']:.3e}",
                        'f1': f"{running_metrics['f1_score']:.3f}",
                        'macro_f1': f"{running_metrics['macro_f1_score']:.3f}",
                    }
                    pbar.set_postfix(tqdm_postfix)
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
        
        num_epochs_total = len(history_dict['learning_rate'])
        history_df = pd.DataFrame(history_dict).set_index(pd.Index(range(num_epochs_total)))
        history_df.to_csv(os.path.join(log_dir, 'history.csv'), index=True)
        profiler.save(os.path.join(log_dir, 'profiler.csv'))
        
        # Only step scheduler after warmup
        if epoch >= args.warmup_epochs:
            scheduler.step(epoch_loss)
        checkpoint = {
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scaler': scaler.state_dict(),
            'scheduler': scheduler.state_dict(),
            'criterion': criterion.state_dict(),
            'best_epoch': best_epoch,
            'best_val_loss': best_val_loss,
            'history_dict': history_dict,
            'profiler_dict': profiler.profiler_history_dict,
        }
        torch.save(checkpoint, os.path.join(log_dir, 'checkpoint.pth'))
        
        if epoch - best_epoch > args.early_stopping_patience:
            logger.log(f'No improvement in validation loss for {args.early_stopping_patience} epochs. Stopping early.')
            break
    
    logger.log(f'Finished training after {epoch} epochs.')
    
    # write a `finished.txt` file to the log directory so that we know the training is finished
    with open(os.path.join(log_dir, 'finished.txt'), 'w') as f:
        f.write(f'Finished training at epoch {epoch}.\n')
        f.write(f'Best validation loss: {best_val_loss:.5f} at epoch {best_epoch}.\n')
    
    
    ### 
    # OLD CODE FOR TESTING - NOT USED IN THIS SCRIPT
    ###
    
    # test_dataset = FineTuneDataset(
    #     data_paths=glob(os.path.join(args.test_dir, 'input', '*.tif')),
    #     target_paths=glob(os.path.join(args.test_dir, 'target', '*.tif')),
    #     mean=mean,
    #     std=std,
    #     transform=None,
    #     preload=not args.load_data_from_disk,
    #     n_threads=args.num_workers,
    #     n_threads=args.num_workers,
    # )
    # logger.log(f'Test dataset: {len(test_dataset)} samples')

    # test_loader = torch.utils.data.DataLoader(
    #     test_dataset,
    #     batch_size=args.mini_batch_size,
    #     batch_size=args.mini_batch_size,
    #     shuffle=False,
    #     # num_workers=args.num_workers,
    #     # pin_memory=True,
    #     # prefetch_factor=4,
    #     # num_workers=args.num_workers,
    #     # pin_memory=True,
    #     # prefetch_factor=4,
    # )
    
    # phase_metrics = {'loss': []}
    # for metric_fn in metric_fns:
    #     phase_metrics[metric_fn.__name__] = []
    # test_metrics = {}
    
    # model.load_state_dict(load_pth(os.path.join(out_dir, 'best_model.pth')))
    # model.eval()
    # torch.set_grad_enabled(False)
    # y_preds = []
    # y_trues = []
    
    # total_steps = math.ceil(len(test_loader) / args.grad_accumulation_steps)
    # with tqdm(total=total_steps, desc='Testing', unit='batch') as pbar:
    #     for step, (X, y) in enumerate(test_loader):
    #         X, y = X.to(device), y.to(device)
    #         y_hat = model(X)
    #         if hasattr(y_hat, 'logits'):
    #             y_hat = y_hat.logits
    #             y_hat = torch.nn.functional.interpolate(y_hat, size=y.shape[-2:], mode='bilinear', align_corners=False)

    #         if hasattr(y_hat, 'logits'):
    #             y_hat = y_hat.logits
    #             y_hat = torch.nn.functional.interpolate(y_hat, size=y.shape[-2:], mode='bilinear', align_corners=False)

            
    #         loss = criterion(y_hat, y)
    #         phase_metrics['loss'].append(loss.detach().cpu().item())
    #         test_metrics['loss'] = sum(phase_metrics['loss']) / ((step * test_loader.batch_size) + len(X))
    #         phase_metrics['loss'].append(loss.detach().cpu().item())
    #         test_metrics['loss'] = sum(phase_metrics['loss']) / ((step * test_loader.batch_size) + len(X))
            
    #         for metric_fn in metric_fns:
    #             phase_metrics[metric_fn.__name__].append(metric_fn(y, torch.argmax(y_hat, dim=1)) * len(X))
    #             test_metrics[metric_fn.__name__] = sum(phase_metrics[metric_fn.__name__]) / ((step * test_loader.batch_size) + len(X))
                
    #         y_preds.append(y_hat.argmax(axis=1).cpu().numpy().flatten())
    #         y_trues.append(y.cpu().numpy().flatten())
            
    #         if (step + 1) % args.grad_accumulation_steps == 0 or step == len(test_loader) - 1:
    #             tqdm_postfix = {
    #                 'loss': f"{test_metrics['loss']:.3e}",
    #                 'f1': f"{test_metrics['f1_score']:.3f}",
    #                 'macro_f1': f"{test_metrics['macro_f1_score']:.3f}",
    #             }
    #             pbar.set_postfix(tqdm_postfix)
    #             pbar.update(1)
    #             phase_metrics[metric_fn.__name__].append(metric_fn(y, torch.argmax(y_hat, dim=1)) * len(X))
    #             test_metrics[metric_fn.__name__] = sum(phase_metrics[metric_fn.__name__]) / ((step * test_loader.batch_size) + len(X))
                
    #         y_preds.append(y_hat.argmax(axis=1).cpu().numpy().flatten())
    #         y_trues.append(y.cpu().numpy().flatten())
            
    #         if (step + 1) % args.grad_accumulation_steps == 0 or step == len(test_loader) - 1:
    #             tqdm_postfix = {
    #                 'loss': f"{test_metrics['loss']:.3e}",
    #                 'f1': f"{test_metrics['f1_score']:.3f}",
    #                 'macro_f1': f"{test_metrics['macro_f1_score']:.3f}",
    #             }
    #             pbar.set_postfix(tqdm_postfix)
    #             pbar.update(1)
    
    # logger.log(f'Test loss: {test_metrics["loss"]:.5f}')
    # for metric_fn in metric_fns:
    #     logger.log(f'Test {metric_fn.__name__}: {test_metrics[metric_fn.__name__]:.5f}')
    
    # test_metrics_df = pd.DataFrame(test_metrics, index=[0])
    # test_metrics_df.to_csv(os.path.join(log_dir, 'test_metrics.csv'), index=False)
    
    # y_preds = np.concatenate(y_preds)
    # y_trues = np.concatenate(y_trues)
    
    
    # if 'cpb' in args.test_dir:
    #     legend_classes = {
    #         1: 'Water',
    #         2: 'Tree canopy',
    #         3: 'Shrubland',
    #         4: 'Low vegetation',
    #         5: 'Barren land',
    #         6: 'Impervious structures',
    #         7: 'Other impervious',
    #     }
    # else:
    #     legend_classes = LEGEND_CLASSES
        
    # y_trues_class_names = [legend_classes[i+1] for i in y_trues]
    # y_preds_class_names = [legend_classes[i+1] for i in y_preds]
    # class_names_list = [legend_classes[i+1] for i in range(num_classes)]

    # cm = confusion_matrix(y_trues_class_names, y_preds_class_names, labels=class_names_list)
    # cm_df = pd.DataFrame(cm, index=class_names_list, columns=class_names_list)
    # cm_df.to_csv(os.path.join(log_dir, 'confusion_matrix.csv'), index=True)
    
    # cr = classification_report(y_trues, y_preds, target_names=class_names_list, output_dict=True, zero_division=0)
    
    # cr_df = pd.DataFrame(cr).transpose()
    # cr_df.to_csv(os.path.join(log_dir, 'classification_report.csv'), index=True)


    
    # y_preds = np.concatenate(y_preds)
    # y_trues = np.concatenate(y_trues)
    
    
    # if 'cpb' in args.test_dir:
    #     legend_classes = {
    #         1: 'Water',
    #         2: 'Tree canopy',
    #         3: 'Shrubland',
    #         4: 'Low vegetation',
    #         5: 'Barren land',
    #         6: 'Impervious structures',
    #         7: 'Other impervious',
    #     }
    # else:
    #     legend_classes = LEGEND_CLASSES
        
    # y_trues_class_names = [legend_classes[i+1] for i in y_trues]
    # y_preds_class_names = [legend_classes[i+1] for i in y_preds]
    # class_names_list = [legend_classes[i+1] for i in range(num_classes)]

    # cm = confusion_matrix(y_trues_class_names, y_preds_class_names, labels=class_names_list)
    # cm_df = pd.DataFrame(cm, index=class_names_list, columns=class_names_list)
    # cm_df.to_csv(os.path.join(log_dir, 'confusion_matrix.csv'), index=True)
    
    # cr = classification_report(y_trues, y_preds, target_names=class_names_list, output_dict=True, zero_division=0)
    
    # cr_df = pd.DataFrame(cr).transpose()
    # cr_df.to_csv(os.path.join(log_dir, 'classification_report.csv'), index=True)



def adjust_backbone_weights(weights):
    new_weights = {}
    for key in weights.keys():
        
        if key.startswith('encoder.'):
            new_key = key.replace('encoder.', '')
        elif key.startswith('online_encoder.0.'):
            new_key = key.replace('online_encoder.0.', '')
        else:
            continue
            
        # initial layers should match up
        if new_key == 'conv1.weight':
            new_key = 'initial.0.weight'
        elif new_key == 'bn1.weight':
            new_key = 'initial.1.weight'
        elif new_key == 'bn1.bias':
            new_key = 'initial.1.bias'
        elif new_key == 'bn1.running_mean':
            new_key = 'initial.1.running_mean'
        elif new_key == 'bn1.running_var':
            new_key = 'initial.1.running_var'
        elif new_key == 'bn1.num_batches_tracked':
            new_key = 'initial.1.num_batches_tracked'
                
        new_weights[new_key] = weights[key]
        
    return new_weights



if __name__ == '__main__':


    main()

