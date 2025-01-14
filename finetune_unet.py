import argparse
from glob import glob
import math
import os
from time import time
import numpy as np
import pandas as pd
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report

from src.mslandcover.utils import Logger, get_torch_device, ProfilerHistory, load_pth
from src.mslandcover.config import HRNET_W18_CONFIG, HRNET_W48_CONFIG, LEGEND_CLASSES
from src.mslandcover.data.datasets import FineTuneDataset
from src.mslandcover.data.transforms import StandardDataAugmentations
from src.mslandcover.models import UNet
from src.mslandcover.loss import FocalLoss, FocalTverskyLoss, UnifiedFocalLoss
from src.mslandcover import metrics


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        '--encoder_weights',
        type=str,
        default=None,
        help='The path to the encoder weights to load for the full model. If `imagenet`, will load ImageNet weights.',
    )
    
    parser.add_argument(
        '--model_weights',
        type=str,
        default=None,
        help='The path to the model weights to load for the full model.',
    )
    
    parser.add_argument(
        '--train_dir',
        type=str,
        default='./data/cpb_tests/splits/train',
        help='The directory containing the training data',
    )
    
    parser.add_argument(
        '--val_dir',
        type=str,
        default='./data/cpb_tests/splits/val',
        help='The directory containing the validation data',
    )
    
    parser.add_argument(
        '--test_dir',
        type=str,
        default='./data/cpb_tests/splits/test',
        help='The directory containing the test data',
    )
    
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
        default=4,
        help='The mini-batch size to use for training (for gradient accumulation)',
    )
    
    parser.add_argument(
        '--full_batch_size',
        type=int,
        default=64,
        help='The effective batch size to use for training',
    )
    
    parser.add_argument(
        '--lr',
        type=float,
        default=1e-4,
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
        default=15,
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
        '--n_train_samples',
        type=int,
        default=None,
        help="If set, will use the first n samples from the training dataset",
    )
    
    parser.add_argument(
        '--load_checkpoint',
        action='store_true',
        help='Load a checkpoint from the log directory and resume training',
    )
    
    parser.add_argument(
        '--minimum_class_proportion',
        type=float,
        default=0.05,
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
        default=2.0,
        help='The power to raise the class weights to',
    )
    
    args = parser.parse_args()
    
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
    
    mean_path = os.path.join('weights', 'pretrain_mean.pth')
    std_path = os.path.join('weights', 'pretrain_std.pth')
    
    mean = load_pth(mean_path) if os.path.exists(mean_path) else None
    std = load_pth(std_path) if os.path.exists(std_path) else None
    
    # n_train_samples = -0 if args.n_train_samples is None else args.n_train_samples
    
    train_dataset = FineTuneDataset(
        data_paths=glob(os.path.join(args.train_dir, 'input', '*.tif'))[:args.n_train_samples],
        target_paths=glob(os.path.join(args.train_dir, 'target', '*.tif'))[:args.n_train_samples],
        mean=mean,
        std=std,
        transform=StandardDataAugmentations(),
        preload=not args.load_data_from_disk,
        n_threads=args.num_workers,
    )
    val_dataset = FineTuneDataset(
        data_paths=glob(os.path.join(args.val_dir, 'input', '*.tif'))[:args.n_train_samples],
        target_paths=glob(os.path.join(args.val_dir, 'target', '*.tif'))[:args.n_train_samples],
        mean=mean,
        std=std,
        transform=None,
        preload=not args.load_data_from_disk,
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
    
    train_dataset.oversample_classes(oversample_classes, oversample_factor=args.oversample_factor, minimum_ratio=minimum_oversample_ratios)
    logger.log(f'Oversampled classes: {oversample_classes}')
    logger.log(f'New class distribution: {train_dataset.get_class_distribution()}')
    logger.log(f'New N_train: {len(train_dataset)}')
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.mini_batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        pin_memory=True,
        prefetch_factor=4,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.mini_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        prefetch_factor=4,
    )
    
    alpha = (1 - class_dist) ** args.alpha_power
    alpha = alpha / alpha.mean()
    logger.log(f'Class weights: {alpha}')
    criterion = UnifiedFocalLoss(alpha=alpha, reduction='sum').to(device)
    
    num_classes = 7 if 'cpb' in args.train_dir else 8
    
    # if full model weights are provided, load them and replace the old
    if args.model_weights is not None:
        pretrained_model_classes = 7 if 'cpb' in args.model_weights else 8
        model = UNet(num_classes=pretrained_model_classes).to(device)
        model.load_state_dict(load_pth(args.model_weights))
        model.classifier = torch.nn.Conv2d(64, num_classes, kernel_size=1)
    
    # if encoder weights only are provided, load them and keep the random decoder
    elif args.encoder_weights is not None and args.encoder_weights != 'imagenet':
        model = UNet(num_classes=num_classes).to(device)
        model.load_encoder_weights(load_pth(args.encoder_weights))
    
    else:
        model = UNet(num_classes=num_classes, pretrained=args.encoder_weights != 'imagenet').to(device)
    
    if args.freeze_encoder:
        for param in model.encoder.parameters():
            param.requires_grad = False
    
    if args.freeze_decoder:
        for param in model.decoder.parameters():
            param.requires_grad = False
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logger.log(f'Total parameters: {total_params}')
    logger.log(f'Trainable parameters: {trainable_params}')

    optimizer = Adam(
        params=model.parameters(),
        lr=args.lr,
    )
    
    scheduler = ReduceLROnPlateau(
        optimizer=optimizer,
        patience=args.reduce_lr_patience,
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
            scheduler.load_state_dict(checkpoint['scheduler'])
            starting_epoch = checkpoint['epoch'] 
            best_epoch = checkpoint['best_epoch']
            best_val_loss = checkpoint['best_val_loss']
            history_dict = checkpoint['history_dict']
            profiler.profiler_history_dict = checkpoint['profiler_dict']
            logger.log(f'Loaded checkpoint from {checkpoint_path} at epoch {starting_epoch}')
        else:
            logger.log(f'No checkpoint found at {checkpoint_path}')
    
    logger.log(f'Starting training from epoch {starting_epoch+1}...')
    
    for epoch in range(starting_epoch+1, args.num_epochs+1):
        
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
            
                    y_hat = model(X)
                    loss = criterion(y_hat, y)
                    
                    if phase == 'train':
                        loss.backward()
                    
                    phase_stats['loss'].append(loss.item())
                    for metric_fn in metric_fns:
                        phase_stats[metric_fn.__name__].append(metric_fn(y, torch.argmax(y_hat, dim=1)) * len(X)) # multiple by samples seen to get true average later

                    running_metrics = {
                        'loss': sum(phase_stats['loss']) / ((step * loader.batch_size) + len(X)),
                    }
                    for metric_fn in metric_fns:
                        running_metrics[metric_fn.__name__] = sum(phase_stats[metric_fn.__name__]) / ((step * loader.batch_size) + len(X))
                    
                    if (step + 1) % args.grad_accumulation_steps == 0:
                        if phase == 'train':
                            optimizer.step()
                            optimizer.zero_grad()
                        tqdm_postfix = {
                            'lr': f"{lr:.0e}",
                            'loss': f"{running_metrics['loss']:.3e}",
                            'f1': f"{running_metrics['f1_score']:.3f}",
                            'macro_f1': f"{running_metrics['macro_f1_score']:.3f}",
                        }
                        pbar.set_postfix(tqdm_postfix)
                        pbar.update(1)
                        
                        if phase == 'train':
                            if len(loader) - step < args.grad_accumulation_steps:
                                profiler.update(epoch, phase, step, time() - phase_start_time)
                                break
                    
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
            'criterion': criterion.state_dict(),
            'epoch': epoch,
            'best_epoch': best_epoch,
            'best_val_loss': best_val_loss,
            'history_dict': history_dict,
            'profiler_dict': profiler.profiler_history_dict,
        }
        torch.save(checkpoint, os.path.join(log_dir, 'checkpoint.pth'))
        
        history_df = pd.DataFrame(history_dict).set_index(pd.Index(range(epoch)))
        history_df.to_csv(os.path.join(log_dir, 'history.csv'), index=True)
        profiler.save(os.path.join(log_dir, 'profiler.csv'))
        
        if epoch - best_epoch > args.early_stopping_patience:
            logger.log(f'No improvement in validation loss for {args.early_stopping_patience} epochs. Stopping early.')
            break
    
    logger.log(f'Finished training after {epoch} epochs.')
    
    test_dataset = FineTuneDataset(
        data_paths=glob(os.path.join(args.test_dir, 'input', '*.tif')),
        target_paths=glob(os.path.join(args.test_dir, 'target', '*.tif')),
        mean=mean,
        std=std,
        transform=None,
        preload=not args.load_data_from_disk,
        n_threads=args.num_workers,
    )
    logger.log(f'Test dataset: {len(test_dataset)} samples')

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.mini_batch_size,
        shuffle=False,
        # num_workers=args.num_workers,
        # pin_memory=True,
        # prefetch_factor=4,
    )
    
    phase_metrics = {'loss': []}
    for metric_fn in metric_fns:
        phase_metrics[metric_fn.__name__] = []
    test_metrics = {}
    
    model.load_state_dict(load_pth(os.path.join(out_dir, 'best_model.pth')))
    model.eval()
    torch.set_grad_enabled(False)
    y_preds = []
    y_trues = []
    
    total_steps = math.ceil(len(test_loader) / args.grad_accumulation_steps)
    with tqdm(total=total_steps, desc='Testing', unit='batch') as pbar:
        for step, (X, y) in enumerate(test_loader):
            X, y = X.to(device), y.to(device)
            y_hat = model(X)
            
            loss = criterion(y_hat, y)
            phase_metrics['loss'].append(loss.item())
            test_metrics['loss'] = sum(phase_metrics['loss']) / ((step * test_loader.batch_size) + len(X))
            
            for metric_fn in metric_fns:
                phase_metrics[metric_fn.__name__].append(metric_fn(y, torch.argmax(y_hat, dim=1)) * len(X))
                test_metrics[metric_fn.__name__] = sum(phase_metrics[metric_fn.__name__]) / ((step * test_loader.batch_size) + len(X))
                
            y_preds.append(y_hat.argmax(axis=1).cpu().numpy().flatten())
            y_trues.append(y.cpu().numpy().flatten())
            
            if (step + 1) % args.grad_accumulation_steps == 0 or step == len(test_loader) - 1:
                tqdm_postfix = {
                    'loss': f"{test_metrics['loss']:.3e}",
                    'f1': f"{test_metrics['f1_score']:.3f}",
                    'macro_f1': f"{test_metrics['macro_f1_score']:.3f}",
                }
                pbar.set_postfix(tqdm_postfix)
                pbar.update(1)
    
    logger.log(f'Test loss: {test_metrics["loss"]:.5f}')
    for metric_fn in metric_fns:
        logger.log(f'Test {metric_fn.__name__}: {test_metrics[metric_fn.__name__]:.5f}')
    
    test_metrics_df = pd.DataFrame(test_metrics, index=[0])
    test_metrics_df.to_csv(os.path.join(log_dir, 'test_metrics.csv'), index=False)
    
    y_preds = np.concatenate(y_preds)
    y_trues = np.concatenate(y_trues)
    
    
    if 'cpb' in args.test_dir:
        legend_classes = {
            1: 'Water',
            2: 'Tree canopy',
            3: 'Shrubland',
            4: 'Low vegetation',
            5: 'Barren land',
            6: 'Impervious structures',
            7: 'Other impervious',
        }
    else:
        legend_classes = LEGEND_CLASSES
        
    y_trues_class_names = [legend_classes[i+1] for i in y_trues]
    y_preds_class_names = [legend_classes[i+1] for i in y_preds]
    class_names_list = [legend_classes[i+1] for i in range(num_classes)]

    cm = confusion_matrix(y_trues_class_names, y_preds_class_names, labels=class_names_list)
    cm_df = pd.DataFrame(cm, index=class_names_list, columns=class_names_list)
    cm_df.to_csv(os.path.join(log_dir, 'confusion_matrix.csv'), index=True)
    
    cr = classification_report(y_trues, y_preds, target_names=class_names_list, output_dict=True, zero_division=0)
    
    cr_df = pd.DataFrame(cr).transpose()
    cr_df.to_csv(os.path.join(log_dir, 'classification_report.csv'), index=True)



if __name__ == '__main__':

    main()
