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
from src.mslandcover.models import HRNetSegmentationModel
from src.mslandcover.loss import FocalLoss, FocalTverskyLoss, UnifiedFocalLoss
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
        default='randinit',
        choices=['randinit', 'imagenet', 'ae', 'dae', 'hsv', 'dae_hsv', 'simclr', 'ae_simclr', 'dae_simclr', 'hsv_simclr', 'dae_hsv_simclr'],
        help='The weights to use for the model',
    )
    
    parser.add_argument(
        '--weights_dir',
        type=str,
        default='./weights/hrnet_w18/20250108',
        help='The directory containing the weights',
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
        '--n_layers_unfrozen',
        type=int,
        default=1,
        help='The number of layers to unfreeze for training',
    )
    
    parser.add_argument(
        '--mini_batch_size',
        type=int,
        default=8,
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
        default=3,
        help='The number of epochs to wait before reducing the learning rate',
    )
    
    parser.add_argument(
        '--log_dir',
        type=str,
        default='./logs/finetune_cpbtests',
        help='The directory to save logs',
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./weights/finetuned_cpbtests',
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
        default=1,
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
        '--train_full_encoder',
        action='store_true',
        help='Overrides n_layers_unfrozen and trains the full encoder',
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
    
    # check to see if script is being run as a job array on a cluster (using SLURM or PBS) and set the random seed accordingly
    job_array_id = None
    if os.getenv('SLURM_ARRAY_TASK_ID') is not None:
        job_array_id = int(os.getenv('SLURM_ARRAY_TASK_ID'))
    elif os.getenv('PBS_ARRAY_INDEX') is not None:
        job_array_id = int(os.getenv('PBS_ARRAY_INDEX'))
    elif os.getenv('TASK_ARRAY_ID') is not None:
        job_array_id = int(os.getenv('TASK_ARRAY_ID'))
    
    if job_array_id is not None:
        print(f'Running as a job array with ID {job_array_id}. Disregarding '\
            '`weights` and `n_layers_unfrozen` arguments and substituting with '\
            'predefined values based on the job array ID.')
        possible_weights = ['randinit', 'imagenet', 'ae', 'dae', 'hsv', 'dae_hsv', 'simclr', 'ae_simclr', 'dae_simclr', 'hsv_simclr', 'dae_hsv_simclr']
        possible_n_layers_unfrozen = range(15)
        
        args.weights = possible_weights[(job_array_id) % len(possible_weights)]
        args.n_layers_unfrozen = possible_n_layers_unfrozen[(job_array_id) // len(possible_weights)]
        print(f'Using weights: {args.weights} and n_layers_unfrozen: {args.n_layers_unfrozen}')
    
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



def main():

    args = parse_arguments()
    
    torch.random.manual_seed(args.seed)
    np.random.seed(args.seed)

    log_dir = os.path.join(args.log_dir, args.model, args.weights, str(args.n_layers_unfrozen))
    out_dir = os.path.join(args.output_dir, args.model, args.weights, str(args.n_layers_unfrozen))

    if args.train_full_encoder:
        log_dir = log_dir + '_full_encoder'
        out_dir = out_dir + '_full_encoder'
    
    if args.n_train_samples is not None:
        log_dir = log_dir + f'_{args.n_train_samples}_samples'
        out_dir = out_dir + f'_{args.n_train_samples}_samples'
    
    log_dir = os.path.join(log_dir, 'test')
    
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
    class_dist = test_dataset.get_class_distribution()
    num_classes = len(class_dist)

    model_config = HRNET_W48_CONFIG if args.model == 'hrnet_w48' else HRNET_W18_CONFIG
    model = HRNetSegmentationModel(
        config=model_config,
        img_decoder_head=True,
        use_simple_decoder=args.n_layers_unfrozen == 0,
        use_se_decoder=args.n_layers_unfrozen > 0,
        unet_like_decoder=True,
        aux_simclr_head=False,
        img_decoder_activation='softmax',
        num_classes=num_classes,
    ).to(device)
    
    model.load_state_dict(load_pth(os.path.join(out_dir, 'best_model.pth')))
    model.eval()
    
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
    
    # phase_metrics = {'loss': []}
    phase_metrics = {}
    for metric_fn in metric_fns:
        phase_metrics[metric_fn.__name__] = []
    test_metrics = {}
    
    torch.set_grad_enabled(False)
    y_preds = []
    y_trues = []
    
    total_steps = math.ceil(len(test_loader) / args.grad_accumulation_steps)
    with tqdm(total=len(test_loader), desc='Testing', unit='batch') as pbar:
        for step, (X, y) in enumerate(test_loader):
            X, y = X.to(device), y.to(device)
            y_hat = model(X)
            
            # loss = criterion(y_hat, y)
            # phase_metrics['loss'].append(loss.item())
            # test_metrics['loss'] = sum(phase_metrics['loss']) / ((step * test_loader.batch_size) + len(X))
            
            for metric_fn in metric_fns:
                phase_metrics[metric_fn.__name__].append(metric_fn(y, torch.argmax(y_hat, dim=1)) * len(X))
                test_metrics[metric_fn.__name__] = sum(phase_metrics[metric_fn.__name__]) / ((step * test_loader.batch_size) + len(X))
                
            y_preds.append(y_hat.argmax(axis=1).cpu().numpy().flatten())
            y_trues.append(y.cpu().numpy().flatten())
            
            if (step + 1) % args.grad_accumulation_steps == 0:
                tqdm_postfix = {
                    # 'loss': f"{test_metrics['loss']:.3e}",
                    'f1': f"{test_metrics['f1_score']:.3f}",
                    'macro_f1': f"{test_metrics['macro_f1_score']:.3f}",
                }
                pbar.set_postfix(tqdm_postfix)
                pbar.update(1)
    
    # logger.log(f'Test loss: {test_metrics["loss"]:.5f}')
    for metric_fn in metric_fns:
        logger.log(f'Test {metric_fn.__name__}: {test_metrics[metric_fn.__name__]:.5f}')
    
    test_metrics_df = pd.DataFrame(test_metrics, index=[0])
    test_metrics_df.to_csv(os.path.join(log_dir, 'test_metrics.csv'), index=False)
    
    y_preds = np.concatenate(y_preds)
    y_trues = np.concatenate(y_trues)
    
    y_trues_class_names = [LEGEND_CLASSES[i+1] for i in y_trues]
    y_preds_class_names = [LEGEND_CLASSES[i+1] for i in y_preds]
    class_names_list = [LEGEND_CLASSES[i+1] for i in range(8)]

    cm = confusion_matrix(y_trues_class_names, y_preds_class_names, labels=class_names_list)
    cm_df = pd.DataFrame(cm, index=class_names_list, columns=class_names_list)
    cm_df.to_csv(os.path.join(log_dir, 'confusion_matrix.csv'), index=True)
    
    cr = classification_report(y_trues, y_preds, target_names=class_names_list, output_dict=True, zero_division=0)
    
    cr_df = pd.DataFrame(cr).transpose()
    cr_df.to_csv(os.path.join(log_dir, 'classification_report.csv'), index=True)



if __name__ == '__main__':
    main()