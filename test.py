import os
from mslandcover.config import LEGEND_CLASSES
from mslandcover.data.datasets import TestDataset
from mslandcover.models import UNet, DeepLabV3Plus, ResNetBackboneUNet, ResNetBackbone
from mslandcover.utils import load_pth, get_torch_device
from argparse import ArgumentParser
import geopandas as gpd
from glob import glob
from torch.utils.data import DataLoader
import torch
import torch.nn.functional as F
from tqdm import tqdm
import pandas as pd
from sklearn import metrics
import json
import numpy as np

def parse_args() -> ArgumentParser:
    parser = ArgumentParser()
    
    parser.add_argument(
        '--ground_truth_shapefile',
        type=str,
        default='./data/shapefiles/assessment_points',
        help='Path to the ground truth shapefile containing assessment points.'
    )
    
    parser.add_argument(
        '--raster_dir',
        type=str,
        default='./data/assessment',
        help='Directory containing the raster files for assessment.'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        choices=['deeplabv3plus', 'unet'],
        default='unet',
        help='Model architecture to use for the assessment.'
    )
    
    parser.add_argument(
        '--model_weights',
        type=str,
        # default='./models/unet_weights.pth', ???
        help='Path to the model weights file.'
    )
    
    parser.add_argument(
        '--n_bands',
        type=int,
        default=3,
        choices=[3, 4],
        help='Number of bands in the raster data (default is 3 for RGB).'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./logs/assessment_results',
        help='Directory to save the assessment results.'
    )
    
    parser.add_argument(
        '--batch_size',
        type=int,
        default=16,
        help='Batch size for the DataLoader.'
    )
    
    return parser



def main():
    parser = parse_args()
    args = parser.parse_args()
    
    points_gdf = gpd.read_file(args.ground_truth_shapefile)
    points_gdf = points_gdf.rename(columns={'ground_tru': 'ground_truth', 'ground_t_1': 'ground_truth_class_name'})
    points_gdf = points_gdf[points_gdf['ground_truth'] != 0]
    
    raster_paths = glob(f"{args.raster_dir}/*.tif")
    if len(raster_paths) == 0:
        raise ValueError(f"No raster files found in the directory: {args.raster_dir}")

    if args.n_bands == 3:
        mean = load_pth('./weights/pretrain_mean.pth')
        std = load_pth('./weights/pretrain_std.pth')
    elif args.n_bands == 4:
        mean = load_pth('./weights/pretrain_mean_4.pt')
        std = load_pth('./weights/pretrain_std_4.pt')

    # Initialize the dataset with the provided arguments
    test_dataset = TestDataset(
        points_gdf=points_gdf,
        raster_paths=raster_paths,
        n_bands=args.n_bands,
        mean=mean,
        std=std,
    )

    if args.model == 'deeplabv3plus':
        model = DeepLabV3Plus(
            backbone=ResNetBackbone(
                in_channels=args.n_bands,
                pretrained=False,
            ),
            num_classes=8
        )
        
    elif args.model == 'unet':
        model = UNet(
            backbone=ResNetBackboneUNet(
                in_channels=args.n_bands,
                pretrained=False,
            ),
            num_classes=8
        )
    
    else:
        raise ValueError(f"Unsupported model type: {args.model}. Choose 'deeplabv3plus' or 'unet'.")
    
    if args.model_weights is not None:
        if not os.path.exists(args.model_weights):
            raise FileNotFoundError(f"Model weights file not found: {args.model_weights}")
        weights = load_pth(args.model_weights)
        model.load_state_dict(weights)
    
    device = get_torch_device()
    model.to(device)
    model.eval()
    
    preds_dict = {
        'point_id': [],
        'predicted_class_idx': [],
        'ground_truth_class_idx': [],
        'ground_truth_class_name': [],
        'cross_entropy': [],
        'brier_score': []
    }

    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
    
    for batch in tqdm(test_loader, desc="Assessing model performance", unit="batch"):
        images = batch['image']
        images = images.to(device)
        with torch.no_grad():
            outputs = model(images)
        
        preds = torch.argmax(outputs, dim=1)
        for i in range(len(preds)):
            point_id = batch['point_id'][i]
            ground_truth_class_idx = batch['class_idx'][i].item()
            ground_truth_class_name = batch['class_name'][i]
            
            pred_idx = preds[i][batch['row'][i], batch['col'][i]].item()
            # hacky way to calculate cross-entropy loss for the specific pixel
            cross_entropy = F.cross_entropy(outputs[i][:, batch['row'][i], batch['col'][i]].unsqueeze(0).to('cpu'), torch.tensor([ground_truth_class_idx - 1])).item()
            # another hacky way to calculate Brier score for the specific pixel (mean squared error between predicted and ground truth)
            brier_score = F.mse_loss(outputs[i][:, batch['row'][i], batch['col'][i]].unsqueeze(0).to('cpu'), torch.nn.functional.one_hot(torch.tensor([ground_truth_class_idx - 1]), num_classes=8).float()).item()
            pred_idx = pred_idx + 1  # Adjusting for zero-based index after calculation
            
            preds_dict['point_id'].append(point_id)
            preds_dict['predicted_class_idx'].append(pred_idx)
            preds_dict['ground_truth_class_idx'].append(ground_truth_class_idx)
            preds_dict['ground_truth_class_name'].append(ground_truth_class_name)
            preds_dict['cross_entropy'].append(cross_entropy)
            preds_dict['brier_score'].append(brier_score)
    
    preds_df = pd.DataFrame(preds_dict)
    
    os.makedirs(args.output_dir, exist_ok=True)
    preds_df.to_csv(f"{args.output_dir}/assessment_results.csv", index=False)

    class_counts = preds_df['ground_truth_class_idx'].value_counts(normalize=True).sort_index()
    weighted_cross_entropy = (
        preds_df.groupby('ground_truth_class_idx')['cross_entropy'].mean() * class_counts
    ).sum()
    weighted_brier_score = (
        preds_df.groupby('ground_truth_class_idx')['brier_score'].mean() * class_counts
    ).sum()

    metrics_dict = {
        # overall metrics
        'accuracy': metrics.accuracy_score(preds_df['ground_truth_class_idx'], preds_df['predicted_class_idx']),
        'f1_score': metrics.f1_score(preds_df['ground_truth_class_idx'], preds_df['predicted_class_idx'], average='micro'),
        'precision': metrics.precision_score(preds_df['ground_truth_class_idx'], preds_df['predicted_class_idx'], average='micro'),
        'recall': metrics.recall_score(preds_df['ground_truth_class_idx'], preds_df['predicted_class_idx'], average='micro'),
        'jaccard': metrics.jaccard_score(preds_df['ground_truth_class_idx'], preds_df['predicted_class_idx'], average='micro'),
        'kappa': metrics.cohen_kappa_score(preds_df['ground_truth_class_idx'], preds_df['predicted_class_idx']),
        'cross_entropy': np.mean(preds_df['cross_entropy']),
        'brier_score': np.mean(preds_df['brier_score']),
        # macro metrics
        'macro_f1_score': metrics.f1_score(preds_df['ground_truth_class_idx'], preds_df['predicted_class_idx'], average='macro'),
        'macro_precision': metrics.precision_score(preds_df['ground_truth_class_idx'], preds_df['predicted_class_idx'], average='macro'),
        'macro_recall': metrics.recall_score(preds_df['ground_truth_class_idx'], preds_df['predicted_class_idx'], average='macro'),
        'macro_jaccard': metrics.jaccard_score(preds_df['ground_truth_class_idx'], preds_df['predicted_class_idx'], average='macro'),
        'macro_cross_entropy': np.mean(preds_df.groupby('ground_truth_class_idx')['cross_entropy'].mean()),
        'macro_brier_score': np.mean(preds_df.groupby('ground_truth_class_idx')['brier_score'].mean()),
        # weighted metrics
        'weighted_f1_score': metrics.f1_score(preds_df['ground_truth_class_idx'], preds_df['predicted_class_idx'], average='weighted'),
        'weighted_precision': metrics.precision_score(preds_df['ground_truth_class_idx'], preds_df['predicted_class_idx'], average='weighted'),
        'weighted_recall': metrics.recall_score(preds_df['ground_truth_class_idx'], preds_df['predicted_class_idx'], average='weighted'),
        'weighted_jaccard': metrics.jaccard_score(preds_df['ground_truth_class_idx'], preds_df['predicted_class_idx'], average='weighted'),
        'weighted_cross_entropy': weighted_cross_entropy,
        'weighted_brier_score': weighted_brier_score,
    }
    with open(f"{args.output_dir}/assessment_metrics.json", 'w') as f:
        json.dump(metrics_dict, f, indent=4)
    
    classification_report = metrics.classification_report(
        preds_df['ground_truth_class_idx'],
        preds_df['predicted_class_idx'],
        target_names=[LEGEND_CLASSES[i] for i in range(1, 9)],
        output_dict=True
    )

    classification_report_df = pd.DataFrame(classification_report).transpose()
    classification_report_df.to_csv(f"{args.output_dir}/classification_report.csv")
    

if __name__ == "__main__":
    main()
