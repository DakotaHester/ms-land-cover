"""Raster-vs-point assessment entrypoint for completed prediction products.

Unlike `test.py` (which evaluates model outputs directly), this script reads a
final land-cover raster and compares sampled values against reference points.
"""

import os
from mslandcover.config import LEGEND_CLASSES
from mslandcover.utils import load_pth
from argparse import ArgumentParser
import geopandas as gpd
import rasterio as rio
import torch
import pandas as pd
from sklearn import metrics
import json
import numpy as np
from tqdm import tqdm

def parse_args() -> ArgumentParser:
    parser = ArgumentParser()
    
    parser.add_argument(
        '--ground_truth_shapefile',
        type=str,
        # default='./data/shapefiles/assessment_points',
        default='./data/assessment/assessment_points_2016/mslc_assessment_2016/',
        help='Path to the ground truth shapefile containing assessment points.'
    )
    
    parser.add_argument(
        '--prediction_raster',
        type=str,
        # required=True,
        default='/home/dhester/server/dbcenter/products/land/ms_hires_landcover/2016.tif',
        help='Path to the completed land cover prediction raster.'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        # default='./logs/assessment_results',
        default='./logs/assessment_results_2016',
        help='Directory to save the assessment results.'
    )
    
    return parser

def extract_predictions_at_points(points_gdf, prediction_raster_path):
    """Extract predicted class values at point locations from the raster."""
    
    predictions = []
    
    with rio.open(prediction_raster_path) as src:
        print(f"\nRaster Information:")
        print(f"  File path: {prediction_raster_path}")
        print(f"  Raster shape: {src.shape}")
        print(f"  Raster CRS: {src.crs}")
        print(f"  Raster bounds: {src.bounds}")
        print(f"  Raster data type: {src.dtypes[0]}")
        print(f"  NoData value: {src.nodata}")
        print(f"  Number of bands: {src.count}")
        
        # Check CRS compatibility
        points_crs = points_gdf.crs
        print(f"  Points CRS: {points_crs}")
        if src.crs != points_crs:
            print(f"  WARNING: CRS mismatch between raster ({src.crs}) and points ({points_crs})")
            points_gdf = points_gdf.to_crs(src.crs)
        
        # Sample the raster at point locations
        coords = [(point.x, point.y) for point in points_gdf.geometry]
        sampled_values = list(src.sample(coords))
        
        # Track statistics
        points_outside_bounds = 0
        unique_pred_values = set()
        
        for i, (idx, point) in enumerate(points_gdf.iterrows()):
            x, y = point.geometry.x, point.geometry.y
            
            # Check if point is within raster bounds
            if not (src.bounds.left <= x <= src.bounds.right and 
                    src.bounds.bottom <= y <= src.bounds.top):
                points_outside_bounds += 1
                print(f"Point {i} ({x}, {y}) is outside raster bounds, skipping...")
                print(point)
                continue
            
            pred_value = sampled_values[i][0]  # Extract the value from the array
            unique_pred_values.add(pred_value)
            
            if pred_value == src.nodata or pred_value == 0:
                print(f"Point {i} ({x}, {y}) has NoData value, skipping...")
                print(point)
                continue
            
            predictions.append({
                'point_id': point.get('id', f'point_{i}'),
                'predicted_class_idx': int(pred_value) if not np.isnan(pred_value) else -9999,
                'ground_truth_class_idx': int(point['ground_truth']),
                'ground_truth_class_name': point['ground_truth_class_name'],
                'x': point.geometry.x,
                'y': point.geometry.y
            })
        
        if len(predictions) > 25000:
            predictions = predictions[:25000]
        
        print(f"\nSampling Results:")
        print(f"  Total points processed: {len(points_gdf)}")
        print(f"  Points outside raster bounds: {points_outside_bounds}")
        print(f"  Unique prediction values found: {sorted(unique_pred_values)}")
        
        # return pd.DataFrame(predictions)
        
        if True:
            # Analyze full raster distribution
            print(f"\nAnalyzing full raster distribution...")
            raster_data = src.read(1)  # Read the first (and only) band
        
            # Get unique values and their counts
            unique_values, counts = np.unique(raster_data, return_counts=True)
            total_pixels = raster_data.size
            
            # Calculate valid class distribution (excluding NoData) first
            if src.nodata is not None:
                valid_mask = (raster_data != src.nodata) & (~np.isnan(raster_data))
            else:
                valid_mask = ~np.isnan(raster_data)
            
            valid_pixels = raster_data[valid_mask]
            valid_pixel_count = len(valid_pixels)
            
            print(f"\nFULL RASTER CLASS DISTRIBUTION:")
            print("-" * 60)
            print(f"  Total pixels in raster: {total_pixels:,}")
            print(f"  Raster dimensions: {raster_data.shape}")
            print(f"  Valid pixels (excluding NoData): {valid_pixel_count:,} ({valid_pixel_count/total_pixels*100:.2f}% of total)")
            print(f"  Pixel area coverage (valid pixels only): {valid_pixel_count * (src.res[0] * src.res[1]):.2f} square units")
            print()
            
            # Sort by value for consistent output
            sorted_indices = np.argsort(unique_values)
            sorted_values = unique_values[sorted_indices]
            sorted_counts = counts[sorted_indices]
        
            for val, count in zip(sorted_values, sorted_counts):
                percentage_total = (count / total_pixels) * 100
                if val == src.nodata or np.isnan(val):
                    continue  # Skip NoData values
                    # class_name = "NoData/Background"
                    # print(f"  Value {val:>3} ({class_name:<20}): {count:>10,} pixels ({percentage_total:>6.2f}% of )")
                else:
                    percentage_valid = (count / valid_pixel_count) * 100 if valid_pixel_count > 0 else 0
                    class_name = LEGEND_CLASSES.get(int(val), f"Unknown_{int(val)}")
                    print(f"  Value {val:>3} ({class_name:<20}): {count:>10,} pixels ({percentage_valid:>6.2f}% of valid)")
            
            if valid_pixel_count > 0:
                print(f"\nVALID CLASS DISTRIBUTION (excluding NoData/NaN):")
                print("-" * 60)
                print(f"  Valid pixels: {valid_pixel_count:,} ({valid_pixel_count/total_pixels*100:.2f}% of total)")
                print(f"  Invalid/NoData pixels: {total_pixels - valid_pixel_count:,} ({(total_pixels - valid_pixel_count)/total_pixels*100:.2f}% of total)")
                print(f"  Valid area coverage: {valid_pixel_count * (src.res[0] * src.res[1]):.2f} square units")
                print()
                
                valid_unique, valid_counts = np.unique(valid_pixels, return_counts=True)
                
                # Sort by class index for consistent output
                valid_sorted_indices = np.argsort(valid_unique)
                valid_sorted_values = valid_unique[valid_sorted_indices]
                valid_sorted_counts = valid_counts[valid_sorted_indices]
                
                # Create distribution dataframe for CSV output
                distribution_data = []
                for val, count in zip(valid_sorted_values, valid_sorted_counts):
                    percentage = (count / valid_pixel_count) * 100
                    area_coverage = count * (src.res[0] * src.res[1])
                    class_name = LEGEND_CLASSES.get(int(val), f"Unknown_{int(val)}")
                    print(f"  Class {int(val):>2} ({class_name:<20}): {count:>10,} pixels ({percentage:>6.2f}%, {area_coverage:>10.2f} sq units)")
                    
                    distribution_data.append({
                        'class_value': int(val),
                        'class_name': class_name,
                        'pixel_count': count,
                        'percentage_of_valid': percentage,
                        'area_coverage_sq_units': area_coverage
                    })
                
                # Save distribution as CSV
                distribution_df = pd.DataFrame(distribution_data)
                return pd.DataFrame(predictions), distribution_df
            else:
                print(f"\nWARNING: No valid pixels found in raster (all are NoData/NaN)")
                # Return empty distribution dataframe if no valid pixels
                return pd.DataFrame(predictions), pd.DataFrame(columns=['class_value', 'class_name', 'pixel_count', 'percentage_of_valid', 'area_coverage_sq_units'])

def calculate_metrics(preds_df):
    """Calculate assessment metrics from predictions dataframe."""
    
    print(f"\n{'='*60}")
    print("DETAILED ACCURACY ASSESSMENT")
    print(f"{'='*60}")
    
    # Print distribution of prediction values
    print("\n1. PREDICTION VALUE DISTRIBUTION:")
    print("-" * 40)
    pred_dist = preds_df['predicted_class_idx'].value_counts().sort_index()
    for val, count in pred_dist.items():
        class_name = LEGEND_CLASSES.get(val, f"Unknown_{val}")
        print(f"  Class {val} ({class_name}): {count} points ({count/len(preds_df)*100:.1f}%)")
    
    print("\n2. GROUND TRUTH VALUE DISTRIBUTION:")
    print("-" * 40)
    gt_dist = preds_df['ground_truth_class_idx'].value_counts().sort_index()
    for val, count in gt_dist.items():
        class_name = LEGEND_CLASSES.get(val, f"Unknown_{val}")
        print(f"  Class {val} ({class_name}): {count} points ({count/len(preds_df)*100:.1f}%)")
    
    # Filter out any invalid predictions (e.g., NoData values)
    valid_mask = (preds_df['predicted_class_idx'] >= 1) & (preds_df['predicted_class_idx'] <= 8)
    preds_df_valid = preds_df[valid_mask].copy()
    
    # Show what was filtered out
    invalid_df = preds_df[~valid_mask]
    if len(invalid_df) > 0:
        print(f"\n3. INVALID PREDICTION VALUES FOUND:")
        print("-" * 40)
        invalid_dist = invalid_df['predicted_class_idx'].value_counts().sort_index()
        for val, count in invalid_dist.items():
            print(f"  Value {val}: {count} points ({count/len(invalid_df)*100:.1f}% of invalid)")
    else:
        print(f"\n3. NO INVALID PREDICTION VALUES FOUND")
        print("-" * 40)
        print("  All prediction values are within the valid range (1-8)")
    
    if len(preds_df_valid) == 0:
        raise ValueError("No valid predictions found. Check that prediction raster has values in range 1-8.")
    
    print(f"\n4. DATA VALIDATION SUMMARY:")
    print("-" * 40)
    print(f"  Total points loaded: {len(preds_df)}")
    print(f"  Valid predictions: {len(preds_df_valid)} ({len(preds_df_valid)/len(preds_df)*100:.1f}%)")
    print(f"  Invalid predictions: {len(preds_df) - len(preds_df_valid)} ({(len(preds_df) - len(preds_df_valid))/len(preds_df)*100:.1f}%)")
    
    # Calculate confusion matrix for detailed analysis
    cm = metrics.confusion_matrix(
        preds_df_valid['ground_truth_class_idx'], 
        preds_df_valid['predicted_class_idx'],
        labels=list(range(1, 9))
    )
    
    print(f"\n5. CONFUSION MATRIX:")
    print("-" * 40)
    print("Rows = Ground Truth, Columns = Predicted")
    print("Class:", end="")
    for i in range(1, 9):
        print(f"{i:>6}", end="")
    print()
    for i, row in enumerate(cm):
        print(f"{i+1:>5}:", end="")
        for val in row:
            print(f"{val:>6}", end="")
        print()
    
    # Calculate per-class metrics
    user_accuracy = []  # precision - how many predicted pixels of class i are actually class i
    producer_accuracy = []  # recall - how many actual pixels of class i were correctly predicted as class i
    f1_scores = []
    
    print(f"\n6. PER-CLASS ACCURACY ASSESSMENT:")
    print("-" * 75)
    print(f"{'Class':<5} {'Name':<20} {'Count':<6} {'User Acc':<9} {'Prod Acc':<9} {'F1-Score':<8}")
    print("-" * 75)
    
    for i in range(len(cm)):
        class_idx = i + 1
        class_name = LEGEND_CLASSES.get(class_idx, f"Class_{class_idx}")
        class_count = cm[i, :].sum()
        
        # User accuracy (precision) = TP / (TP + FP) = diagonal / column sum
        if cm[:, i].sum() > 0:
            ua = cm[i, i] / cm[:, i].sum()
        else:
            ua = 0.0
        user_accuracy.append(ua)
        
        # Producer accuracy (recall) = TP / (TP + FN) = diagonal / row sum
        if cm[i, :].sum() > 0:
            pa = cm[i, i] / cm[i, :].sum()
        else:
            pa = 0.0
        producer_accuracy.append(pa)
        
        # F1 score
        if ua + pa > 0:
            f1 = 2 * (ua * pa) / (ua + pa)
        else:
            f1 = 0.0
        f1_scores.append(f1)
        
        print(f"{class_idx:<5} {class_name:<20} {class_count:<6} {ua:<9.3f} {pa:<9.3f} {f1:<8.3f}")
    
    print("-" * 75)
    
    # Calculate class distribution for weighted metrics
    class_counts = preds_df_valid['ground_truth_class_idx'].value_counts(normalize=True).sort_index()
    
    # Print overall accuracy metrics
    overall_accuracy = metrics.accuracy_score(
        preds_df_valid['ground_truth_class_idx'], 
        preds_df_valid['predicted_class_idx']
    )
    kappa = metrics.cohen_kappa_score(
        preds_df_valid['ground_truth_class_idx'], 
        preds_df_valid['predicted_class_idx']
    )
    
    print(f"\n7. OVERALL ACCURACY ASSESSMENT:")
    print("-" * 40)
    print(f"  Overall Accuracy: {overall_accuracy:.4f} ({overall_accuracy*100:.2f}%)")
    print(f"  Kappa Coefficient: {kappa:.4f}")
    print(f"  Average User Accuracy: {np.mean(user_accuracy):.4f} ({np.mean(user_accuracy)*100:.2f}%)")
    print(f"  Average Producer Accuracy: {np.mean(producer_accuracy):.4f} ({np.mean(producer_accuracy)*100:.2f}%)")
    print(f"  Average F1-Score: {np.mean(f1_scores):.4f}")
    print(f"  Standard Deviation User Accuracy: {np.std(user_accuracy):.4f}")
    print(f"  Standard Deviation Producer Accuracy: {np.std(producer_accuracy):.4f}")
    print(f"  Standard Deviation F1-Score: {np.std(f1_scores):.4f}")
    
    # Calculate error matrix statistics
    total_correct = np.trace(cm)
    total_samples = cm.sum()
    errors_of_commission = []
    errors_of_omission = []
    
    print(f"\n8. DETAILED ERROR ANALYSIS:")
    print("-" * 40)
    print(f"  Total correct classifications: {total_correct}")
    print(f"  Total misclassifications: {total_samples - total_correct}")
    print(f"  Error rate: {(total_samples - total_correct)/total_samples*100:.2f}%")
    print()
    
    for i in range(len(cm)):
        class_idx = i + 1
        class_name = LEGEND_CLASSES.get(class_idx, f"Class_{class_idx}")
        
        # Errors of commission (false positives)
        commission_errors = cm[:, i].sum() - cm[i, i]
        commission_rate = commission_errors / cm[:, i].sum() if cm[:, i].sum() > 0 else 0
        errors_of_commission.append(commission_rate)
        
        # Errors of omission (false negatives)
        omission_errors = cm[i, :].sum() - cm[i, i]
        omission_rate = omission_errors / cm[i, :].sum() if cm[i, :].sum() > 0 else 0
        errors_of_omission.append(omission_rate)
        
        print(f"  Class {class_idx} ({class_name}):")
        print(f"    True Positives: {cm[i, i]}")
        print(f"    False Positives (Commission): {commission_errors} ({commission_rate*100:.1f}%)")
        print(f"    False Negatives (Omission): {omission_errors} ({omission_rate*100:.1f}%)")
        print(f"    Total Reference: {cm[i, :].sum()}")
        print(f"    Total Classified: {cm[:, i].sum()}")
        print()
    
    # Calculate class imbalance information
    print(f"\n9. CLASS DISTRIBUTION ANALYSIS:")
    print("-" * 40)
    total_ref_points = len(preds_df_valid)
    for class_idx, proportion in class_counts.items():
        class_name = LEGEND_CLASSES.get(class_idx, f"Class_{class_idx}")
        count = int(proportion * total_ref_points)
        print(f"  Class {class_idx} ({class_name}):")
        print(f"    Reference points: {count} ({proportion*100:.1f}% of total)")
        print(f"    Predicted points: {cm[:, class_idx-1].sum()} ({cm[:, class_idx-1].sum()/total_ref_points*100:.1f}% of total)")
    
    # Calculate micro, macro, and weighted averages
    print(f"\n10. AGGREGATED METRICS:")
    print("-" * 40)
    
    # Micro metrics
    micro_ua = metrics.precision_score(preds_df_valid['ground_truth_class_idx'], preds_df_valid['predicted_class_idx'], average='micro')
    micro_pa = metrics.recall_score(preds_df_valid['ground_truth_class_idx'], preds_df_valid['predicted_class_idx'], average='micro')
    micro_f1 = metrics.f1_score(preds_df_valid['ground_truth_class_idx'], preds_df_valid['predicted_class_idx'], average='micro')
    
    print(f"  Micro-averaged metrics:")
    print(f"    User Accuracy (Precision): {micro_ua:.4f} ({micro_ua*100:.2f}%)")
    print(f"    Producer Accuracy (Recall): {micro_pa:.4f} ({micro_pa*100:.2f}%)")
    print(f"    F1-Score: {micro_f1:.4f}")
    
    # Macro metrics
    macro_ua = metrics.precision_score(preds_df_valid['ground_truth_class_idx'], preds_df_valid['predicted_class_idx'], average='macro')
    macro_pa = metrics.recall_score(preds_df_valid['ground_truth_class_idx'], preds_df_valid['predicted_class_idx'], average='macro')
    macro_f1 = metrics.f1_score(preds_df_valid['ground_truth_class_idx'], preds_df_valid['predicted_class_idx'], average='macro')
    
    print(f"  Macro-averaged metrics:")
    print(f"    User Accuracy (Precision): {macro_ua:.4f} ({macro_ua*100:.2f}%)")
    print(f"    Producer Accuracy (Recall): {macro_pa:.4f} ({macro_pa*100:.2f}%)")
    print(f"    F1-Score: {macro_f1:.4f}")
    
    # Weighted metrics
    weighted_ua = metrics.precision_score(preds_df_valid['ground_truth_class_idx'], preds_df_valid['predicted_class_idx'], average='weighted')
    weighted_pa = metrics.recall_score(preds_df_valid['ground_truth_class_idx'], preds_df_valid['predicted_class_idx'], average='weighted')
    weighted_f1 = metrics.f1_score(preds_df_valid['ground_truth_class_idx'], preds_df_valid['predicted_class_idx'], average='weighted')
    
    print(f"  Weighted-averaged metrics:")
    print(f"    User Accuracy (Precision): {weighted_ua:.4f} ({weighted_ua*100:.2f}%)")
    print(f"    Producer Accuracy (Recall): {weighted_pa:.4f} ({weighted_pa*100:.2f}%)")
    print(f"    F1-Score: {weighted_f1:.4f}")
    
    # Jaccard indices
    micro_jaccard = metrics.jaccard_score(preds_df_valid['ground_truth_class_idx'], preds_df_valid['predicted_class_idx'], average='micro')
    macro_jaccard = metrics.jaccard_score(preds_df_valid['ground_truth_class_idx'], preds_df_valid['predicted_class_idx'], average='macro')
    weighted_jaccard = metrics.jaccard_score(preds_df_valid['ground_truth_class_idx'], preds_df_valid['predicted_class_idx'], average='weighted')
    
    print(f"  Jaccard Index (IoU):")
    print(f"    Micro: {micro_jaccard:.4f}")
    print(f"    Macro: {macro_jaccard:.4f}")
    print(f"    Weighted: {weighted_jaccard:.4f}")
    
    # Calculate per-class Jaccard
    print(f"\n11. PER-CLASS JACCARD INDEX (IoU):")
    print("-" * 40)
    try:
        per_class_jaccard = metrics.jaccard_score(
            preds_df_valid['ground_truth_class_idx'], 
            preds_df_valid['predicted_class_idx'], 
            average=None,
            labels=list(range(1, 9))
        )
        
        if hasattr(per_class_jaccard, '__iter__'):
            for i, jaccard in enumerate(per_class_jaccard):
                class_idx = i + 1
                class_name = LEGEND_CLASSES.get(class_idx, f"Class_{class_idx}")
                print(f"  Class {class_idx} ({class_name}): {jaccard:.4f}")
            
            jaccard_list = per_class_jaccard.tolist() if hasattr(per_class_jaccard, 'tolist') else list(per_class_jaccard)
        else:
            print(f"  Single class Jaccard: {per_class_jaccard:.4f}")
            jaccard_list = [per_class_jaccard]
            
    except Exception as e:
        print(f"  Could not calculate per-class Jaccard scores: {e}")
        jaccard_list = [0.0] * 8
    
    print(f"\n12. STATISTICAL SIGNIFICANCE:")
    print("-" * 40)
    print(f"  Total sample size: {len(preds_df_valid)}")
    print(f"  Degrees of freedom: {len(preds_df_valid) - 1}")
    print(f"  95% Confidence interval for Overall Accuracy:")
    
    # Calculate confidence interval for overall accuracy
    p = overall_accuracy
    n = len(preds_df_valid)
    se = np.sqrt(p * (1 - p) / n)  # Standard error
    ci_95 = 1.96 * se  # 95% confidence interval
    print(f"    {p - ci_95:.4f} to {p + ci_95:.4f}")
    print(f"    Margin of error: ±{ci_95:.4f} ({ci_95*100:.2f}%)")
    
    metrics_dict = {
        # Overall metrics
        'overall_accuracy': overall_accuracy,
        'f1_score_micro': micro_f1,
        'user_accuracy_micro': micro_ua,
        'producer_accuracy_micro': micro_pa,
        'jaccard_micro': micro_jaccard,
        'kappa': kappa,
        
        # Macro metrics (unweighted average across classes)
        'f1_score_macro': macro_f1,
        'user_accuracy_macro': macro_ua,
        'producer_accuracy_macro': macro_pa,
        'jaccard_macro': macro_jaccard,
        
        # Weighted metrics (weighted by class frequency)
        'f1_score_weighted': weighted_f1,
        'user_accuracy_weighted': weighted_ua,
        'producer_accuracy_weighted': weighted_pa,
        'jaccard_weighted': weighted_jaccard,
        
        # Per-class metrics
        'per_class_user_accuracy': user_accuracy,
        'per_class_producer_accuracy': producer_accuracy,
        'per_class_f1_scores': f1_scores,
        'per_class_jaccard': per_class_jaccard.tolist(),
        'per_class_commission_errors': errors_of_commission,
        'per_class_omission_errors': errors_of_omission,
        
        # Statistical measures
        'user_accuracy_std': np.std(user_accuracy),
        'producer_accuracy_std': np.std(producer_accuracy),
        'f1_score_std': np.std(f1_scores),
        'overall_accuracy_ci_95': [p - ci_95, p + ci_95],
        'overall_accuracy_margin_error': ci_95,
        
        # Additional info
        'total_points': len(preds_df),
        'valid_points': len(preds_df_valid),
        'invalid_points': len(preds_df) - len(preds_df_valid),
        'confusion_matrix': cm.tolist(),
        'class_distribution': class_counts.to_dict(),
        'total_correct': int(total_correct),
        'total_misclassified': int(total_samples - total_correct),
        'error_rate': float((total_samples - total_correct)/total_samples)
    }
    
    return metrics_dict, preds_df_valid

def main():
    parser = parse_args()
    args = parser.parse_args()
    # Parse assessment configuration and load reference points.
    
    # Load ground truth points
    print("Loading ground truth points...")
    points_gdf = gpd.read_file(args.ground_truth_shapefile)
    
    print(f"Original shapefile columns: {list(points_gdf.columns)}")
    print(f"Original point count: {len(points_gdf)}")
    
    points_gdf = points_gdf.rename(columns={
        'ground_tru': 'ground_truth', 
        'ground_t_1': 'ground_truth_class_name'
    })
    
    print(f"After renaming columns: {list(points_gdf.columns)}")
    
    # Filter out points with ground_truth value of 0
    original_count = len(points_gdf)
    points_gdf = points_gdf[points_gdf['ground_truth'] != 0]
    filtered_count = len(points_gdf)
    
    print(f"Filtered out {original_count - filtered_count} points with ground_truth = 0")
    print(f"Remaining points: {filtered_count}")
    
    # Print ground truth class distribution
    # Display the distribution of ground truth classes
    print("\nGround Truth Class Distribution:")
    gt_counts = points_gdf['ground_truth'].value_counts().sort_index()
    for class_idx, count in gt_counts.items():
        class_name = LEGEND_CLASSES.get(class_idx, f"Unknown_{class_idx}")
        print(f"  Class {class_idx} ({class_name}): {count} points ({count/len(points_gdf)*100:.1f}%)")
    
    print(f"\nLoaded {len(points_gdf)} ground truth points.")
    
    # Check if prediction raster exists
    print(f"\nChecking prediction raster: {args.prediction_raster}")
    if not os.path.exists(args.prediction_raster):
        raise FileNotFoundError(f"Prediction raster not found: {args.prediction_raster}")
    
    # Extract predictions at point locations
    print("Extracting predictions from raster...")
    if True:
        preds_df, distribution_df = extract_predictions_at_points(points_gdf, args.prediction_raster)
        # Predictions extracted successfully
    else:
        preds_df = extract_predictions_at_points(points_gdf, args.prediction_raster)
    
    print(f"\nPrediction extraction complete:")
    print(f"  Total rows in prediction dataframe: {len(preds_df)}")
    print(f"  Columns: {list(preds_df.columns)}")
    
    # Calculate metrics
    print("\nCalculating assessment metrics...")
    metrics_dict, preds_df_valid = calculate_metrics(preds_df)
    # Metrics calculated successfully
    
    # Create output directory
    print(f"\nCreating output directory: {args.output_dir}")
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save results
    print("Saving results...")
    
    # Save the detailed predictions dataframe
    # Save predictions to CSV
    pred_csv_path = f"{args.output_dir}/assessment_results.csv"
    preds_df.to_csv(pred_csv_path, index=False)
    print(f"  Saved predictions CSV: {pred_csv_path}")
    
    if True:
        # Save the raster class distribution
        # Save distribution data if available
        if not distribution_df.empty:
            dist_csv_path = f"{args.output_dir}/raster_class_distribution.csv"
            distribution_df.to_csv(dist_csv_path, index=False)
            print(f"  Saved raster distribution CSV: {dist_csv_path}")
        else:
            print(f"  Warning: No valid raster distribution to save")
    
    # Save the metrics dictionary
    metrics_json_path = f"{args.output_dir}/assessment_metrics.json"
    with open(metrics_json_path, 'w') as f:
        json.dump(metrics_dict, f, indent=4)
    print(f"  Saved metrics JSON: {metrics_json_path}")
    
    # Generate classification report
    try:
        classification_report = metrics.classification_report(
            preds_df['ground_truth_class_idx'],
            preds_df['predicted_class_idx'],
            target_names=[LEGEND_CLASSES[i] for i in range(1, 9)],
            output_dict=True,
            zero_division=0
        )
        
        classification_report_df = pd.DataFrame(classification_report).transpose()
        report_csv_path = f"{args.output_dir}/classification_report.csv"
        classification_report_df.to_csv(report_csv_path)
        print(f"  Saved classification report: {report_csv_path}")
        
    except Exception as e:
        print(f"Warning: Could not generate classification report: {e}")
    
    # Print final summary
    print("\n" + "="*60)
    print("FINAL ASSESSMENT SUMMARY")
    print("="*60)
    print(f"Total points: {metrics_dict['total_points']}")
    print(f"Valid predictions: {metrics_dict['valid_points']}")
    print(f"Invalid predictions: {metrics_dict['invalid_points']}")
    print(f"Overall Accuracy: {metrics_dict['overall_accuracy']:.4f} ({metrics_dict['overall_accuracy']*100:.2f}%)")
    print(f"Kappa Coefficient: {metrics_dict['kappa']:.4f}")
    print(f"Macro F1-Score: {metrics_dict['f1_score_macro']:.4f}")
    print(f"Weighted F1-Score: {metrics_dict['f1_score_weighted']:.4f}")
    print(f"Micro User Accuracy: {metrics_dict['user_accuracy_micro']:.4f}")
    print(f"Macro User Accuracy: {metrics_dict['user_accuracy_macro']:.4f}")
    print(f"Weighted User Accuracy: {metrics_dict['user_accuracy_weighted']:.4f}")
    print(f"Micro Producer Accuracy: {metrics_dict['producer_accuracy_micro']:.4f}")
    print(f"Macro Producer Accuracy: {metrics_dict['producer_accuracy_macro']:.4f}")
    print(f"Weighted Producer Accuracy: {metrics_dict['producer_accuracy_weighted']:.4f}")
    print(f"\nAll results saved to: {args.output_dir}")
    print("="*60)

if __name__ == "__main__":
    main()