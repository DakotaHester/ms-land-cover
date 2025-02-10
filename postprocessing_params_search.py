from concurrent.futures import ThreadPoolExecutor
from functools import partial
import os
import time
import optuna
import numpy as np
import geopandas as gpd
import rasterio as rio
from rasterio.mask import mask
from rasterio.transform import xy
from rasterio.features import shapes, geometry_mask
from shapely.geometry import Polygon, Point, shape
from skimage.segmentation import felzenszwalb, quickshift, slic
from skimage.filters import unsharp_mask
from skimage.filters import median
from skimage.morphology import disk
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
import torch
from tqdm import tqdm
from multiprocessing.shared_memory import SharedMemory
import multiprocess as mp
from numba import prange, njit
from typing import Optional, Tuple, Callable, Dict, Any, Union
from src.mslandcover.config import LEGEND_CLASSES, LEGEND_COLORS_RGBA
from src.mslandcover.utils import Logger, load_pth, get_torch_device
from src.mslandcover.models import UNet
from src.mslandcover.data.datasets import FineTuneDataset
from torch.utils.data import DataLoader
from sklearn.preprocessing import OneHotEncoder
from glob import glob
import argparse
import json
import pickle


def parse_args() -> argparse.Namespace:
    
    parser = argparse.ArgumentParser(description='Postprocessing parameter search')
    
    parser.add_argument(
        '--data_path',
        type=str,
        default='./data/splits',
        help='Path to the directory containing the land cover data.'
    )
    
    parser.add_argument(
        '--model_path',
        type=str,
        default='./weights/finetuned_unet2/best_model.pth',
        help='Path to the trained model weights.'
    )
    
    parser.add_argument(
        '--mean_path',
        type=str,
        default='./weights/pretrain_mean.pth',
        help='Path to the mean pixel values.'
    )
    
    parser.add_argument(
        '--std_path',
        type=str,
        default='./weights/pretrain_std.pth',
        help='Path to the standard deviation of pixel values.'
    )
    
    parser.add_argument(
        '--log_dir',
        type=str,
        default='./logs/postprocessing_params_search',
        help='Directory to save log files.'
    )
    
    parser.add_argument(
        '--n_trials',
        type=int,
        default=2,
        help='Number of trials to run for the optimization.'
    )
    
    parser.add_argument(
        '--batch_size',
        type=int,
        default=16,
        help='Batch size for the data loader.'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=1701,
        help='Random seed for reproducibility.'
    )
    
    parser.add_argument(
        '--skip_search',
        action='store_true',
        help='Skip the parameter search and use the best parameters found so far.'
    )
    
    return parser.parse_args()


def main():

    args = parse_args()
    
    os.makedirs(args.log_dir, exist_ok=True)
    logger = Logger(os.path.join(args.log_dir, 'out.log'))
    
    device = get_torch_device()
    model = UNet()
    model.load_state_dict(load_pth(args.model_path, map_location='cpu'))
    model.to(device)
    model.eval()
    torch.set_grad_enabled(False)
    
    mean = load_pth(args.mean_path)
    std = load_pth(args.std_path)
    
    train_dataset = FineTuneDataset(
        data_paths=glob(os.path.join(args.data_path, 'train', 'input', '*.tif')),
        target_paths=glob(os.path.join(args.data_path, 'train', 'target', '*.tif')),
        transform=None,
        mean=mean,
        std=std,
    )
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    
    X_train = []
    y_hat_train = []
    y_train = []
    for i, (X, y) in tqdm(enumerate(train_loader), desc='Getting LC probs', total=len(train_loader)):
        
        X_train.append(X.numpy())
        y_hat_train.append(model(X.to(device)).detach().cpu().numpy())
        y_train.append(y.numpy())
    
    X_train = np.concatenate(X_train, axis=0)
    y_hat_train = np.concatenate(y_hat_train, axis=0)
    y_train = np.concatenate(y_train, axis=0)
    
    # y_train.shape = (n_samples, height, width)
    # y_train_one_hot.shape = (n_samples, n_classes, height, width)
    y_train_one_hot = np.stack([np.eye(8)[y_i] for y_i in y_train]).transpose(0, 3, 1, 2)

    train_ce = cross_entropy(y_train_one_hot, y_hat_train)
    train_f1 = f1_score(y_train.flatten(), np.argmax(y_hat_train, axis=1).flatten(), average='macro')
    
    print(f'Starting CE: {train_ce}')
    print(f'Starting F1 (macro): {train_f1}')
    
    if not args.skip_search:
        
        def objective(trial: optuna.Trial) -> float:
            
            # scale = trial.suggest_float("scale", 10, 500)
            # min_size = trial.suggest_int("min_size", 5, 100)
            
            # slic params
            n_segments = trial.suggest_int("n_segments", 10, 1000)
            compactness = trial.suggest_float("compactness", 1e-3, 10.0, log=True)
            
            # quickshift params
            # ratio = trial.suggest_float("ratio", 0, 1)
            # kernel_size = trial.suggest_int("kernel_size", 1, 5)
            # max_dist = trial.suggest_int("max_dist", 1, 5)
            
            median_radius = trial.suggest_int("median_radius", 0, 10)
            unsharp_radius = trial.suggest_int("unsharp_radius", 0, 20)
            unsharp_amount = trial.suggest_float("unsharp_amount", 0.0, 10.0)
            
            median_radius = trial.params.get('median_radius', 0)
            params_trial = {
                ## felzenszwalb params
                # 'scale': scale,
                # 'min_size': min_size,
                # 'sigma': 0,
                ## slic params
                'n_segments': n_segments,
                'compactness': compactness,
                'max_num_iter': 3,
                ## quickshift params
                # 'ratio': ratio,
                # 'kernel_size': kernel_size,
                # 'max_dist': max_dist,
                'median_radius': median_radius,
                'unsharp_radius': unsharp_radius,
                'unsharp_amount': unsharp_amount,
            }
            
            try:
                
                start_time = time.time()
                
                with mp.Pool(mp.cpu_count()) as pool:
                    processed_probs = np.array(pool.starmap(
                        segment_and_process,
                        [(img, probs, slic, params_trial) for img, probs in zip(X_train, y_hat_train)]
                    ))
                
                stop_time = time.time()
                
                algorithm_run_time = stop_time - start_time
                
                loss_value = cross_entropy(y_train_one_hot, processed_probs)
                
                trial.set_user_attr('seg_params', params_trial)
                trial.set_user_attr('alg_runtime', algorithm_run_time)
                
                acc = accuracy_score(y_train.flatten(), np.argmax(processed_probs, axis=1).flatten())
                trial.set_user_attr('accuracy', acc)
                
                for average_type in ['micro', 'macro', 'weighted']:
                    precision = precision_score(y_train.flatten(), np.argmax(processed_probs, axis=1).flatten(), average=average_type, zero_division=0)
                    recall = recall_score(y_train.flatten(), np.argmax(processed_probs, axis=1).flatten(), average=average_type, zero_division=0)
                    f1 = f1_score(y_train.flatten(), np.argmax(processed_probs, axis=1).flatten(), average=average_type, zero_division=0)
                    
                    trial.set_user_attr(f'{average_type}_precision', precision)
                    trial.set_user_attr(f'{average_type}_recall', recall)
                    trial.set_user_attr(f'{average_type}_f1_score', f1)
                
            except Exception as e:
                loss_value = np.inf
                logger.log(f'====================')
                logger.log(f'{type(e).__name__}: {e}')
                logger.log(f'Trial params: {params_trial}')
                logger.log(f'====================')
            
            return loss_value
        
        sampler = optuna.samplers.TPESampler(seed=args.seed)
        study = optuna.create_study(study_name='seg_alg_eval_slic', direction='minimize', sampler=sampler)
        study.optimize(objective, n_trials=args.n_trials)
        
        new_ce = study.best_trial.value
        new_f1 = study.best_trial.user_attrs.get('macro_f1_score')
        
        ce_improvement = (train_ce - new_ce) / train_ce
        f1_improvement = (new_f1 - train_f1) / train_f1
        
        logger.log(f'{"="*5} Train Set {"="*5}')
        logger.log(f'Original CE: {train_ce}')
        logger.log(f'New CE: {new_ce}')
        logger.log(f'CE Improvement: {ce_improvement:.2%}')
        
        logger.log(f'Original F1 (macro): {train_f1}')
        logger.log(f'New F1 (macro): {new_f1}')
        logger.log(f'F1 Improvement: {f1_improvement:.2%}')
        
        test_dataset = FineTuneDataset(
            data_paths=glob(os.path.join(args.data_path, 'test', 'input', '*.tif')) + glob(os.path.join(args.data_path, 'val', 'input', '*.tif')),
            target_paths=glob(os.path.join(args.data_path, 'test', 'target', '*.tif')) + glob(os.path.join(args.data_path, 'val', 'target', '*.tif')),
            transform=None,
            mean=mean,
            std=std,
        )
        
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)
        
        X_test = []
        y_hat_test = []
        y_test = []
        
        for i, (X, y) in tqdm(enumerate(test_loader), desc='Getting LC probs', total=len(test_loader)):
            
            X_test.append(X.numpy())
            y_hat_test.append(model(X.to(device)).detach().cpu().numpy())
            y_test.append(y.numpy())
        
        X_test = np.concatenate(X_test, axis=0)
        y_hat_test = np.concatenate(y_hat_test, axis=0)
        y_test = np.concatenate(y_test, axis=0)
        
        y_test_one_hot = np.stack([np.eye(8)[y_i] for y_i in y_test]).transpose(0, 3, 1, 2)
        
        test_ce = cross_entropy(y_test_one_hot, y_hat_test)
        test_f1 = f1_score(y_test.flatten(), np.argmax(y_hat_test, axis=1).flatten(), average='macro')
        
        # get seg_params from best trial
        best_params = study.best_trial.user_attrs.get('seg_params')
        
        with mp.Pool(mp.cpu_count()) as pool:
            processed_probs = np.array(pool.starmap(
                segment_and_process,
                [(img, probs, slic, best_params) for img, probs in zip(X_test, y_hat_test)]
            ))
        
        new_test_ce = cross_entropy(y_test_one_hot, processed_probs)
        new_test_f1 = f1_score(y_test.flatten(), np.argmax(processed_probs, axis=1).flatten(), average='macro')
        
        new_ce_improvement = (test_ce - new_test_ce) / test_ce
        new_f1_improvement = (new_test_f1 - test_f1) / test_f1
        
        logger.log(f'{"="*5} Test Set {"="*5}')
        logger.log(f'Original CE: {test_ce}')
        logger.log(f'New CE: {new_test_ce}')
        logger.log(f'CE Improvement: {new_ce_improvement:.2%}')
        logger.log(f'Original F1 (macro): {test_f1}')
        logger.log(f'New F1 (macro): {new_test_f1}')
        logger.log(f'F1 Improvement: {new_f1_improvement:.2%}')
        
        
        out_path = os.path.join(args.log_dir, 'best_params.json')
        json.dump(study.best_trial.params, open(out_path, 'w'))
        logger.log(f'Best params saved to {out_path}')
        
        out_path = os.path.join(args.log_dir, 'study.pkl')
        pickle.dump(study, open(out_path, 'wb'))
        logger.log(f'Study results saved to {out_path}')




def old_parse_args() -> argparse.Namespace:
    
    parser = argparse.ArgumentParser(description='Postprocessing parameter search')
    
    parser.add_argument(
        '--okt_probs_raster_path',
        type=str,
        default='./data/postprocessing_trials/okt_probs_clipped.tif',
        help='Path to the raster file containing the Oktibehha County, MS land cover probabilities.'
    )
    
    parser.add_argument(
        '--okt_imagery_raster_path',
        type=str,
        default='./data/postprocessing_trials/okt_imagery_clipped.tif',
        help='Path to the raster file containing the Oktibehha County, MS NAIP imagery.'
    )
    
    parser.add_argument(
        '--sampled_points_path',
        type=str,
        default='./data/shapefiles/okt_ms_low_confidence_point_samples.gpkg',
        help='Path to the GeoPackage containing the annotated sample points.'
    )
    
    parser.add_argument(
        '--ms_places_path',
        type=str,
        default='./data/shapefiles/tl_2024_28_place',
        help='Path to the shapefile containing the Mississippi places.'
    )
    
    parser.add_argument(
        '--log_dir',
        type=str,
        default='./logs/postprocessing_params_search_slic_felzenszwalb',
        help='Directory to save log files.'
    )
    
    parser.add_argument(
        '--n_trials',
        type=int,
        default=1024,
        help='Number of trials to run for the optimization.'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=1701,
        help='Random seed for reproducibility.'
    )
    
    parser.add_argument(
        '--skip_search',
        action='store_true',
        help='Skip the parameter search and use the best parameters found so far.'
    )
    
    return parser.parse_args()



def old_main():
        
    args = parse_args()
    
    # args.skip_search = True
    
    os.makedirs(args.log_dir, exist_ok=True)
    logger = Logger(os.path.join(args.log_dir, 'out.log'))
    
    lc_probs, okt_imagery, geom, transform, sampled_points_annotated_gdf = load_data(
        args.okt_probs_raster_path, args.okt_imagery_raster_path, args.ms_places_path, args.sampled_points_path
    )
    bounding_mask = get_bounding_mask(geom, lc_probs.shape[1:], transform)
    crs = sampled_points_annotated_gdf.crs
    
    if not args.skip_search:
        
        def objective(trial: optuna.Trial) -> float:
            # sample segmentation parameters (felzenszwalb)
            # scale = trial.suggest_float("scale", 10, 500)
            # min_size = trial.suggest_int("min_size", 5, 100)
            # sigma = 0
            # params_trial = {'scale': scale, 'min_size': min_size, 'sigma': sigma}
            
            # segment parameters (quickshift)
            # ratio = trial.suggest_float("ratio", 0, 1)
            # kernel_size = trial.suggest_int("kernel_size", 1, 10)
            # max_dist = trial.suggest_int("max_dist", 1, 10)
            # sigma = 0
            # params_trial = {'ratio': ratio, 'kernel_size': kernel_size, 'max_dist': max_dist, 'sigma': sigma}
            
            # segment parameters (slic)
            # n_segments = trial.suggest_int("n_segments", 10, 1000)
            # compactness = trial.suggest_float("compactness", 0.0, 10.0)
            
            algorithm = trial.suggest_categorical("algorithm", ['felzenszwalb', 'slic'])
            if algorithm == 'felzenszwalb':
                seg_func = felzenszwalb
                scale = trial.suggest_float("scale", 10, 500)
                min_size = trial.suggest_int("min_size", 5, 100)
                seg_func_params = {'scale': scale, 'min_size': min_size, 'sigma': 0}
            
            elif algorithm == 'slic':
                seg_func = slic
                
                n_segments = trial.suggest_int("n_segments", 10, 1000)
                compactness = trial.suggest_float("compactness", 0.0, 10.0)
                seg_func_params = {'n_segments': n_segments, 'compactness': compactness}
            
            else:
                raise ValueError(f"Invalid algorithm: {algorithm}")
            
            
            median_radius = trial.suggest_int("median_radius", 0, 20)
            unsharp_radius = trial.suggest_int("unsharp_radius", 0, 20)
            unsharp_amount = trial.suggest_float("unsharp_amount", 0.0, 10.0)
            seg_func_params.update({"median_radius": median_radius, "unsharp_radius": unsharp_radius, "unsharp_amount": unsharp_amount})
            
            # sigma = 0
            
            
            try:
                
                start_time = time.time()
                
                # filtering and sharpening is now handled in the process_chunk function for multi-processing
                # if median_radius > 0:
                #     # apply median filter over each channel
                #     with mp.Pool(mp.cpu_count()) as pool:
                #         filtered_channels = pool.starmap(
                #             median,
                #             [(channel, disk(median_radius)) for channel in okt_imagery.transpose(2, 0, 1)]
                #         )
                #     okt_imagery_filtered = np.stack(filtered_channels).transpose(1, 2, 0)
                # else:
                #     okt_imagery_filtered = okt_imagery
                
                # if unsharp_radius < 0.01 or unsharp_amount == 0.0:
                #     okt_imagery_sharpened = okt_imagery_filtered
                # else:
                #     okt_imagery_sharpened = unsharp_mask(okt_imagery_filtered, radius=unsharp_radius, amount=unsharp_amount)
                
                # run segmentation
                segmented_image, _, segment_mapping = parallel_segment(
                    okt_imagery, lc_probs, seg_func, seg_func_params, chunk_size=256, n_procs=mp.cpu_count()
                )
                
                stop_time = time.time()
                algorithm_run_time = stop_time - start_time
                
                start_time = time.time()
                segment_id_geoms = [
                    (shape(geom), seg_id) 
                    for geom, seg_id in shapes(segmented_image, mask=bounding_mask, connectivity=4, transform=transform)
                ]
                
                segmented_gdf = gpd.GeoDataFrame({
                    'segment_id': [seg_id for _, seg_id in segment_id_geoms],
                    'mean_probs': [segment_mapping[seg_id] for _, seg_id in segment_id_geoms],
                }, geometry=[geom for geom, _ in segment_id_geoms], crs=crs)
                segmented_gdf['new_pred_class_index'] = segmented_gdf['mean_probs'].apply(lambda x: np.argmax(x) + 1)
                
                stop_time = time.time()
                vectorization_time = stop_time - start_time
                
                # spatial join with annotated sample points and compute loss
                results_gdf = gpd.sjoin(sampled_points_annotated_gdf, segmented_gdf, how='left')
                y_true = np.stack(results_gdf['one_hot'].values)
                y_pred = np.stack(results_gdf['mean_probs'].values)
                loss_value = cross_entropy(y_true, y_pred)
                
                # compute accuracy and weighted f1 score
                true_labels = np.argmax(y_true, axis=1)
                pred_labels = np.argmax(y_pred, axis=1)
                
                acc = accuracy_score(true_labels, pred_labels)
                trial.set_user_attr('accuracy', acc)
                trial.set_user_attr('alg_runtime', algorithm_run_time)
                trial.set_user_attr('vectorization_runtime', vectorization_time)
                trial.set_user_attr('total_runtime', algorithm_run_time + vectorization_time)
                
                for average_type in ['micro', 'macro', 'weighted']:
                    precision = precision_score(true_labels, pred_labels, average=average_type, zero_division=0)
                    recall = recall_score(true_labels, pred_labels, average=average_type, zero_division=0)
                    f1 = f1_score(true_labels, pred_labels, average=average_type, zero_division=0)
                    
                    trial.set_user_attr(f'{average_type}_precision', precision)
                    trial.set_user_attr(f'{average_type}_recall', recall)
                    trial.set_user_attr(f'{average_type}_f1_score', f1)
                
            except Exception as e:
                loss_value = np.inf  # if any error occurs, use infinity to indicate a failed trial
                logger.log(f'====================')
                logger.log(f'{type(e).__name__}: {e}')
                logger.log(f'Trial params: {seg_func_params}')
                logger.log(f'====================')
                
                trial.set_user_attr('error', e)
                
            return loss_value

        sampler = optuna.samplers.TPESampler(seed=args.seed)
        study = optuna.create_study(study_name='seg_alg_eval_slic_felz', direction='minimize', sampler=sampler)
        study.optimize(objective, n_trials=args.n_trials)
        
        out_path = os.path.join(args.log_dir, 'best_params.json')
        json.dump(study.best_trial.params, open(out_path, 'w'))
        logger.log(f'Best params saved to {out_path}')
        
        out_path = os.path.join(args.log_dir, 'study.pkl')
        pickle.dump(study, open(out_path, 'wb'))
        logger.log(f'Study results saved to {out_path}')
    
    else:
        
        best_params = json.load(open('./logs/postprocessing_params_search/best_params.json', 'r'))
        study = pickle.load(open('./logs/postprocessing_params_search/study.pkl', 'rb'))
        
        logger.log(f"Best trial:")
        logger.log(f"Params: {study.best_trial.params}")
        logger.log(f"Loss: {study.best_trial.value}")
        logger.log(f"Macro F1 Score: {study.best_trial.user_attrs.get('macro_f1_score')}")
        logger.log(f"Accuracy: {study.best_trial.user_attrs.get('accuracy')}")

        original_f1 = f1_score(sampled_points_annotated_gdf['true_class_index'], sampled_points_annotated_gdf['pred_class_index'], average='macro')
        original_acc = accuracy_score(sampled_points_annotated_gdf['true_class_index'], sampled_points_annotated_gdf['pred_class_index'])
        
        print(np.stack(sampled_points_annotated_gdf['one_hot'].values).dtype)
        print(np.stack(sampled_points_annotated_gdf['pred_probs'].values).dtype)
        
        # pred probs stored as string of list of floats i.e., '[0.08 0.07 0.22 0.21 0.05 0.22 0.04 0.07]'
        original_pred_probs = np.stack(sampled_points_annotated_gdf['pred_probs'].apply(lambda x: np.array(x[1:-1].split()).astype(np.float32)).values)
        print(original_pred_probs)
        # original_pred_probs = np.stack(sampled_points_annotated_gdf['pred_probs'].values.astype(np.float32))
        original_ce = cross_entropy(np.stack(sampled_points_annotated_gdf['one_hot'].values), original_pred_probs)
        
        f1_improvement = (study.best_trial.user_attrs.get('macro_f1_score') - original_f1) / original_f1
        acc_improvement = (study.best_trial.user_attrs.get('accuracy') - original_acc) / original_acc
        ce_improvement = -(study.best_trial.value - original_ce) / original_ce

        logger.log(f"Original F1 Score: {original_f1} ({f1_improvement:.2%} improvement)")
        logger.log(f"Original Accuracy: {original_acc} ({acc_improvement:.2%} improvement)")
        logger.log(f"Original Cross-Entropy: {original_ce} ({ce_improvement:.2%} improvement)")
    
    # now, postprocess the image with the best parameters
    best_params = study.best_trial.params
    
    profile = rio.open(args.okt_probs_raster_path).profile
    profile.update(
        dtype=rio.uint8,
        count=1,
        compress='lzw',
        nodata=0,
    )
    
    segment_image_only(okt_imagery, lc_probs, best_params, profile, args.log_dir, bounding_geom=geom)



def mask_rasters(
    ms_places: gpd.GeoDataFrame,
    okt_confidence_raster_path: str, 
    okt_classes_raster_path: str, 
    okt_probs_raster_path: str, 
    okt_imagery_raster_path: str, 
    out_dir: str='./data/postprocessing_trials',
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    
    # okt_confidence_raster_path = r"Z:\guser\dh\MS_HiRes_LC_Prelim\105\lc_confidence_clipped.tif"
    with rio.open(okt_confidence_raster_path) as src:
        okt_confidence_profile = src.profile

    places_of_interest = ms_places.loc[ms_places['NAME'].isin(['Starkville', 'Mississippi State'])].to_crs(okt_confidence_profile['crs'])

    places_of_interest = places_of_interest.dissolve()
    geom = Polygon(places_of_interest['geometry'].iloc[0].exterior.coords)

    with rio.open(okt_confidence_raster_path) as src:
        lc_confidence, out_transform = mask(src, [geom], crop=True, all_touched=True)
        out_profile = src.profile
        out_profile.update(
            height=lc_confidence.shape[1],
            width=lc_confidence.shape[2],
            transform=out_transform
        )
        
    # with rio.open(r"G:\postprocessing_tests\okt_confidence_clipped.tif", 'w', **out_profile) as dst:
    with rio.open(os.path.join(out_dir, 'okt_confidence_clipped.tif'), 'w', **out_profile) as dst:
        dst.write(lc_confidence)

    # okt_classes_raster_path = okt_confidence_raster_path.replace('confidence', 'classes')
    with rio.open(okt_classes_raster_path) as src:
        lc_classes, out_transform = mask(src, [geom], crop=True, all_touched=True)
        out_profile = src.profile
        out_profile.update(
            height=lc_classes.shape[1],
            width=lc_classes.shape[2],
            transform=out_transform
        )
        cmap = src.colormap(1)

    # with rio.open(r"G:\postprocessing_tests\okt_classes_clipped.tif", 'w', **out_profile) as dst:
    with rio.open(os.path.join(out_dir, 'okt_classes_clipped.tif'), 'w', **out_profile) as dst:
        dst.write(lc_classes)
        dst.write_colormap(1, cmap)

    # okt_probs_raster_path = okt_confidence_raster_path.replace('confidence', 'probs')
    with rio.open(okt_probs_raster_path) as src:
        lc_probs, out_transform = mask(src, [geom], crop=True, all_touched=True)
        out_profile = src.profile
        out_profile.update(
            height=lc_probs.shape[1],
            width=lc_probs.shape[2],
            transform=out_transform
        )

    # with rio.open(r"G:\postprocessing_tests\okt_probs_clipped.tif", 'w', **out_profile) as dst:
    with rio.open(os.path.join(out_dir, 'okt_probs_clipped.tif'), 'w', **out_profile) as dst:
        dst.write(lc_probs)

    # okt_imagery_raster_path = r"G:\NAIP_MS_2023\ortho_1-1_hc_s_ms105_2023_1\ortho_1-1_hc_s_ms105_2023_1_1m.tif"
    with rio.open(okt_imagery_raster_path) as src:
        okt_imagery, out_transform = mask(src, [geom], crop=True, all_touched=True)
        out_profile = src.profile
        out_profile.update(
            height=okt_imagery.shape[1],
            width=okt_imagery.shape[2],
            transform=out_transform
        )

    # with rio.open(r"G:\postprocessing_tests\okt_imagery_clipped.tif", 'w', **out_profile) as dst:
    with rio.open(os.path.join(out_dir, 'okt_imagery_clipped.tif'), 'w', **out_profile) as dst:
        dst.write(okt_imagery)


    lc_confidence = lc_confidence.squeeze()
    lc_classes = lc_classes.squeeze()
    lc_probs = lc_probs.squeeze()
    okt_imagery = okt_imagery.transpose(1, 2, 0)
    
    return lc_confidence, lc_classes, lc_probs, okt_imagery, 



def load_data(
    okt_probs_raster_path: str='./data/postprocessing_trials/okt_probs_clipped.tif',
    okt_imagery_raster_path: str='./data/postprocessing_trials/okt_imagery_clipped.tif',
    ms_places_path: str='./data/shapefiles/tl_2024_28_place',
    sampled_points_path: str='./data/postprocessing_trials/okt_ms_low_confidence_point_samples.gpkg',
) -> Tuple[np.ndarray, np.ndarray, Polygon, rio.Affine, gpd.GeoDataFrame]:
    
    
    
    # okt_probs_raster_path = '/Volumes/dhester_ssd/postprocessing_tests/okt_probs_clipped.tif'
    with rio.open(okt_probs_raster_path) as src:
        lc_probs = src.read().squeeze()

    # okt_classes_raster_path: str='/Volumes/dhester_ssd/postprocessing_tests/okt_probs_clipped.tif',
    # # okt_classes_raster_path = '/Volumes/dhester_ssd/postprocessing_tests/okt_classes_clipped.tif'
    # with rio.open(okt_classes_raster_path) as src:
    #     lc_classes = src.read().squeeze()

    # okt_imagery_raster_path = '/Volumes/dhester_ssd/postprocessing_tests/okt_imagery_clipped.tif'
    with rio.open(okt_imagery_raster_path) as src:
        okt_imagery = src.read().transpose(1, 2, 0)
        crs = src.crs
        out_transform = src.transform
        
    # ms_places_gdf = gpd.read_file('./data/shapefiles/tl_2024_28_place')
    ms_places_gdf = gpd.read_file(ms_places_path)
    places_of_interest = ms_places_gdf.loc[ms_places_gdf['NAME'].isin(['Starkville', 'Mississippi State'])].to_crs(crs)
    places_of_interest = places_of_interest.dissolve()
    geom = Polygon(places_of_interest['geometry'].iloc[0].exterior.coords)

    # sampled_points_annotated_gdf = gpd.read_file('./data/shapefiles/okt_ms_low_confidence_point_samples.gpkg')
    sampled_points_annotated_gdf = gpd.read_file(sampled_points_path)
    sampled_points_annotated_gdf['one_hot'] = sampled_points_annotated_gdf['true_class_index'].apply(lambda x: np.eye(8)[x-1])
    
    return lc_probs, okt_imagery, geom, out_transform, sampled_points_annotated_gdf


def sample_points(
    lc_confidence: np.ndarray,
    lc_classes: np.ndarray,
    transform: rio.Affine,
    crs: rio.crs.CRS,
    n_samples: int = 200,
    max_confidence: int = 25,
    out_path: Optional[str] = None,
    seed: int = 1701,
) -> gpd.GeoDataFrame:
    
    np.random.seed(seed)

    # randomly sample points where confidence is below threshold and class is not 0
    pop_points = np.argwhere((lc_confidence < max_confidence) & (lc_classes != 0))
    sample_indices = np.random.choice(range(pop_points.shape[0]), n_samples, replace=False)
    sample_points = pop_points[sample_indices]

    # get predicted classes for each sample point
    sample_classes = lc_classes[sample_points[:, 0], sample_points[:, 1]]
    sample_class_names = [LEGEND_CLASSES[cls] for cls in sample_classes]

    # convert sample points to coordinates in raster crs
    sample_coords = [Point(xy(transform, point[0], point[1])) for point in sample_points]

    sampled_points_gdf = gpd.GeoDataFrame({
        'confidence': lc_confidence[sample_points[:, 0], sample_points[:, 1]],
        'pred_class_index': sample_classes,
        'pred_class_name': sample_class_names,
    },geometry=sample_coords, crs=crs)
    
    if out_path is not None:
        sampled_points_gdf.to_file(out_path)
    return sampled_points_gdf


@njit(parallel=True)
def process_chunk_vectorized(
    chunk_seg: np.ndarray, 
    lc_probs_chunk: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Update probabilities in a chunk based on segment averages.

    For each unique segment in the chunk, compute the mean probability 
    across all pixels for each channel and assign the computed mean to 
    all pixels in that segment.

    Parameters
    ----------
    chunk_seg : np.ndarray
        2D array representing segmentation labels for the current image chunk.
    lc_probs_chunk : np.ndarray
        3D array with shape (channels, height, width) containing probability values 
        corresponding to the image chunk.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        A tuple containing:
        - The segmentation chunk with unchanged labels.
        - The updated probability chunk with each pixel replaced by its channel mean.
    """
    unique_segments = np.unique(chunk_seg)
    
    # Loop over each unique segment in parallel.
    for s in prange(unique_segments.shape[0]):
        segment = unique_segments[s]
        # Boolean mask: True for pixels belonging to the current segment.
        mask = (chunk_seg == segment)
        
        # Process each probability channel.
        for i in range(lc_probs_chunk.shape[0]):
            s_val = 0.0
            cnt = 0
            # Sum and count probability values over all pixels in the segment.
            for idx in range(mask.size):
                if mask.flat[idx]:
                    s_val += lc_probs_chunk[i].flat[idx]
                    cnt += 1
            # Compute mean probability for the segment; avoid division by zero.
            mean_val = s_val / cnt if cnt > 0 else 0.0
            # Assign the mean value back into each pixel of the segment.
            for idx in range(mask.size):
                if mask.flat[idx]:
                    lc_probs_chunk[i].flat[idx] = mean_val
    return chunk_seg, lc_probs_chunk


def process_chunk(
    image_chunk: np.ndarray,
    seg_func: Callable[..., np.ndarray],
    seg_params: Dict[str, Any],
    lc_probs_chunk: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Segment a chunk of the image and update its corresponding probability values.

    Applies a segmentation function to a chunk of the image and then uses the 
    vectorized processing to update the probability chunk based on segment means.

    Parameters
    ----------
    image_chunk : np.ndarray
        A sub-array of the overall image to be segmented.
    seg_func : Callable[..., np.ndarray]
        Segmentation function (e.g., felzenszwalb) that accepts the image chunk and parameters.
    seg_params : dict
        Dictionary of parameters to be passed to the segmentation function.
    lc_probs_chunk : np.ndarray
        A sub-array of the probability array corresponding to the image chunk.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        A tuple containing:
        - The segmentation mask for the chunk.
        - The updated probability chunk.
    """
    # Segment the image chunk with the provided segmentation function.
    median_radius = seg_params.pop('median_radius')
    if median_radius > 0:
        structuring_element = disk(median_radius)
        image_chunk_filtered = np.stack([
            median(
                image_chunk[:, :, i], 
                structuring_element,
            ) 
            for i in range(image_chunk.shape[2])
        ]).transpose(1, 2, 0)

    else:
        image_chunk_filtered = image_chunk
    
    unsharp_radius = seg_params.pop('unsharp_radius')
    unsharp_amount = seg_params.pop('unsharp_amount')
    if unsharp_radius < 0.01 or unsharp_amount == 0.0:
        image_chunk_sharpened = image_chunk_filtered
    else:
        image_chunk_sharpened = unsharp_mask(image_chunk_filtered, radius=unsharp_radius, amount=unsharp_amount)
        
    chunk_seg = seg_func(image_chunk_sharpened, **seg_params)
    # Update probability values based on segment statistics.
    chunk_seg, lc_probs_chunk = process_chunk_vectorized(chunk_seg, lc_probs_chunk)
    return chunk_seg, lc_probs_chunk


def parallel_segment(
    image: np.ndarray,
    lc_probs: np.ndarray,
    seg_func: Callable[..., np.ndarray],
    seg_params: Dict[str, Any],
    chunk_size: int = 256,
    n_procs: int = 16,
    enable_pbar: bool = True,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Segment a full image in parallel and update probability values accordingly.

    Divides the image and its corresponding probability array into chunks, processes each 
    chunk in parallel, and then reassembles the full segmented image, updated probabilities, 
    and a mapping from segment IDs to normalized mean probabilities.

    Parameters
    ----------
    image : np.ndarray
        The full image data as a numpy array with shape (height, width, channels) or similar.
    lc_probs : np.ndarray
        The probability array with shape (channels, height, width).
    seg_func : Callable[..., np.ndarray]
        The segmentation function to apply to each image chunk.
    seg_params : dict
        Parameters for the segmentation function.
    chunk_size : int, optional
        The size of each chunk (default is 1024). The image is divided into non-overlapping 
        chunks of shape (chunk_size, chunk_size).
    n_procs : int, optional
        Number of processes to run in parallel (default is 16).
    enable_pbar : bool, optional
        Whether to display a progress bar (default is True).

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, dict]
        A tuple containing:
        - segmented_image: np.ndarray with unique segment labels for the whole image.
        - processed_probs: np.ndarray with updated probability values.
        - segment_mapping: Dictionary mapping each segment ID to its normalized mean probabilities.
    """
    img_height, img_width = image.shape[:2]
    segmented_image = np.zeros((img_height, img_width), dtype=np.int32)
    processed_probs = np.zeros(lc_probs.shape, dtype=lc_probs.dtype)
    segment_mapping: Dict = {}

    # Generate a list of chunk starting indices.
    chunks = [(i, j) for i in range(0, img_height, chunk_size) for j in range(0, img_width, chunk_size)]
    
    # Create shared memory for the lc_probs array to avoid duplicating large arrays.
    shm_probs = SharedMemory(create=True, size=lc_probs.nbytes)
    shared_probs = np.ndarray(lc_probs.shape, dtype=lc_probs.dtype, buffer=shm_probs.buf)
    shared_probs[:] = lc_probs[:]

    # Prepare arguments for processing each chunk.
    args = []
    for i, j in chunks:
        img_chunk = image[i:i+chunk_size, j:j+chunk_size]
        # Copy the corresponding chunk of probability data.
        lc_chunk = shared_probs[:, i:i+chunk_size, j:j+chunk_size].copy()
        args.append((img_chunk, seg_func, seg_params, lc_chunk, i, j))

    def process_wrapper(
        arg: Tuple[np.ndarray, Callable[..., np.ndarray], Dict[str, Any], np.ndarray, int, int]
    ) -> Tuple[int, int, np.ndarray, np.ndarray]:
        """
        Unpack arguments and process a single image-chunk.

        Parameters
        ----------
        arg : tuple
            A tuple containing:
            - image_chunk: np.ndarray
            - seg_func: segmentation function
            - seg_params: parameters for seg_func
            - lc_probs_chunk: np.ndarray for current chunk probabilities
            - i, j: starting indices in the full image

        Returns
        -------
        tuple
            A tuple containing:
            - i: row index of chunk start.
            - j: column index of chunk start.
            - chunk_seg: processed segmentation of the chunk.
            - updated_chunk_probs: updated probability chunk.
        """
        img_chunk, seg_func, seg_params, lc_chunk, i, j = arg
        chunk_seg, updated_chunk_probs = process_chunk(img_chunk, seg_func, seg_params, lc_chunk)
        return i, j, chunk_seg, updated_chunk_probs

    running_max = 0
    # Process chunks in parallel using a process pool.
    with mp.Pool(n_procs) as pool:
        for i, j, chunk_seg, updated_chunk_probs in tqdm(
            pool.imap_unordered(process_wrapper, args),
            total=len(args),
            desc="Processing chunks",
            unit="chunk",
            disable=not enable_pbar,
        ):
            # Ensure unique segment labels across chunks.
            chunk_seg += running_max
            segmented_image[i:i+chunk_size, j:j+chunk_size] = chunk_seg
            processed_probs[:, i:i+chunk_size, j:j+chunk_size] = updated_chunk_probs
            
            # Parallel computation of segment statistics
            unique_ids = np.unique(chunk_seg)
            segment_stats = []
            # Loop through segments and compute mean probs
            for seg in unique_ids:
                mask = (chunk_seg == seg)
                mean_probs = np.mean(updated_chunk_probs[:, mask], axis=1)
                total = np.sum(mean_probs)
                # Normalize probabilities
                norm_probs = mean_probs / total if total != 0 else mean_probs
                # segment_mapping.update({seg: norm_probs})
                segment_stats.append((seg, norm_probs))
            
            # Update mapping with computed statistics
            segment_mapping.update(dict(segment_stats))
            running_max = np.max(unique_ids) + 1

    # Clean up shared memory.
    shm_probs.close()
    shm_probs.unlink()

    return segmented_image, processed_probs, segment_mapping


def cross_entropy(y_true, y_pred, eps=1e-10):
    return -np.sum(y_true * np.log(y_pred + eps))


def get_bounding_mask(geom: Polygon, shape: Tuple[int, int], transform: rio.Affine) -> np.ndarray:
    return geometry_mask(
        [geom], 
        out_shape=shape, 
        transform=transform, 
        invert=True, 
        all_touched=True
    )



def segment_image_only(okt_imagery: np.ndarray, lc_probs: np.ndarray, best_params: dict, profile: dict, out_path: str, bounding_geom: Optional[Polygon] = None):
    
    algorithm = best_params.pop('algorithm')
    if algorithm == 'felzenszwalb':
        seg_func = felzenszwalb
    elif algorithm == 'slic':
        seg_func = slic
    else:
        raise ValueError(f"Invalid algorithm: {algorithm}")
    
    # with mp.Pool(mp.cpu_count()) as pool:
    #     filtered_channels = pool.starmap(
    #         median,
    #         [(channel, disk(best_params['median_radius'])) for channel in okt_imagery.transpose(2, 0, 1)]
    #     )
    # okt_imagery_filtered = np.stack(filtered_channels).transpose(1, 2, 0)
    
    # okt_imagery_sharpened = unsharp_mask(okt_imagery_filtered, radius=best_params.pop('unsharp_radius'), amount=best_params.pop('unsharp_amount'))
    
    _, processed_probs, __ = parallel_segment(
        okt_imagery, lc_probs, seg_func, best_params, chunk_size=256, n_procs=mp.cpu_count()
    )
    
    processed_classes = (np.argmax(processed_probs, axis=0) + 1).squeeze().astype(np.uint8)
    
    
    with rio.open(os.path.join(out_path, 'processed_classes.tif'), 'w', **profile) as dst:
        dst.write(processed_classes, 1)
        dst.write_colormap(1, LEGEND_COLORS_RGBA)
    
    if bounding_geom is not None:
        with rio.open(os.path.join(out_path, 'processed_classes.tif')) as src:
            out_data, out_transform = mask(src, [bounding_geom], crop=True, all_touched=True)
            out_profile = profile.copy()
            out_profile.update(
                height=out_data.shape[1],
                width=out_data.shape[2],
                transform=out_transform
            )
        
        with rio.open(os.path.join(out_path, 'processed_classes_clipped.tif'), 'w', **out_profile) as dst:
            dst.write(out_data)
            dst.write_colormap(1, LEGEND_COLORS_RGBA)
        


@njit(parallel=False)
def postprocess_probs(segmented_image: np.ndarray, probs: np.ndarray) -> np.ndarray:
    
    unique_segments = np.unique(segmented_image)
    # for seg in unique_segments:
    for i in prange(unique_segments.shape[0]):
        mask = (segmented_image == i)
        for j in range(probs.shape[0]):
            s_val = 0.0
            cnt = 0
            for idx in range(mask.size):
                if mask.flat[idx]:
                    s_val += probs[j].flat[idx]
                    cnt += 1
            mean_val = s_val / cnt if cnt > 0 else 0.0
            for idx in range(mask.size):
                if mask.flat[idx]:
                    probs[j].flat[idx] = mean_val
    return probs

def segment_image(
    img: np.ndarray,
    seg_func: Callable[..., np.ndarray],
    params: Dict[str, Union[int, float]],
) -> np.ndarray:
    
    median_radius = params.pop('median_radius')
    if median_radius > 0:
        for i in range(3):
            img[:, :, i] = median(img[:, :, i], disk(median_radius))
    
    unsharp_radius = params.pop('unsharp_radius')
    unsharp_amount = params.pop('unsharp_amount')
    if unsharp_radius > 0 and unsharp_amount > 0:
        img = unsharp_mask(img, radius=unsharp_radius, amount=unsharp_amount)
    
    return seg_func(img, **params)



def segment_and_process(
    img: np.ndarray,
    probs: np.ndarray,
    seg_func: Callable[..., np.ndarray],
    params: Dict[str, Union[int, float]],
):
    segmented_image = segment_image(img.copy().transpose(1, 2, 0), seg_func, params.copy())
    return postprocess_probs(segmented_image, probs)
    



if __name__ == '__main__':
    main()
