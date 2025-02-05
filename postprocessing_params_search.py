import os
import optuna
import numpy as np
import geopandas as gpd
import rasterio as rio
from rasterio.mask import mask
from rasterio.transform import xy
from rasterio.features import shapes, geometry_mask
from shapely.geometry import Polygon, Point, shape
from skimage.segmentation import felzenszwalb
from skimage.filters import unsharp_mask
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from tqdm import tqdm
from multiprocessing.shared_memory import SharedMemory
import multiprocess as mp
import numba
from numba import prange
from typing import Optional, Tuple, Callable, Dict, Any
from src.mslandcover.config import LEGEND_CLASSES
from src.mslandcover.utils import Logger
import argparse
import json
import pickle


def parse_args() -> argparse.Namespace:
    
    parser = argparse.ArgumentParser(description='Postprocessing parameter search')
    
    parser.add_argument(
        '--okt_probs_raster_path',
        type=str,
        default='/Volumes/dhester_ssd/postprocessing_tests/okt_classes_clipped.tif',
        help='Path to the raster file containing the Oktibehha County, MS land cover probabilities.'
    )
    
    parser.add_argument(
        '--okt_imagery_raster_path',
        type=str,
        default='/Volumes/dhester_ssd/postprocessing_tests/okt_imagery_clipped.tif',
        help='Path to the raster file containing the Oktibehha County, MS NAIP imagery.'
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
        default='./logs/postprocessing_params_search',
        help='Directory to save log files.'
    )
    
    parser.add_argument(
        '--n_trials',
        type=int,
        default=128,
        help='Number of trials to run for the optimization.'
    )
    
    return parser.parse_args()


def main():
    
    args = parse_args()
    os.makedirs(args.log_dir, exist_ok=True)
    logger = Logger(os.path.join(args.log_dir, 'out.log'))
    
    lc_probs, _, okt_imagery, geom, transform, sampled_points_annotated_gdf = load_data(
        args.okt_probs_raster_path, args.okt_imagery_raster_path, args.ms_places_path
    )
    bounding_mask = get_bounding_mask(geom, lc_probs.shape[1:], transform)
    crs = sampled_points_annotated_gdf.crs
    
    def objective(trial):
        # sample segmentation parameters
        scale = trial.suggest_float("scale", 10, 500)
        min_size = trial.suggest_int("min_size", 5, 100)
        sigma = trial.suggest_float("sigma", 0.01, 5.0, log=True)
        # chunk_size = trial.suggest_categorical("chunk_size", [256, 512, 1024])
        unsharp_radius = trial.suggest_int("unsharp_radius", 0, 20)
        unsharp_amount = trial.suggest_float("unsharp_amount", 0.0, 2.0)
        params_trial = {'scale': scale, 'min_size': min_size, 'sigma': sigma}
        
        try:
            
            if unsharp_radius < 0.01 or unsharp_amount == 0.0:
                okt_imagery_sharpened = okt_imagery
            else:
                okt_imagery_sharpened = unsharp_mask(okt_imagery, radius=unsharp_radius, amount=unsharp_amount)
            
            # run segmentation
            segmented_image, _, segment_mapping = parallel_segment(
                okt_imagery_sharpened, lc_probs, felzenszwalb, params_trial, chunk_size=256, n_procs=mp.cpu_count()
            )
            
            
            segment_id_geoms = [
                (shape(geom), seg_id) 
                for geom, seg_id in shapes(segmented_image, mask=bounding_mask, connectivity=4, transform=transform)
            ]
            
            segmented_gdf = gpd.GeoDataFrame({
                'segment_id': [seg_id for _, seg_id in segment_id_geoms],
                'mean_probs': [segment_mapping[seg_id] for _, seg_id in segment_id_geoms],
            }, geometry=[geom for geom, _ in segment_id_geoms], crs=crs)
            segmented_gdf['new_pred_class_index'] = segmented_gdf['mean_probs'].apply(lambda x: np.argmax(x) + 1)
            
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
            logger.log(f'Trial params: {params_trial}')
            logger.log(f'====================')
            
            trial.set_user_attr('error', e)
            
        return loss_value

    
    study = optuna.create_study(study_name='seg_alg_test', direction='minimize')
    study.optimize(objective, n_trials=args.n_trials)

    logger.log("Best trial:")
    logger.log("Params: ", study.best_trial.params)
    logger.log("Loss: ", study.best_trial.value)
    logger.log("Macro F1 Score: ", study.best_trial.user_attrs.get('macro_f1_score'))
    logger.log("Accuracy: ", study.best_trial.user_attrs.get('accuracy'))

    original_f1 = f1_score(sampled_points_annotated_gdf['true_class_index'], sampled_points_annotated_gdf['pred_class_index'], average='macro')
    original_acc = accuracy_score(sampled_points_annotated_gdf['true_class_index'], sampled_points_annotated_gdf['pred_class_index'])

    f1_improvement = (study.best_trial.user_attrs.get('macro_f1_score') - original_f1) / original_f1
    acc_improvement = (study.best_trial.user_attrs.get('accuracy') - original_acc) / original_acc

    logger.log(f"Original F1 Score: {original_f1} ({f1_improvement:.2%} improvement)")
    logger.log(f"Original Accuracy: {original_acc} ({acc_improvement:.2%} improvement)")
    
    out_path = os.path.join(args.log_dir, 'best_params.json')
    json.dump(study.best_trial.params, open(out_path, 'w'))
    logger.log(f'Best params saved to {out_path}')
    
    out_path = os.path.join(args.log_dir, 'study.pkl')
    pickle.dump(study, open(out_path, 'wb'))
    logger.log(f'Study results saved to {out_path}')



def mask_rasters(
    ms_places: gpd.GeoDataFrame,
    okt_confidence_raster_path: str, 
    okt_classes_raster_path: str, 
    okt_probs_raster_path: str, 
    okt_imagery_raster_path: str, 
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
        
    with rio.open(r"G:\postprocessing_tests\okt_confidence_clipped.tif", 'w', **out_profile) as dst:
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

    with rio.open(r"G:\postprocessing_tests\okt_classes_clipped.tif", 'w', **out_profile) as dst:
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

    with rio.open(r"G:\postprocessing_tests\okt_probs_clipped.tif", 'w', **out_profile) as dst:
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

    with rio.open(r"G:\postprocessing_tests\okt_imagery_clipped.tif", 'w', **out_profile) as dst:
        dst.write(okt_imagery)


    lc_confidence = lc_confidence.squeeze()
    lc_classes = lc_classes.squeeze()
    lc_probs = lc_probs.squeeze()
    okt_imagery = okt_imagery.transpose(1, 2, 0)
    
    return lc_confidence, lc_classes, lc_probs, okt_imagery, 



def load_data(
    okt_probs_raster_path: str,
    okt_imagery_raster_path: str,
    ms_places_path: str='./data/shapefiles/tl_2024_28_place',
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

    sampled_points_annotated_gdf = gpd.read_file('./data/shapefiles/okt_ms_low_confidence_point_samples.gpkg')
    sampled_points_annotated_gdf['one_hot'] = sampled_points_annotated_gdf['true_class_index'].apply(lambda x: np.eye(8)[x-1])
    
    return lc_probs, okt_imagery, geom, out_transform, sampled_points_annotated_gdf


def sample_points(
    lc_confidence: np.ndarray,
    lc_classes: np.ndarray,
    transform: rio.transform.Transform,
    crs: rio.crs.CRS,
    n_samples: int = 200,
    max_confidence: int = 25,
    out_path: Optional[str] = r"G:\postprocessing_tests\sampled_points.gpkg",
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
        sampled_points_gdf.to_file(r"G:\postprocessing_tests\sampled_points.gpkg")
    return sampled_points_gdf


@numba.njit(parallel=True)
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
    chunk_seg = seg_func(image_chunk, **seg_params)
    # Update probability values based on segment statistics.
    chunk_seg, lc_probs_chunk = process_chunk_vectorized(chunk_seg, lc_probs_chunk)
    return chunk_seg, lc_probs_chunk


def parallel_segment(
    image: np.ndarray,
    lc_probs: np.ndarray,
    seg_func: Callable[..., np.ndarray],
    seg_params: Dict[str, Any],
    chunk_size: int = 1024,
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
            # For each segment, compute normalized mean probabilities.
            unique_ids = np.unique(chunk_seg)
            for seg in unique_ids:
                mask = (chunk_seg == seg)
                mean_probs = np.mean(updated_chunk_probs[:, mask], axis=1)
                total = np.sum(mean_probs)
                segment_mapping[seg] = mean_probs / total if total != 0 else mean_probs
            running_max = np.max(unique_ids) + 1

    # Clean up shared memory.
    shm_probs.close()
    shm_probs.unlink()

    return segmented_image, processed_probs, segment_mapping


def cross_entropy(y_true, y_pred):
    return -np.sum(y_true * np.log(y_pred))


def get_bounding_mask(geom: Polygon, shape: Tuple[int, int], transform: rio.transform.Transform) -> np.ndarray:
    return geometry_mask(
        [geom], 
        out_shape=shape, 
        transform=transform, 
        invert=True, 
        all_touched=True
    )


if __name__ == '__main__':
    main()