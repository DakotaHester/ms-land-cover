from concurrent.futures import ThreadPoolExecutor
from functools import partial
import math
import numpy as np
import multiprocessing as mp
from multiprocessing.pool import Pool
import rasterio
from rasterio.mask import mask
from rasterio.features import geometry_mask, rasterize
import torch
import torch.nn as nn
import torch.nn.functional as F
from shapely import Polygon
from tqdm import tqdm
from src.mslandcover.utils import Logger, get_torch_device
from typing import Callable, Dict, List, Tuple, Optional, Union
from queue import Queue
from threading import Thread
import cv2 as cv
from shapely.geometry import shape
import geopandas as gpd
from src.mslandcover.config import LEGEND_CLASSES
from src.mslandcover.data.transforms import normalize
from multiprocessing import Pool, shared_memory, cpu_count
from scipy.special import softmax
from scipy.ndimage import convolve
from joblib import Parallel, delayed
from skimage.segmentation import quickshift, felzenszwalb
from skimage.filters import unsharp_mask, median
from skimage.morphology import disk
from numba import jit, njit, prange


def process_batch(args):
    """
    Process a batch of segment IDs using shared memory.
    
    Args:
        args: Tuple containing:
            - segment_ids: List of segment IDs to process
            - shm_raster_name: Name of shared memory for raster data
            - shm_segments_name: Name of shared memory for segments array
            - raster_shape: Shape of raster array
            - segments_shape: Shape of segments array
            - raster_dtype: Dtype of raster array
    """
    segment_ids, shm_raster_name, shm_segments_name, raster_shape, segments_shape, raster_dtype, segments_dtype = args
    
    # Access shared memory
    raster_shm = shared_memory.SharedMemory(name=shm_raster_name)
    segments_shm = shared_memory.SharedMemory(name=shm_segments_name)
    
    raster = np.ndarray(raster_shape, dtype=raster_dtype, buffer=raster_shm.buf)
    segments = np.ndarray(segments_shape, dtype=segments_dtype, buffer=segments_shm.buf)
    
    try:
        results = []
        # Create a mask for all segment_ids at once
        masks = np.stack([(segments == segment_id) for segment_id in segment_ids])
        # Compute means for all segments in one operation
        means = np.array([raster[:, mask].mean(axis=1) if mask.any() else np.zeros(raster_shape[0]) 
                         for mask in masks])
        # Create results list
        results = list(zip(segment_ids, means))
        
        return results
    
    finally:
        raster_shm.close()
        segments_shm.close()

def compute_segment_means(raster, segments, n_processes=None, batch_size=1000):
    """
    Compute mean values for each segment in parallel using shared memory.
    
    Args:
        raster: Input raster array (classes, height, width)
        segments: Segmentation array (height, width) with unique IDs
        n_processes: Number of processes (defaults to CPU count)
        batch_size: Number of segments to process in each batch
    
    Returns:
        Dict mapping segment IDs to their mean class probabilities
    """
    if n_processes is None:
        n_processes = cpu_count()
    
    # Create shared memory for input arrays
    raster_shm = shared_memory.SharedMemory(create=True, size=raster.nbytes)
    segments_shm = shared_memory.SharedMemory(create=True, size=segments.nbytes)
    
    shared_raster = np.ndarray(raster.shape, dtype=raster.dtype, buffer=raster_shm.buf)
    shared_segments = np.ndarray(segments.shape, dtype=segments.dtype, buffer=segments_shm.buf)
    
    np.copyto(shared_raster, raster)
    np.copyto(shared_segments, segments)
    
    try:
        # Get unique segment IDs and create batches
        unique_segments = np.unique(segments)
        if unique_segments[0] == 0:  # Skip background if present
            unique_segments = unique_segments[1:]
        
        if batch_size is None:
            batch_size = len(unique_segments) // n_processes
        
        batch_args = []
        for i in range(0, len(unique_segments), batch_size):
            batch_segments = unique_segments[i:i + batch_size]
            batch_args.append((
                batch_segments,
                raster_shm.name,
                segments_shm.name,
                raster.shape,
                segments.shape,
                raster.dtype,
                segments.dtype
            ))
        
        # Process batches in parallel
        results = {}
        with Pool(n_processes) as pool:
            # for batch_results in tqdm(
            #     pool.imap_unordered(process_batch, batch_args),
            #     total=len(batch_args),
            #     desc="Computing segment means",
            #     unit="batches",
            #     unit_scale=batch_size,
            # ):             
            for batch_results in pool.imap_unordered(process_batch, batch_args):   
                for segment_id, means in batch_results:
                    results[segment_id] = means
        
        return results
    
    finally:
        raster_shm.close()
        raster_shm.unlink()
        segments_shm.close()
        segments_shm.unlink()




def mean_shift(image, spatial_radius=10, color_radius=5):
    """
    Perform mean shift segmentation using OpenCV.
    Args:
        image (numpy array): Aerial image (H x W x 3).
        spatial_radius (int): Spatial window radius.
        color_radius (int): Color window radius.
    Returns:
        numpy array: Segmented image with unique labels.
    """
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)
    
    segmented = cv.pyrMeanShiftFiltering(image, sp=spatial_radius, sr=color_radius)

    # Convert the segmented image to a format suitable for labeling
    # (e.g., ensuring each unique color is represented as a single label)
    reshaped_segmented = segmented.reshape(-1, 3)
    _, labels = np.unique(reshaped_segmented, axis=0, return_inverse=True)

    # Reshape labels back to the original image dimensions
    return labels.reshape(segmented.shape[:2]).astype(np.uint16)


def create_feature(geom, value):
    return {"geometry": shape(geom), "segment_id": value}



def segments_to_polygons(segments, transform):
    """
    Convert raster segments to polygons.
    Args:
        segments (numpy array): Segmented image (H x W).
        transform (Affine): Affine transform of the raster.
    Returns:
        list: List of polygons with segment IDs.
    """
    shapes = rasterio.features.shapes(segments, transform=transform, connectivity=8)
    # polygons = [{"geometry": shape(geom), "segment_id": value} for geom, value in shapes]
    polygons = []
    with Pool(mp.cpu_count()) as pool:
        for polygon in tqdm(pool.starmap(create_feature, shapes), total=len(segments)):
            polygons.append(polygon)
    return polygons

def process_polygon(poly, land_cover_probs, segments):
    segment_id = poly["segment_id"]
    mask = segments == segment_id

    mean_probs = land_cover_probs[:, mask].mean(axis=1)
    predicted_class = np.argmax(mean_probs)

    return {
        "geometry": poly["geometry"],
        "segment_id": segment_id,
        **{f"{LEGEND_CLASSES[i]}_mean_prob": prob for i, prob in enumerate(mean_probs)},
        "predicted_class": predicted_class,
    }


def create_geodataframe(polygons: List[Polygon], land_cover_probs: np.ndarray, segments: np.ndarray, crs: str):
    """
    Create a GeoDataFrame from polygons and land cover probabilities.
    Args:
        polygons (list): List of polygons with segment IDs.
        land_cover_probs (numpy array): Land cover probabilities (H x W x C).
        segments (numpy array): Segmented image (H x W).
    Returns:
        GeoDataFrame: GeoDataFrame with polygons and class probabilities.
    """
    worker = partial(process_polygon, land_cover_probs=land_cover_probs, segments=segments)
    with Pool(mp.cpu_count()) as pool:
        rows = list(tqdm(pool.imap(worker, polygons), total=len(polygons), desc='Processing polygons'))
    
    gdf = gpd.GeoDataFrame(rows, crs=crs)  # Use appropriate CRS
    return gdf



class RasterProcessor:
    def __init__(
        self, 
        input_raster_path: str,
        output_path: str,
        bounding_polygon: Polygon = None,
        tile_size: int = 256,
        stride: int = 64,
        gaussian_sigma: float = 128,
        num_workers: int = None,
        mean: torch.Tensor = None,
        std: torch.Tensor = None,
        logger: Logger = None,
        device: torch.device = torch.device('cpu'),
    ):
        self.input_path = input_raster_path
        self.output_path = output_path
        self.bounding_polygon = bounding_polygon
        self.tile_size = tile_size
        self.stride = stride
        self.sigma = gaussian_sigma
        self.num_workers = num_workers or mp.cpu_count()
        self.mean = mean
        self.std = std
        self.logger = logger
        self.device = device

        logger.log('Loading data and metadata...')
        # Load data and metadata
        with rasterio.open(input_raster_path) as src:
            self.profile = src.profile.copy()
            self.raster_data = src.read()
        logger.log('Data and metadata loaded.')
        
        # Create weight matrix once
        x = np.linspace(-tile_size/2, tile_size/2, tile_size)
        y = np.linspace(-tile_size/2, tile_size/2, tile_size)
        X, Y = np.meshgrid(x, y)
        self.weights = np.exp(-(X**2 + Y**2)/(2*self.sigma**2))
        self.weights /= self.weights.max()
        logger.log('Weight matrix created.')
    
    @staticmethod
    def process_tile(args):
        tile, weights, model, mean, std = args
        if (tile == 0).all():
            return np.zeros((model.num_classes, tile.shape[1], tile.shape[2]))
        
        # Normalize and process
        tile = torch.from_numpy(tile.astype(np.float32) / 255.0)
        
        model_device = next(model.parameters()).device
        if tile.device != model_device:
            tile = tile.to(model_device)
        
        with torch.no_grad():
            probs = model(tile.unsqueeze(0)).cpu()
        
        return probs.squeeze(0).numpy() * weights

    def process_raster(self, model: nn.Module):
        height, width = self.raster_data.shape[1:]
        num_classes = model.num_classes
        
        # Add padding
        pad = self.tile_size - self.stride
        self.logger.log(f'Padding raster data with {pad} pixels...')
        padded = np.pad(self.raster_data, ((0,0), (pad,pad), (pad,pad)), mode='reflect')
        
        # Initialize outputs
        outputs = np.zeros((num_classes, height, width))
        weights_sum = np.zeros((height, width))
        self.logger.log('Initialized tiles...')
        
        # Generate tile coordinates
        self.logger.log('Generating tile coordinates...')
        coords = []
        for y in range(0, height, self.stride):
            for x in range(0, width, self.stride):
                # Extract tile with padding
                tile = padded[:, y:y+self.tile_size, x:x+self.tile_size]
                if tile.shape[1:] == (self.tile_size, self.tile_size):  # Only process full tiles
                    coords.append((tile, self.weights, model, self.mean, self.std))
        self.logger.log('Generated tile coordinates.')
        
        self.logger.log('Processing tiles...')
        # Process tiles in parallel
        with tqdm(total=len(coords), desc='Processing tiles', unit='tiles') as pbar:
            with Pool(self.num_workers) as pool:
                results = pool.imap(self.process_tile, coords)
                for _ in results:
                    pbar.update(1)
        
        self.logger.log('Accumulating results...')
        # Accumulate results
        for idx, weighted_probs in enumerate(results):
            y = (idx // (width // self.stride)) * self.stride
            x = (idx % (width // self.stride)) * self.stride
            
            outputs[:, y:y+self.tile_size, x:x+self.tile_size] += weighted_probs
            weights_sum[y:y+self.tile_size, x:x+self.tile_size] += self.weights
        
        self.logger.log('Finalizing predictions...')
        # Get final predictions
        final_outputs = outputs / (weights_sum + 1e-10)
        predictions = np.argmax(final_outputs, axis=0).astype(np.uint8)
        
        self.logger.log('Saving results...')
        # Save results
        profile = self.profile.copy()
        profile.update(count=1, dtype=rasterio.uint8)
        
        if self.bounding_polygon:
            self.logger.log('Masking predictions by bounding polygon...')
            mask = geometry_mask([self.bounding_polygon], out_shape=(height, width), transform=profile['transform'], invert=True, all_touched=True)
            predictions[~mask] = 0
        
        self.logger.log(f'Writing results to {self.output_path}...')
        with rasterio.open(self.output_path, 'w', **profile) as dst:
            dst.write(predictions.astype(rasterio.uint8), 1)
        
        
        if self.bounding_polygon:
            self.logger.log('Clipping raster...')
            with rasterio.open(self.output_path.replace('.tif', '_mask.tif'), 'w', **profile) as dst:
                clipped_data, clipped_transform = mask(rasterio.open(self.output_path), [self.bounding_polygon], crop=True)
                
            profile.update({
                'transform': clipped_transform,
                'height': clipped_data.shape[1],
                'width': clipped_data.shape[2]
            })
            
            self.logger.log(f'Writing clipped results to {self.output_path.replace(".tif", "_mask.tif")}...')
            with rasterio.open(self.output_path.replace('.tif', '_mask.tif'), 'w', **profile) as dst:
                dst.write(clipped_data)
        
        self.logger.log('Processing complete.')
        
        import numpy as np



class GPURasterProcessor:
    """A class for processing large rasters using GPU-accelerated deep learning models."""
    
    def __init__(
        self, 
        model: nn.Module,
        tile_size: int=256,
        stride: int=64,
        gaussian_sigma: float=128,
        batch_size: int=32,
        mean: Optional[torch.Tensor]=None,
        std: Optional[torch.Tensor]=None,
        device: torch.device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        seg_func: Optional[Callable]=None,
        seg_func_params: Optional[Dict[str, Union[int, float]]]=None,
        apply_random_transformations: bool=False,
    ):
        self.tile_size = tile_size
        self.stride = stride
        self.sigma = gaussian_sigma
        self.batch_size = batch_size
        self.device = device
        self.seg_func = seg_func
        self.seg_func_params = seg_func_params
        self.apply_random_transformations = apply_random_transformations
        
        # Move model to device and set to eval mode
        self.model = model.to(device)
        self.model.eval()
        
        self.pad_size = tile_size - stride

        self.mean = mean.cpu().numpy() if mean is not None else None
        self.std = std.cpu().numpy() if std is not None else None

        # Create weight matrix and move to GPU
        x = np.linspace(-tile_size/2, tile_size/2, tile_size)
        y = np.linspace(-tile_size/2, tile_size/2, tile_size)
        X, Y = np.meshgrid(x, y)
        weights = np.exp(-(X**2 + Y**2)/(2*self.sigma**2))
        weights /= weights.max()
        weights = weights.astype(np.float32)
        self.weights_cpu = torch.from_numpy(weights)
        self.weights_gpu = torch.from_numpy(weights).unsqueeze(0).unsqueeze(0).to(device)
        self.weights_cpu.requires_grad = False
        self.weights_gpu.requires_grad = False

    def generate_tile_batches(self, raster_data: np.ndarray):
        height, width = raster_data.shape[1:]
        batch_tiles = []
        batch_coords = []

        for y in range(0, height, self.stride):
            for x in range(0, width, self.stride):
                tile = raster_data[:, y:y+self.tile_size, x:x+self.tile_size]

                if (tile == 0).all() or 0 in tile.shape:
                    continue

                if tile.shape[1:] != (self.tile_size, self.tile_size):
                    continue

                batch_tiles.append(tile)
                batch_coords.append((y, x))

                if len(batch_tiles) == self.batch_size:
                    yield (torch.from_numpy(np.stack(batch_tiles)), batch_coords)
                    batch_tiles = []
                    batch_coords = []
        
        if batch_tiles:
            yield (torch.from_numpy(np.stack(batch_tiles)), batch_coords)    

    def process_batch(self, batch_tiles: torch.Tensor, batch_coords: List[Tuple[int, int]]) -> Tuple[torch.Tensor, List[Tuple[int, int]]]:
        """Process a batch of tiles on GPU.
        
        Parameters
        ----------
        batch_tiles : torch.Tensor
            Batch of tiles to process, shape (B, C, H, W)
        batch_coords : List[Tuple[int, int]]
            List of (y, x) coordinates for each tile
            
        Returns
        -------
        Tuple[torch.Tensor, List[Tuple[int, int]]]
            Tuple containing:
            - Processed tiles with weights applied, shape (B, num_classes, H, W)
            - Original coordinates for each tile
        """
        batch_tiles_norm = normalize(batch_tiles, self.mean, self.std)
        
        if self.apply_random_transformations:
            # randomly flip and rotate individual tiles
            is_hflip = torch.randint(0, 2, (batch_tiles.shape[0],)).bool()
            is_vflip = torch.randint(0, 2, (batch_tiles.shape[0],)).bool()
            rot_angle = torch.randint(0, 4, (batch_tiles.shape[0],))
            
            for i in range(batch_tiles.shape[0]):
                if is_hflip[i]:
                    batch_tiles_norm[i] = torch.flip(batch_tiles_norm[i], [2])
                if is_vflip[i]:
                    batch_tiles_norm[i] = torch.flip(batch_tiles_norm[i], [1])
                batch_tiles_norm[i] = torch.rot90(batch_tiles_norm[i], rot_angle[i], [1, 2])
            
        batch_tiles_norm = batch_tiles_norm.to(self.device)
        
        with torch.no_grad():
            probs = self.model(batch_tiles_norm).cpu()
            # weighted_probs = probs * self.weights_gpu
        
        if self.apply_random_transformations:
            # undo transformations
            for i in range(batch_tiles.shape[0]):
                probs[i] = torch.rot90(probs[i], -rot_angle[i], [1, 2])
                if is_vflip[i]:
                    probs[i] = torch.flip(probs[i], [1])
                if is_hflip[i]:
                    probs[i] = torch.flip(probs[i], [2])
            
        if self.seg_func:
            # Apply segmentation function if provided
            batch_tiles = batch_tiles.numpy()
            probs = probs.numpy()
            
            # segmented_images = segment_images(batch_tiles, self.seg_func, self.seg_func_params)
            # with mp.Pool(mp.cpu_count()) as pool:
            with ThreadPoolExecutor(max_workers=4*mp.cpu_count()) as pool:
                # segmented_images = pool.map(partial(segment_image, seg_func=self.seg_func, params=self.seg_func_params), batch_tiles)
                probs = pool.map(partial(segment_and_process, seg_func=self.seg_func, params=self.seg_func_params), batch_tiles, probs)
            
            # for i, segmented_image in enumerate(segmented_images):
                # probs[i] = postprocess_probs(segmented_image, probs[i])
            
            # for i, tile in enumerate(batch_tiles):
                # probs[i] = segment_and_process(tile, probs[i], self.seg_func, self.seg_func_params)
            # with mp.Pool(mp.cpu_count()) as pool:
                # probs = pool.starmap(segment_and_process, [(tile, probs[i], self.seg_func, self.seg_func_params) for i, tile in enumerate(batch_tiles)])
            # probs = torch.from_numpy(np.stack(probs))
        
        probs = torch.from_numpy(np.array(list(probs)))
        weighted_probs = probs * self.weights_cpu
        return weighted_probs, batch_coords

    def process_raster(self, raster_data: np.ndarray) -> np.ndarray:
        """Process a raster array and return probabilities for each class.
        
        Parameters
        ----------
        raster_data : np.ndarray
            Input raster of shape (channels, height, width)
            
        Returns
        -------
        np.ndarray
            Class probabilities of shape (num_classes, height, width)
        """
        if raster_data.dtype != np.float32:
            raster_data = raster_data.astype(np.float32) / 255.0
        
        # Normalize if mean and std provided
        # if self.mean is not None and self.std is not None:
        #     raster_data = np.transpose(raster_data, (1, 2, 0))
        #     raster_data = (raster_data - self.mean) / self.std
        #     raster_data = np.transpose(raster_data, (2, 0, 1))
        
        height, width = raster_data.shape[1:]
        num_classes = self.model.num_classes
        
        # Initialize outputs
        outputs = torch.zeros((num_classes, height, width))
        weights_sum = torch.zeros((height, width))
        
        # determine total number of batches
        total_batches = 0
        for _ in self.generate_tile_batches(raster_data):
            total_batches += 1
        
        # Process batches
        for batch_tiles, batch_coords in tqdm(self.generate_tile_batches(raster_data), total=total_batches, desc='Processing', unit='batches'):
            weighted_probs, coords = self.process_batch(batch_tiles, batch_coords)
            
            # Accumulate results
            for idx, (y, x) in enumerate(coords):
                outputs[:, y:y+self.tile_size, x:x+self.tile_size] += weighted_probs[idx]
                weights_sum[y:y+self.tile_size, x:x+self.tile_size] += self.weights_cpu
        
        # Get final probabilities
        return get_probabilites(outputs.numpy(), weights_sum.numpy())

def process_chunk_shared(args):
    (y_start, y_end, x_start, x_end), outputs_name, weights_name, final_name, output_shape, weights_shape = args
    
    # Attach to shared memory
    outputs_shm = shared_memory.SharedMemory(name=outputs_name)
    weights_shm = shared_memory.SharedMemory(name=weights_name)
    final_shm = shared_memory.SharedMemory(name=final_name)

    try:
        # Create numpy arrays from shared memory
        outputs = np.ndarray(output_shape, dtype=np.float32, buffer=outputs_shm.buf)
        weights = np.ndarray(weights_shape, dtype=np.float32, buffer=weights_shm.buf)
        final_outputs = np.ndarray(output_shape, dtype=np.float32, buffer=final_shm.buf)

        # Process chunk
        chunk_outputs = outputs[:, y_start:y_end, x_start:x_end]
        chunk_weights = weights[y_start:y_end, x_start:x_end]
        result = chunk_outputs / (chunk_weights + 1e-10)[np.newaxis, :, :]

        # Write result back to shared memory
        final_outputs[:, y_start:y_end, x_start:x_end] = result

    finally:
        # Close (but don't unlink) shared memory
        outputs_shm.close()
        weights_shm.close()
        final_shm.close()


def get_probabilites(outputs, weights_sum, chunk_size=16384):
    # Get final probabilities
    # split outputs and weights_sum into chunks to avoid memory issues
    # Define chunk size and create empty array for final outputs
    # chunk_size = 8192  # Adjust based on available memory
    output_shape = outputs.shape
    weights_sum_shape = weights_sum.shape

    # Create shared memory blocks
    outputs_shm = shared_memory.SharedMemory(create=True, size=outputs.nbytes)
    weights_sum_shm = shared_memory.SharedMemory(create=True, size=weights_sum.nbytes)
    final_outputs_shm = shared_memory.SharedMemory(create=True, size=outputs.nbytes)

    try:
        # Create numpy arrays using shared memory
        shared_outputs = np.ndarray(output_shape, dtype=outputs.dtype, buffer=outputs_shm.buf)
        shared_weights = np.ndarray(weights_sum_shape, dtype=weights_sum.dtype, buffer=weights_sum_shm.buf)
        shared_final = np.ndarray(output_shape, dtype=outputs.dtype, buffer=final_outputs_shm.buf)

        # Copy data to shared memory
        np.copyto(shared_outputs, outputs)
        np.copyto(shared_weights, weights_sum)

        # Process chunks in parallel
        chunk_args = []
        for y_start in range(0, output_shape[1], chunk_size):
            y_end = min(y_start + chunk_size, output_shape[1])
            for x_start in range(0, output_shape[2], chunk_size):
                x_end = min(x_start + chunk_size, output_shape[2])
                chunk_args.append((
                    (y_start, y_end, x_start, x_end),
                    outputs_shm.name,
                    weights_sum_shm.name,
                    final_outputs_shm.name,
                    output_shape,
                    weights_sum_shape
                ))

        with Pool(cpu_count()) as pool:
            pool.map(process_chunk_shared, chunk_args)

        # Copy result back from shared memory
        result = np.copy(shared_final)

    finally:
        # Clean up shared memory
        outputs_shm.close()
        outputs_shm.unlink()
        weights_sum_shm.close()
        weights_sum_shm.unlink()
        final_outputs_shm.close()
        final_outputs_shm.unlink()
    
    return result



def create_shared_array(data):
    """Create a shared memory array from input data."""
    shm = shared_memory.SharedMemory(create=True, size=data.nbytes)
    shared_array = np.ndarray(data.shape, dtype=data.dtype, buffer=shm.buf)
    shared_array[:] = data[:]
    return shm, shared_array

def get_chunk_bounds(chunk_idx, chunk_size, max_size, overlap):
    """
    Calculate chunk boundaries with overlap, handling edge cases.
    Returns start and end indices, and the actual chunk start and end without overlap.
    """
    chunk_start = chunk_idx * chunk_size
    chunk_end = min(chunk_start + chunk_size, max_size)
    
    # Add overlap
    start_with_overlap = max(0, chunk_start - overlap)
    end_with_overlap = min(max_size, chunk_end + overlap)
    
    return (start_with_overlap, end_with_overlap, chunk_start, chunk_end)

def process_chunk(args):
    """Process a single chunk of the image using quickshift."""
    y_bounds, x_bounds, shape, dtype, shm_name, kernel_size, max_dist, ratio = args
    y_start, y_end, y_chunk_start, y_chunk_end = y_bounds
    x_start, x_end, x_chunk_start, x_chunk_end = x_bounds
    
    # Attach to shared memory
    existing_shm = shared_memory.SharedMemory(name=shm_name)
    data = np.ndarray(shape, dtype=dtype, buffer=existing_shm.buf)
    
    # Extract chunk with overlap
    chunk = data[:, y_start:y_end, x_start:x_end]  # (C, chunk_H, chunk_W)
    chunk = chunk.transpose(1, 2, 0)  # Convert to (chunk_H, chunk_W, C) for quickshift
    
    # Process chunk
    result = quickshift(chunk, kernel_size=kernel_size, 
                       max_dist=max_dist, ratio=ratio)
    
    # Remove overlap regions from result
    result = result[y_chunk_start-y_start:y_chunk_end-y_start,
                   x_chunk_start-x_start:x_chunk_end-x_start]
    
    # Adjust segment labels to avoid conflicts between chunks
    result = result.astype(np.int32) + (y_chunk_start * shape[2] + x_chunk_start)
    
    # Clean up
    existing_shm.close()
    
    return result, (y_chunk_start, y_chunk_end, x_chunk_start, x_chunk_end)

def parallel_quickshift(raster_data, kernel_size=3, max_dist=6, ratio=0.5, chunk_size=1000, n_processes=None):
    """
    Parallel implementation of quickshift segmentation using shared memory and square chunks.
    
    Parameters:
    -----------
    raster_data : numpy.ndarray
        Input raster data with shape (channels, height, width)
    kernel_size : int
        Size of kernel for quickshift
    max_dist : float
        Maximum distance between pixels for quickshift
    ratio : float
        Ratio parameter for quickshift
    chunk_size : int, optional
        Size of square chunks. If None, calculated based on image size and n_processes
    n_processes : int, optional
        Number of processes to use. Defaults to CPU count
        
    Returns:
    --------
    numpy.ndarray
        Segmented image with unique labels
    """
    if n_processes is None:
        n_processes = mp.cpu_count()
    
    height, width = raster_data.shape[1:3]
    
    # Calculate chunk size if not provided
    if chunk_size is None:
        # Aim for roughly n_processes total chunks
        chunk_size = int(np.sqrt((height * width) / n_processes))
    
    # Calculate overlap based on kernel_size and max_dist
    overlap = max(kernel_size, int(max_dist))
    
    # Create shared memory for input data
    shm, shared_array = create_shared_array(raster_data)
    
    # Calculate number of chunks in each dimension
    n_chunks_y = math.ceil(height / chunk_size)
    n_chunks_x = math.ceil(width / chunk_size)
    
    # Prepare arguments for each chunk
    args = []
    for i in range(n_chunks_y):
        y_bounds = get_chunk_bounds(i, chunk_size, height, overlap)
        for j in range(n_chunks_x):
            x_bounds = get_chunk_bounds(j, chunk_size, width, overlap)
            args.append((y_bounds, x_bounds, raster_data.shape, raster_data.dtype,
                        shm.name, kernel_size, max_dist, ratio))
    
    # Process chunks in parallel
    with mp.Pool(n_processes) as pool:
        results = pool.map(process_chunk, args)
    
    # Clean up shared memory
    shm.close()
    shm.unlink()
    
    # Initialize final segments array
    final_segments = np.zeros((height, width), dtype=np.int32)
    
    # Combine results
    for result, (y_start, y_end, x_start, x_end) in results:
        final_segments[y_start:y_end, x_start:x_end] = result
    
    # Relabel segments to ensure continuity
    unique_labels = np.unique(final_segments)
    label_map = {old: new for new, old in enumerate(unique_labels)}
    vectorized_map = np.vectorize(lambda x: label_map[x])
    final_segments = vectorized_map(final_segments)
    
    return final_segments


@njit(parallel=True)
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
    
    # img = img.transpose(1, 2, 0)  # Convert to (H, W, C)
    # params = params.copy()
    median_radius = params.pop('median_radius')
    if median_radius > 0:
        for i in range(3):
            img[:, :, i] = median(img[:, :, i], disk(median_radius))
    
    unsharp_radius = params.pop('unsharp_radius')
    unsharp_amount = params.pop('unsharp_amount')
    if unsharp_radius > 0 and unsharp_amount > 0:
        img = unsharp_mask(img, radius=unsharp_radius, amount=unsharp_amount)
    
    return seg_func(img, **params)


@njit(parallel=True)
def segment_images(imgs, seg_func, params):
    segmented_images = np.empty_like(imgs[:, 0, :, :])
    for i in prange(imgs.shape[0]):
        segmented_images[i] = segment_image(imgs[i], seg_func, params.copy())
    return segmented_images



def segment_and_process(
    img: np.ndarray,
    probs: np.ndarray,
    seg_func: Callable[..., np.ndarray],
    params: Dict[str, Union[int, float]],
):
    segmented_image = segment_image(img.copy().transpose(1, 2, 0), seg_func, params.copy())
    return postprocess_probs(segmented_image, probs)
