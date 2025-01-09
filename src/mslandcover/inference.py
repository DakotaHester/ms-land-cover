import numpy as np
import multiprocessing as mp
from multiprocessing.pool import Pool
import rasterio
from rasterio.mask import mask
from rasterio.features import geometry_mask
import torch
import torch.nn as nn
import torch.nn.functional as F
from shapely import Polygon
from tqdm import tqdm
from src.mslandcover.utils import Logger, get_torch_device
from typing import Dict, List, Tuple, Optional, Union
from queue import Queue
from threading import Thread

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
            probs = F.softmax(model(tile.unsqueeze(0)), dim=1).cpu()
        
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
    """A class for processing large rasters using GPU-accelerated deep learning models.
    
    This class handles the efficient processing of large raster images by:
    1. Dividing the raster into overlapping tiles
    2. Processing tiles in batches on GPU
    3. Merging results using Gaussian weights
    4. Optionally masking and clipping results
    
    Parameters
    ----------
    model : torch.nn.Module
        The PyTorch model to use for processing. Must have a `num_classes` attribute.
    input_raster_path : str
        Path to the input raster file.
    output_path : str
        Path where the processed raster will be saved.
    bounding_polygons : List[Polygon], optional
        List of Shapely polygons for masking/clipping the output, by default None
    tile_size : int, optional
        Size of the square tiles to process, by default 256
    stride : int, optional
        Stride between consecutive tiles, by default 64
    gaussian_sigma : float, optional
        Standard deviation for Gaussian weighting, by default 128
    batch_size : int, optional
        Number of tiles to process simultaneously, by default 32
    mean : torch.Tensor, optional
        Mean values for input normalization, by default None
    std : torch.Tensor, optional
        Standard deviation values for input normalization, by default None
    logger : object, optional
        Logger object with a log method, by default None
    device : torch.device, optional
        Device to use for processing, by default cuda if available
    colormap : Dict[int, Tuple[int, int, int, int]], optional
        Colormap to use for saving the output raster, by default None
    
    Attributes
    ----------
    model : torch.nn.Module
        The PyTorch model used for processing
    input_path : str
        Path to the input raster
    output_path : str
        Path where results will be saved
    bounding_polygon : List[Polygon] or None
        List of Polygons used for masking/clipping
    tile_size : int
        Size of processing tiles
    stride : int
        Stride between tiles
    sigma : float
        Gaussian sigma for weight calculation
    batch_size : int
        Size of processing batches
    mean : torch.Tensor or None
        Normalization mean values
    std : torch.Tensor or None
        Normalization standard deviation values
    logger : object or None
        Logger instance
    device : torch.device
        Processing device
    colormap : Dict[int, Tuple[int, int, int, int]] or None
        Colormap for saving the output raster
    profile : dict
        Raster metadata profile
    raster_data : numpy.ndarray
        Input raster data
    weights : torch.Tensor
        Gaussian weights for tile merging
    """
    
    def __init__(
        self, 
        model: nn.Module,
        input_raster_path: str,
        output_path: str,
        bounding_polygons: Optional[List[Polygon]] = None,
        tile_size: int = 256,
        stride: int = 64,
        gaussian_sigma: float = 128,
        batch_size: int = 32,
        mean: Optional[torch.Tensor] = None,
        std: Optional[torch.Tensor] = None,
        logger: Optional[object] = None,
        device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        colormap: Optional[Dict[int, Tuple[int, int, int, int]]] = None,
        prefetch_factor: int = 2,
    ):
        self.input_path = input_raster_path
        self.output_path = output_path
        self.bounding_polygons = bounding_polygons
        self.tile_size = tile_size
        self.stride = stride
        self.sigma = gaussian_sigma
        self.batch_size = batch_size
        self.device = device
        self.logger = logger
        self.colormap = colormap
        self.prefetch_factor = prefetch_factor
        
        # Move model to device and set to eval mode
        self.model = model.to(device)
        self.model.eval()

        self.mean = mean.cpu().numpy() if mean is not None else None
        self.std = std.cpu().numpy() if std is not None else None

        if logger:
            logger.log(f'Using device: {device}')
            logger.log('Loading data and metadata...')
            
        # Load data and metadata
        with rasterio.open(input_raster_path) as src:
            self.profile = src.profile.copy()
            self.raster_data = (src.read() / 255.0).astype(np.float32)
        
        if self.bounding_polygons:
            
            masking_polygons = [p.buffer(tile_size*(2**0.5)) for p in self.bounding_polygons]
            
            if logger:
                logger.log('Masking raster by bounding polygons...')
            geom_mask = geometry_mask(
                masking_polygons, 
                out_shape=self.raster_data.shape[1:], 
                transform=self.profile['transform'], 
                all_touched=True,
            )
            self.raster_data[:, geom_mask] = 0
        
        # pad by tile size to handle edge cases
        self.raster_data = np.pad(self.raster_data, ((0,0), (self.tile_size, self.tile_size), (self.tile_size,self.tile_size)), mode='reflect')
        
        if self.mean is not None and self.std is not None:
            self.raster_data = np.transpose(self.raster_data, (1, 2, 0))
            self.raster_data = (self.raster_data - self.mean) / self.std
            self.raster_data = np.transpose(self.raster_data, (2, 0, 1))
            
        if logger:
            logger.log('Data and metadata loaded.')
        
        # Create weight matrix and move to GPU
        x = np.linspace(-tile_size/2, tile_size/2, tile_size)
        y = np.linspace(-tile_size/2, tile_size/2, tile_size)
        X, Y = np.meshgrid(x, y)
        weights = np.exp(-(X**2 + Y**2)/(2*self.sigma**2))
        weights /= weights.max()
        self.weights = torch.from_numpy(weights.astype(np.float32)).to(device)
        
        if logger:
            logger.log('Weight matrix created and moved to GPU.')

    def preload_batches(self, queue: Queue):
        height, width = self.raster_data.shape[1:]
        batch_tiles = []
        batch_coords = []

        for y in range(0, height, self.stride):
            for x in range(0, width, self.stride):
                tile = self.raster_data[:, y:y+self.tile_size, x:x+self.tile_size]

                if (tile == 0).all() or 0 in tile.shape:
                    continue

                if tile.shape[1:] != (self.tile_size, self.tile_size):
                    continue

                batch_tiles.append(tile)
                batch_coords.append((y, x))

                if len(batch_tiles) == self.batch_size:
                    queue.put((torch.from_numpy(np.stack(batch_tiles)), batch_coords))
                    batch_tiles = []
                    batch_coords = []

        if batch_tiles:
            queue.put((torch.from_numpy(np.stack(batch_tiles)), batch_coords))
        queue.put(None)  # Signal that loading is done

    def generate_tile_batches(self):
        """Generate batches of tiles with their corresponding coordinates.
        
        Yields
        -------
        Tuple[torch.Tensor, List[Tuple[int, int]]]
            Tuple containing:
            - Batch of tiles as torch.Tensor of shape (B, C, H, W)
            - List of (y, x) coordinates for each tile in the batch
        """

        queue = Queue(maxsize=self.prefetch_factor)
        loader_thread = Thread(target=self.preload_batches, args=(queue,))
        loader_thread.start()

        while True:
            batch = queue.get()
            if batch is None:
                break
            yield batch

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
        batch_tiles = batch_tiles.to(self.device)
        
        with torch.no_grad():
            probs = F.softmax(self.model(batch_tiles), dim=1)
            weighted_probs = probs * self.weights.unsqueeze(0).unsqueeze(0)
            
        return weighted_probs, batch_coords

    def process_raster(self) -> None:
        """Process the entire raster using GPU-accelerated batch processing.
        
        This method:
        1. Divides the raster into overlapping tiles
        2. Processes tiles in batches using the model
        3. Merges results using Gaussian weights
        4. Optionally applies masking/clipping
        5. Saves the results to disk
        """
        height, width = self.raster_data.shape[1:]
        num_classes = self.model.num_classes
        
        # Initialize outputs
        outputs = torch.zeros((num_classes, height, width), device=self.device)
        weights_sum = torch.zeros((height, width), device=self.device)
        
        # calculate total number of batches based on height, width, and stride
        self.logger.log('Calculating total number of batches...')
        total_batches = 0
        for y in range(0, height, self.stride):
            for x in range(0, width, self.stride):
                tile = self.raster_data[:, y:y+self.tile_size, x:x+self.tile_size]
                if (tile == 0).all() or tile.shape[1:] != (self.tile_size, self.tile_size):
                    continue
                total_batches += 1
        total_batches = (total_batches // self.batch_size) + 1
        
        if self.logger:
            self.logger.log(f'Processing {total_batches} batches...')
        
        # Process batches
        for batch_tiles, batch_coords in tqdm(self.generate_tile_batches(), desc='Processing batches', unit='batch', total=total_batches):
            weighted_probs, coords = self.process_batch(batch_tiles, batch_coords)
            
            # Accumulate results
            for idx, (y, x) in enumerate(coords):
                outputs[:, y:y+self.tile_size, x:x+self.tile_size] += weighted_probs[idx]
                weights_sum[y:y+self.tile_size, x:x+self.tile_size] += self.weights
        
        if self.logger:
            self.logger.log('Finalizing predictions...')
        
        # Get final predictions
        final_outputs = outputs / (weights_sum + 1e-10)
        predictions = torch.argmax(final_outputs, dim=0).cpu().numpy().astype(np.uint8) + 1
        
        # remove padding from the predictions
        predictions = predictions[self.tile_size:-self.tile_size, self.tile_size:-self.tile_size]
        
        if self.logger:
            self.logger.log('Saving results...')
        
        # Save results
        profile = self.profile.copy()
        profile.update(count=1, dtype=rasterio.uint8)
        
        if self.bounding_polygons:
            if self.logger:
                self.logger.log('Masking predictions by bounding polygons...')
            geom_mask = geometry_mask(
                self.bounding_polygons, 
                out_shape=predictions.shape,
                transform=profile['transform'],
                all_touched=True
            )
            predictions[geom_mask] = 0
        
        if self.logger:
            self.logger.log(f'Writing results to {self.output_path}...')
        with rasterio.open(self.output_path, 'w', **profile) as dst:
            dst.write(predictions, 1)
            if self.colormap:
                dst.write_colormap(1, self.colormap)
        
        if self.bounding_polygons:
            if self.logger:
                self.logger.log('Clipping raster...')
            with rasterio.open(self.output_path) as src:
                clipped_data, clipped_transform = mask(src, self.bounding_polygons, crop=True)
                profile = src.profile.copy()
            
            profile.update({
                'transform': clipped_transform,
                'height': clipped_data.shape[1],
                'width': clipped_data.shape[2]
            })
            
            if self.logger:
                self.logger.log(f'Writing clipped results to {self.output_path}...')
            with rasterio.open(self.output_path, 'w', **profile) as dst:
                dst.write(clipped_data)
                if self.colormap:
                    dst.write_colormap(1, self.colormap)