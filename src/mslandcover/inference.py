import numpy as np
import torch
import torch.nn as nn
from numpy.lib.stride_tricks import sliding_window_view
from tqdm import tqdm
from typing import List, Tuple, Optional
import logging
import sys
from datetime import datetime
from zoneinfo import ZoneInfo



class RasterProcessor:
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
    ):
        self.tile_size = tile_size
        self.stride = stride
        self.sigma = gaussian_sigma
        self.batch_size = batch_size
        self.device = device
        
        # Move model to device and set to eval mode
        self.model = model.to(device)
        self.model.eval()

        self.mean = mean if mean is not None else None
        self.std = std if std is not None else None

        # Create weight matrix and move to GPU
        x = np.linspace(-tile_size/2, tile_size/2, tile_size)
        y = np.linspace(-tile_size/2, tile_size/2, tile_size)
        X, Y = np.meshgrid(x, y)
        weights = np.exp(-(X**2 + Y**2)/(2*self.sigma**2))
        weights /= weights.max()
        weights = weights.astype(np.float32)
        self.weights = torch.from_numpy(weights)

    def generate_tile_batches(self, raster_data: np.ndarray, nodata: int=np.nan):
        height, width = raster_data.shape[1:]
        batch_tiles = []
        batch_coords = []

        for y in range(0, height, self.stride):
            for x in range(0, width, self.stride):
                tile = raster_data[:, y:y+self.tile_size, x:x+self.tile_size]

                if (tile == nodata).all() or 0 in tile.shape:
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
        if self.mean is not None and self.std is not None:
            batch_tiles = torch.subtract(batch_tiles, self.mean[None, :, None, None])
            batch_tiles = torch.divide(batch_tiles, self.std[None, :, None, None])
        
        batch_tiles = batch_tiles.to(self.device)
        
        with torch.no_grad():
            probs = self.model(batch_tiles).cpu()

        probs = torch.from_numpy(np.array(list(probs)))
        weighted_probs = probs * self.weights
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
            raster_data = raster_data.astype(np.float32)
        
        height, width = raster_data.shape[1:]
        num_classes = self.model.num_classes
        
        # Initialize outputs
        outputs = torch.zeros((num_classes, height, width))
        # weights_sum = torch.zeros((height, width))
        
        # determine total number of batches
        # total_batches = 0
        # for _ in self.generate_tile_batches(raster_data):
        #     total_batches += 1
        
        # Process batches
        # for batch_tiles, batch_coords in tqdm(self.generate_tile_batches(raster_data), total=total_batches, desc='Processing', unit='batches', disable=True):
        for batch_tiles, batch_coords in self.generate_tile_batches(raster_data):
            weighted_probs, coords = self.process_batch(batch_tiles, batch_coords)

            # Accumulate results
            for idx, (y, x) in enumerate(coords):
                outputs[:, y:y+self.tile_size, x:x+self.tile_size] += weighted_probs[idx]
                # weights_sum[y:y+self.tile_size, x:x+self.tile_size] += self.weights

        # Get final probabilities
        # return outputs / weights_sum.unsqueeze(0)
        # return torch.argmax(outputs, dim=0)  # Return the class with the highest probability
        outputs = outputs / torch.clamp(outputs.sum(dim=0, keepdim=True), min=1e-8)  # Normalize to get probabilities



class CSTFormatter(logging.Formatter):
    def formatTime(self, record, datefmt=None):
        # Convert to CST (UTC-6 / UTC-5 for DST)
        # Assumes system time is UTC
        cst_time = datetime.fromtimestamp(record.created, tz=ZoneInfo("America/Chicago"))
        return cst_time.strftime('%Y-%m-%d %H:%M:%S %Z')

    def format(self, record):
        timestamp = self.formatTime(record)
        message = super().format(record)
        return f"[ {timestamp} ] {message}"



def create_stdout_logger(name: str = "stdout_logger", level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = CSTFormatter('%(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.propagate = False  # Prevents duplicate logs if root logger has handlers

    return logger



def mode_filter(raster: np.ndarray, size: int=3, n_classes: int=9, nodata: int=None):
    """
    Apply a mode filter to a categorical raster using histogram-based counting.
    The output will have the same shape as the input, with nodata padding for the edges.

    Parameters:
    - raster: 2D numpy array of integers (HxW), categorical raster.
    - size: odd integer, window size.
    - n_classes: int, max number of categorical classes (exclusive upper bound).
    - nodata: value to ignore during mode computation, optional.

    Returns:
    - 2D numpy array (same shape as the input raster).
    """
    pad = size // 2  # Calculate the padding for edges
    padded_raster = np.pad(raster, pad_width=pad, constant_values=nodata)  # Pad with nodata

    # Sliding window view
    windows = sliding_window_view(padded_raster, (size, size))
    H, W = windows.shape[:2]
    windows = windows.reshape(H, W, -1)

    # Prepare the output raster
    output = np.full((H, W), nodata, dtype=np.uint8)  # Initialize with nodata

    # Count the frequency of each class in the window
    for c in range(n_classes):
        count_c = (windows == c).sum(axis=-1)
        if c == 0:
            counts = count_c[..., None]
        else:
            counts = np.concatenate([counts, count_c[..., None]], axis=-1)

    # Get the mode (most frequent value)
    output = counts.argmax(axis=-1)
    
    return output.astype(np.uint8)