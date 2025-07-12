from argparse import ArgumentParser
import os
import tempfile
import numpy as np
from shapely import Polygon, unary_union
import torch
import torch.nn as nn
from numpy.lib.stride_tricks import sliding_window_view
from typing import List, Tuple, Optional
from tqdm import tqdm
import xarray as xr
import rioxarray as rxr
from scipy.interpolate import interp1d
import rasterio as rio
import geopandas as gpd
from subprocess import Popen

from mslandcover.config import LEGEND_COLORS_RGBA
from mslandcover.data.utils import load_histogram_data
from mslandcover.models import ResNetBackboneUNet, UNet
from mslandcover.utils import get_torch_device, load_pth


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
        tta: bool=False,
        device: torch.device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        enable_pbar: bool=False,
    ):
        self.tile_size = tile_size
        self.stride = stride
        self.sigma = gaussian_sigma
        self.batch_size = batch_size
        self.device = device
        self.tta = tta  # Test Time Augmentation flag
        self.enable_pbar = enable_pbar
        
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
        """Process a batch of tiles on GPU with optional random TTA and reversal."""
        if self.mean is not None and self.std is not None:
            batch_tiles_norm = torch.subtract(batch_tiles, self.mean[None, :, None, None])
            batch_tiles_norm = torch.divide(batch_tiles_norm, self.std[None, :, None, None])
        else:
            batch_tiles_norm = batch_tiles.clone()

        if self.tta:
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
        batch_tiles = batch_tiles.to(self.device)

        with torch.no_grad():
            probs = self.model(batch_tiles_norm).cpu()

        if self.tta:
            # undo transformations
            for i in range(batch_tiles.shape[0]):
                probs[i] = torch.rot90(probs[i], -int(rot_angle[i]), [1, 2])
                if is_vflip[i]:
                    probs[i] = torch.flip(probs[i], [1])
                if is_hflip[i]:
                    probs[i] = torch.flip(probs[i], [2])

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
        
        
        # estimate total batches based on raster size, tile size, and stride
        total_batches = ((height - self.tile_size) // self.stride + 1)
        total_batches *= ((width - self.tile_size) // self.stride + 1)
        total_batches = (total_batches / self.batch_size) + 1
        
        # Process batches
        for batch_tiles, batch_coords in tqdm(self.generate_tile_batches(raster_data), total=total_batches, desc='Processing', unit='batches', disable=not self.enable_pbar):
            weighted_probs, coords = self.process_batch(batch_tiles, batch_coords)

            # Accumulate results
            for idx, (y, x) in enumerate(coords):
                outputs[:, y:y+self.tile_size, x:x+self.tile_size] += weighted_probs[idx]
                # weights_sum[y:y+self.tile_size, x:x+self.tile_size] += self.weights

        # Get final probabilities
        # return outputs / weights_sum.unsqueeze(0)
        # return torch.argmax(outputs, dim=0)  # Return the class with the highest probability
        outputs = outputs / torch.clamp(outputs.sum(dim=0, keepdim=True), min=1e-8)  # Normalize to get probabilities
        return outputs.numpy()


def histogram_match_image(image_data, source_histograms, target_histograms, output_path=None, 
                            bins=255, value_range=(1, 255), chunks=None):
    """
    Apply histogram matching to an image using precomputed source and target histograms.
    
    Parameters
    ----------
    image_data : xarray.DataArray
        Input image data
    source_histograms : np.ndarray
        Source histograms for each band, shape (n_bands, n_bins)
    target_histograms : np.ndarray  
        Target histograms for each band, shape (n_bands, n_bins)
    output_path : str, optional
        Path to save output. If None, returns xarray.DataArray
    bins : int, default 255
        Number of histogram bins
    value_range : tuple, default (1, 255)
        Range of values for histogram bins
    chunks : dict, optional
        Chunk sizes for dask arrays, e.g. {'x': 2048, 'y': 2048}
        
    Returns
    -------
    xarray.DataArray or str
        Matched image data or output path if saved
    """
    
    # Default chunking
    if chunks is None:
        chunks = {'x': 2048, 'y': 2048}
    
    # Convert histograms to numpy arrays if needed
    if isinstance(source_histograms, list):
        source_histograms = np.array(source_histograms)
    if isinstance(target_histograms, list):
        target_histograms = np.array(target_histograms)
    
    # Create bin edges
    bin_edges = np.linspace(value_range[0], value_range[1], bins)
    
    def create_lut_for_band(source_hist, target_hist):
        """Create lookup table for a single band using provided histograms"""
        # Convert to CDFs
        source_cdf = source_hist.cumsum() / source_hist.sum()
        target_cdf = target_hist.cumsum() / target_hist.sum()
        
        # Handle edge cases
        source_cdf = np.clip(source_cdf, 1e-10, 1.0)
        target_cdf = np.clip(target_cdf, 1e-10, 1.0)
        
        # Create interpolation function
        interp_func = interp1d(target_cdf, bin_edges, bounds_error=False, 
                                fill_value=(bin_edges[0], bin_edges[-1]))
        
        # Create lookup table
        lut_values = interp_func(source_cdf)
        
        return bin_edges, lut_values
    
    def apply_lut(chunk, bin_edges, lut_values):
        """Apply lookup table to a chunk"""
        indices = np.searchsorted(bin_edges, chunk, side='right') - 1
        indices = np.clip(indices, 0, len(lut_values) - 1)
        result = lut_values[indices]
        # Ensure the output has the same shape as input
        return result.astype(chunk.dtype)
    
    # Process each band
    matched_bands = []
    for band_idx in range(image_data.sizes['band']):
        band = image_data.isel(band=band_idx)
        
        # Create LUT for this band using provided histograms
        bin_edges_band, lut_values = create_lut_for_band(
            source_histograms[band_idx], 
            target_histograms[band_idx]
        )
        
        # Apply LUT using xr.apply_ufunc as an alternative to map_blocks
        matched_band = xr.apply_ufunc(
            lambda chunk: apply_lut(chunk, bin_edges_band, lut_values),
            band,
            dask='allowed',
            output_dtypes=[band.dtype]
        )
        # Convert to uint8 while preserving DataArray structure
        matched_band = matched_band.astype(np.uint8)
        matched_bands.append(matched_band)
    
    # Combine bands
    matched_image = xr.concat(matched_bands, dim='band')
    matched_image = matched_image.rio.write_crs(image_data.rio.crs)
    matched_image = matched_image.rio.write_transform(image_data.rio.transform())
    matched_image = matched_image.rio.write_nodata(image_data.rio.nodata)
    
    # Save or return
    if output_path:
        matched_image.rio.to_raster(
            output_path,
            compress='LZW',
            bigtiff=True,
            tiled=True,
            blockxsize=512,
            blockysize=512
        )
        return output_path
    else:
        return matched_image
    


def process_single_raster(path, args, gdf: Optional[gpd.GeoDataFrame]=None):
    """Process a single raster file with proper CUDA handling"""
    filename = os.path.basename(path)
    out_path = os.path.join(args.output_dir, filename)
    
    # Skip if output already exists (unless overwrite is specified)
    if os.path.exists(out_path) and not args.overwrite:
        if args.skip_existing:
            return f"Skipped {filename} (already exists)"
    
    try:
        # Clear any existing CUDA context in this process
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Load model for this process
        weights = load_pth(args.weights_path, map_location='cpu')
        model = UNet(
            backbone=ResNetBackboneUNet(in_channels=args.in_channels, pretrained=False),
            num_classes=args.num_classes,
        )
        model.load_state_dict(weights, strict=True)
        
        # Load normalization parameters
        mean = load_pth(args.mean_path)
        std = load_pth(args.std_path)
        
        # Get device for this worker process
        device = get_torch_device()
        
        processor = RasterProcessor(
            model=model,
            tile_size=args.tile_size,
            stride=args.stride,
            gaussian_sigma=args.gaussian_sigma,
            batch_size=args.batch_size,
            mean=mean,
            std=std,
            tta=args.tta,
            device=device,
            enable_pbar=args.enable_pbar,
        )
        
        in_data = load_raster_for_processing(path, args, gdf)
        
        # # Process the raster
        # in_data = rxr.open_rasterio(path)
        
        # if args.match_histograms:
        #     source_histograms = load_histogram_data(state='MS', year=2016)
        #     target_histograms = load_histogram_data(state='MS', year=2023)
            
        #     in_data = histogram_match_image(
        #         in_data,
        #         source_histograms=source_histograms,
        #         target_histograms=target_histograms,
        #         output_path=None,  # Return as xarray.DataArray
        #         bins=255,
        #         value_range=(1, 255),
        #         # chunks={'x': args.tile_size, 'y': args.tile_size}
        #     )
        
        # in_data = in_data.sel(band=args.bands)
        
        probs = processor.process_raster(in_data.values / args.scale_factor)
        
        lc_class = (np.argmax(probs, axis=0) + 1).astype(np.uint8)  # Convert to class indices (1-indexed)
        # lc_prob = np.clip(np.round(probs.max(axis=0) * 100), 1, 100).astype(np.uint8)  # Convert to percentage
        
        # Apply nodata mask before mode filter
        nodata_mask = (in_data == in_data.rio.nodata).all(dim='band').values
        lc_class[nodata_mask] = 0
        # lc_prob[nodata_mask] = 0
        
        # Apply mode filter only to land cover class
        if args.mode_filter_size > 0:
            lc_class = mode_filter(lc_class, size=args.mode_filter_size, nodata=0)
        
        # Stack class and confidence
        # out = np.stack([lc_class, lc_prob], axis=0)
        out = lc_class
        
        # Create output DataArray
        out_da = xr.DataArray(
            out,
            dims=['y', 'x'],
            coords={
                # 'band': ['land_cover_class'],
                'y': in_data.y,
                'x': in_data.x,
            },
            attrs={
                'long_name': 'Land Cover Classification',
                'units': '1',
                'crs': in_data.rio.crs.to_proj4(),
            },
        )
        out_da = out_da.rio.write_nodata(0)
        
        # Save to file
        os.makedirs(args.output_dir, exist_ok=True)
        out_da.rio.to_raster(
            out_path,
            driver='GTiff',
            dtype='uint8',
            compress=args.compress,
            blockxsize=args.block_size,
            blockysize=args.block_size,
            tiled=True,
            bigTiff=True,
        )
        
        # Add colormap to the first band (land cover class)
        with rio.open(out_path, 'r+') as dst:
            dst.write_colormap(1, LEGEND_COLORS_RGBA)
        
        # Clean up CUDA memory for this process
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        return f"Processed {filename}"
        
    except Exception as e:
        # return f"Error processing {filename}: {str(e)}"
        raise e



def load_raster_for_processing(path: str, args: ArgumentParser, gdf: Optional[gpd.GeoDataFrame]=None):
    
    if path.endswith('.sid'):
        
        if args.match_histograms:
            raise ValueError("Histogram matching is not supported for .sid files.")
        
        # first - convret mrsid to tiff
        with tempfile.NamedTemporaryFile(suffix='.tif', delete=True) as tmp_file:
            
            env = os.environ.copy()
            env['PATH'] = '/home/dhester/.local/bin:' + env.get('PATH', '')
            env['LD_LIBRARY_PATH'] = '/home/dhester/etc/mrsid_sdk/Raster_DSDK/bin:/home/dhester/etc/mrsid_sdk/Raster_DSDK/lib' + env.get('LD_LIBRARY_PATH', '')
            Popen([
                'mrsiddecode', 
                '-i', path, 
                '-o', tmp_file.name,
                '-of', 'tifg', 
                '-quiet',
            ], env=env).wait()
            
            # check resolution - if not 1m, resample to 1m
            with rio.open(tmp_file.name) as src:
                resolution = src.res
            
            if resolution != (1.0, 1.0):
                
                with tempfile.NamedTemporaryFile(suffix='.tif', delete=True) as tmp_resampled_file:
                    Popen([
                        'gdalwarp',
                        '-q',
                        '-tr', '1.0', '1.0',
                        '-r', 'bilinear',
                        # '-t_srs', crs,
                        # input_path,
                        tmp_file.name,
                        # output_path,
                        tmp_resampled_file.name,
                        '-co', 'COMPRESS=LZW',
                        '-co', 'TILED=YES',
                        '-co', 'BIGTIFF=YES',
                        '-co', 'BLOCKXSIZE=256',
                        '-co', 'BLOCKYSIZE=256',
                        '-wo', 'NUM_THREADS=ALL_CPUS',
                        '-wm', '4096', # 4 GiB
                    ]).wait()
                                        
                    in_data = rxr.open_rasterio(tmp_resampled_file.name)
            
            else:
                in_data = rxr.open_rasterio(tmp_file.name)
                
            # if gdf is not None:
            #     in_data = rxr.open_rasterio(tmp_file.name).rio.clip_box(**gdf.buffer(args.tile_size).bounds, crs=gdf.crs)
            #     in_data = in_data.rio.clip(gdf.buffer(args.tile_size), crs=gdf.crs, all_touched=True)
            # else:
            #     in_data = rxr.open_rasterio(tmp_file.name)
        
        # NOTE: Only here due to issue with ban ordering - remove later
        if args.bands == [1, 2, 4]:
            in_data = in_data.sel(band=[2, 3, 1])
        
        # # check and resample to 1m 
        # if in_data.rio.resolution() != (1.0, 1.0):
        #     in_data = in_data.rio.reproject(
        #         in_data.rio.crs,
        #         resolution=(1.0, 1.0),
        #         resampling=rio.enums.Resampling.bilinear,
        #         num_threads=8,
        #     )
    
    elif path.endswith('.tif'):
        
        if gdf is not None:
            raise NotImplementedError("Clipping with GeoDataFrame is not implemented for .tif files just yet.")
            in_data = rxr.open_rasterio(path).rio.clip_box(*unary_union(buffered_geoms).bounds)
            in_data = in_data.rio.clip(geoms=buffered_geoms, all_touched=True)
        else:
            in_data = rxr.open_rasterio(path)
        
        if args.match_histograms:
            source_histograms = load_histogram_data(state='MS', year=2016)
            target_histograms = load_histogram_data(state='MS', year=2023)
            
            in_data = histogram_match_image(
                in_data,
                source_histograms=source_histograms,
                target_histograms=target_histograms,
                output_path=None,  # Return as xarray.DataArray
                bins=255,
                value_range=(1, 255),
                # chunks={'x': args.tile_size, 'y': args.tile_size}
            )
            
        in_data = in_data.sel(band=args.bands)
    
    return in_data



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