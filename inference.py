import os
import numpy as np
import xarray as xr
import rioxarray as rxr
import rasterio as rio
from mslandcover.models import UNet, ResNetBackboneUNet
from mslandcover.inference import RasterProcessor, mode_filter
from mslandcover.config import LEGEND_COLORS_RGBA
from mslandcover.utils import load_pth, get_torch_device
from tqdm import tqdm
import torch
from glob import glob
import multiprocessing as mp
from functools import partial
from argparse import ArgumentParser
import torch

# Set multiprocessing start method to 'spawn' for CUDA compatibility
mp.set_start_method('spawn', force=True)
# torch.set_num_threads(1)


# def process_single_raster(path, args):
#     """Process a single raster file"""
#     filename = os.path.basename(path)
#     out_path = os.path.join(args.output_dir, filename)
    
#     # Skip if output already exists
#     if os.path.exists(out_path):
#         return f"Skipped {filename} (already exists)"
    
#     try:
#         # Load model for this process
#         weights = load_pth(args.weights_path, map_location='cpu')
#         model = UNet(
#             backbone=ResNetBackboneUNet(in_channels=3, pretrained=False),
#             num_classes=8,
#         )
#         model.load_state_dict(weights, strict=True)
        
#         # Load normalization parameters
#         mean = load_pth(args.mean_path)
#         std = load_pth(args.std_path)
        
#         processor = RasterProcessor(
#             model=model,
#             tile_size=256,
#             stride=128,
#             gaussian_sigma=64,
#             batch_size=64,
#             mean=mean,
#             std=std,
#             device=get_torch_device()
#         )
        
#         # Process the raster
#         in_data = rxr.open_rasterio(path).sel(band=[1, 2, 4])
#         out = processor.process_raster(in_data.values / 255.0) + 1
        
#         # Apply nodata mask and mode filter
#         out[(in_data == in_data.rio.nodata).all(dim='band').values] = 0
#         out = out.astype(np.uint8)
#         out = mode_filter(out, size=3, nodata=0)
        
#         # Create output DataArray
#         out_da = xr.DataArray(
#             out,
#             dims=['y', 'x'],
#             coords={
#                 'y': in_data.y,
#                 'x': in_data.x,
#             },
#             attrs={
#                 'long_name': 'Land Cover Class',
#                 'units': '1',
#                 'crs': in_data.rio.crs.to_proj4(),
#             },
#         )
        
#         # Save to file
#         os.makedirs(args.output_dir, exist_ok=True)
#         out_da.rio.to_raster(
#             out_path,
#             driver='GTiff',
#             dtype='uint8',
#             compress='lzw',
#             blockxsize=256,
#             blockysize=256,
#             tiled=True,
#             bigTiff=True,
#         )
        
#         # Add colormap
#         with rio.open(out_path, 'r+') as dst:
#             dst.write_colormap(1, LEGEND_COLORS_RGBA)
            
#         return f"Processed {filename}"
        
#     except Exception as e:
#         # return f"Error processing {filename}: {str(e)}"
#         raise e
        


def parse_arguments():
    """Parse command line arguments for land cover inference."""
    parser = ArgumentParser(
        description='Perform land cover classification inference on multispectral raster imagery.',
        epilog='Example usage: python inference.py --input_dir /data/naip/2016 --output_dir /data/landcover/2016'
    )
    
    # Input/Output directories
    parser.add_argument(
        '--input_dir',
        type=str,
        default='/scratch/dhester/ms_naip/2016',
        help='Directory containing input raster files (.tif format). Default: /scratch/dhester/ms_naip/2016'
    )
    
    parser.add_argument(
        '--output_dir', 
        type=str,
        default='/scratch/dhester/ms_lc/2016',
        help='Directory where classified land cover rasters will be saved. Default: /scratch/dhester/ms_lc/2016'
    )
    
    # Model and weights
    parser.add_argument(
        '--weights_path',
        type=str,
        default='./weights/finetune_20250607/unet_fe.pth',
        help='Path to the trained model weights file (.pth format). Default: ./weights/finetune_20250607/unet_fe.pth'
    )
    
    parser.add_argument(
        '--mean_path',
        type=str,
        default='./weights/pretrain_mean.pth',
        help='Path to the normalization mean values file. Default: ./weights/pretrain_mean.pth'
    )
    
    parser.add_argument(
        '--std_path',
        type=str,
        default='./weights/pretrain_std.pth',
        help='Path to the normalization standard deviation values file. Default: ./weights/pretrain_std.pth'
    )
    
    # Model architecture parameters
    parser.add_argument(
        '--num_classes',
        type=int,
        default=8,
        help='Number of land cover classes in the model. Default: 8'
    )
    
    parser.add_argument(
        '--in_channels',
        type=int,
        default=3,
        help='Number of input channels (spectral bands). Default: 3 (NIR, Red, Green)'
    )
    
    # Processing parameters
    parser.add_argument(
        '--tile_size',
        type=int,
        default=256,
        help='Size of tiles for processing large rasters (pixels). Larger tiles use more memory but may be faster. Default: 256'
    )
    
    parser.add_argument(
        '--stride',
        type=int,
        default=128,
        help='Stride between tiles (pixels). Should be <= tile_size. Smaller stride provides more overlap but takes longer. Default: 128'
    )
    
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='Batch size for model inference. Larger batches are faster but use more GPU memory. Default: 256'
    )
    
    parser.add_argument(
        '--gaussian_sigma',
        type=float,
        default=64.0,
        help='Sigma for Gaussian blending of overlapping tiles. Higher values create smoother transitions. Default: 64.0'
    )
    
    # Input data preprocessing
    parser.add_argument(
        '--bands',
        type=int,
        nargs=3,
        default=[1, 2, 4],
        help='Spectral bands to use from input rasters (1-indexed). Default: [1, 2, 4] (Red, Green, NIR)'
    )
    
    parser.add_argument(
        '--scale_factor',
        type=float,
        default=255.0,
        help='Factor to divide pixel values by for normalization (e.g., 255 for 8-bit to 0-1 range). Default: 255.0'
    )
    
    # Post-processing
    parser.add_argument(
        '--mode_filter_size',
        type=int,
        default=0,
        help='Size of mode filter for post-processing (odd numbers only). Set to 0 to disable. Default: 3'
    )
    
    # Parallel processing
    parser.add_argument(
        '--num_processes',
        type=int,
        default=None,
        help='Number of parallel processes for inference. If not specified, automatically determined based on CPU cores (max 32).'
    )
    
    # Output format options
    parser.add_argument(
        '--compress',
        type=str,
        default='lzw',
        choices=['none', 'lzw', 'deflate', 'jpeg'],
        help='Compression method for output GeoTIFF files. Default: lzw'
    )
    
    parser.add_argument(
        '--block_size',
        type=int,
        default=256,
        help='Block size for tiled GeoTIFF output (pixels). Default: 256'
    )
    
    # Miscellaneous
    parser.add_argument(
        '--skip_existing',
        action='store_true',
        default=True,
        help='Skip processing files that already exist in output directory. Default: True'
    )
    
    parser.add_argument(
        '--overwrite',
        action='store_true',
        default=False,
        help='Overwrite existing output files. Overrides --skip_existing. Default: False'
    )
    
    parser.add_argument(
        '--file_pattern',
        type=str,
        default='*.tif',
        help='File pattern to match input files (glob pattern). Default: *.tif'
    )
    
    return parser.parse_args()


def process_single_raster(path, args):
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
            device=device
        )
        
        # Process the raster
        in_data = rxr.open_rasterio(path).sel(band=args.bands)
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


def main():
    # Set multiprocessing start method at the beginning of main
    if mp.get_start_method(allow_none=True) != 'spawn':
        mp.set_start_method('spawn', force=True)
    
    args = parse_arguments()
    
    # Print configuration
    print("Land Cover Inference Configuration:")
    print(f"  Input Directory: {args.input_dir}")
    print(f"  Output Directory: {args.output_dir}")
    print(f"  Model Weights: {args.weights_path}")
    print(f"  Processing {args.in_channels} bands: {args.bands}")
    print(f"  Tile Size: {args.tile_size}x{args.tile_size}")
    print(f"  Batch Size: {args.batch_size}")
    print(f"  Mode Filter: {'Enabled' if args.mode_filter_size > 0 else 'Disabled'}")
    print("-" * 50)
    
    # Get all paths to process
    paths = glob(os.path.join(args.input_dir, args.file_pattern))
    print(f"Found {len(paths)} files to process")
    
    if len(paths) == 0:
        print(f"No files found matching pattern '{args.file_pattern}' in {args.input_dir}")
        return
    
    # Create partial function with fixed arguments
    process_func = partial(process_single_raster, args=args)
    
    # Determine number of processes
    if args.num_processes is None:
        num_processes = min(mp.cpu_count() - 1, len(paths), 8)  # Cap at 8 processes
    else:
        num_processes = min(args.num_processes, len(paths))
    
    if num_processes > 1:
        print(f"Using {num_processes} parallel processes")
    
        # Process in parallel
        with mp.Pool(processes=num_processes) as pool:
            results = list(tqdm(
                pool.imap(process_func, paths),
                total=len(paths),
                desc="Processing rasters"
            ))
        
        # Print results summary
        successful = sum(1 for r in results if r.startswith("Processed"))
        skipped = sum(1 for r in results if r.startswith("Skipped"))
        errors = sum(1 for r in results if r.startswith("Error"))
        
        print(f"\nProcessing Summary:")
        print(f"  Successfully Processed: {successful}")
        print(f"  Skipped (already exist): {skipped}")
        print(f"  Errors: {errors}")
        print(f"  Total Files: {len(paths)}")
        
        # Print any errors
        if errors > 0:
            print(f"\nErrors Encountered ({errors}):")
            for result in results:
                if result.startswith("Error"):
                    print(f"  {result}")
    
    else:
        for path in tqdm(paths, desc="Processing rasters"):
            try:
                result = process_single_raster(path, args)
                print(result)
            except Exception as e:
                print(f"Error processing {os.path.basename(path)}: {str(e)}")
    
    print(f"\nOutput files saved to: {args.output_dir}")


if __name__ == '__main__':
    main()