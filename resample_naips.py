import rasterio
from rasterio.enums import Resampling
import glob
import os
from tqdm import tqdm

def resample_raster(input_path, output_path, target_resolution=30):
    """Resample the input raster to the target resolution and save it to output_path."""
    with rasterio.open(input_path) as src:
        # Calculate the new transform and dimensions
        scale_factor = target_resolution / src.res[0]
        new_width = int(src.width / scale_factor)
        new_height = int(src.height / scale_factor)
        new_transform = src.transform * src.transform.scale(scale_factor, scale_factor)

        # Resample the dataset
        data = src.read(
            out_shape=(src.count, new_height, new_width),
            resampling=Resampling.bilinear
        )

        # Update metadata
        profile = src.profile
        profile.update(
            transform=new_transform,
            width=new_width,
            height=new_height
        )

        # Save the resampled raster
        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(data)

def main():
    """Find and resample all rasters matching the pattern."""
    path = '/home/dhester/server/guser/dh/NAIP_MS_2023'
    raster_files = glob.glob(f"{path}/*/*_1m.tif")
    
    if not raster_files:
        print("No matching rasters found.")
        return
    
    for input_path in tqdm(raster_files, desc="Resampling rasters", unit="file"):
        output_path = input_path.replace("1m.tif", "30m.tif")
        resample_raster(input_path, output_path)

if __name__ == "__main__":
    main()
