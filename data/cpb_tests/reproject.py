import math
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject
from rasterio.windows import Window
from rasterio.transform import Affine
from tqdm import tqdm
import os

def main(landcover_path, naip_path, out_landcover_path, out_naip_path):
    # Open NAIP to get target CRS, bounds, resolution, etc.
    with rasterio.open(naip_path) as naip_src:
        naip_crs = naip_src.crs
        naip_bounds = naip_src.bounds
        # original_transform = naip_src.transform

        # Compute new transform at 1m resolution, aligned to top-left
        left, bottom, right, top = naip_bounds
        new_res = 1.0
        width = int(math.ceil((right - left) / new_res))
        height = int(math.ceil((top - bottom) / new_res))
        new_transform = Affine(new_res, 0, left, 0, -new_res, top)

    if not os.path.exists(out_landcover_path):
        # Reproject and clip landcover to NAIP's extent, CRS, and new transform
        with rasterio.open(landcover_path) as lc_src:
            if lc_src.crs != naip_crs:
                print("CRS mismatch detected. Reprojecting landcover...")

            profile = lc_src.profile.copy()
            profile.update({
                'crs': naip_crs,
                'transform': new_transform,
                'width': width,
                'height': height,
                'compress': 'lzw'
            })

            with rasterio.open(out_landcover_path, 'w', **profile) as dst:
                for i in tqdm(range(1, lc_src.count + 1), desc="Reprojecting LULC bands"):
                    reproject(
                        source=rasterio.band(lc_src, i),
                        destination=rasterio.band(dst, i),
                        src_transform=lc_src.transform,
                        src_crs=lc_src.crs,
                        dst_transform=new_transform,
                        dst_crs=naip_crs,
                        resampling=Resampling.nearest,
                        num_threads=16,
                    )
                
                try: # Copy colormap if it exists
                    dst.write_colormap(1, lc_src.colormap(1))
                except ValueError: # No colormap
                    dst.write_colormap(1, {
                        1: (5, 67, 255, 255),
                        2: (21, 154, 113, 255),
                        3: (32, 98, 3, 255),
                        4: (68, 230, 2, 255),
                        5: (152, 248, 103, 255),
                        6: (253, 154, 10, 255),
                        7: (251, 0, 6, 255),
                        8: (163, 163, 163, 255),
                        9: (0, 0, 0, 255),
                        10: (96, 98, 2, 255),
                        11: (194, 197, 83, 255),
                        12: (255, 255, 97, 255),
                        13: (181, 0, 255, 255),
                    })

    # Resample NAIP from 0.6m -> 1.0m so pixels align with landcover
    if not os.path.exists(out_naip_path) or True:
        with rasterio.open(naip_path) as naip_src:
            if naip_src.crs != naip_crs:
                print("Unexpected CRS difference. Double-check input files.")
            
            # Setup output parameters
            profile = naip_src.profile.copy()
            profile.update({
                'transform': new_transform,
                'width': width,
                'height': height,
                'compress': 'lzw',
                'BIGTIFF': 'YES',
                'TILED': 'YES',
                'BLOCKXSIZE': '256',
                'BLOCKYSIZE': '256'
            })

            # Create destination array
            dest = np.zeros((naip_src.count, height, width), dtype=naip_src.dtypes[0])

            with rasterio.open(out_naip_path, 'w', **profile) as dst:
                # Reproject entire image at once
                reproject(
                    source=rasterio.band(naip_src, list(range(1, naip_src.count + 1))),
                    destination=dest,
                    src_transform=naip_src.transform,
                    src_crs=naip_src.crs,
                    dst_transform=new_transform,
                    dst_crs=naip_crs,
                    resampling=Resampling.bilinear,
                    num_threads=16
                )
                
                # Write output
                dst.write(dest)
                #         pbar.update(1)
                # pbar.close()

    print("Done. Both rasters should have the exact same affine transform.")

if __name__ == "__main__":
    # Example usage:
    main(
        landcover_path="./va_lc_2018_2022-Edition/va_lc_2018_2022-Edition.tif",
        naip_path="./ortho_1-1_hc_s_va059_2018_1/ortho_1-1_hc_s_va059_2018_1.tif",
        out_landcover_path="landcover_aligned.tif",
        out_naip_path="naip_resampled.tif"
    )