import numpy as np
import rasterio as rio
from rasterio import windows
import os
from tqdm import tqdm

def main() -> None:
    
    naip_raster_path = './naip_resampled.tif'
    landcover_raster_path = './landcover_aligned.tif'
    out_path = './splits'
    tile_size = 256
    # total_samples = 10000
    
    # reclassify land cover rasters to match our products
    reclassify_1 = {
        1: 1,   # Water
        2: 5,   # Emergent Wetlands -> low vegetation
        3: 3,   # Tree canopy (woody vegetation)
        4: 4,   # Shrubland
        5: 5,   # Low vegetation
        6: 6,   # Barren land
        7: 7,   # Imperious structures
        8: 8,   # Other Impervious
        9: 8,   # Roads -> Imperious surfaces
        10: 3,  # Tree canopy over imperious structures -> Tree canopy (woody vegetation)
        11: 3,  # Tree canopy over other impervious -> Tree canopy (woody vegetation)
        12: 3,  # Tree canopy over roads -> Tree canopy (woody vegetation)
    }
    original_class_names = {
        1: 'Water',
        2: 'Emergent Wetlands',
        3: 'Tree canopy',
        4: 'Shrubland',
        5: 'Low vegetation',
        6: 'Barren land',
        7: 'Imperious structures',
        8: 'Other Impervious',
        9: 'Roads',
        10: 'Tree canopy over imperious structures',
        11: 'Tree canopy over other impervious',
        12: 'Tree canopy over roads',
    }
    reclassify_1_class_names = {k: original_class_names[v] for k, v in reclassify_1.items()}
    reclassify_2 = {v: i for i, v in enumerate(set(reclassify_1.values()), 1)}
    reclassify_2_class_names = {v: original_class_names[k] for k, v in reclassify_2.items()}    
    
    print('Reclassification mapping:')
    for starting_index, intermediate_index in reclassify_1.items():
        final_index = reclassify_2[intermediate_index]
        print(f'{starting_index}: {original_class_names[starting_index]} -> {intermediate_index}: {reclassify_1_class_names[intermediate_index]} -> {final_index}: {reclassify_2_class_names[final_index]}')
    
    reclassify_full = {k: reclassify_2[v] for k, v in reclassify_1.items()}
    reclassify_func = np.vectorize(reclassify_full.get)
    for k, v in reclassify_full.items():
        print(f'{k}: {original_class_names[k]} -> {v}: {reclassify_2_class_names[v]}')
    return
    with rio.open(naip_raster_path) as naip_src, rio.open(landcover_raster_path) as landcover_src:
        
        naip_width, naip_height = naip_src.width, naip_src.height
        candidate_indices = [(x, y) for x in range(0, naip_width, tile_size) for y in range(0, naip_height, tile_size)]
        np.random.shuffle(candidate_indices)

        sampled_tiles = 0
        pbar = tqdm(total=len(candidate_indices), desc='Sampling tiles', unit='tiles')
        while len(candidate_indices) > 0:
            pbar.set_postfix({'sampled_tiles': sampled_tiles})
            pbar.update(1)
            candidate_index = candidate_indices.pop()
            
            window = windows.Window(*candidate_index, tile_size, tile_size) 
            naip_tile = naip_src.read(window=window)
            
            if naip_tile.shape != (3, tile_size, tile_size):
                continue
            
            # if more than 5% of the tile is black (0, 0, 0), skip
            nd_values = np.array([naip_src.nodata] * 3)
            pixels_with_nd = np.equal(naip_tile.transpose(1, 2, 0).reshape(-1, 3), nd_values).all(axis=1)
            if pixels_with_nd.sum(): # > (0.05 * len(pixels_with_nd)):
                continue
            
            lc_tile = landcover_src.read(window=window)
            if lc_tile.shape != (1, tile_size, tile_size):
                continue
            
            pixels_with_nd = np.equal(lc_tile.reshape(-1), landcover_src.nodata)
            if pixels_with_nd.sum(): # > (0.05 * len(pixels_with_nd)):
                continue
            
            # valid_tiles += 1
            # continue
            
            lc_tile = reclassify_func(lc_tile)
            
            if sampled_tiles % 4 in (0, 1):
                out_dir = os.path.join(out_path, 'train')
            elif sampled_tiles % 4 == 2:
                out_dir = os.path.join(out_path, 'val')
            else:
                out_dir = os.path.join(out_path, 'test')
            
            image_id = f'{sampled_tiles:05d}'
            naip_out_path = os.path.join(out_dir, 'input')
            os.makedirs(naip_out_path, exist_ok=True)
            
            profile = naip_src.profile.copy()
            profile.update({
                'width': tile_size,
                'height': tile_size,
                'transform': rio.windows.transform(window, naip_src.transform),
                'compress': 'lzw',
                'BIGTIFF': 'NO',
                'TILED': 'YES',
                'BLOCKXSIZE': '256',
                'BLOCKYSIZE': '256'
            })
            with rio.open(os.path.join(naip_out_path, f'{image_id}.tif'), 'w', **profile) as dst:
                dst.write(naip_tile)
                
            lc_out_path = os.path.join(out_dir, 'target')
            os.makedirs(lc_out_path, exist_ok=True)
            profile = landcover_src.profile.copy()
            profile.update({
                'width': tile_size,
                'height': tile_size,
                'transform': rio.windows.transform(window, landcover_src.transform),
                'compress': 'lzw',
                'BIGTIFF': 'NO',
                'TILED': 'YES',
                'BLOCKXSIZE': '256',
                'BLOCKYSIZE': '256'
            })
            with rio.open(os.path.join(lc_out_path, f'{image_id}.tif'), 'w', **profile) as dst:
                dst.write(lc_tile)
            
            sampled_tiles += 1
            pbar.update(1)
    
    with open(os.path.join(out_path, 'sampled_tiles.txt'), 'w') as f:
        f.write(f'{sampled_tiles}\n')
    
if __name__ == '__main__':
    main()