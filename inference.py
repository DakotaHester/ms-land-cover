from functools import partial
from threading import Lock
from time import sleep, time
from warnings import warn
import pandas as pd
import torch
import geopandas as gpd
import shapely
import rasterio as rio
from rasterio.mask import geometry_mask, mask
from rasterio.features import shapes, rasterize
from rasterio.io import MemoryFile
from multiprocessing import Pool, cpu_count, shared_memory
import os
import numpy as np
import cv2 as cv
from concurrent.futures import ThreadPoolExecutor
from skimage.segmentation import quickshift, felzenszwalb
from tqdm import tqdm


from src.mslandcover.inference import GPURasterProcessor, compute_segment_means, process_batch
from src.mslandcover.models import UNet
from src.mslandcover.config import MSTM_PROJ4, HRNET_W18_CONFIG, LEGEND_COLORS_RGBA, LEGEND_CLASSES
from src.mslandcover.utils import load_pth, get_torch_device, Logger

# if land cover probabilities are already computed, set this to True to skip inference



def main():
    SKIP_INFERENCE = True
    SKIP_CLIPPING = False
    SKIP_POSTPROCESSING = True
    
        
    model_weights_path = './weights/multistage_finetuning_stage2/dae/s1_full_train/s2_decoder_train/best_model.pth'
    # model_weights_path = './weights/multistage_unet/best_model.pth'
    
    if os.environ.get('MSLC_INFERENCE_COUNTY_INDEX') is not None:
        county_index = int(os.environ.get('MSLC_INFERENCE_COUNTY_INDEX'))
    else:
        county_index = 74 # warren county
        # county_index = 52 # oktibbeha county
    
    log_dir = f'./data/inference_results/logs/{county_index}'
    
    os.makedirs(log_dir, exist_ok=True)
    logger = Logger(os.path.join(log_dir, 'inference.log'))
    logger.log(f'Starting inference for county index {county_index}...')

    device = get_torch_device()
    logger.log(f'Using device: {device}')
    
    logger.log('Loading county boundaries...')
    ms_counties_gdf = gpd.read_file('./data/shapefiles/ms_counties.gpkg')
    county_series = ms_counties_gdf.iloc[county_index]
    
    county_geom = county_series['geometry']
    county_name = county_series['NAME']
    county_fp_code = county_series['COUNTYFP']
    raster_path = county_series['raster_path']

    
    # load starkville and msu countuy geom
    # okt_raster_path = '/Volumes/dhester_ssd/NAIP_MS_2023/ortho_1-1_hc_s_ms105_2023_1/ortho_1-1_hc_s_ms105_2023_1_1m.tif'
    # census_ms_places_shp_path = './data/shapefiles/tl_2024_28_place/tl_2024_28_place.shp'
    #census_usa_counties_shp_path = '/Users/dak/Downloads/tl_2024_us_county/tl_2024_us_county.shp'

    # census_counties_gdf = gpd.read_file(census_usa_counties_shp_path).to_crs(ms_counties_gdf.crs)
    # ms_places_gdf = gpd.read_file(census_ms_places_shp_path).to_crs(ms_counties_gdf.crs)
    # # okt_county_geom = census_counties_gdf[(census_counties_gdf['STATEFP'] == '28') & (census_counties_gdf['COUNTYFP'] == '105')]['geometry'].values[0]
    # okt_county_places = ms_places_gdf[ms_places_gdf.intersects(county_geom)]
    # starville_msu_gdf = okt_county_places[okt_county_places['NAME'].isin(['Starkville', 'Mississippi State'])]
    # starville_msu_gdf = starville_msu_gdf.dissolve()
    # county_geom = shapely.geometry.Polygon(starville_msu_gdf.loc[0, 'geometry'].exterior)
    # county_geom = county_geom.buffer(-1000)
    
    
    if county_geom.geom_type == 'Polygon':
        bounding_polygons = [shapely.geometry.Polygon(county_geom.exterior)]
    else:
        bounding_polygons = [shapely.geometry.Polygon(polygon.exterior) for polygon in county_geom.geoms]
    
    out_path = f'/home/dhester/server/guser/dh/MS_HiRes_LC_Prelim/{int(county_fp_code):03}'
    # use whole state for now
    # bounding_polygons = [ms_counties_gdf.unary_union]
    
    logger.log(f'Loaded county {county_name} with FIPS code {county_fp_code}')
    
    logger.log('Loading model...')
    model = UNet(use_extended_decoder=False).to(get_torch_device())
    # if len(model.decoder_blocks > 5):
        
    model.load_state_dict(load_pth(model_weights_path, map_location=device))
    model.eval()
    
    logger.log('Starting inference...')
    processor = GPURasterProcessor(
        model=model,
        tile_size=256,
        stride=64,
        gaussian_sigma=128,
        batch_size=128,
        mean=load_pth('./weights/pretrain_mean.pth'),
        std=load_pth('./weights/pretrain_std.pth'),
        device=device,
    )
    
    logger.log('Loading raster data...')
    # raster_path = '/Volumes/dhester_ssd/mslc_inf_test/starkville_msu_2023_reduced.tif'
    # raster_path = r"G:\mslc_inf_test\starkville_msu_2023_reduced.tif"
    # raster_path = '/Volumes/dhester_ssd/NAIP_MS_2023/ortho_1-1_hc_s_ms105_2023_1/ortho_1-1_hc_s_ms105_2023_1_1m.tif'
    # raster_path = './data/NAIP_MS/ortho_1-1_hc_s_ms149_2023_1/ortho_1-1_hc_s_ms149_2023_1_1m.tif'
    # raster_path = r"Z:\guser\dh\NAIP_MS_2023\ortho_1-1_hc_s_ms105_2023_1\ortho_1-1_hc_s_ms105_2023_1_1m.tif"
    
    raster_path = f'/home/dhester/server/guser/dh/NAIP_MS_2023/ortho_1-1_hc_s_ms{int(county_fp_code):03}_2023_1/ortho_1-1_hc_s_ms{int(county_fp_code):03}_2023_1_1m.tif'
    with rio.open(raster_path) as src:
        profile = src.profile.copy()
        # raster_data = src.read()
        raster_data, transform = mask(
            src,
            [county_geom.buffer(processor.tile_size * 1.5)], # clip to county boundary for now TODO account for multipolygonsor polygons with holes???
            crop=True,
            all_touched=True,
        )
    
    profile.update({
        'BIGTIFF': 'YES',
        'compress': 'LZW',
        'transform': transform,
        'height': raster_data.shape[1],
        'width': raster_data.shape[2],
    })
    
    
    if SKIP_INFERENCE and not os.path.exists(os.path.join(out_path, 'lc_probs.tif')):
        warn('SKIP_INFERENCE is set to True, but land cover probabilities are not found. Continuing with inference...')
        SKIP_INFERENCE = False
        
    if SKIP_INFERENCE and os.path.exists(os.path.join(out_path, 'lc_probs.tif')):
        logger.log('Land cover probabilities already computed. Skipping inference...')
        with rio.open(os.path.join(out_path, 'lc_probs.tif')) as src:
            lc_probs = src.read().astype(np.float32) / 100
    else:
        logger.log('Processing raster data...')
        lc_probs = processor.process_raster(raster_data)
        
        logger.log('Saving results...')
        os.makedirs(out_path, exist_ok=True)
            
        lc_probs_profile = profile.copy()
        lc_probs_profile.update(count=lc_probs.shape[0], dtype=rio.uint8)
        with rio.open(os.path.join(out_path, 'lc_probs.tif'), 'w', **lc_probs_profile) as dst:
            for band in range(lc_probs.shape[0]):
                dst.write(np.clip((lc_probs[band] * 100), 1, 100).astype(rio.uint8), band + 1)
        logger.log(f'Saved land cover probabilities to {os.path.join(out_path, "lc_probs.tif")}')


    if SKIP_INFERENCE and os.path.exists(os.path.join(out_path, 'lc_classes.tif')):
        logger.log('Land cover classes already computed. Skipping final classification...')
    else:
        logger.log('Classifying land cover...')
        lc_classes = lc_probs.argmax(axis=0).astype(rio.uint8) + 1
        lc_classes_profile = profile.copy()
        lc_classes_profile.update(count=1, dtype=rio.uint8, nodata=0)
        with rio.open(os.path.join(out_path, 'lc_classes.tif'), 'w', **lc_classes_profile) as dst:
            dst.write(lc_classes, 1)
            dst.write_colormap(1, LEGEND_COLORS_RGBA)
            
        logger.log(f'Saved land cover classes to {os.path.join(out_path, "lc_classes.tif")}')
        del lc_classes
    
    
    if SKIP_INFERENCE and os.path.exists(os.path.join(out_path, 'lc_confidence.tif')):
        logger.log('Land cover confidence already computed. Skipping final classification...')
    else:
        logger.log('Computing land cover confidence...')
        lc_confidence = np.clip((lc_probs.max(axis=0) * 100).astype(rio.uint8), 1, 100)
        
        lc_confidence_profile = profile.copy()
        lc_confidence_profile.update(count=1, dtype=rio.uint8)
        with rio.open(os.path.join(out_path, 'lc_confidence.tif'), 'w', **lc_confidence_profile) as dst:
            dst.write(lc_confidence, 1)
        
        # free up memory as lc_classes and lc_confidence are no longer needed
        logger.log(f'Saved land cover confidence to {os.path.join(out_path, "lc_confidence.tif")}')
        del lc_confidence
    
    
    
    # now, mask outputs by polygon boundary
    if SKIP_CLIPPING and os.path.exists(os.path.join(out_path, 'lc_probs_clipped.tif')):
        logger.log('Skipping clipping of land cover probabilities...')
    else:
        logger.log('Masking predictions by bounding polygons...')
        with rio.open(os.path.join(out_path, 'lc_probs.tif')) as src:
            clipped_data, clipped_transform = mask(
                src,
                bounding_polygons, 
                crop=True,
                all_touched=True
            )
            
            updated_profile = src.profile.copy()
            updated_profile.update({
                'transform': clipped_transform,
                'height': clipped_data.shape[1],
                'width': clipped_data.shape[2],
                'BIGTIFF': 'YES',
                'count': lc_probs.shape[0],
            })
            
            with rio.open(os.path.join(out_path, 'lc_probs_clipped.tif'), 'w', **updated_profile) as dst:
                dst.write(clipped_data)
        
        logger.log(f'Saved clipped land cover probabilities to {os.path.join(out_path, "lc_probs_clipped.tif")}')
    
    
    if SKIP_CLIPPING and os.path.exists(os.path.join(out_path, 'lc_classes_clipped.tif')):
        logger.log('Skipping clipping of land cover classes...')
    else:
        logger.log('Masking land cover classes..')
        with rio.open(os.path.join(out_path, 'lc_classes.tif')) as src:
            clipped_data, clipped_transform = mask(
                src,
                bounding_polygons, 
                crop=True,
                all_touched=True
            )
            
            updated_profile = src.profile.copy()
            updated_profile.update({
                'transform': clipped_transform,
                'height': clipped_data.shape[1],
                'width': clipped_data.shape[2],
                'BIGTIFF': 'YES',
                'count': 1,
            })
            
            with rio.open(os.path.join(out_path, 'lc_classes_clipped.tif'), 'w', **updated_profile) as dst:
                dst.write(clipped_data)
                dst.write_colormap(1, LEGEND_COLORS_RGBA)
            
        logger.log(f'Saved clipped land cover classes to {os.path.join(out_path, "lc_classes_clipped.tif")}')
    
    
    if SKIP_CLIPPING and os.path.exists(os.path.join(out_path, 'lc_confidence_clipped.tif')):
        logger.log('Skipping clipping of land cover confidence...')
    else:
        logger.log('Masking land cover confidence...')
        with rio.open(os.path.join(out_path, 'lc_confidence.tif')) as src:
            clipped_data, clipped_transform = mask(
                src,
                bounding_polygons, 
                crop=True,
                all_touched=True
            )
            
            updated_profile = src.profile.copy()
            updated_profile.update({
                'transform': clipped_transform,
                'height': clipped_data.shape[1],
                'width': clipped_data.shape[2],
                'BIGTIFF': 'YES',
                'count': 1,
            })
            
            with rio.open(os.path.join(out_path, 'lc_confidence_clipped.tif'), 'w', **updated_profile) as dst:
                dst.write(clipped_data)
                
        logger.log(f'Saved clipped land cover confidence to {os.path.join(out_path, "lc_confidence_clipped.tif")}')
    
    
    # if SKIP_POSTPROCESSING and os.path.exists(os.path.join(out_path, 'features.gpkg')):
        # logger.log('Features already computed. Skipping post-processing...')
        # return
    if SKIP_POSTPROCESSING:
        logger.log(f'SKIP_POSTPROCESSING set to True, ending now..')
        return
    
    features_dict = {
        'geometry': [],
        'predicted_class': [],
        'confidence': [],
    }
    for i in range(1, len(LEGEND_CLASSES)):
        features_dict[LEGEND_CLASSES[i]] = []
    
    # prepare for multiprocessing - global vars are a bad practice I know, but its easier to do this than to work with shared memory
    bounding_mask = geometry_mask(bounding_polygons, out_shape=raster_data.shape[1:], transform=profile['transform'], all_touched=True, invert=True)
    
    raster_transform = profile['transform']
    
    # chunk the arrays into 10k x 10k pixel chunks for processing
    chunk_size = 1024
    # print(raster_data.shape, lc_probs.shape)
    # with tqdm(total=(lc_probs.shape[2] // chunk_size) * (lc_probs.shape[1] // chunk_size), desc='Post-processing land cover', unit='chunks') as pbar:
        
    with Pool(cpu_count()) as pool:
        active_tasks = []
        max_concurrent_tasks = cpu_count()
        features_list = []
        
        with tqdm(total=(lc_probs.shape[2] // chunk_size) * (lc_probs.shape[1] // chunk_size), desc='Post-processing land cover', unit='chunks') as pbar:
            for j in range(0, lc_probs.shape[2], chunk_size):
                for i in range(0, lc_probs.shape[1], chunk_size):
                    while len(active_tasks) >= max_concurrent_tasks:
                        done_tasks = []
                        for task in active_tasks:
                            if task.ready():
                                features_list.extend(task.get())
                                done_tasks.append(task)
                                pbar.update(1)
                        
                        active_tasks = [task for task in active_tasks if not task in done_tasks]
                        if len(active_tasks) >= max_concurrent_tasks:
                            sleep(0.1)
                            
                    chunk_args = (
                        raster_data[:, i:i+chunk_size, j:j+chunk_size],
                        lc_probs[:, i:i+chunk_size, j:j+chunk_size],
                        bounding_mask[i:i+chunk_size, j:j+chunk_size],
                        rio.Affine.translation(j, -i) * raster_transform,
                    )
                    active_tasks.append(pool.apply_async(postprocess_batch, chunk_args))
            
            for task in active_tasks:
                features_list.extend(task.get())
                pbar.update(1)

    for feature in features_list:
        for key in features_dict.keys():
            features_dict[key].append(feature[key])
            
        # for i in range(0, lc_probs.shape[2], chunk_size):
        #     for j in range(0, lc_probs.shape[1], chunk_size):
                
                
            # features_gdf_chunk = gpd.GeoDataFrame(features_dict_chunk, crs=profile['crs'], geometry='geometry')
            # features_gdfs.append(features_gdf_chunk)
            # features_gdf = pd.concat(features_gdfs)
            # features_gdf.to_file(os.path.join(out_path, 'features_running.gpkg'), driver='GPKG')
                
            
            
            
    # # now, object-based segmentation using quickshift
    # logger.log('Segmenting image...')
    # # segments = quickshift(raster_data.transpose(1, 2, 0), kernel_size=3, max_dist=6, ratio=0.5).astype(np.uint16)
    # segments = parallel_quickshift(raster_data).astype(np.int32)
    
    # # use zonal statistics to get mean probabilities for each segment
    # logger.log('Extracting zonal land cover probability means...')
    # class_means = compute_segment_means(lc_probs, segments) # Dict[int, np.ndarray]
    
    # logger.log('Generating polygons from segments...')
    # bounding_mask = geometry_mask(bounding_polygons, out_shape=segments.shape, transform=profile['transform'], all_touched=True, invert=True)
    # geoms = [(geom, int(id)) for geom, id in shapes(segments.astype(np.float64), mask=bounding_mask, connectivity=4, transform=profile['transform'])]
    
    # logger.log('Compiling to feature class')
    # features_dict = {
    #     'geometry': [],
    #     'predicted_class': [],
    #     'confidence': [],
    # }
    # for i in range(1, len(LEGEND_CLASSES)):
    #     features_dict[LEGEND_CLASSES[i]] = []
    
    # for i, (g, segment_id) in enumerate(tqdm(geoms, desc='Compiling features', unit='geometries')):
    #     if segment_id not in class_means.keys():
    #         logger.log(f'Segment ID {segment_id} not found in class means! Skipping...')
    #         continue
        
    #     features_dict['geometry'].append(shapely.geometry.shape(g))
    #     probs = class_means[segment_id]
    #     for j, prob in enumerate(probs):
    #         class_label = LEGEND_CLASSES[j + 1]
    #         features_dict[class_label].append(prob)
    #     features_dict['predicted_class'].append(LEGEND_CLASSES[np.argmax(probs) + 1])
    #     features_dict['confidence'].append(np.max(probs))
    
    
    # features_gdf = gpd.GeoDataFrame(features_dict, crs=profile['crs'], geometry='geometry')
    
    
    logger.log(f'Compiling features into GeoDataFrame...')
    features_gdf = gpd.GeoDataFrame(features_dict, crs=profile['crs'], geometry='geometry')
    
    logger.log(f'Saving features to {os.path.join(out_path, "features.gpkg")}...')
    features_gdf.to_file(os.path.join(out_path, 'features.gpkg'), driver='GPKG')
    
    # features_dissolved_gdf = features_gdf[['predicted_class', 'geometry']]
    logger.log('Dissolving features...')
    # TODO: split features gdf into chunks then dissolve using multiprocessing
    features_dissolved_gdf = features_gdf.dissolve(by='predicted_class', aggfunc='mean')
    
    logger.log(f'Saving reduced features to {os.path.join(out_path, "features_dissolved.gpkg")}...')
    features_dissolved_gdf.to_file(os.path.join(out_path, 'features_dissolved.gpkg'), driver='GPKG')

# TODO: Move helper functions to a separate file
def process_segment_mean(segments, segment_ids, raster):
    
    # Create a mask for all segment_ids at once
    masks = np.stack([(segments == segment_id) for segment_id in segment_ids])
    # Compute means for all segments in one operation
    means = np.array([raster[:, mask].mean(axis=1) if mask.any() else np.zeros(raster.shape[0]) 
                        for mask in masks])
    # Create results list
    results = list(zip(segment_ids, means))
    return {segment_id: mean for segment_id, mean in results}



def postprocess_batch(
    raster_data_chunk,
    lc_probs_chunk,
    chunk_bounding_mask,
    chunk_transform,
    pbar=None
):
    
    if (raster_data_chunk == 0).all():
        if pbar is not None:
            pbar.update(1)
        # logger.log(f'Chunk {i}, {j} is all zeros. Skipping...')
        return []
    
    segments_chunk = felzenszwalb(raster_data_chunk.transpose(1, 2, 0), scale=75, sigma=0.8)
    unique_segments_chunk = np.unique(segments_chunk)
    segment_masks = np.stack([(segments_chunk == segment_id) for segment_id in unique_segments_chunk])
    segment_means = np.array(
        [lc_probs_chunk[:, mask].mean(axis=1) if mask.any() else np.zeros(lc_probs_chunk.shape[0])
        for mask in segment_masks]
    )
    class_means_chunk = {segment_id: mean for segment_id, mean in zip(unique_segments_chunk, segment_means)}
    
    geoms_chunk = shapes(segments_chunk.astype(np.float64), mask=chunk_bounding_mask, transform=chunk_transform)
    geoms_chunk = [(geom, int(id)) for geom, id in geoms_chunk]
    
    if len(geoms_chunk) == 0:
        if pbar is not None:
            pbar.update(1)
        return []
    
    features_list = []
    for g, segment_id in geoms_chunk:
        
        feature = {}
        
        if segment_id == 0:
            continue
        
        if segment_id not in class_means_chunk.keys():
            
            continue
        
        feature['geometry'] = shapely.geometry.shape(g)
        probs = class_means_chunk[segment_id]
        for l, prob in enumerate(probs):
            class_label = LEGEND_CLASSES[l + 1]
            feature[class_label] = prob
        feature['predicted_class'] = LEGEND_CLASSES[np.argmax(probs) + 1]
        feature['confidence'] = np.max(probs)
        features_list.append(feature)
        
    if pbar is not None:
        pbar.update(1)
        
    return features_list

if __name__ == '__main__':
    main()
