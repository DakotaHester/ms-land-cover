from functools import partial
from threading import Lock
from time import time
import torch
import geopandas as gpd
import shapely
import rasterio as rio
from rasterio.mask import geometry_mask, mask
from rasterio.features import shapes, rasterize
from rasterio.io import MemoryFile
from multiprocessing import cpu_count, shared_memory
import os
import numpy as np
import cv2 as cv
from concurrent.futures import ThreadPoolExecutor
from skimage.segmentation import quickshift
from tqdm import tqdm


from src.mslandcover.inference import GPURasterProcessor, compute_zonal_means, parallel_quickshift
from src.mslandcover.models import UNet
from src.mslandcover.config import MSTM_PROJ4, HRNET_W18_CONFIG, LEGEND_COLORS_RGBA, LEGEND_CLASSES
from src.mslandcover.utils import load_pth, get_torch_device, Logger


def main():
    
    # model_weights_path = './weights/multistage_finetuning_stage2/dae/s1_full_train/s2_decoder_train/best_model.pth'
    model_weights_path = './weights/multistage_unet/best_model.pth'
    
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
    # census_ms_places_shp_path = '/Users/dak/Downloads/tl_2024_28_place/tl_2024_28_place.shp'
    # census_usa_counties_shp_path = '/Users/dak/Downloads/tl_2024_us_county/tl_2024_us_county.shp'

    # census_counties_gdf = gpd.read_file(census_usa_counties_shp_path).to_crs(ms_counties_gdf.crs)
    # ms_places_gdf = gpd.read_file(census_ms_places_shp_path).to_crs(ms_counties_gdf.crs)
    # okt_county_geom = census_counties_gdf[(census_counties_gdf['STATEFP'] == '28') & (census_counties_gdf['COUNTYFP'] == '105')]['geometry'].values[0]
    # okt_county_places = ms_places_gdf[ms_places_gdf.intersects(okt_county_geom)]
    # starville_msu_gdf = okt_county_places[okt_county_places['NAME'].isin(['Starkville', 'Mississippi State'])]
    # starville_msu_gdf = starville_msu_gdf.dissolve()
    # county_geom = shapely.geometry.Polygon(starville_msu_gdf.loc[0, 'geometry'].exterior)
    # # county_geom = county_geom.buffer(-2000)
    
    
    if county_geom.geom_type == 'Polygon':
        bounding_polygons = [shapely.geometry.Polygon(county_geom.exterior)]
    else:
        bounding_polygons = [shapely.geometry.Polygon(polygon.exterior) for polygon in county_geom.geoms]
    
    # use whole state for now
    # bounding_polygons = [ms_counties_gdf.unary_union]
    
    logger.log(f'Loaded county {county_name} with FIPS code {county_fp_code}')
    
    logger.log('Loading model...')
    model = UNet().to(get_torch_device())
    model.load_state_dict(load_pth(model_weights_path, map_location=device))
    model.eval()
    
    logger.log('Starting inference...')
    processor = GPURasterProcessor(
        model=model,
        tile_size=256,
        stride=64,
        gaussian_sigma=192,
        batch_size=32,
        mean=load_pth('./weights/pretrain_mean.pth'),
        std=load_pth('./weights/pretrain_std.pth'),
        device=device,
    )
    
    logger.log('Loading raster data...')
    # raster_path = '/Volumes/dhester_ssd/mslc_inf_test/starkville_msu_2023_reduced.tif'
    # raster_path = r"G:\mslc_inf_test\starkville_msu_2023_reduced.tif"
    # raster_path = '/Volumes/dhester_ssd/NAIP_MS_2023/ortho_1-1_hc_s_ms105_2023_1/ortho_1-1_hc_s_ms105_2023_1_1m.tif'
    # raster_path = 
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
    
    logger.log('Processing raster data...')
    lc_probs = processor.process_raster(raster_data)
    
    out_path = f'./data/inference_results/{county_fp_code}'
    # out_path = f'./data/inference_results/starkville_msu_2023_less_reduced'
    logger.log('Saving results...')
    os.makedirs(out_path, exist_ok=True)
    
    lc_probs_profile = profile.copy()
    lc_probs_profile.update(count=lc_probs.shape[0], dtype=rio.uint8)
    with rio.open(os.path.join(out_path, 'lc_probs.tif'), 'w', **lc_probs_profile) as dst:
        dst.write(np.clip((lc_probs * 100).astype(rio.uint8), 1, 100))
    logger.log(f'Saved land cover probabilities to {os.path.join(out_path, "lc_probs.tif")}')
    
    lc_classes = lc_probs.argmax(axis=0).astype(rio.uint8) + 1
    lc_confidence = np.clip((lc_probs.max(axis=0) * 100).astype(rio.uint8), 1, 100)
    
    lc_classes_profile = profile.copy()
    lc_classes_profile.update(count=2, dtype=rio.uint8)
    with rio.open(os.path.join(out_path, 'lc_classes.tif'), 'w', **lc_classes_profile) as dst:
        dst.write(lc_classes, 1)
        dst.write(lc_confidence, 2)
        dst.write_colormap(1, LEGEND_COLORS_RGBA)
    logger.log(f'Saved land cover classes to {os.path.join(out_path, "lc_classes.tif")}')
    
    # free up memory as lc_classes and lc_confidence are no longer needed
    del lc_classes, lc_confidence
    
    # now, mask outputs by polygon boundary
    logger.log('Masking predictions by bounding polygons...')
    
    with rio.open(os.path.join(out_path, 'lc_probs.tif')) as src:
        lc_probs, lc_probs_transform = mask(
            src,
            bounding_polygons, 
            crop=True,
            all_touched=True
        )
        updated_profile = src.profile.copy()
        updated_profile.update({
            'transform': lc_probs_transform,
            'height': lc_probs.shape[1],
            'width': lc_probs.shape[2]
        })
        with rio.open(os.path.join(out_path, 'lc_probs_clipped.tif'), 'w', **updated_profile) as dst:
            dst.write(lc_probs)
    
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
            'width': clipped_data.shape[2]
        })
        with rio.open(os.path.join(out_path, 'lc_classes_clipped.tif'), 'w', **updated_profile) as dst:
            dst.write(clipped_data)
            dst.write_colormap(1, LEGEND_COLORS_RGBA)
    
    # now, object-based segmentation using quickshift
    logger.log('Segmenting image...')
    # segments = quickshift(raster_data.transpose(1, 2, 0), kernel_size=3, max_dist=6, ratio=0.5).astype(np.uint16)
    segments = parallel_quickshift(raster_data)
    
    logger.log('Generating polygons from segments...')
    bounding_mask = geometry_mask(bounding_polygons, out_shape=segments.shape, transform=profile['transform'], all_touched=True, invert=True)
    geoms = [shapely.geometry.shape(geom) for geom, _ in shapes(segments, mask=bounding_mask, connectivity=4, transform=profile['transform'])]
    
    # use zonal statistics to get mean probabilities for each segment
    logger.log('Extracting zonal land cover probability means...')
    class_means = compute_zonal_means(lc_probs, geoms, lc_probs_transform)
    
    logger.log('Compiling to feature class')
    class_means = np.array(list(class_means)).T
    features_dict = {'geometry': geoms}
    for i in range(class_means.shape[0]):
        class_label = LEGEND_CLASSES[i + 1]
        features_dict[class_label] = class_means[i]
    
    features_dict['predicted_class'] = [LEGEND_CLASSES[pred] for pred in np.argmax(class_means, axis=0) + 1]
    features_dict['confidence'] = np.max(class_means, axis=0)
    
    features_gdf = gpd.GeoDataFrame(features_dict, crs=profile['crs'], geometry='geometry')
    
    logger.log(f'Saving features to {os.path.join(out_path, "features.gpkg")}...')
    features_gdf.to_file(os.path.join(out_path, 'features.gpkg'), driver='GPKG')
    
    features_dissolved_gdf = features_gdf[['predicted_class', 'geometry']]
    features_dissolved_gdf = features_dissolved_gdf.dissolve(by='predicted_class')
    
    logger.log(f'Saving reduced features to {os.path.join(out_path, "features_dissolved.gpkg")}...')
    features_dissolved_gdf.to_file(os.path.join(out_path, 'features_dissolved.gpkg'), driver='GPKG')




if __name__ == '__main__':
    main()