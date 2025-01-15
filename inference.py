from threading import Lock
import torch
import geopandas as gpd
import shapely
import rasterio as rio
from rasterio.mask import geometry_mask, mask
from rasterio.features import shapes
from rasterio.io import MemoryFile
from multiprocessing import cpu_count
import os
import numpy as np
import cv2 as cv
from concurrent.futures import ThreadPoolExecutor

from tqdm import tqdm


from src.mslandcover.inference import GPURasterProcessor, extract_zonal_mean
from src.mslandcover.models import HRNetSegmentationModel
from src.mslandcover.config import MSTM_PROJ4, HRNET_W18_CONFIG, LEGEND_COLORS_RGBA, LEGEND_CLASSES
from src.mslandcover.utils import load_pth, get_torch_device, Logger


def main():
    
    log_dir = './data/inference_results/test'
    os.makedirs(log_dir, exist_ok=True)
    logger = Logger(os.path.join(log_dir, 'inference.log'))
    
    logger.log('Loading Starkville and Mississippi State boundary...')
    census_ms_places_shp_path = './data/shapefiles/tl_2024_28_place/tl_2024_28_place.shp'
    ms_places_gdf = gpd.read_file(census_ms_places_shp_path)
    starville_msu_gdf = ms_places_gdf[ms_places_gdf['NAME'].isin(['Starkville', 'Mississippi State'])]
    
    starville_msu_reproj_gdf = starville_msu_gdf.to_crs(MSTM_PROJ4)
    try:
        starville_msu_geom = starville_msu_reproj_gdf.union_all()
    except:
        starville_msu_geom = starville_msu_reproj_gdf.unary_union

    # if multipolygon, convert to a list of polygons
    if isinstance(starville_msu_geom, shapely.MultiPolygon):
        bounding_polygons = [shapely.Polygon(geom.exterior) for geom in starville_msu_geom.geoms]
    else:
        bounding_polygons = [shapely.Polygon(starville_msu_geom)]
    
    logger.log('Loading model...')
    model = HRNetSegmentationModel(
        config=HRNET_W18_CONFIG,
        num_classes=8,
        img_decoder_head=True,
        use_simple_decoder=True,
        use_se_decoder=False,
        unet_like_decoder=False,
        img_decoder_activation='softmax',
    )
    logger.log('Loading model weights...')
    model.load_state_dict(load_pth('./weights/finetuned/hrnet_w18/dae/14/best_model.pth', map_location=torch.device('cpu')))
    model.eval()
    model.to(get_torch_device())
    
    logger.log('Starting inference...')
    processor = GPURasterProcessor(
        model=model,
        tile_size=256,
        stride=128,
        gaussian_sigma=192,
        batch_size=32,
        mean=load_pth('./weights/pretrain_mean.pth'),
        std=load_pth('./weights/pretrain_std.pth'),
        device=get_torch_device(),
    )
    
    logger.log('Loading raster data...')
    raster_path = '/Volumes/dhester_ssd/mslc_inf_test/starkville_msu_2023_reduced.tif'
    with rio.open(raster_path) as src:
        profile = src.profile
        raster_data = src.read()
    
    
    
    logger.log('Processing raster data...')
    lc_probs = processor.process_raster(raster_data)
    
    out_path = './data/inference_results/test/starkville_msu_2023_reduced'
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
    
    # now, object-based segmentation using mean shift
    logger.log('Segmenting mean shift image...')
    segmented_image = cv.pyrMeanShiftFiltering(raster_data.transpose(1, 2, 0), 10, 10)
    segmented_image_reshaped = segmented_image.reshape(-1, 3)
    unique_values, segments = np.unique(segmented_image_reshaped, axis=0, return_inverse=True)
    segments = segments.reshape(segmented_image.shape[:2]).astype(np.uint16)
    print(segmented_image.shape, len(unique_values), segments.shape)
    
    logger.log('Generating polygons from segments...')
    bounding_mask = geometry_mask(bounding_polygons, out_shape=segments.shape, transform=profile['transform'], all_touched=True, invert=True)
    geoms = [geom for geom, _ in shapes(segments, mask=bounding_mask, connectivity=4, transform=profile['transform'])]
    
    # use zonal statistics to get mean probabilities for each segment
    logger.log('Extracting zonal land cover probability means...')
    lock = Lock()
    n_cpus = cpu_count()
    with MemoryFile(open(os.path.join(out_path, 'lc_probs.tif'), 'rb')) as memfile:
        with rio.open(memfile) as src:
            with ThreadPoolExecutor(4 * n_cpus) as executor:
                class_means = []
                results = executor.map(
                    lambda geom: extract_zonal_mean(geom, src, lock),
                    geoms,
                )
                with tqdm(total=len(geoms), desc='Extracting zonal land cover probability means', unit='geoms') as pbar:
                    for res in results:
                        class_means.append(res)
                        pbar.update(1)
                        # for j in range(class_mean.shape[0]):
                        #     class_label = LEGEND_CLASSES[j + 1]
                        #     features_gdf[class_label].iloc[i] = class_mean[j]
    
    logger.log('Compiling to feature class')
    # class_mean_shape = (N_geoms, N_classes)
    class_means = np.array(list(class_means)).T
    print(class_means.shape)
    features_dict = {'geometry': geoms}
    for i in range(class_means.shape[0]):
        class_label = LEGEND_CLASSES[i + 1]
        features_dict[class_label] = class_means[i]
    
    features_dict['predicted_class'] = [LEGEND_CLASSES[pred] for pred in np.argmax(class_means, axis=0) + 1]
    features_dict['confidence'] = np.max(class_means, axis=0)
    
    for key, val in features_dict.items():
        print(key, len(val), val[0])
    features_gdf = gpd.GeoDataFrame(features_dict, crs=profile['crs'])
    
    logger.log(f'Saving features to {os.path.join(out_path, "features.gpkg")}...')
    features_gdf.to_file(os.path.join(out_path, 'features.gpkg'), driver='GPKG')
    
    
    
    
    
    
    # final_outputs = final_outputs.cpu().numpy()
    # final_outputs = final_outputs[:, self.tile_size:-self.tile_size, self.tile_size:-self.tile_size]
    # image_segments = mean_shift(self.raster_data.transpose(1, 2, 0)[self.tile_size:-self.tile_size, self.tile_size:-self.tile_size, :] * self.std + self.mean)
    # segments_polys = segments_to_polygons(image_segments, self.profile['transform'])
    # features_gdf = create_geodataframe(segments_polys, final_outputs, image_segments, self.profile['crs'])
    # features_gdf.to_file(self.output_path.replace('.tif', '.gpkg'), driver='GPKG')
    
    # if self.logger:
    #     self.logger.log('Saving results...')
    
    # # Save results
    # profile = self.profile.copy()
    # profile.update(count=1, dtype=rasterio.uint8)
    
    # if self.bounding_polygons:
    #     if self.logger:
    #         self.logger.log('Masking predictions by bounding polygons...')
    #     geom_mask = geometry_mask(
    #         self.bounding_polygons, 
    #         out_shape=predictions.shape,
    #         transform=profile['transform'],
    #         all_touched=True
    #     )
    #     predictions[geom_mask] = 0
    
    # if self.logger:
    #     self.logger.log(f'Writing results to {self.output_path}...')
    # with rasterio.open(self.output_path, 'w', **profile) as dst:
    #     dst.write(predictions, 1)
    #     if self.colormap:
    #         dst.write_colormap(1, self.colormap)
    
    # if self.bounding_polygons:
    #     if self.logger:
    #         self.logger.log('Clipping raster...')
    #     with rasterio.open(self.output_path) as src:
    #         clipped_data, clipped_transform = mask(src, self.bounding_polygons, crop=True)
    #         profile = src.profile.copy()
        
    #     profile.update({
    #         'transform': clipped_transform,
    #         'height': clipped_data.shape[1],
    #         'width': clipped_data.shape[2]
    #     })
        
    #     if self.logger:
    #         self.logger.log(f'Writing clipped results to {self.output_path}...')
    #     with rasterio.open(self.output_path, 'w', **profile) as dst:
    #         dst.write(clipped_data)
    #         if self.colormap:
    #             dst.write_colormap(1, self.colormap)

if __name__ == '__main__':
    main()