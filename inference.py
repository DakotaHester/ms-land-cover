import torch
import geopandas as gpd
import shapely
import os

from src.mslandcover.inference import RasterProcessor, GPURasterProcessor
from src.mslandcover.models import HRNetSegmentationModel
from src.mslandcover.config import MSTM_PROJ4, HRNET_W18_CONFIG, LEGEND_COLORS_RGBA
from src.mslandcover.utils import load_pth, get_torch_device, Logger


def main():
    
    log_dir = './data/inference_results/test'
    os.makedirs(log_dir, exist_ok=True)
    logger = Logger(os.path.join(log_dir, 'inference.log'))
    
    logger.log('Loading Starkville and Mississippi State boundary...')
    census_ms_places_shp_path = '/Users/dak/Downloads/tl_2024_28_place/tl_2024_28_place.shp'
    ms_places_gdf = gpd.read_file(census_ms_places_shp_path)
    starville_msu_gdf = ms_places_gdf[ms_places_gdf['NAME'].isin(['Mississippi State'])]
    
    starville_msu_reproj_gdf = starville_msu_gdf.to_crs(MSTM_PROJ4)
    try:
        starville_msu_geom = starville_msu_reproj_gdf.union_all()
    except:
        starville_msu_geom = starville_msu_reproj_gdf.unary_union

    # if multipolygon, convert to a list of polygons
    if isinstance(starville_msu_geom, shapely.MultiPolygon):
        starville_msu_geom = [shapely.Polygon(geom.exterior) for geom in starville_msu_geom.geoms]
    else:
        starville_msu_geom = [shapely.Polygon(starville_msu_geom)]
    
    logger.log('Loading model...')
    model = HRNetSegmentationModel(
        config=HRNET_W18_CONFIG,
        num_classes=8,
        img_decoder_head=True,
        use_simple_decoder=False,
        use_se_decoder=True,
        unet_like_decoder=True,
        img_decoder_activation='softmax',
    )
    logger.log('Loading model weights...')
    model.load_state_dict(load_pth('./weights/finetuned_unetlike/hrnet_w18/dae_hsv_simclr/14/best_model.pth', map_location=torch.device('cpu')))
    model.eval()
    model.to(get_torch_device())
    
    logger.log('Starting inference...')
    processor = GPURasterProcessor(
        model=model,
        input_raster_path='/Volumes/dhester_ssd/mslc_inf_test/starkville_msu_2023.tif',
        output_path='./data/inference_results/starkville_msu_2023_LC.tif',
        bounding_polygons=starville_msu_geom,
        tile_size=256,
        stride=64,
        gaussian_sigma=192,
        batch_size=32,
        mean=load_pth('./weights/pretrain_mean.pth'),
        std=load_pth('./weights/pretrain_std.pth'),
        device=get_torch_device(),
        logger=logger,
        colormap=LEGEND_COLORS_RGBA,
        prefetch_factor=8,
    )
    
    processor.process_raster()

if __name__ == '__main__':
    main()