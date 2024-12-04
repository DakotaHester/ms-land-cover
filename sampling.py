import os
import geopandas as gpd
import pandas as pd
import numpy as np
from glob import glob
import rasterio as rio
from rasterio.mask import mask
from rasterio.io import MemoryFile
from rasterio.plot import show
from PIL import Image
import json
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from threading import Lock
from src.mslandcover.config import MSTM_PROJ4, LEGEND_CLASSES
from src.mslandcover.utils import raise_if_not_exists
from src.mslandcover.data.preprocessing import LargeRasterDataset
import cv2 as cv
import matplotlib.pyplot as plt
from multiprocessing import Pool
from tqdm import tqdm

from sklearn.preprocessing import RobustScaler
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from os import cpu_count

if __name__ == '__main__':
    
    if False: # TODO: FINISH ME!!! Have most of the code in sampling.ipynb under old_notebooks, will complete later
        # load the shapefilse with boundaries of the rasters
        raster_boundaries_path = './data/sampling/regions_boundaries.gpkg'
        if not os.path.exists(raster_boundaries_path):
            shapefiles = glob(r'Z:\\guser\\dh\\NAIP_MS_2023\\*\\*.shp')
            gdfs = []
            for shapefile in shapefiles:
                gdf = gpd.read_file(shapefile).to_crs(MSTM_PROJ4) # convert to the same projection
                raster_path = glob(os.path.join(os.path.dirname(shapefile), '*_1m.tif'))[0]
                gdf['raster_path'] = raster_path
                gdfs.append(gdf)

            raster_boundaries_gdf = gpd.GeoDataFrame(pd.concat(gdfs))[['raster_path', 'geometry']] # only keep the relevant columns
            raster_boundaries_gdf = raster_boundaries_gdf.dissolve(by='raster_path').reset_index() # dissolve the geometries to get the boundaries of the raster
            raster_boundaries_gdf.to_file('data/sampling/regions_boundaries.gpkg', driver='GPKG')
        
        
        raster_boundaries_gdf = gpd.read_file('data/sampling/regions_boundaries.gpkg')
        # load the samples parquet file (produced from stratification process)
        samples = gpd.read_parquet('./data/sampling/samples.par')

        # spatial join with the raster boundaries
        raster_boundaries_gdf['raster_geometry'] = raster_boundaries_gdf['geometry'] # make a copy of the geometry
        samples = gpd.sjoin(samples, raster_boundaries_gdf, predicate='intersects', how='left')
    
    pretrain_h5_path = '/scratch/dhester/mslc/pretrain.hdf5'
    pretrain_h5_dataset = LargeRasterDataset(pretrain_h5_path)
    pretrain_h5_dataset.create_groups_from_folders(
        ['/scratch/dhester/mslc/pretrain', '/scratch/dhester/mslc/pretrain_val'],
        n_threads=32, chunk_size=4096,
    )