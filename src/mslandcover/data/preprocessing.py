from contextlib import nullcontext
import os
from zipfile import ZipFile
from subprocess import Popen
from warnings import warn
from rasterio import mask
import rasterio as rio
import numpy as np
import pandas as pd
from threading import Lock
from time import sleep

from ..utils import raise_if_not_exists

# typing
from typing import Optional, Literal
from tqdm.std import tqdm



def unzip(zip_path: str, dest_path: Optional[str]=None) -> None:
    '''
    Unzip a file to a destination path. If no destination path is provided, the
    contents of the zip file will be placed in a folder with the same name as
    the zip file in the same directory as the zip file, minus the '.zip' 
    extension.
    
    Parameters
    ----------
    zip_path : str
        The path to the zip file to be unzipped.
    dest_path : str, optional
        The path to the directory where the contents of the zip file will be
        extracted. If not provided, the contents will be extracted to a folder
        with the same name as the zip file in the same directory as the zip file,
        minus the '.zip' extension.
    
    Returns
    -------
    None
    '''
    
    raise_if_not_exists(zip_path)
    if dest_path is None:
        dest_path = zip_path.replace('.zip', '')
    with ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(dest_path)



def mrsid_to_tiff(mrsid_path: str, tiff_path: Optional[str]=None) -> None:
    '''Convert a .sid file to a .tif file using the mrsiddecode command line tool.
    
    Use the mrsiddecode command line tool to convert a .sid file to a .tif file.
    
    Parameters
    ----------
    mrsid_path : str
        The path to the input .sid file.
    tiff_path : Optional[str], optional
        The path to the output .tif file. If not provided, the output file will
        be saved in the same directory as the input file with the .tif extension.
    
    Returns
    -------
    None
    
    Raises
    ------
    ValueError
        If the input file is not a .sid file.
    '''

    raise_if_not_exists(mrsid_path)
    
    if mrsid_path.split('.')[-1] != 'sid':
        raise ValueError('Input file must be a .sid file')
    
    if tiff_path is None:
        tiff_path = mrsid_path.replace('.sid', '.tif')
    
    # 'gdal_translate -of GTiff {mrsid_path} {tiff_path}')
    Popen([
        'mrsiddecode', 
        '-i', mrsid_path, 
        '-o', tiff_path, 
        '-of', 'tifg', 
        '-quiet'
    ]).wait()



def resample(
    input_path: str,
    output_path: Optional[str]=None, 
    target_resolution: int=1, 
    resampling: Literal[
        'bilinear', 
        'cubic', 
        'cubicspline', 
        'lanczos', 
        'average', 
        'rms', 
        'mode'
    ]='bilinear',
    crs: str='+proj=tmerc +lat_0=32.5 +lon_0=-89.75 +k=0.9998335 +x_0=500000 +y_0=1300000', # default to mississippi transverse mercator
    pbar: Optional[tqdm]=None,
) -> None:
    '''Resample a tiff file to a target resolution using GDAL.
    
    Use the GDAL command line tool `gdalwarp` to resample a tiff file to a
    target resolution. The resampling method can be specified using the
    `resampling` parameter.
    
    Parameters
    ----------
    input_path : str
        The path to the input tiff file.
    output_path : Optional[str], optional
        The path to the output tiff file. If not provided, the output file will
        be saved in the same directory as the input file with the target
        resolution appended to the filename.
    target_resolution : int, optional
        The target resolution of the output tiff file in meters. The default is 1.
    resampling : {'bilinear', 'cubic', 'cubicspline', 'lanczos', 'average', 'rms', 'mode'}, optional
        The resampling method to use. The default is 'bilinear'.
    pbar : Optional[tqdm], optional
        A tqdm progress bar to update when the function completes. The default is None.
    
    Returns
    -------
    None
    
    Raises
    ------
    ValueError
        If the input file is not a .tif or .tiff file.
        If the output file is not a .tif or .tiff file.
        
    
    '''
    
    try:
        
        if input_path.split('.')[-1] != 'tif' and input_path.split('.')[-1] != 'tiff':
            # print(input_path.split('.')[-1])
            # print(input_path.split('.')[-1] == 'tif')
            raise ValueError('Input file must be a .tif or .tiff file, got: ' + input_path)
        
        if output_path is None:
            output_path = input_path.replace('.tif', f'_{target_resolution}m.tif')
        
        if output_path.split('.')[-1] != 'tif' and output_path.split('.')[-1] != 'tiff':
            raise ValueError('Output file must be a .tif or .tiff file')

        # print(input_path)
        # print(f'DOES IT EXIST? {os.path.exists(input_path)}')  
        Popen([
            'gdalwarp',
            '-quiet',
            '-tr', str(target_resolution), str(target_resolution),
            '-r', resampling,
            '-t_srs', crs,
            input_path,
            output_path,
            '-co', 'COMPRESS=LZW',
            '-co', 'TILED=YES',
            '-co', 'BIGTIFF=YES',
            '-co', 'BLOCKXSIZE=256',
            '-co', 'BLOCKYSIZE=256',
            '-wo', 'NUM_THREADS=ALL_CPUS',
        ]).wait()

    except Exception as e:
        warn(f'Error resampling tiff file: {type(e)}: {e}', UserWarning)
        
    finally:
        if pbar:
            pbar.update(1)


def preprocess_file(
    zip_path: str, 
    target_resolution: int=1, 
    resampling: Literal[
        'bilinear', 
        'cubic', 
        'cubicspline', 
        'lanczos', 
        'average', 
        'rms', 
        'mode'
    ]='bilinear',
    pbar: Optional[tqdm]=None,
) -> None:
    '''Preprocess a file by unzipping, converting to tiff, and resampling.
    
    Preprocess a file in a zip archive by unzipping the archive, converting any
    .sid files to .tif files, and resampling the .tif files to a target
    resolution.
    
    Parameters
    ----------
    zip_path : str
        The path to the zip archive containing the file to preprocess.
    target_resolution : int, optional
        The target resolution of the output tiff file in meters. The default is 1.
    resampling : {'bilinear', 'cubic', 'cubicspline', 'lanczos', 'average', 'rms', 'mode'}, optional
        The resampling method to use. The default is 'bilinear'.
    
    Returns
    -------
    None
    '''
    
    raise_if_not_exists(zip_path)

    # convert to absolute path
    zip_path = os.path.abspath(zip_path)

    # define file names    
    file_name = os.path.basename(zip_path).replace('.zip', '')
    folder_dir = os.path.join(os.path.dirname(zip_path), file_name)
    mrsid_path = os.path.join(folder_dir, file_name + '.sid')
    tiff_path = mrsid_path.replace('.sid', '.tif')
    
    if not os.path.exists(folder_dir):
        unzip(zip_path)
        os.remove(zip_path)
    
    if not os.path.exists(tiff_path):
        mrsid_to_tiff(mrsid_path)
    
    if not os.path.exists(tiff_path.replace('.tif', f'_{target_resolution}m.tif')):
        resample(tiff_path, target_resolution=target_resolution, resampling=resampling)
        os.remove(tiff_path)
    
    if pbar:
        pbar.update(1)



def extract_mask(
    sample: pd.Series, 
    raster_dataset: rio.DatasetReader, 
    lock: Optional[Lock]=None, 
    pbar: Optional[tqdm]=None,
) -> None:
    """
    Extracts a mask from a raster dataset for a given sample.

    Parameters
    ----------
    sample : pd.Series
        A pandas Series containing the sample data. Should have the following
        columns:
            - 'geometry': the geometry of the sample
            - 'split': the split of the sample.
    raster_dataset : rio.io.DatasetReader
        A rasterio dataset reader object for the raster file. Should be 3-band imagery.
    lock : Optional[Lock], optional
        A threading lock to ensure thread-safe operations, by default None.
    pbar : Optional[tqdm], optional
        A tqdm progress bar object to update progress, by default None.

    Returns
    -------
    None
    """
    
    max_tries = 5
    for i in range(max_tries):
        try:
            out_image, out_transform = mask(raster_dataset, [sample['geometry']], crop=True, all_touched=True)
                
        except Exception as e:
            print(f'EXCEPTION: type({e}) raised while extracting mask for sample {sample.name} in split {sample['split']}: {e}')
            print(f'Continuing to the next sample...')
            if pbar is not None:
                pbar.update(1)
            return
        
        except Warning: # catch warnings and retry just in case
            print(f'WARNING: type({e}) raised while extracting mask for sample {sample.name} in split {sample['split']}')
            if i < max_tries:
                print(f'Retrying... ({i+1}/{max_tries})')
                sleep(1)
                continue
            
            else:
                print(f'ERROR: Failed after {max_tries} tries, continuing to the next sample...')
                if pbar is not None:
                    pbar.update(1)
                return
    
    out_meta = raster_dataset.meta.copy()

    if out_image.shape[1] > 256 or out_image.shape[2] > 256:
        # crop the image to 256x256
        out_image = out_image[:, :256, :256]

    filename = str(sample.name)
    if out_image.shape != (3, 256, 256):\
        return
    
    # image nodata values are set to 0, even though 0 is a valid value for the image
    # in order to check, we assume that any image that has at least 5% of pixels 
    # where all three bands are set to the nodata value is invalid 
    nd_values = np.array([raster_dataset.nodata] * 3)
    pixels_with_nd = np.equal(out_image.transpose(1, 2, 0).reshape(-1, 3), nd_values).all(axis=1)
    if pixels_with_nd.sum() > 0.05 * len(pixels_with_nd):
        return
    
    out_meta.update({
        'driver': 'GTiff',
        'height': out_image.shape[1],
        'width': out_image.shape[2],
        'transform': out_transform,
    })
    
    # need to segment and convert imgery to polygons for annotation later,
    # having Null nodata values makes this process easier
    if sample['split'] in ('train', 'val', 'test'):
        out_meta['nodata'] = None
    
    out_path = os.path.join('data', 'splits', sample['split'], filename + '.tif')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    
    if lock is None: # if no lock is provided, create an empty context manager 
        lock = nullcontext()
    
    with lock:
        if os.path.exists(out_path):
            os.remove(out_path) # remove the file if it already exists - this is a workaround for a bug in rasterio
            
        with rio.open(out_path, 'w', **out_meta) as dst:
            dst.write(out_image)
    
    if pbar is not None:
        pbar.update(1)



def extract_masks_from_raster(
    samples_group: pd.DataFrame, 
    lock: Optional[Lock]=None, 
    pbar: Optional[tqdm]=None,
) -> None:
    """
    Extracts masks from a raster dataset for a group of samples.

    Parameters
    ----------
    samples_group : pd.DataFrame
        A pandas DataFrame containing the group of samples. Should have the following 
        columns:
            - 'geometry': the geometry of the sample
            - 'split': the split of the sample
            - 'raster_path': the path to the raster file
    lock : Optional[Lock], optional
        A threading lock to ensure thread-safe operations, by default None.
    pbar : Optional[tqdm], optional
        A tqdm progress bar object to update progress, by default None.

    Returns
    -------
    None
    """
    raster_path = samples_group[0]
    with rio.open(raster_path) as raster_dataset:
        samples_group[1].apply(lambda x: extract_mask(x, raster_dataset, lock=lock, pbar=pbar), axis=1)
