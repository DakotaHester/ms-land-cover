import os
from zipfile import ZipFile
from subprocess import Popen
from warnings import warn
from ..utils import raise_if_not_exists
import h5py
from .utils import read_images
from tqdm import trange
from concurrent.futures import ThreadPoolExecutor
from functools import partial

# typing
from typing import Iterable, Optional, Literal
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



class LargeRasterDataset:
    """
    Class for creating HDF5 dataset from a large amount of geotiff images.
    
    Parameters
    ----------
    h5_path : str
        Path to the output HDF5 file
    """
    def __init__(self, h5_path: str):
        self.h5_path = h5_path
        self.n_threads = 32
        self.chunk_size = 4096
    
    def create_group(self, name: str, data_paths: Iterable[str], n_threads: int = 4, chunk_size: int = 4096):
        """
        Create a group in the HDF5 file and populate it with image data.
        
        Parameters
        ----------
        name : str
            Name of the group to create
        data_paths : Iterable[str]
            Paths to image files
        n_threads : int, optional
            Number of threads to use for reading images
        chunk_size : int, optional
            Number of images to process in each chunk
        """
        with h5py.File(self.h5_path, 'a') as h5_file:
            # Create or get the group
            if name not in h5_file:
                group = h5_file.create_group(name)
            else:
                group = h5_file[name]
            
            # Process images in chunks
            for i in trange(0, len(data_paths), chunk_size, desc=f'Creating group {name}', unit='images', leave=False, unit_scale=chunk_size):
                chunk_paths = data_paths[i:i+chunk_size]
                chunk_ids = [os.path.basename(path).replace('.tif', '') for path in chunk_paths]
                chunk_images = read_images(chunk_paths, n_threads=n_threads)
                
                for id, img in zip(chunk_ids, chunk_images):
                    group.create_dataset(id, data=img, compression='gzip', chunks=True)
                
                # # Add each image as a dataset in the group
                # func = lambda id, img: group.create_dataset(id, data=img, compression='gzip', chunks=True)
                # with ThreadPoolExecutor(n_threads) as executor:
                #     results = executor.map(func, chunk_ids, chunk_images)
                #     for _ in results:
                #         pass # iterate through results to trigger exceptions if any
                    

    def create_groups_from_folders(self, folder_paths: Iterable[str], n_threads: int = 4, chunk_size: int = 4096):
        """
        Create groups for multiple folders, with each group containing 
        images from that folder.
        
        Parameters
        ----------
        folder_paths : Iterable[str]
            Paths to folders containing .tif images
        n_threads : int, optional
            Number of threads to use for reading images
        chunk_size : int, optional
            Number of images to process in each chunk
        """
        for folder_path in folder_paths:
            # Get all .tif files in the folder
            data_paths = [
                os.path.join(folder_path, f) 
                for f in os.listdir(folder_path) 
                if f.endswith('.tif')
            ]
            # Use folder name as group name
            group_name = os.path.basename(folder_path)
            self.create_group(group_name, data_paths, n_threads=n_threads, chunk_size=chunk_size)