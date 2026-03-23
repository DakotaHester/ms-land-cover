"""Imagery preprocessing entrypoint.

Pipeline:
1) unzip source NAIP archives
2) convert MrSID -> GeoTIFF
3) resample to 1m
4) reproject to Mississippi Transverse Mercator (EPSG:3813)
"""

from mslandcover.data.preprocessing import preprocess_file
from concurrent.futures import ThreadPoolExecutor
from subprocess import Popen
from glob import glob
from tqdm import tqdm

def main():
    
    # Validate that GDAL tools are available before launching threaded jobs.
    try:
        Popen(['gdalinfo', '--version']).wait()
    except:
        try:
            Popen(['conda', 'activate', 'gdal']).wait()
        except:
            raise RuntimeError('GDAL is not active or installed.')
    
    zip_files_dir = './data/MS_NAIP_2023'
    zip_files_list = glob(f'{zip_files_dir}/*.zip')
    
    n_files = len(zip_files_list)
    # n_threads = os.cpu_count()
    n_threads = 4
    
    print(f'Processing {n_files} files with {n_threads} threads.')
    # Fan out file-level preprocessing work across threads for throughput.
    with tqdm(total=n_files, desc='files processed', unit='file') as pbar:
        with ThreadPoolExecutor(max_workers=n_threads) as executor:
            # print('Processing files...')
            executor.map(
                lambda x: preprocess_file(x, pbar=pbar), 
                zip_files_list
            )

if __name__ == '__main__':
    main()