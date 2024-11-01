from src.mslandcover.data.preprocessing import preprocess_file
from concurrent.futures import ThreadPoolExecutor
import os
from glob import glob
from tqdm import tqdm

def main():
    
    zip_files_dir = './data/MS_NAIP_2023'
    zip_files_list = glob(f'{zip_files_dir}/*.zip')
    
    n_files = len(zip_files_list)
    n_cpus = os.cpu_count()
    
    print(f'Processing {n_files} files with {n_cpus} CPUs.')
    with tqdm(total=n_files, desc='files processed', unit='file') as pbar:
        with ThreadPoolExecutor(max_workers=n_cpus) as executor:
            files = executor.map(preprocess_file, zip_files_list)

if __name__ == '__main__':
    main()