import numpy as np
import dask
import dask.array as da
from dask.distributed import Client
import rasterio as rio
from glob import glob
import torch

def main():
    client = Client(n_workers=16, memory_limit='4GB')
    print(client)

    filenames = glob('/scratch/dhester/mslc_data_v2/pretrain/*.tif')

    # Create a delayed function to read each file
    @dask.delayed
    def read_image(filename):
        with rio.open(filename) as src:
            # Read the data and append an extra dimension
            return src.read()[np.newaxis, ...] / 255.0

    # determine samples shapes
    with rio.open(filenames[0]) as src:
        sample = src.read() / 255.0
    sample_shape = sample.shape
    sample_dtype = sample.dtype

    lazy_arrays = [read_image(filename) for filename in filenames]

    # each file becomes one chunk
    arrays = [
        da.from_delayed(
            lazy_array, 
            shape=(1, *sample_shape), 
            dtype=sample_dtype,
        ) 
        for lazy_array in lazy_arrays
    ]

    # Stack all arrays into a single array along the first axis
    data_da = da.concatenate(arrays, axis=0).rechunk()
    print(data_da)

    mean = data_da.mean(axis=(0, 2, 3)).compute()
    std = data_da.std(axis=(0, 2, 3)).compute()
    
    print("Mean: ", mean)
    print("Std: ", std)

    mean_tensor = torch.tensor(mean, dtype=torch.float32)
    std_tensor = torch.tensor(std, dtype=torch.float32)

    torch.save(mean_tensor, r"./weights/pretrain_mean_4.pt")
    torch.save(std_tensor, r"./weights/pretrain_std_4.pt")



if __name__ == "__main__":
    main()