# CREATED: 10-JUL-2023
# LAST EDIT: 9-AUG-2023
# AUTHORS: DUANE RINEHART, MBA (drinehart@ucsd.edu), KUI QIAN
# CREATES OME-Zarr FORMAT (FOLDER) FROM SINGLE CHANNEL BRAIN IMAGE STACK (ALONG z-AXIS)
# INCLUDES MIP GENERATION (GAUSSIAN OR LAPLACIAN)

# TODO (9-AUG-2023):
#     -FIND BETTER WAY TO LIMIT RAM USAGE DURING CREATION (CURRENTLY AROUND 750GB FOR DK55 STACK - 100MB chunks, 900GB w/ 25MB chunks)
#     -ADD META-DATA USING PROVENANCE PACKAGE FROM PRECOMPUTED LAYER + META-DATA EXTRACTION FROM CZI FILES (INSTRUMENTATION)


import os, time
from pathlib import Path
import glob
from datetime import datetime, timedelta
import tifffile
from imagecodecs import tiff_decode
import dask
import dask.array as da
from dask import delayed
import dask.distributed as dd
from dask.distributed import Client, LocalCluster
from dask.diagnostics import ProgressBar
import zarr
import ome_zarr
import ome_zarr.scale
import ome_zarr.writer
from ome_zarr.writer import write_image, write_multiscale
from aicsimageio import AICSImage
from aicsimageio.writers import OmeZarrWriter
import numpy as np


##########################################################
input_tiff_path = '/data_in/DK55/preps/CH1/full_aligned'
output_dir = '/data_out/pipeline_data/test'
scratch_dir = '/scratch'
output_zarr_folder = 'ome-test.zarr'

dims_order = "XYZ"  # ONLY EXTRACT Y,X DIMENSIONS FOR SINGLE FILE, SECTION IS FILENAME
transformation_method = 'gaussian' #options 'gaussian', or if not defined 'laplacian'
n_levels = 4
units = {'x': 'nanometer', 'y': 'nanometer', 'z': 'micrometer'}
resolution = {'x': 325, 'y': 325, 'z': 20}
max_client_ram = "32GB"  # e.g. "32GB"
n_workers = 1
# META-DATA (PULL FROM PRECOMPUTED CLOUD VOLUME: provenance)
perf_lab = 'Performance Lab: UCSD'
description = 'DK55 brain stack'
name='deadmouse'
##########################################################
print('USING PARAMETERS:')
print(f'SRC IMAGE PATH: {input_tiff_path}')
print(f'DEST ZARR PATH: {Path(output_dir,output_zarr_folder)}')
print(f'DIMENSIONS ORDER: {dims_order}')
##########################################################


def sizeof_fmt(num, suffix="B"):
    #ref: https://stackoverflow.com/questions/1094841/get-human-readable-version-of-file-size
    for unit in ("", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei", "Zi"):
        if abs(num) < 1024.0:
            return f"{num:3.1f}{unit}{suffix}"
        num /= 1024.0
    return f"{num:.1f}Yi{suffix}"


def get_meta_data(axes, transformations, transformation_method: str, description, perf_lab):
    # META-DATA COMPILATION ('provenance' FROM precomputed FORMAT)

    #TODO: write_multiscales_metadata() from ome_zarr?
    meta_datasets = {'coordinateTransformations': transformations}

    now = datetime.now()  # datetime object containing current date and time
    dt_string = now.strftime("%m-%d-%Y %H:%M:%S")
    processing_meta = {'description': description, 'method': transformation_method, 'owners': perf_lab,
            'processing': {'date': dt_string}, 'kwargs': {'multichannel': True}}

    meta_multiscales = {'axes': axes, 'datasets': meta_datasets, 'metadata': processing_meta}
    return meta_multiscales


def get_storage_opts(axis_names: tuple[str]) -> dict:
    '''
    CALCULATES OPTIMAL CHUNK SIZE FOR IMAGE STACK (TARGET IS ~25MB EACH)
    N.B. CHUNK DIMENSION ORDER (XYZ) SHOULD CORRESPOND TO DASK DIMENSION ORDER (XYZ)
    
    ref: https://forum.image.sc/t/deciding-on-optimal-chunk-size/63023/7

    :param axis_names: tuple[str]
    :return: dict
    '''

    z_section_chunk = 20
    byte_per_pixel = 2
    target_chunk_size_mb = 25
    chunk_dim = (target_chunk_size_mb*10**6 / byte_per_pixel / z_section_chunk)**(1/2) #1MB / BYTES PER PIXEL / kui_constant, SPLIT (SQUARE ROOT) BETWEEN LAST 2 DIMENSIONS
    
    if len(axis_names) > 2:
        new_chunks = (int(chunk_dim), int(chunk_dim), z_section_chunk)
        print(F'EACH CHUNK MEM SIZE: {sizeof_fmt(new_chunks[0] * new_chunks[1] * new_chunks[2] * byte_per_pixel)}')
    else:
        new_chunks = (int(chunk_dim), int(chunk_dim))    
        print(F'EACH CHUNK MEM SIZE: {sizeof_fmt(new_chunks[0] * new_chunks[1] * byte_per_pixel)}')

    return {"chunks": new_chunks}


def get_axes_and_transformations(mip, axis_names, units, resolution, n_levels) -> tuple[dict,dict]:
    '''
    GENERATES META-INFO FOR PYRAMID

    :param mip:
    :param axis_names:
    :param units:
    :param resolution:
    :param n_levels:
    :return: tuple[dict,dict]
    '''
    axes = []
    # units = {k.lower():v for k,v in units.items()} #FORCE LOWER-CASE KEYS
    resolution = {k.lower():v for k,v in resolution.items()} #FORCE LOWER-CASE KEYS
    for ax in axis_names:
        axis = {"name": ax, "type": "channel" if ax == "C" else "time" if ax == "T" else "space"}
        unit = units.get(ax, None)
        if unit is not None:
            axis["unit"] = unit
        axes.append(axis)

    is_scaled = {"t": False, "c": False, "z": False, "y": True, "x": True}

    transformations = []
    for scale_level in range(n_levels + 1):
        scale = []
        for ax in axis_names:
            if is_scaled[ax] and ax in resolution:
                scale.append(resolution[ax] / 2**scale_level)
            else:
                scale.append(resolution.get(ax, 1))
        transformations.append([{"scale": scale, "type": "scale"}])
    return axes, transformations


def load_image(path, dims_order):
    #ONLY EXTRACT DIMENSIONS OF INTEREST - ORG. IMAGE DIMENSIONS ORDERING: TCZYX
    aics_image = AICSImage(path)
    return aics_image.get_image_data(dims_order) #ONLY YZ DIMS


def get_dask_image_stack(scratch_dir, max_client_ram, storage_opts, dims_order):
    base_path_with_pattern = str(input_tiff_path) + '/*.tif'
    image_paths = [files for files in sorted(glob.glob(base_path_with_pattern, recursive=False))]

    #ALL IMAGES IN STACK SHOULD HAVE SAME DIMENSIONS (XY) - GET SAMPLE FROM FIRST IMAGE
    img = AICSImage(image_paths[0])
    y_dim = img.dims.X
    x_dim = img.dims.Y

    def imread(fname):
        with open(fname, 'rb') as fh:
            data = fh.read()
        return tiff_decode(data)

    with tifffile.imread(base_path_with_pattern, aszarr=True, imread=imread) as store:
        da_stack = dask.array.from_zarr(store)

    if da_stack.shape[1] == x_dim and da_stack.shape[2] == y_dim:
        #ORG. STACK HAS ZYX ORDERING (MUST BE RE-ORDERED TO XYZ)
        da_stack_revised = np.transpose(da_stack, (1, 2, 0))
    else:
        da_stack_revised = da_stack #NO TRANSPOSE

    new_chunk_size = (storage_opts['chunks'])
    da_stack_revised = da_stack_revised.rechunk(new_chunk_size) #USE SMALLER CHUNKS [THAT FIT IN RAM]

    #DEBUG:
    #print(f'x_dim: {x_dim}, y_dim: {y_dim}')
    #print(f'ORG STACK SHAPE: {da_stack.shape}')
    #print(f'REVISED STACK SHAPE: {da_stack_revised, type(da_stack_revised)}')

    return da_stack_revised


def main():
    axis_names = tuple(dims_order.lower())
    storage_opts = get_storage_opts(axis_names)
    
    da_stack = get_dask_image_stack(scratch_dir, max_client_ram, storage_opts, dims_order)

    print(f'INPUT IMAGE INFO: {da_stack.shape} ({da_stack})')

    axes, transformations = get_axes_and_transformations(da_stack, axis_names, units, resolution, n_levels)

    print(f'axes count: {len(axes)}')
    print(f'dimensions count: {len(transformations)}')
    print(f'transformations: {transformations}')
    print(f'axes: {axes}')

    meta = get_meta_data(axes, transformations, transformation_method, description, perf_lab)
    zgroup = zarr.open(Path(output_dir, output_zarr_folder), mode='w')
    write_image(image=da_stack, group=zgroup,
                axes=axes, storage_options=storage_opts, compute=True, metadata=meta
                )


if __name__ == "__main__":
    start = time.perf_counter()
    main()
    finish = time.perf_counter()
    exec_time = timedelta(seconds=(finish - start))
    print(f"EXECUTION TIME: {round(finish - start, 2)}s ({exec_time})")