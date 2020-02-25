#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 12:15:35 2020

@author: arthur
"""

import xarray as xr
import dask
import dask.array as da
import dask.bag as db
from scipy.ndimage import gaussian_filter
import numpy as np

def advections(u_v_dataset):
    """Computes advection terms"""
    gradient_x = u_v_dataset.differentiate('x')
    gradient_y = u_v_dataset.differentiate('y')
    u, v = u_v_dataset['usurf'], u_v_dataset['vsurf']
    adv_x = u * gradient_x['usurf'] + v * gradient_y['usurf']
    adv_y = v * gradient_x['vsurf'] + v * gradient_y['vsurf']
    return xr.Dataset({'adv_x': adv_x, 'adv_y' :adv_y})


# def spatial_filter(data, scale):
#     print('scale', scale)
#     result = None
#     for time in range(data.shape[0]):
#         print(time)
#         gf = dask.delayed(gaussian_filter)(data[time, ...], scale)
#         if result is None:
#             result = dask.array.from_delayed(gf, shape=data.shape[1:],
#                                              dtype=float)
#         else:
#             gf = dask.array.from_delayed(gf, shape = data.shape[1:], 
#                                          dtype=float)
#             result = dask.array.concatenate((result, gf))
#     return result

def spatial_filter(data, sigma):
    result = np.zeros_like(data)
    for t in range(data.shape[0]):
        result[t, ...] = gaussian_filter(data[t, ...], sigma,
                                         mode='constant')
    return result

def spatial_filter_dataset(dataset, sigma: float):
    """Applies spatial filtering to the dataset across the spatial dimensions
    """
    return xr.apply_ufunc(lambda x: spatial_filter(x, sigma), dataset, 
                                  dask='parallelized', 
                                  output_dtypes=[float,])

def compute_grid_steps(u_v_dataset):
    """Computes the grid steps for the (x,y) grid"""
    grid_step = [0, 0]
    steps_x = u_v_dataset.coords['x'].diff('x')
    steps_y = u_v_dataset.coords['y'].diff('y')
    grid_step[0] = abs(steps_x.mean().item())
    grid_step[1] = abs(steps_y.mean().item())
    return tuple(grid_step)


def eddy_forcing(u_v_dataset, scale: float, method='mean'):
    """Computes the eddy forcing terms on high resolution"""
    # High res advection terms
    adv = advections(u_v_dataset)
    # Grid steps
    grid_steps = compute_grid_steps(u_v_dataset)
    # Filtered u,v field
    u_v_filtered = spatial_filter_dataset(u_v_dataset, 
                                          (scale / grid_steps[0],
                                          scale / grid_steps[1]))
    # Advection term from filtered
    adv_filtered = advections(u_v_filtered)
    # Forcing
    forcing = adv_filtered - adv
    forcing = forcing.rename({'adv_x' : 'S_x', 'adv_y' : 'S_y'})
    # Merge filtered u,v and forcing terms
    forcing = forcing.merge(u_v_filtered)
    # Coarsen
    forcing = forcing.coarsen({'x' : int(scale / grid_steps[0]),
                            'y' : int(scale / grid_steps[1])},
                            boundary='trim')
    if method == 'mean':
        forcing = forcing.mean()
    else:
        raise('Passed method does not correspond to anything.')
    return forcing

if __name__ == '__main__':
    test = da.random.randint(0, 10, (200, 20, 20), chunks = (1, 20, 20))
    filtered = spatial_filter(test, 2)