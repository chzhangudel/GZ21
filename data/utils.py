#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 12 17:40:39 2020

@author: arthur
"""
import mlflow
import xarray as xr

def load_data_from_run(run_id):
    mlflow_client = mlflow.tracking.MlflowClient()
    data_file = mlflow_client.download_artifacts(run_id, 'forcing')
    xr_dataset = xr.open_zarr(data_file)
    return xr_dataset

def load_data_from_runs(run_ids):
    xr_datasets = list()
    for run_id in run_ids:
        xr_datasets.append(load_data_from_run(run_id))
    return xr_datasets

def cyclize_dataset(ds: xr.Dataset, coord_name: str, nb_points: int):
    """
    Return a cyclic dataset, with nb_points added on each end, along 
    the coordinate specified by coord_name

    Parameters
    ----------
    ds : xr.Dataset
        Dataset to process.
    coord_name : str
        Name of the coordinate along which the data is made cyclic.
    nb_points : int
        Number of points added on each end.

    Returns
    -------
    New extended dataset.

    """
    left = ds.roll({coord_name: nb_points}, roll_coords=True)
    right = ds.roll({coord_name: nb_points}, roll_coords=True)
    right = right.isel({coord_name: slice(0, 2 * nb_points)})
    left[coord_name] = xr.concat((left[coord_name][:nb_points] - len(left[coord_name]),left[coord_name][nb_points:]), 'x')
    right[coord_name] = xr.concat((right[coord_name][:nb_points], right[coord_name][nb_points:] + len(left[coord_name])), 'x')
    new_ds = xr.concat((left, right), coord_name)
    return new_ds