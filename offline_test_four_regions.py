import mlflow
import xarray as xr
import matplotlib.pyplot as plt
from data.utils import load_training_datasets
import os
import xarray as xr
import numpy as np
import math

raw_data = xr.open_zarr('/scratch/gpfs/cz3321/CM2P6/forcing.zarr')
raw_datasets = load_training_datasets(raw_data, 'training_subdomains.yaml')

low_rez = raw_datasets[0]
u = low_rez['usurf']
v = low_rez['vsurf']

import torch
import importlib
#load the neural network
def load_model_cls(model_module_name: str, model_cls_name: str):
    try:
        module = importlib.import_module(model_module_name)
        model_cls = getattr(module, model_cls_name)
    except ModuleNotFoundError as e:
        raise type(e)('Could not retrieve the module in which the trained model \
                      is defined: ' + str(e))
    except AttributeError as e:
        raise type(e)('Could not retrieve the model\'s class. ' + str(e))
    return model_cls
def load_paper_net(device: str = 'gpu'):
    """
        Load the neural network from the paper
    """
    print('In load_paper_net()')
    model_module_name = 'models.models1'
    model_cls_name = 'FullyCNN'
    model_cls = load_model_cls(model_module_name, model_cls_name)
    print('After load_model_cls()')
    net = model_cls(2, 4)
    print('After net')
    if device == 'cpu':
        transformation = torch.load('/scratch/cimes/cz3321/MOM6/MOM6-examples/src/MOM6/config_src/external/ML_Forpy/Forpy_CNN_GZ21/final_transformation.pth')
        print('After torch.load()')
    else:
        transformation = pickle_artifact(MODEL_RUN_ID, 'models/transformation')
    net.final_transformation = transformation
    print('After transformation')

    # Load parameters of pre-trained model
    print('Loading the neural net parameters')
    # logging.info('Loading the neural net parameters')
    # client = mlflow.tracking.MlflowClient()
    print('After mlflow.tracking.MlflowClient()')
#    model_file = client.download_artifacts(MODEL_RUN_ID,
#                                           'nn_weights_cpu.pth')
    model_file = '/scratch/cimes/cz3321/MOM6/MOM6-examples/src/MOM6/config_src/external/ML_Forpy/Forpy_CNN_GZ21/trained_model.pth'
    print('After download_artifacts()')
    if device == 'cpu':
        print('Device: CPU')
        model_file = '/scratch/cimes/cz3321/MOM6/MOM6-examples/src/MOM6/config_src/external/ML_Forpy/Forpy_CNN_GZ21/nn_weights_cpu.pth'
        net.load_state_dict(torch.load(model_file, map_location=torch.device('cpu')))
    else:
        net.load_state_dict(torch.load(model_file))
    # print(torch.load(model_file, map_location=torch.device('cpu')).keys())
    # print(net.state_dict().keys())
    print(net)
    return net
net = load_paper_net('cpu')

device = torch.device('cpu')
from train.losses import HeteroskedasticGaussianLossV2
criterion = HeteroskedasticGaussianLossV2(n_target_channels=2)
from testing.utils import (create_large_test_dataset, create_test_dataset)
from torch.utils.data import DataLoader
from data.datasets import (RawDataFromXrDataset, DatasetTransformer,
                           Subset_, DatasetWithTransform, ComposeTransforms,
                           MultipleTimeIndices, DatasetPartitioner)
low_rez = low_rez.fillna(0)
dataset = RawDataFromXrDataset(low_rez * 10.)
dataset.index = 'time'
dataset.add_input('usurf')
dataset.add_input('vsurf')
dataset.add_output('S_x')
dataset.add_output('S_y')
features_transform_ = ComposeTransforms()
targets_transform_ = ComposeTransforms()
transform = DatasetTransformer(features_transform_, targets_transform_)
transform.fit(dataset)
dataset = DatasetWithTransform(dataset, transform)
test = create_large_test_dataset(net.to(device=device), criterion, [dataset, ], [DataLoader(dataset)], device)
test = test.rename(dict(longitude='xu_ocean', latitude='yu_ocean'))
test

from dask.diagnostics import ProgressBar
with ProgressBar():
    test = test.compute()

test.to_netcdf('test_four_regions.nc')