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
import time
from torch.nn import Parameter

args_no_cuda = False #True when manually turn off cuda
use_cuda = not args_no_cuda and torch.cuda.is_available()
if use_cuda:
    print('device for inference on',torch.cuda.device_count(),'GPU(s)')
else:
    print('device for inference on CPU')

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
    net = model_cls(3,4,batch_norm=1)

    # final_transform= '/scratch/cimes/cz3321/MOM6/MOM6-examples/src/MOM6/config_src/external/ML_Forpy/Forpy_CNN_GZ21/final_transformation_04292023.pth'
    # print('After net')
    # if device == 'cpu':
    #     transformation = torch.load(final_transform)
    #     print('After torch.load()')
    # else:
    #     transformation = pickle_artifact(MODEL_RUN_ID, 'models/transformation')
    # net.final_transformation = transformation
    print('After transformation')
    # Load parameters of pre-trained model
    print('After mlflow.tracking.MlflowClient()')
    
    
    # ----------------- CHANGE THIS PATH TO TRAINED MODEL ----------------- #
    model_file = '/scratch/cimes/cz3321/MOM6/MOM6-examples/src/MOM6/config_src/external/ML_Forpy/Forpy_CNN_GZ21/best_trained_model_masks.pth'
    # ---------------------------------------------------- #

    model_module_name = 'models.transforms'
    model_cls_name = 'SoftPlusTransform'
    model_cls = load_model_cls(model_module_name, model_cls_name)
    model_cls_name = 'PrecisionTransform'
    model_cls1 = load_model_cls(model_module_name, model_cls_name)
    transform = model_cls.__new__(model_cls,)
    model_cls1.__init__(transform,)
    state_dict = torch.load(model_file, map_location=torch.device('cpu'))
    transform._min_value = Parameter(state_dict.pop('final_transformation._min_value'))
    transform.indices = slice(2,4)
    print('After download_artifacts()')
    print(state_dict.keys())
    print(net.state_dict().keys())
    net.load_state_dict(state_dict)
    net.final_transformation = transform
    print(net)
    return net
net = load_paper_net('cpu')
net.eval()

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
dataset.add_landmask_input()
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

test.to_netcdf('/scratch/gpfs/cz3321/CM2P6/test_global.nc')