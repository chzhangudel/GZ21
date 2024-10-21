import mlflow
import xarray as xr
import matplotlib.pyplot as plt
from data.utils import load_training_datasets
import os
import xarray as xr
import numpy as np
import math

# raw_data = xr.open_zarr('/scratch/cimes/cz3321/MOM6/experiments/double_gyre/postprocess/offline_test/cm2p6/forcing.zarr')
raw_data = xr.open_zarr('/scratch/gpfs/cz3321/CM2P6/forcing.zarr')
raw_datasets = load_training_datasets(raw_data, 'training_subdomains.yaml')

# low_rez = raw_datasets[0]
first_dataset = raw_datasets[0]
#pick randomly
# np.random.seed(42)
# random_indices = np.random.choice(first_dataset.time.size, 1000, replace=False)
#pick test dataset (20% in the end)
start_index = int(first_dataset.time.size*0.8)
random_indices = slice(start_index, None)
random_snapshots = first_dataset.isel(time=random_indices)
low_rez = random_snapshots
u = low_rez['usurf']
v = low_rez['vsurf']

import torch
import importlib
# #load the neural network
# def load_model_cls(model_module_name: str, model_cls_name: str):
#     try:
#         module = importlib.import_module(model_module_name)
#         model_cls = getattr(module, model_cls_name)
#     except ModuleNotFoundError as e:
#         raise type(e)('Could not retrieve the module in which the trained model \
#                       is defined: ' + str(e))
#     except AttributeError as e:
#         raise type(e)('Could not retrieve the model\'s class. ' + str(e))
#     return model_cls
# def load_paper_net(device: str = 'gpu'):
#     """
#         Load the neural network from the paper
#     """
#     print('In load_paper_net()')
#     model_module_name = 'models.models1'
#     model_cls_name = 'FullyCNN_BC'
#     model_cls = load_model_cls(model_module_name, model_cls_name)
#     print('After load_model_cls()')
#     net = model_cls(2, 4)
#     print('After net')
#     if device == 'cpu':
#         transformation = torch.load('/scratch/cimes/cz3321/MOM6/MOM6-examples/src/MOM6/config_src/external/ML_Forpy/Forpy_CNN_GZ21/final_transformation.pth')
#         print('After torch.load()')
#     else:
#         transformation = pickle_artifact(MODEL_RUN_ID, 'models/transformation')
#     net.final_transformation = transformation
#     print('After transformation')

#     # Load parameters of pre-trained model
#     print('Loading the neural net parameters')
#     # logging.info('Loading the neural net parameters')
#     # client = mlflow.tracking.MlflowClient()
#     print('After mlflow.tracking.MlflowClient()')
# #    model_file = client.download_artifacts(MODEL_RUN_ID,
# #                                           'nn_weights_cpu.pth')
#     model_file = '/scratch/cimes/cz3321/MOM6/MOM6-examples/src/MOM6/config_src/external/ML_Forpy/Forpy_CNN_GZ21/trained_model.pth'
#     print('After download_artifacts()')
#     if device == 'cpu':
#         print('Device: CPU')
#         model_file = '/scratch/cimes/cz3321/MOM6/MOM6-examples/src/MOM6/config_src/external/ML_Forpy/Forpy_CNN_GZ21/nn_weights_cpu.pth'
#         state_dict = torch.load(model_file, map_location=torch.device('cpu'))
#     else:
#         state_dict = torch.load(model_file)
#     #change the key name->
#     print(model_cls_name)
#     if model_cls_name.endswith("_BC"):
#         from collections import OrderedDict
#         new_name=["conv1.weight", "conv1.bias", "conv2.weight", "conv2.bias", "conv3.weight", "conv3.bias", "conv4.weight", "conv4.bias", "conv5.weight", "conv5.bias", "conv6.weight", "conv6.bias", "conv7.weight", "conv7.bias", "conv8.weight", "conv8.bias",'final_transformation.min_value']
#         new_state_dict = OrderedDict()
#         i=0
#         for k, v in state_dict.items():
#             print(k,i)
#             name = new_name[i]
#             new_state_dict[name] = v
#             i = i+1
#         state_dict = new_state_dict
#         # print(state_dict.keys())
#     #<-
#     net.load_state_dict(state_dict)
#     print(net)
#     return net

#load the neural network
from torch.nn import Parameter
batch_norm = 0
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
    model_cls_name = 'FullyCNN_BC'
    model_cls = load_model_cls(model_module_name, model_cls_name)
    print('After load_model_cls()')
    net = model_cls(2,4,batch_norm=batch_norm)
    
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
    model_file = '/scratch/cimes/cz3321/MOM6/MOM6-examples/src/MOM6/config_src/external/ML_Forpy/Forpy_CNN_GZ21/cem_1_four_regions_fixed_branch.pth'
    # ---------------------------------------------------- #
    
    
    print('Loading final transformation')
    model_module_name = 'models.transforms'
    model_cls_name1 = 'SoftPlusTransform'
    model_cls = load_model_cls(model_module_name, model_cls_name1)
    model_cls_name2 = 'PrecisionTransform'
    model_cls1 = load_model_cls(model_module_name, model_cls_name2)
    transform = model_cls.__new__(model_cls,)
    model_cls1.__init__(transform,)
    state_dict = torch.load(model_file, map_location=torch.device('cpu'))
    transform._min_value = Parameter(state_dict.pop('final_transformation._min_value'))
    transform.indices = slice(2,4)
    print('After download_artifacts()')
    #change the key name->
    print(model_cls_name)
    if model_cls_name.endswith("_BC"):
        from collections import OrderedDict
        if batch_norm == 1:
            keys_to_delete = ['2.num_batches_tracked', '5.num_batches_tracked', '8.num_batches_tracked', '11.num_batches_tracked', '14.num_batches_tracked', '17.num_batches_tracked', '20.num_batches_tracked']
            for key in keys_to_delete:
                if key in state_dict:
                    del state_dict[key]
            new_name=["conv1.weight", "conv1.bias", "batch_norm1.weight", "batch_norm1.bias", "batch_norm1.running_mean", "batch_norm1.running_var", "conv2.weight", "conv2.bias", "batch_norm2.weight", "batch_norm2.bias", "batch_norm2.running_mean", "batch_norm2.running_var", "conv3.weight", "conv3.bias", "batch_norm3.weight", "batch_norm3.bias", "batch_norm3.running_mean", "batch_norm3.running_var", "conv4.weight", "conv4.bias", "batch_norm4.weight", "batch_norm4.bias", "batch_norm4.running_mean", "batch_norm4.running_var", "conv5.weight", "conv5.bias", "batch_norm5.weight", "batch_norm5.bias", "batch_norm5.running_mean", "batch_norm5.running_var", "conv6.weight", "conv6.bias", "batch_norm6.weight", "batch_norm6.bias", "batch_norm6.running_mean", "batch_norm6.running_var", "conv7.weight", "conv7.bias", "batch_norm7.weight", "batch_norm7.bias", "batch_norm7.running_mean", "batch_norm7.running_var", "conv8.weight", "conv8.bias"]
        elif batch_norm == 0:
            new_name=["conv1.weight", "conv1.bias", "conv2.weight", "conv2.bias", "conv3.weight", "conv3.bias", "conv4.weight", "conv4.bias", "conv5.weight", "conv5.bias", "conv6.weight", "conv6.bias", "conv7.weight", "conv7.bias", "conv8.weight", "conv8.bias"]
        new_state_dict = OrderedDict()
        i=0
        for k, v in state_dict.items():
            name = new_name[i]
            new_state_dict[name] = v
            i = i+1
        state_dict = new_state_dict
    net.load_state_dict(state_dict)
    net.final_transformation = transform
    print(net)
    return net

net = load_paper_net('cpu')
net.eval()

device = torch.device('cpu')
from train.losses import HeteroskedasticGaussianLossV2
criterion = HeteroskedasticGaussianLossV2(n_target_channels=2)
from testing.utils_bc import (create_large_test_dataset, create_test_dataset)
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
test1 = create_large_test_dataset(net.to(device=device), criterion, [dataset, ], [DataLoader(dataset)], device,mask=False, replicate=False)
test1 = test1.rename(dict(longitude='xu_ocean', latitude='yu_ocean'))
test2 = create_large_test_dataset(net.to(device=device), criterion, [dataset, ], [DataLoader(dataset)], device,mask=True, replicate=False)
test2 = test2.rename(dict(longitude='xu_ocean', latitude='yu_ocean'))
test3 = create_large_test_dataset(net.to(device=device), criterion, [dataset, ], [DataLoader(dataset)], device,mask=True, replicate=True)
test3 = test3.rename(dict(longitude='xu_ocean', latitude='yu_ocean'))

from dask.diagnostics import ProgressBar
with ProgressBar():
    test1 = test1.compute()
    test2 = test2.compute()
    test3 = test3.compute()

test1.to_netcdf('test_four_regions_test1_nobc.nc')
test2.to_netcdf('test_four_regions_test1_0pad.nc')
test3.to_netcdf('test_four_regions_test1_rpad.nc')