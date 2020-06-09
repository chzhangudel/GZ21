# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 20:25:09 2020

@author: Arthur
Here we test a trained model on an unseen region. The user is prompted to
select a trained model within a list and a new region to test that model.
Fine-tuning is an option through the n_epochs parameter of the script.

TODO:
    - Allow to test on all regions at once with one command
"""
import mlflow
import torch
import torch.nn
from torch.utils.data import DataLoader, Subset
import numpy as np
import xarray as xr
from analysis.utils import select_run
from train.utils import learning_rates_from_string
from data.datasets import (RawDataFromXrDataset, DatasetTransformer,
                            Subset_, ConcatDataset_, DatasetWithTransform,
                            MultipleTimeIndices)
from train.base import Trainer
from train.losses import (HeteroskedasticGaussianLoss, 
                          HeteroskedasticGaussianLossV2)

from testing.utils import create_test_dataset

from models.utils import load_model_cls

import os.path
import importlib

import tempfile
import logging

import argparse
import pickle

from sys import modules

# Parse arguments
# n_epochs : Number of epochs we fine-tune the model on the new data
parser = argparse.ArgumentParser()
parser.add_argument('--n_epochs', type=int, default=0)
parser.add_argument('--lr_ratio', type=float, default=1)
parser.add_argument('--models_experiment_name', type=str, default='training')
script_params = parser.parse_args()
n_epochs = script_params.n_epochs
lr_ratio = script_params.lr_ratio
models_experiment_name = script_params.models_experiment_name

# Location used to write generated data before it is logged through MLFlow
data_location = tempfile.mkdtemp(dir='/scratch/ag7531/temp/')
model_output_dir = 'model_output'

# Select the device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Prompt user to retrieve a trained model based on a run id for the default
# experiment (folder mlruns/0)
cols = ['metrics.test loss', 'start_time', 'params.time_indices',
        'params.model_cls_name', 'params.source.run_id']
# Recover experiment id of the models
models_experiment = mlflow.get_experiment_by_name(models_experiment_name)
models_experiment_id = models_experiment.experiment_id
model_run = select_run(sort_by='start_time', cols=cols,
                       experiment_ids=[models_experiment_id, ])

# Load some extra parameters of the model.
# TODO allow general time_indices
time_indices = [0, ]
train_split = float(model_run['params.train_split'])
test_split = float(model_run['params.test_split'])
batch_size = int(model_run['params.batchsize'])
source_data_id = model_run['params.source.run_id']
model_module_name = model_run['params.model_module_name']
model_cls_name = model_run['params.model_cls_name']
loss_cls_name = model_run['params.loss_cls_name']
learning_rates = learning_rates_from_string(model_run['params.learning_rate'])
weight_decay = float(model_run['params.weight_decay'])

learning_rate = learning_rates[0] * lr_ratio

# Load the model's file
client = mlflow.tracking.MlflowClient()
model_file = client.download_artifacts(model_run.run_id,
                                       'models/trained_model.pth')
transformation_file = client.download_artifacts(model_run.run_id,
                                                'models/transformation')
features_transform_file = client.download_artifacts(model_run.run_id,
                                                    'models/features_transform')
try:
    targets_transform_file = client.download_artifacts(model_run.run_id,
                                                       'models/targets_transform')
except FileNotFoundError:
    targets_transform_file = None

with torch.no_grad():
    with open(transformation_file, 'rb') as f:
        transformation = pickle.load(f)
with open(features_transform_file, 'rb') as f:
    features_transform = pickle.load(f)
if targets_transform_file is not None:
    with open(targets_transform_file, 'rb') as f:
        targets_transform = pickle.load(f)
else:
    targets_transform = None

# Prompt user to select the test dataset
mlflow.set_experiment('forcingdata')
cols = ['params.lat_min', 'params.lat_max', 'params.long_min',
        'params.long_max', 'params.scale']

i_test = 0
while True:
    i_test += 1
    data_run = select_run(cols=cols)
    if isinstance(data_run, int):
        break
    # Recover the data (velocities and forcing)
    client = mlflow.tracking.MlflowClient()
    data_file = client.download_artifacts(data_run.run_id, 'forcing')

    # Set the experiment to 'multiregion'
    print('Logging to experiment multiregion')
    mlflow.set_experiment('multiregion')
    mlflow.start_run()
    mlflow.log_param('model_run_id', model_run.run_id)
    mlflow.log_param('data_run_id', data_run.run_id)
    mlflow.log_param('n_epochs', n_epochs)
    
    # Read the dataset file
    xr_dataset = xr.open_zarr(data_file).load()
    
    # To PyTorch Dataset
    dataset = RawDataFromXrDataset(xr_dataset)
    dataset.index = 'time'
    dataset.add_input('usurf')
    dataset.add_input('vsurf')
    dataset.add_output('S_x')
    dataset.add_output('S_y')

    train_index = int(train_split * len(dataset))
    test_index = int(test_split * len(dataset))
    train_dataset = Subset_(dataset, np.arange(train_index))

    transform = DatasetTransformer(features_transform, targets_transform)
    transform.fit(train_dataset)
    dataset = DatasetWithTransform(dataset, transform)
    dataset = MultipleTimeIndices(dataset)
    dataset.time_indices = [0, ]
    train_dataset = Subset_(dataset, np.arange(train_index))
    test_dataset = Subset_(dataset, np.arange(test_index, len(dataset)))

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size,
                                  shuffle=True, drop_last=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size,
                                 shuffle=False, drop_last=True)
    print('Size of training data: {}'.format(len(train_dataset)))
    print('Size of validation data : {}'.format(len(test_dataset)))
    print('Height: {}'.format(train_dataset.height))
    print('Width: {}'.format(train_dataset.width))
    print(train_dataset[0][0].shape)
    print(train_dataset[0][1].shape)
    print('Features transform: ', transform.transforms['features'].transforms)
    print('Targets transform: ', transform.transforms['targets'].transforms)

    # Load the model itself
    logging.info('Creating the neural network model')
    model_cls = load_model_cls(model_module_name, model_cls_name)
    net = model_cls(dataset.n_features, 2*dataset.n_targets)

    logging.info('Loading the neural net parameters')
    # Load parameters of pre-trained model
    net.final_transformation = transformation
    net.load_state_dict(torch.load(model_file))

    # Adding transforms required by the model
    dataset.add_targets_transform_from_model(net)

    # Net to GPU
    net.to(device)

    # Set up training criterion and select parameters to train
    try:
        criterion = getattr(modules['__main__'], loss_cls_name)()
    except AttributeError as e:
        raise type(e)('Could not find the loss class used for training.')

    print('width: {}, height: {}'.format(dataset.width, dataset.height))

    trainer = Trainer(net, device)
    trainer.criterion = criterion
    if n_epochs > 0:
        print('Fine-tuning whole network')
        parameters = net.parameters()
        optimizer = torch.optim.Adam(parameters, lr=learning_rate)

    # Training itself
    for i_epoch in range(n_epochs):
        train_loss = trainer.train_for_one_epoch(train_dataloader, optimizer)
        test_loss, metrics_results = trainer.test(test_dataloader)
        print('Epoch {}'.format(i_epoch))
        print('Train loss for this epoch is {}'.format(train_loss))
        print('Test loss for this epoch is {}'.format(test_loss))
        mlflow.log_metric('train loss', train_loss, i_epoch)
        mlflow.log_metric('test loss', test_loss, i_epoch)

    # Final test
    train_loss, train_metrics_results = trainer.test(train_dataloader)
    test_loss, test_metrics_results = trainer.test(test_dataloader)
    print(f'Final train loss is {train_loss}')
    print(f'Final test loss is {test_loss}')

    # Do the predictions for that dataset using the loaded model
    out = create_test_dataset(net, xr_dataset, test_dataset,
                              test_dataloader, test_index, device)
    if i_test == 1:
        output_dataset = out
    else:
        output_dataset = output_dataset.merge(out)

# Save dataset
file_path = os.path.join(data_location, 'test_output')
output_dataset.to_zarr(file_path)
mlflow.log_artifact(file_path)
mlflow.end_run()
