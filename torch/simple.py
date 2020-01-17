# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 16:13:28 2019

@author: Arthur
Torch implementation of a similar form of NN as in Bolton et al
"""

import numpy as np
import mlflow
import os.path
from datetime import datetime

import torch
from torch.utils.data import DataLoader, Subset
import torch.optim as optim
import torch.nn

import matplotlib.pyplot as plt


# Import our Dataset class and neural network
from full_cnn1 import Dataset_psi_s, FullyCNN, MLFlowDataset

# Import some utils functions
from utils_nn import print_every, RunningAverage, DEVICE_TYPE

# Training parameters
# Note that we use two indices for the train/test split. This is because we
# want to avoid the time correlation to play in our favour during test.
batch_size = 8
learning_rates = {0: 1e-3}
n_epochs = 100
train_split = 0.7
test_split = 0.8

# Parameters specific to the input data
# past specifies the indices from the past that are used for prediction
indices = [0, -1]

# Other parameters
print_loss_every = 20
data_location = '/data/ag7531/'
figures_directory = 'figures'

# Device selection. If available we use the GPU.
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device_type = DEVICE_TYPE.GPU if torch.cuda.is_available() \
                              else DEVICE_TYPE.CPU
print('Selected device type: ', device_type.value)

# Log the parameters with mlflow
mlflow.log_param('batch_size', batch_size)
mlflow.log_param('learning_rate', learning_rates)
mlflow.log_param('device', device)
mlflow.log_param('time_indices', indices)
mlflow.log_param('device', device_type.value)
mlflow.log_param('train_split', train_split)
mlflow.log_param('test_split', test_split)


# Load data from disk
dataset_ = Dataset_psi_s('/data/ag7531/processed_data',
                        'psi_coarse.npy', 'sx_coarse.npy', 'sy_coarse.npy')
# Specifies which time indices to use for the prediction
dataset_.set_indices(indices)

# Convert to MLFlow dataset
dataset = MLFlowDataset(dataset_)

# Split train/test
n_indices = len(dataset)
split_index = int(train_split * n_indices)
test_index = int(test_split * n_indices)
dataset.pre_process(split_index)
train_dataset = Subset(dataset, np.arange(split_index))
test_dataset = Subset(dataset, np.arange(split_index, n_indices))

# Dataloaders are responsible for sending batches of data to the NN
train_dataloader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size,
                             shuffle=False)
# Neural network
net = FullyCNN(len(indices), dataset.width, dataset.height,
               dataset.n_output_targets)
print('--------------------')
print(net)
print('--------------------')
# To GPU
net.to(device)

# Log the text representation of the net into a txt artifact
with open('nn_architecture.txt', 'w') as f:
    print('Writing neural net architecture into txt file.')
    f.write(str(net))
mlflow.log_artifact('nn_architecture.txt')

# MSE criterion + Adam optimizer
criterion = torch.nn.MSELoss()
optimizers = {i: optim.Adam(net.parameters(), lr=v) for (i,v) in
              learning_rates.items()}

# Training
for i_epoch in range(n_epochs):
    if i_epoch in optimizers:
        optimizer = optimizers[i_epoch]
    print('Epoch number {}.'.format(i_epoch))
    # reset running loss to zero at each epoch
    running_loss = RunningAverage()
    for i_batch, batch in enumerate(train_dataloader):
        # Zero the gradients
        net.zero_grad()
        # Get a batch and move it to the GPU (if possible)
        X = batch[0].to(device, dtype=torch.float)
        Y = batch[1].to(device, dtype=torch.float)
        # Compute loss
        loss = criterion(net(X), Y)
        running_loss.update(loss.item(), X.size(0))
        # Print current loss
        loss_text = 'Current loss value is {}'.format(running_loss)
        print_every(loss_text, print_loss_every, i_batch)
        # Backpropagate
        loss.backward()
        # Update parameters
        optimizer.step()
    # Log the training loss
    print('Train loss for this epoch is ', running_loss)
    mlflow.log_metric('train mse', running_loss.value, i_epoch)

    # At the end of each epoch we compute the test loss and print it
    with torch.no_grad():
        nb_samples = 0
        running_loss = RunningAverage()
        for i, data in enumerate(test_dataloader):
            X = data[0].to(device, dtype=torch.float)
            Y = data[1].to(device, dtype=torch.float)
            loss = criterion(net(X), Y)
            running_loss.update(loss.item(), X.size(0))
    print('Test loss for this epoch is ', running_loss)
    mlflow.log_metric('test mse', running_loss.value, i_epoch)

    # We also save a snapshot figure to the disk and log it
    # TODO rewrite this bit, looks confusing for now
    ids_data = (np.random.randint(0, len(test_dataset)), 300)
    with torch.no_grad():
        for i, id_data in enumerate(ids_data):
            data = test_dataset[id_data]
            X = torch.tensor(data[0][np.newaxis, ...]).to(device,
                                                          dtype=torch.float)
            Y = data[1][np.newaxis, ...]
            pred = net(X).cpu().numpy()
            fig = dataset.plot_true_vs_pred(Y, pred)
            f_name = 'image{}-{}.png'.format(i_epoch, i)
            file_path = os.path.join(data_location, figures_directory, f_name)
            plt.savefig(file_path)
            plt.close(fig)
    # log the epoch
    mlflow.log_param('n_epochs', i_epoch + 1)


# Save the trained model to disk
print('Saving the neural network learnt parameters to disk...')
model_name = str(datetime.now()).split('.')[0] + '.pth'
full_path = os.path.join(data_location, 'models', model_name)
torch.save(net.state_dict(), full_path)
mlflow.log_artifact(full_path)
print('Neural network saved and logged in the artifacts.')

# Post analysis (Correlation map)
pred = np.zeros((len(test_dataset), 2, dataset.width, dataset.height))
truth = np.zeros((len(test_dataset), 2, dataset.width, dataset.height))

# Predictions on the test set using the trained model
with torch.no_grad():
    for i, data in enumerate(test_dataloader):
        X = data[0].to(device, dtype=torch.float)
        pred_i = net(X)
        pred_i = pred_i.cpu().numpy()
        pred_i = np.reshape(pred_i, (-1, 2, dataset.width, dataset.height))
        pred[i * batch_size:(i+1) * batch_size] = pred_i
        Y = np.reshape(data[1], (-1, 2, dataset.width, dataset.height))
        truth[i * batch_size:(i+1) * batch_size] = Y

# TODO log the predictions as artifacts?

# Correlation map, shape (2, dataset.width, dataset.height)
correlation_map = np.mean(truth * pred, axis=0)
correlation_map -= np.mean(truth, axis=0) * np.mean(pred, axis=0)
correlation_map /= np.maximum(np.std(truth, axis=0) * np.std(pred, axis=0),
                              1e-20)

print('Saving correlation map to disk')
# Save the correlation map to disk and its plot as well.
np.save('/data/ag7531/analysis/correlation_map', correlation_map)

fig = plt.figure()
plt.subplot(121)
plt.imshow(correlation_map[0], vmin=0, vmax=1, origin='lower')
plt.colorbar()
plt.title('Correlation map for S_x')
plt.subplot(122)
plt.imshow(correlation_map[1], vmin=0, vmax=1, origin='lower')
plt.colorbar()
plt.title('Correlation map for S_y')
f_name = 'Correlation_maps.png'
file_path = os.path.join(data_location, figures_directory, f_name)
plt.savefig(file_path)
plt.close(fig)

# log the figures as artifacts
mlflow.log_artifact(os.path.join(data_location, figures_directory))
# log the correlation map figure
mlflow.log_artifact(file_path)
if 'y' in input('register as success?').lower():
    mlflow.set_tag('success', 'True')
else:
    mlflow.set_tag('success', 'False')
