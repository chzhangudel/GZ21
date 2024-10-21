# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 17:24:37 2020

@author: Arthur

TODOs:
-Try some standard image classification network whose last layer you'll change
- change the color map of plots
- study different values of time indices
------BUGS-----
-when we run less than 100 epochs the figures from previous runs are
logged.
"""
# TODO Log the data run that is used to create the dataset. Log any
# transformation applied to the data. Later we might want to allow from
# stream datasets.

import torch
from torch.nn import Module, Parameter, Sequential, ModuleList
from torch.nn import functional as F
from torch.nn.modules.utils import _pair
from torch.nn.functional import pad
import torch.nn as nn

import numpy as np
from .base import DetectOutputSizeMixin, FinalTransformationMixin
import pickle
import os

class Identity(Module):
    def __init__(self):
        super().__init__()

    def forward(self, input: torch.Tensor):
        return input


class ScaledModule(Module):
    def __init__(self, factor: float, module: torch.nn.Module):
        super().__init__()
        self.factor = factor
        self.module = module

    def forward(self, input: torch.Tensor):
        return self.factor * self.module.forward(input)

class LocallyConnected2d(nn.Module):
    """Class based on the code provided on the following link:
        https://discuss.pytorch.org/t/locally-connected-layers/26979
    """
    def __init__(self, input_h, input_w, in_channels, out_channels,
                 kernel_size, padding, stride, bias=False):
        super(LocallyConnected2d, self).__init__()
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        def padding_extract(padding):
            new = []
            for el in padding:
                new.append(int(el))
                new.append(int(el))
            return tuple(new)
        self.padding_long = padding_extract(self.padding)
        output_size = self.calculate_output_size(input_h, input_w, 
                                                 self.kernel_size, 
                                                 self.padding,
                                                 self.stride)
        self.weight = nn.Parameter(
            torch.randn(1, out_channels, in_channels, output_size[0],
                        output_size[1], 
                        self.kernel_size[0] * self.kernel_size[1])
        )
        # Scaling of the weight parameters according to number of inputs
        self.weight.data = self.weight / np.sqrt(in_channels * 
                                                 self.kernel_size[0]
                                                 * self.kernel_size[1])
        if bias:
            self.bias = nn.Parameter(
                torch.randn(1, out_channels, output_size[0], output_size[1])
            )
        else:
            self.register_parameter('bias', None)

    def forward(self, x):
        _, c, h, w = x.size()
        kh, kw = self.kernel_size
        dh, dw = self.stride
        x = pad(x, self.padding_long)
        x = x.unfold(2, kh, dh).unfold(3, kw, dw)
        x = x.contiguous().view(*x.size()[:-2], -1)
        # Sum in in_channel and kernel_size dims
        out = (x.unsqueeze(1) * self.weight).sum([2, -1])
        if self.bias is not None:
            out += self.bias
        return out

    @staticmethod
    def calculate_output_size(input_h : int, input_w : int, kernel_size : int,
                              padding : int = None, stride : int = 1):
        # TODO add the stride bit. Right now it assumes 1.
        output_h = int(input_h - (kernel_size[0] - 1) / 2 + padding[0])
        output_w = int(input_w - (kernel_size[1] - 1) / 2 + padding[1])
        return output_h, output_w


class MLFlowNN(Module):
    """Class for a pytorch NN whose characteristics are automatically
    logged through MLFLOW."""
    def __init__(self, input_depth: int, output_size: int, height : int = None,
                 width: int = None):
        super().__init__()
        self.input_depth = input_depth
        self.output_size = output_size
        self.layers = torch.nn.ModuleList()
        self._n_layers = 0
        self.conv_layers = []
        self.linear_layer = None
        self.activation_choices = {'relu': torch.nn.ReLU(),
                                   'selu': torch.nn.SELU(),
                                   'tanh': torch.nn.Tanh(),
                                   '2tanh': ScaledModule(2, torch.nn.Tanh()),
                                   'identity': Identity()}
        self.activations = []
        self.params_to_log = {'max_pool': False,
                              'max_kernel_size': 0,
                              'max_depth': 1,
                              'groups': False,
                              'batch_normalization': False
                              }
        self.logged_params = False
        if width is not None and height is not None:
            self.image_dims = (width, height)
            self.image_size = width * height
            self.width = width
            self.height = height
        self._log_structure = False
        self._final_transformation = lambda x: x

    @property
    def n_layers(self) -> int:
        """Returns the number of layers. Note that we consider that
        activations are not layers in this count, but are part of a layer,
        hence the division by two."""
        return self._n_layers

    @n_layers.setter
    def n_layers(self, value: int) -> None:
        self._n_layers = value

    @property
    def transformation(self):
        return self._final_transformation

    @transformation.setter
    def transformation(self, transformation):
        self._final_transformation = transformation

    @property
    def log_structure(self):
        return self._log_structure

    @log_structure.setter
    def log_structure(self, value: bool):
        self._log_structure = value

    def add_activation(self, activation: str) -> None:
        self.layers.append(self.activation_choices[activation])
        self.params_to_log['default_activation'] = activation
        self.activations.append(activation)

    def add_final_activation(self, activation: str) -> None:
        """Use this funtion to specify the final activation. This is
        required to log a specific parameter through mlflow corresponding
        to this activation, as it plays a specific role."""
        self.layers.append(self.activation_choices[activation])
        self.params_to_log['last_layer_activation'] = activation

    def add_linear_layer(self, in_features : int, out_features : int, 
                         bias : bool = True, do_not_load : bool = False):
        layer = torch.nn.Linear(in_features, out_features, bias)
        if do_not_load:
            # The following line prevents the layer from loading its
            # parameters from the state dict on calls to load_state_dict
            layer._load_from_state_dict = lambda *args : None
        i_layer = self.n_layers
        self.params_to_log['layer{}'.format(i_layer)] = 'Linear'
        self.layers.append(torch.nn.Flatten())
        self.layers.append(layer)
        self.n_layers += 1
        self.linear_layer = layer

    def add_locally_connected2d(self, input_h, input_w, in_channels: int, 
                                out_channels : int, kernel_size : int,
                                padding : int, stride : int = 1,
                                bias : bool = True, do_not_load : bool = True):
        layer = LocallyConnected2d(input_h, input_w, in_channels,
                                   out_channels, kernel_size, padding, 
                                   stride, bias)
        if do_not_load:
            layer._load_from_state_dict = lambda *args : None
        i_layer = self.params_to_log['layer{}'.format(i_layer)] = 'Local lin'
        self.layers.append(layer)
        self.n_layers += 1
        self.linear_layer = layer
        

    def add_conv2d_layer(self, in_channels: int, out_channels: int,
                         kernel_size: int, stride: int = 1, padding: int = 0,
                         dilation: int = 1, groups: int = 1, bias: bool = True):
        """Adds a convolutional layer. Same parameters as the torch Conv2d,
        the difference is that we log some of these parameters through mlflow.
        """
        conv_layer = torch.nn.Conv2d(in_channels, out_channels, kernel_size,
                                     stride, padding, dilation, groups, bias)
        self.layers.append(conv_layer)
        i_layer = self.n_layers
        self.params_to_log['layer{}'.format(i_layer)] = 'Conv2d'
        self.params_to_log['kernel{}'.format(i_layer)] = str(kernel_size)
        self.params_to_log['groups{}'.format(i_layer)] = groups
        self.params_to_log['depth{}'.format(i_layer)] = out_channels
        self.params_to_log['bias{}'.format(i_layer)] = bias
        if groups > 1:
            self.params_to_log['groups'] = True
        if kernel_size > self.params_to_log['max_kernel_size']:
            self.params_to_log['max_kernel_size'] = kernel_size
        if out_channels > self.params_to_log['max_depth']:
            self.params_to_log['max_depth'] = out_channels
        # Register that we have added a layer
        self.n_layers += 1
        self.conv_layers.append(conv_layer)

    def add_divergence2d_layer(self, n_input_channels: int,
                               n_output_channels: int):
        div2d_layer = Divergence2d(n_input_channels, n_output_channels)
        self.layers.append(div2d_layer)
        self.params_to_log['divergence2d'] = True
        # Register we have added a layer
        self.n_layers += 1

    def add_max_pool_layer(self, kernel_size: int, stride=None,
                           padding: int = 0, dilation: int = 1):
        """Adds a max pool layer and logs some parameters corresponding to
        this layer."""
        layer = torch.nn.MaxPool2d(kernel_size, stride, padding, dilation)
        self.layers.append(layer)
        i_layer = self.n_layers
        self.params_to_log['layer{}'.format(i_layer)] = 'MaxPoolv2d'
        self.params_to_log['kernel{}'.format(i_layer)] = str(kernel_size)
        self.params_to_log['max_pool'] = True

    def add_batch_norm_layer(self, num_features: int, eps: float = 1e-5,
                             momentum: float = 0.1, affine: bool = True,
                             track_running_stats: bool = True):
        """Adds a batch normalization layer and makes some logs accordingly"""
        layer = torch.nn.BatchNorm2d(num_features, eps, momentum, affine,
                                     track_running_stats)
        self.layers.append(layer)
        self.params_to_log['batch_normalization'] = True

    def log_params(self):
        """Logs the parameters for the built neural net."""
        print('Logging neural net parameters...')
        mlflow.log_param('n_layers', self.n_layers)
        for param_name, param_value in self.params_to_log.items():
            mlflow.log_param(param_name, str(param_value))
        self.logged_params = True

    def forward(self, input: torch.Tensor):
        """Overwrites the abstract method of the Module class."""
        # Log the params if it has not been done already.
        if not self.logged_params and self.log_structure:
            self.log_params()
        # Propagate through the layers
        output = input
        for i_layer,  layer in enumerate(self.layers):
            output = layer(output)
        output = self.transformation(output)
        return output


class Divergence2d(Module):
    """Class that defines a fixed layer that produces the divergence of the
    input field. Note that the padding is set to 2, hence the spatial dim
    of the output is larger than that of the input."""
    def __init__(self, n_input_channels: int, n_output_channels: int):
        super().__init__()
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.n_input_channels = n_input_channels
        self.n_output_channels = n_output_channels
        factor = n_input_channels // n_output_channels
        lambda_ = 1 / (factor * 2)
        shape = (1, n_input_channels//4, 1, 1)
        self.lambdas1x = Parameter(torch.ones(shape)) * lambda_
        self.lambdas2x = Parameter(torch.ones(shape)) * lambda_
        self.lambdas1y = Parameter(torch.ones(shape)) * lambda_
        self.lambdas2y = Parameter(torch.ones(shape)) * lambda_
        self.lambdas1x = self.lambdas1x.to(device=device)
        self.lambdas2x = self.lambdas2x.to(device=device)
        self.lambdas1y = self.lambdas1y.to(device=device)
        self.lambdas2y = self.lambdas2y.to(device=device)
        x_derivative = torch.tensor([[[[0, 0, 0],
                                       [-1, 0, 1],
                                       [0, 0, 0]]]])
        x_derivative = x_derivative.expand(1, n_input_channels // 4, -1, -1)
        y_derivative = torch.tensor([[0, 1, 0],
                                     [0, 0, 0],
                                     [0, -1, 0]])
        y_derivative = y_derivative.expand(1, n_input_channels // 4, -1, -1)
        x_derivative = x_derivative.to(dtype=torch.float32,
                                                 device=device)
        y_derivative = y_derivative.to(dtype=torch.float32,
                                                 device=device)
        self.x_derivative = x_derivative
        self.y_derivative = y_derivative

    def forward(self, input: torch.Tensor):
        n, c, h, w = input.size()
        y_derivative1 = self.y_derivative * self.lambdas1y
        x_derivative2 = self.x_derivative * self.lambdas2x
        y_derivative2 = self.y_derivative * self.lambdas2y
        output11 = F.conv2d(input[:, :c//4, :, :], self.x_derivative * self.lambdas1x,
                           padding=2)
        output12 = F.conv2d(input[:, c//4:c//2, :, :], self.y_derivative,
                            padding=2)
        output1 = output11 + output12
        output21 = F.conv2d(input[:, c//2:c//2+c//4, :, :], self.x_derivative,
                           padding=2)
        output22 = F.conv2d(input[:, c//2+c//4:, :, :], self.y_derivative,
                            padding=2)
        output2 = output21 + output22
        res =  torch.stack((output1, output2), dim=1)
        res = res[:,:, 0, :, :]
        return res

#new FullyCNN to solve boundary issue
#->
def replicate_nans(_x):
  '''
  x is 4D tensor N x C x Ny x Nx
  returns nans filled with nearest neighbour
  We assume that NaNs occur consistently
  in the full NxC dimensitons
  '''
  mask_array = torch.isnan(_x[0,0,:,:]).clone()
  ny, nx = mask_array.shape
  def mask(j,i):
    if j < ny and j > -1 and i < nx and i > -1:
      return mask_array[j,i]
    else:
      return True
  x = _x.clone()
  for j in range(x.shape[-2]):
    for i in range(x.shape[-1]):
      if mask(j,i):
        if mask(j,i+1) and mask(j,i-1) and mask(j+1,i) and mask(j-1,i) and mask(j+1,i+1) and mask(j+1,i-1) and mask(j-1,i+1) and mask(j-1,i-1):
          continue # nothing to interoplate

        x[:,:,j,i] = 0.
        n = 0
        if (not mask(j,i+1)):
          x[:,:,j,i] += x[:,:,j,i+1]
          n += 1

        if (not mask(j,i-1)):
          x[:,:,j,i] += x[:,:,j,i-1]
          n += 1

        if (not mask(j+1,i)):
          x[:,:,j,i] += x[:,:,j+1,i]
          n += 1

        if (not mask(j-1,i)):
          x[:,:,j,i] += x[:,:,j-1,i]
          n += 1

        if (not mask(j+1,i+1)):
          x[:,:,j,i] += x[:,:,j+1,i+1]
          n += 1

        if (not mask(j+1,i-1)):
          x[:,:,j,i] += x[:,:,j+1,i-1]
          n += 1

        if (not mask(j-1,i+1)):
          x[:,:,j,i] += x[:,:,j-1,i+1]
          n += 1

        if (not mask(j-1,i-1)):
          x[:,:,j,i] += x[:,:,j-1,i-1]
          n += 1

        if n==0:
          print('Error')
        else:
          x[:,:,j,i] *= 1/n
  return x

def replicate_mat(maskn):
  '''
  create coefficient matrix A for maskn
  Then filling nan in x with nearest neighbour
  can be achieved by A*x
  '''
  mask_array = torch.isnan(maskn[0,0,:,:]).clone()
  ny, nx = mask_array.shape
  def mask(j,i):
    if j < ny and j > -1 and i < nx and i > -1:
      return mask_array[j,i]
    else:
      return True
    
  idi = []
  idj = []
  val = []
  
  for j in range(ny):
    for i in range(nx):
      if mask(j,i):

        n = 0
        k = j*nx+i

        if mask(j,i+1) and mask(j,i-1) and mask(j+1,i) and mask(j-1,i) and mask(j+1,i+1) and mask(j+1,i-1) and mask(j-1,i+1) and mask(j-1,i-1):
          idi.append(k)
          idj.append(k)

        if (not mask(j,i+1)):
          idi.append(k+1)
          idj.append(k)
          n += 1

        if (not mask(j,i-1)):
          idi.append(k-1)
          idj.append(k)
          n += 1

        if (not mask(j+1,i)):
          idi.append(k+nx)
          idj.append(k)
          n += 1

        if (not mask(j-1,i)):
          idi.append(k-nx)
          idj.append(k)
          n += 1

        if (not mask(j+1,i+1)):
          idi.append(k+nx+1)
          idj.append(k)
          n += 1

        if (not mask(j+1,i-1)):
          idi.append(k+nx-1)
          idj.append(k)
          n += 1

        if (not mask(j-1,i+1)):
          idi.append(k-nx+1)
          idj.append(k)
          n += 1

        if (not mask(j-1,i-1)):
          idi.append(k-nx-1)
          idj.append(k)
          n += 1

        if n==0:
          val.append(torch.tensor(float('nan')))
        else:
          n_val = [1/n] * n
          val.extend(n_val)
      else:
          k = j*nx+i
          idi.append(k)
          idj.append(k)
          val.append(1)

  indices = torch.tensor([idj,idi])
  values = torch.tensor(val, dtype=torch.float32)
  A = torch.sparse_coo_tensor(indices=indices, values=values, size=[nx*ny,nx*ny])
  return A

def replicate_mat_new(maskn):
  '''
  create coefficient matrix A for mask0 (without include any nan)
  mask0 has values either 1 or 0
  can be achieved by A*x to fill land points with nearest neighbours
  '''
  mask_array = torch.isnan(maskn[0,0,:,:]).clone()
#   mask_array = maskn[0,0,:,:] == 0
  ny, nx = mask_array.shape
  def mask(j,i):
    if j < ny and j > -1 and i < nx and i > -1:
      return mask_array[j,i]
    else:
      return True
    
  idi = []
  idj = []
  val = []
  
  for j in range(ny):
    for i in range(nx):
      if mask(j,i):

        n = 0
        k = j*nx+i

        if mask(j,i+1) and mask(j,i-1) and mask(j+1,i) and mask(j-1,i) and mask(j+1,i+1) and mask(j+1,i-1) and mask(j-1,i+1) and mask(j-1,i-1):
          idi.append(k)
          idj.append(k)

        if (not mask(j,i+1)):
          idi.append(k+1)
          idj.append(k)
          n += 1

        if (not mask(j,i-1)):
          idi.append(k-1)
          idj.append(k)
          n += 1

        if (not mask(j+1,i)):
          idi.append(k+nx)
          idj.append(k)
          n += 1

        if (not mask(j-1,i)):
          idi.append(k-nx)
          idj.append(k)
          n += 1

        if (not mask(j+1,i+1)):
          idi.append(k+nx+1)
          idj.append(k)
          n += 1

        if (not mask(j+1,i-1)):
          idi.append(k+nx-1)
          idj.append(k)
          n += 1

        if (not mask(j-1,i+1)):
          idi.append(k-nx+1)
          idj.append(k)
          n += 1

        if (not mask(j-1,i-1)):
          idi.append(k-nx-1)
          idj.append(k)
          n += 1

        if n==0:
          val.append(torch.tensor(float(0.0)))
        else:
          n_val = [1/n] * n
          val.extend(n_val)
      else:
          k = j*nx+i
          idi.append(k)
          idj.append(k)
          val.append(1)

  indices = torch.tensor([idj,idi])
  values = torch.tensor(val, dtype=torch.float32)
  A = torch.sparse_coo_tensor(indices=indices, values=values, size=[nx*ny,nx*ny])
  return A

def replicate_nans_new(x,A):
  '''
  x is 4D tensor N x C x Ny x Nx
  returns nans filled with nearest neighbour
  We assume that NaNs occur consistently
  in the full NxC dimensitons
  We use A*x to update x
  '''
#   print('x.shape',_x.shape)
  _x=x.clone()
#   _A=A.clone()
#   flat_x = _x.reshape(_x.size(0), _x.size(1), -1)
#   print(flat_x.shape)
#   print('_x.size(2)* _x.size(3)',_x.size(2)* _x.size(3))
  flat_x = _x.reshape(-1, _x.size(2)* _x.size(3))
#   print(flat_x.shape)
  # print(A.shape)
  xx = torch.matmul(A, flat_x.T).T
  # print(xx.shape)
  _x = xx.reshape(_x.size(0), _x.size(1), _x.size(2), _x.size(3))
  return _x


class FullyCNN_BC(DetectOutputSizeMixin, Module):
    def __init__(self, n_in_channels: int = 2, n_out_channels: int = 4,
                 padding=None, batch_norm=False):
        super(FullyCNN_BC, self).__init__()
        if padding is None:
            padding_5 = 0
            padding_3 = 0
        elif padding == 'same':
            padding_5 = 2
            padding_3 = 1
        else:
            raise ValueError('Unknow value for padding parameter.')
        self.n_in_channels = n_in_channels
        self.batch_norm = batch_norm

        self.conv1 = nn.Conv2d(n_in_channels, 128, 5, padding=padding_5)
        if self.batch_norm:
          self.batch_norm1 = nn.BatchNorm2d(self.conv1.out_channels) 
        self.conv2 = nn.Conv2d(128, 64, 5, padding=padding_5)
        if self.batch_norm:
          self.batch_norm2 = nn.BatchNorm2d(self.conv2.out_channels) 
        self.conv3 = nn.Conv2d(64, 32, 3, padding=padding_3)
        if self.batch_norm:
          self.batch_norm3 = nn.BatchNorm2d(self.conv3.out_channels) 
        self.conv4 = nn.Conv2d(32, 32, 3, padding=padding_3)
        if self.batch_norm:
          self.batch_norm4 = nn.BatchNorm2d(self.conv4.out_channels) 
        self.conv5 = nn.Conv2d(32, 32, 3, padding=padding_3)
        if self.batch_norm:
          self.batch_norm5 = nn.BatchNorm2d(self.conv5.out_channels) 
        self.conv6 = nn.Conv2d(32, 32, 3, padding=padding_3)
        if self.batch_norm:
          self.batch_norm6 = nn.BatchNorm2d(self.conv6.out_channels) 
        self.conv7 = nn.Conv2d(32, 32, 3, padding=padding_3)
        if self.batch_norm:
          self.batch_norm7 = nn.BatchNorm2d(self.conv7.out_channels) 
        self.conv8 = nn.Conv2d(32, n_out_channels, 3, padding=padding_3)
        self.relu = nn.ReLU()

    @property
    def final_transformation(self):
        return self._final_transformation

    @final_transformation.setter
    def final_transformation(self, transformation):
        self._final_transformation = transformation

    def forward(self, x, maskn=None, replicate=True, use_cuda=False,gpu_id=0):
        # halo_old = 0
        halo = 0
        _cache = 0
        if os.path.exists('/scratch/cimes/cz3321/MOM6/experiments/double_gyre/postprocess/offline_test/subgrid2/matrix_dict_global.pkl'):
            with open('/scratch/cimes/cz3321/MOM6/experiments/double_gyre/postprocess/offline_test/subgrid2/matrix_dict_global.pkl', 'rb') as f:
                matrix_dict = pickle.load(f)
            # print("load matrix for A")
        else:
            matrix_dict={}
        for i in range(1, 9): #8 depth
            """
            #old way
            x_old=x
            conv = getattr(self, f'conv{i}') #get the conv layer name
            if i<8 and self.batch_norm:
                batchnorm = getattr(self, f'batch_norm{i}') #get the batch norm layer name
            if maskn is not None:
                #to adapt the mask size to the input of each layer
                mask = maskn[0,0,:,:].unsqueeze(0).unsqueeze(0).expand(x_old.shape[0], x_old.shape[1], -1, -1)
                mask_halo = mask[:, :, halo_old:-halo_old, halo_old:-halo_old] if halo_old > 0 else mask
                #fill values to land points by nearest values
                if replicate is True:
                  for n in range(1,conv.kernel_size[0]//2+1):# 3x3 kernel needs one replicate and 5x5 needs two replicates
                    if f'A_{i}_{n}_old' not in matrix_dict:
                        matrix_dict[f'A_{i}_{n}_old'] = replicate_mat(x_old*mask_halo) if n == 1 else replicate_mat(x_old)
                        # print('mask_halo',mask_halo[0,0,:,:])
                        if use_cuda:
                            matrix_dict[f'A_{i}_{n}_old']=matrix_dict[f'A_{i}_{n}_old'].cuda(gpu_id)
                    x_old = replicate_nans_new(x_old,matrix_dict[f'A_{i}_{n}_old'])
                else:
                    x_old = x_old * mask_halo.nan_to_num(0.)
            # print('before conv: x_old',x_old[0,0,:,:])
            x_old = conv(x_old)
            # print('after conv: x_old',x_old[0,0,:,:])
            halo_old = halo_old + conv.kernel_size[0]//2
            if maskn is not None and replicate is False:
                x_old = x_old.nan_to_num(0.)
            if i<8:
              x_old = self.relu(x_old)
              x_old = batchnorm(x_old) if self.batch_norm else x_old
            else:
              mask = maskn[0,0,:,:].unsqueeze(0).unsqueeze(0).expand(x_old.shape[0], x_old.shape[1], -1, -1)
              x_old = x_old.nan_to_num(0.)*mask[:, :, halo_old:-halo_old, halo_old:-halo_old]
            """
            
            #new way
            conv = getattr(self, f'conv{i}') #get the conv layer name
            if i<8 and self.batch_norm:
                batchnorm = getattr(self, f'batch_norm{i}') #get the batch norm layer name
            if maskn is not None:
                #to adapt the mask size to the input of each layer
                mask = maskn[0,0,:,:].unsqueeze(0).unsqueeze(0).expand(x.shape[0], x.shape[1], -1, -1)
                mask_halo = mask[:, :, halo:-halo, halo:-halo] if halo > 0 else mask
                #fill values to land points by nearest values
                if replicate is True:
                  for n in range(1,conv.kernel_size[0]//2+1):# 3x3 kernel needs one replicate and 5x5 needs two replicates
                    if f'A_{i}_{n}' not in matrix_dict:
                        _cache += 1
                        if i>1 and n==1:
                            mask_halo = mask_halo*mask_halo1
                            mask_halo[(mask_halo != 0) & (~torch.isnan(mask_halo))] = 1.0
                        matrix_dict[f'A_{i}_{n}'] = replicate_mat_new(mask_halo) #create matrix and save in a tuple only in first time
                        # print('mask_halo',mask_halo[0,0,:,:])
                        if use_cuda:
                            matrix_dict[f'A_{i}_{n}']=matrix_dict[f'A_{i}_{n}'].cuda(gpu_id)
                        mask_halo = replicate_nans_new(mask_halo,matrix_dict[f'A_{i}_{n}'])
                        if n==conv.kernel_size[0]//2 and i==8:
                           with open('/scratch/cimes/cz3321/MOM6/experiments/double_gyre/postprocess/offline_test/subgrid2/matrix_dict_global.pkl', 'wb') as f:
                               pickle.dump(matrix_dict, f)
                              #  print("create matrix for A")
                    # print('mask_halo',mask_halo[0,0,:,:])
                    x = replicate_nans_new(x,matrix_dict[f'A_{i}_{n}'])
                    # print(i,n,x[0,0,:,:])
                else:
                  # x = x * mask_halo.nan_to_num(0.)
                  mask_halo0=mask_halo.clone()
                  mask_halo0[mask_halo0 != mask_halo0] = 0
                  x = x * mask_halo0
                  # x = x * mask_halo
            # x = x.nan_to_num(0.)
            # print('before conv: x',x[0,0,:,:])
            x = conv(x)
            if maskn is not None and _cache>0:
                # print(f'A_{i+1}_{1}')
                mask_halo1 = conv(mask_halo)
                # print(maskn_halo1[0,0,:,:])
            # print('after conv: x',x[0,0,:,:])
            halo = halo + conv.kernel_size[0]//2
            if maskn is not None and replicate is False:
                # x = x.nan_to_num(0.)
                x[x != x] = 0
            if i<8:
              x = self.relu(x)
              x = batchnorm(x) if self.batch_norm else x
              if maskn is not None and _cache>0:
                  mask_halo1 = self.relu(mask_halo1)
                  mask_halo1 = batchnorm(mask_halo1) if self.batch_norm else mask_halo1
            else:
              if maskn is not None:
                  mask = maskn[0,0,:,:].unsqueeze(0).unsqueeze(0).expand(x.shape[0], x.shape[1], -1, -1)
                  # x = x.nan_to_num(0.)*mask[:, :, halo:-halo, halo:-halo].nan_to_num(0.)
                  mask_halo2=mask[:, :, halo:-halo, halo:-halo].clone()
                  mask_halo2[mask_halo2 != mask_halo2] = 0
                  x0=x.clone()
                  x0[x0 != x0] = 0
                  x = x0 * mask_halo2
              x = self.final_transformation(x)

        """
        #Compare old and new
        # x_old=x_old
        x_new = x.clone()
        x_new[(torch.isnan(x_old))] = torch.tensor(float('nan'))
        x_old=x_old.nan_to_num(0.)
        x_new=x_new.nan_to_num(0.)
        are_equal = torch.equal(x_old, x_new)
        if are_equal is False:
           print('old and new is same:',are_equal)
        """
            
        return self.final_transformation(x)
#<-


class FullyCNN(DetectOutputSizeMixin, Sequential):
  
    def __init__(self, n_in_channels: int = 3, n_out_channels: int = 4,
                 padding=None, batch_norm=False):
        if padding is None:
            padding_5 = 0
            padding_3 = 0
        elif padding == 'same':
            padding_5 = 2
            padding_3 = 1
        else:
            raise ValueError('Unknow value for padding parameter.')
        self.n_in_channels = n_in_channels
        self.batch_norm = batch_norm
        conv1 = torch.nn.Conv2d(n_in_channels, 128, 5, padding=padding_5)
        block1 = self._make_subblock(conv1)
        conv2 = torch.nn.Conv2d(128, 64, 5, padding=padding_5)
        block2 = self._make_subblock(conv2)
        conv3 = torch.nn.Conv2d(64, 32, 3, padding=padding_3)
        block3 = self._make_subblock(conv3)
        conv4 = torch.nn.Conv2d(32, 32, 3, padding=padding_3)
        block4 = self._make_subblock(conv4)
        conv5 = torch.nn.Conv2d(32, 32, 3, padding=padding_3)
        block5 = self._make_subblock(conv5)
        conv6 = torch.nn.Conv2d(32, 32, 3, padding=padding_3)
        block6 = self._make_subblock(conv6)
        conv7 = torch.nn.Conv2d(32, 32, 3, padding=padding_3)
        block7 = self._make_subblock(conv7)
        conv8 = torch.nn.Conv2d(32, n_out_channels, 3, padding=padding_3)
        Sequential.__init__(self, *block1, *block2, *block3, *block4, *block5,
                            *block6, *block7, conv8)
        

    @property
    def final_transformation(self):
        return self._final_transformation

    @final_transformation.setter
    def final_transformation(self, transformation):
        self._final_transformation = transformation

    def forward(self, x):
        x = super().forward(x)
        
        # mean,prec = torch.split(x,x.shape[1]//2,dim = 1)
        # prec = torch.nn.functional.softplus(prec)
        # return mean,prec
        # Temporary fix for the student loss
        return self.final_transformation(x)

    def _make_subblock(self, conv):
        subbloc = [conv, nn.ReLU()]
        if self.batch_norm:
            subbloc.append(nn.BatchNorm2d(conv.out_channels))
        return subbloc


# class FullyCNN(MLFlowNN, DetectOutputSizeMixin):
#     def __init__(self, input_depth: int, output_size: int, width: int = None,
#                  height: int = None, do_not_load_linear: bool = False):
#         super().__init__(input_depth, output_size, width, height)
#         DetectOutputSizeMixin.__init__(self, height, width)
#         self.do_not_load_linear = do_not_load_linear
#         self.build()

#     def build(self):
#         self.add_conv2d_layer(self.input_depth, 128, 5, padding=2+0)
#         self.add_activation('relu')
#         self.add_batch_norm_layer(128)

#         self.add_conv2d_layer(128, 64, 5, padding=2+0)
#         self.add_activation('relu')
#         self.add_batch_norm_layer(64)

#         self.add_conv2d_layer(64, 32, 3, padding=1+0)
#         self.add_activation('relu')
#         self.add_batch_norm_layer(32)

#         self.add_conv2d_layer(32, 32, 3, padding=1+0)
#         self.add_activation('relu')
#         self.add_batch_norm_layer(32)

#         self.add_conv2d_layer(32, 32, 3, padding=1+(0))
#         self.add_activation('relu')
#         self.add_batch_norm_layer(32)

#         self.add_conv2d_layer(32, 32, 3, padding=1)
#         self.add_activation('relu')
#         self.add_batch_norm_layer(32)

#         self.add_conv2d_layer(32, 32, 3, padding=1)
#         self.add_activation('relu')
#         self.add_batch_norm_layer(32)

#         self.add_conv2d_layer(32, 4, 3, padding=1)

#         self.add_final_activation('identity')


if __name__ == '__main__':
    import numpy as np
    # net = FullyCNN(1, 100*100, 100, 100)
    # input_ = torch.randint(-3, 3, (8, 1, 100, 100))
    # input_ = input_.to(dtype=torch.float32)
    # output = net(input_)
    # output_ = output.detach().numpy()
    # print(output.size())
    # s = torch.sum(output)
    # print(s.item())
    # s.backward()
    from transforms import SoftPlusTransform

    net = FullyCNN()
    net._final_transformation = lambda x: x
    input_ = torch.randint(0, 10, (17, 2, 35, 30)).to(dtype=torch.float)
    input_[0, 0, 0, 0] = np.nan
    output = net(input_)
