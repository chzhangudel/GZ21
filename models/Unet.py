#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 15 21:22:59 2020

@author: arthur
Implementation of the U-net structure
"""


import torch
from torch.nn import (Module, ModuleList, Parameter, Upsample, Sequential)
from torch.nn import functional as F
from torch.nn.modules.utils import _pair
from torch.nn.functional import pad
import torch.nn as nn
import numpy as np

class Unet(Module):
    def __init__(self, n_in_channels: int = 2, n_out_channels: int = 4,
                 height=0, width=0, n_scales: int = 2, batch_norm=True):
        super().__init__()
        self.n_in_channels = n_in_channels
        self.n_out_channels = n_out_channels
        self.n_scales = n_scales
        self.down_convs = ModuleList()
        self.up_convs = ModuleList()
        self.up_samplers = ModuleList()
        self.final_convs = None
        self.conv_layers = []
        self.linear_layer = None
        self.batch_norm = batch_norm
        self._build_convs()
        self.linear_layer = None
        

    @property
    def transformation(self):
        return self._final_transformation

    @transformation.setter
    def transformation(self, transformation):
        self._final_transformation = transformation

    def forward(self, x : torch.Tensor):
        blocks = list()
        for i in range(self.n_scales):
            x = self.down_convs[i](x)
            if i != self.n_scales - 1:
                blocks.append(x)
                x = self.down(x)
        blocks.reverse()
        for i in range(self.n_scales - 1):
            x = self.up(x, i)
            x = torch.cat((x, blocks[i]), 1)
            x = self.up_convs[i](x)
        final = self.final_convs(x)
        return self.transformation(final)

    def down(self, x):
        return F.max_pool2d(x, 2)

    def up(self, x, i):
        return self.up_samplers[i](x)

    def _make_subblock(self, conv):
        subbloc = [conv, nn.ReLU()]
        if self.batch_norm:
            subbloc.append(nn.BatchNorm2d(conv.out_channels))
        return subbloc

    def _build_convs(self):
        for i in range(self.n_scales):
            if i == 0:
                n_in_channels = self.n_in_channels
                n_out_channels = 64
            else:
                n_in_channels = n_out_channels
                n_out_channels = 2 * n_out_channels
            conv1 = torch.nn.Conv2d(n_in_channels, n_out_channels, 3, padding=1)
            conv2 = torch.nn.Conv2d(n_out_channels, n_out_channels, 3, padding=1)
            block1 = self._make_subblock(conv1)
            block2 = self._make_subblock(conv2)
            submodule = Sequential(*block1, *block2)
            self.down_convs.append(submodule)
            self.conv_layers.append(conv1)
            self.conv_layers.append(conv2)
        for i in range(self.n_scales - 1):
            # Add the upsampler
            up_sampler = Upsample(mode='bilinear', scale_factor=2)
            conv = torch.nn.Conv2d(n_out_channels, n_out_channels // 2, 1)
            self.up_samplers.append(Sequential(up_sampler, conv))
            # The up convs
            n_in_channels = n_out_channels
            n_out_channels = n_out_channels // 2
            conv1 = torch.nn.Conv2d(n_in_channels, n_out_channels, 3, padding=1)
            conv2 = torch.nn.Conv2d(n_out_channels, n_out_channels, 3, padding=1)
            block1 = self._make_subblock(conv1)
            block2 = self._make_subblock(conv2)
            submodule = Sequential(*block1, *block2)
            self.up_convs.append(submodule)
            self.conv_layers.append(conv1)
            self.conv_layers.append(conv2)
        #Final convs
        conv1 = torch.nn.Conv2d(n_out_channels, n_out_channels,
                                3, padding=1)
        conv2 = torch.nn.Conv2d(n_out_channels, n_out_channels,
                                3, padding=1)
        conv3 = torch.nn.Conv2d(n_out_channels, self.n_out_channels,
                                3, padding=1)
        block1 = self._make_subblock(conv1)
        block2 = self._make_subblock(conv2)
        self.final_convs = Sequential(*block1, *block2, conv3)
            