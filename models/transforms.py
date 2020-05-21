#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  6 19:38:09 2020

@author: arthur
In this file we define some transformations applied to the output of our 
models. This allows us to keep separate these from the models themselves.
In particular, when we use a heteroskedastic loss, we compare two
transformations that ensure that the precision is positive.
"""

from abc import ABC, abstractmethod
from torch.nn import Module, Parameter
import torch
from torch.nn.functional import softplus


class Transform(Module, ABC):
    """Abstract Base Class for all transforms"""

    def __init__(self):
        super().__init__()

    @abstractmethod
    def transform(self, input):
        pass

    def forward(self, input_):
        return self.transform(input_)

    @abstractmethod
    def __repr__(self):
        pass


class PrecisionTransform(Transform):
    def __init__(self, min_value=0.1):
        super().__init__()
        self._min_value = Parameter(torch.tensor(min_value))

    @property
    def min_value(self):
        return self._min_value

    @min_value.setter
    def min_value(self, value):
        self._min_value = value

    def transform(self, input_):
        # Split in sections of size 2 along channel dimension
        # Careful: the split argument is the size of the sections, not the
        # number of them (although does not matter for 4 channels)
        mean, precision = torch.split(input_, 2, dim=1)
        if (precision != precision).any():
            raise ValueError('Pre-processed precision contains nan')
        precision = self.transform_precision(precision)
        precision.add_(softplus(self.min_value))
        return torch.cat((mean, precision), dim=1)

    @staticmethod
    @abstractmethod
    def transform_precision(precision):
        pass


class SoftPlusTransform(PrecisionTransform):
    def __init__(self, min_value=0.1):
        super().__init__(min_value)

    @staticmethod
    def transform_precision(precision):
        return torch.log(1 + torch.exp(precision))

    def __repr__(self):
        return ''.join(('SoftPlusTransform(', str(self.min_value), ')'))


class SquareTransform(PrecisionTransform):
    def __init__(self, min_value=0.1):
        super().__init__(min_value)

    @staticmethod
    def transform_precision(precision):
        return precision**2

    def __repr__(self):
        return ''.join(('SquareTransform(', str(self.min_value), ')'))
