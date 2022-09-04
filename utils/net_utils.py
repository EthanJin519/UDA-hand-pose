# Copyright (c) Liuhao Ge. All Rights Reserved.
r"""
Network utilities
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os.path as osp
from collections import OrderedDict

# import numpy
# from numpy import *
# import numpy as np

import math


import torch
import torch.nn as nn


class ConvLayer(nn.Sequential):
    def __init__(self, inputs, num_outputs, kernel_sz=3, use_normalizer=True, use_activation=True):
        super(ConvLayer, self).__init__()
        padding = int(kernel_sz // 2)
        self.add_module('conv', nn.Conv2d(inputs, num_outputs, kernel_size=kernel_sz, padding=padding))
        if use_normalizer: self.add_module('norm', nn.BatchNorm2d(num_outputs, eps=1e-03, momentum=0.1))
        if use_activation: self.add_module('relu', nn.ReLU(
            inplace=True))  # nn.PReLU(init=0.0)) # 0.01, 0.25 # nn.ReLU(inplace=True)

    def forward(self, x):
        return super(ConvLayer, self).forward(x)


class FCLayer(nn.Sequential):
    def __init__(self, inputs, num_outputs, use_activation=True, use_dropout=True):
        super(FCLayer, self).__init__()
        self.add_module('fc', nn.Linear(inputs, num_outputs))
        if use_activation: self.add_module('relu', nn.ReLU(
            inplace=True))  # nn.PReLU(init=0.0)) # 0.01, 0.25 # nn.ReLU(inplace=True)
        if use_dropout: self.add_module('dropout', nn.Dropout(p=0.5, inplace=False))

    def forward(self, x):
        return super(FCLayer, self).forward(x)


class Residual(nn.Module):
    def __init__(self, numIn, numOut):
        super(Residual, self).__init__()
        self.numIn = numIn
        self.numOut = numOut
        self.bn = nn.BatchNorm2d(self.numIn)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(self.numIn, self.numOut // 2, bias=True, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(self.numOut // 2)
        self.conv2 = nn.Conv2d(self.numOut // 2, self.numOut // 2, bias=True, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(self.numOut // 2)
        self.conv3 = nn.Conv2d(self.numOut // 2, self.numOut, bias=True, kernel_size=1)

        if self.numIn != self.numOut:
            self.conv4 = nn.Conv2d(self.numIn, self.numOut, bias=True, kernel_size=1)

    def forward(self, x):
        residual = x
        out = self.bn(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)

        if self.numIn != self.numOut:
            residual = self.conv4(x)

        return out + residual



class Residual2(nn.Module):
    def __init__(self, numIn, numOut):
        super(Residual2, self).__init__()
        self.numIn = numIn
        self.numOut = numOut
        self.bn = nn.BatchNorm2d(self.numIn)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(self.numIn, self.numOut // 2, bias=True, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(self.numOut // 2)
        self.conv2 = nn.Conv2d(self.numOut // 2, self.numOut // 2, bias=True, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(self.numOut // 2)
        self.conv3 = nn.Conv2d(self.numOut // 2, self.numOut, bias=True, kernel_size=1)


        self.conv4 = nn.Conv2d(self.numIn, self.numOut, bias=True, kernel_size=3, stride=2, padding=1)
            #self.conv5 = nn.Conv2d(self.numIn, self.numOut, bias=True, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        residual = x
        out = self.bn(x)
        out = self.relu(out)
        out = self.conv1(out)
        o1 = out.shape

        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)

        o2  = out.shape
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        o3 = out.shape

       # if self.numIn != self.numOut:
        residual = self.conv4(x)
        o4 = residual.shape

        return out + residual


class make_head(nn.Module):
    def __init__(self, num_layers, channel_dim, num_keypoints):
        super(make_head, self).__init__()
        self.conv1 = nn.Conv2d(channel_dim, channel_dim, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(channel_dim)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(
                in_channels=channel_dim,
                out_channels=num_keypoints,
                kernel_size=1,
                stride=1,
                padding=0
            )
        nn.init.normal_(self.conv1.weight, std=0.001)
        nn.init.constant_(self.conv1.bias, 0)

        nn.init.normal_(self.conv2.weight, std=0.001)
        nn.init.constant_(self.conv2.bias, 0)

    def forward(self, x):

        x0 = self.conv1(x)
        x0 = self.bn1(x0)
        x0 = self.relu(x0)
        x1 = self.conv2(x0)

        return x0, x1


class make_head2(nn.Module):
    def __init__(self, num_layers, channel_dim, num_keypoints):
        super(make_head2, self).__init__()
        self.conv1 = nn.Conv2d(channel_dim, channel_dim, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(channel_dim)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(
                in_channels=channel_dim,
                out_channels=num_keypoints,
                kernel_size=1,
                stride=1,
                padding=0
            )
        nn.init.normal_(self.conv1.weight, std=0.001)
        nn.init.constant_(self.conv1.bias, 0)

        nn.init.normal_(self.conv2.weight, std=0.001)
        nn.init.constant_(self.conv2.bias, 0)

    def forward(self, x):

        x0 = self.conv1(x)
        x1 = self.bn1(x0)
        x1 = self.relu(x1)
        x1 = self.conv2(x1)

        return x0, x1


class make_head3(nn.Module):
    def __init__(self, num_layers, channel_dim, num_keypoints):
        super(make_head3, self).__init__()
        self.conv1 = nn.Conv2d(channel_dim, channel_dim, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(channel_dim)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(channel_dim, channel_dim, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(channel_dim)
        self.relu = nn.ReLU()
        self.conv3 = nn.Conv2d(
                in_channels=channel_dim,
                out_channels=num_keypoints,
                kernel_size=1,
                stride=1,
                padding=0
            )
        nn.init.normal_(self.conv1.weight, std=0.001)
        nn.init.constant_(self.conv1.bias, 0)

        nn.init.normal_(self.conv2.weight, std=0.001)
        nn.init.constant_(self.conv2.bias, 0)

        nn.init.normal_(self.conv3.weight, std=0.001)
        nn.init.constant_(self.conv3.bias, 0)

    def forward(self, x):

        x = self.conv1(x)
        x0 = self.bn1(x)
        x0 = self.relu(x0)
        x1 = self.conv2(x0)
        x1 = self.bn2(x1)
        x1 = self.relu(x1)
        x1 = self.conv3(x1)

        return x, x0, x1


class Residualx(nn.Module):
    def __init__(self, numIn, numOut):
        super(Residualx, self).__init__()
        self.numIn = numIn
        self.numOut = numOut
      #  self.bn = nn.BatchNorm2d(self.numIn)
        
        self.conv1 = nn.Conv2d(self.numIn, self.numOut // 2, bias=True, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(self.numOut // 2)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(self.numOut // 2, self.numOut // 2, bias=True, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(self.numOut // 2)
        self.conv3 = nn.Conv2d(self.numOut // 2, self.numOut, bias=True, kernel_size=1)

        if self.numIn != self.numOut:
            self.conv4 = nn.Conv2d(self.numIn, self.numOut, bias=True, kernel_size=1)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        if self.numIn != self.numOut:
            residual = self.conv4(x)

        return out + residual

def load_net_model(model_path, net):
    assert (osp.isfile(model_path)), ('The model does not exist! Error path:\n%s' % model_path)

    model_dict = torch.load(model_path, map_location='cpu')
    module_prefix = 'module.'
    module_prefix_len = len(module_prefix)

    for k in model_dict.keys():
        if k[:module_prefix_len] != module_prefix:
            net.load_state_dict(model_dict)
            return 0

    del_keys = filter(lambda x: 'num_batches_tracked' in x, model_dict.keys())
    for k in del_keys:
        del model_dict[k]

    model_dict = OrderedDict([(k[module_prefix_len:], v) for k, v in model_dict.items()])
    net.load_state_dict(model_dict)
    return 0


class my_sparse_mm(torch.autograd.Function):
    """
    this function is forked from https://github.com/xbresson/spectral_graph_convnets
    Implementation of a new autograd function for sparse variables,
    called "my_sparse_mm", by subclassing torch.autograd.Function
    and implementing the forward and backward passes.
    """

    def forward(self, W, x):  # W is SPARSE
        self.save_for_backward(W, x)
        y = torch.mm(W, x)
        return y

    def backward(self, grad_output):
        W, x = self.saved_tensors
        grad_input = grad_output.clone()
        grad_input_dL_dW = torch.mm(grad_input, x.t())
        grad_input_dL_dx = torch.mm(W.t(), grad_input)
        return grad_input_dL_dW, grad_input_dL_dx


def load_net_model(model_path, net):
    assert (osp.isfile(model_path)), ('The model does not exist! Error path:\n%s' % model_path)

    model_dict = torch.load(model_path, map_location='cpu')
    module_prefix = 'module.'
    module_prefix_len = len(module_prefix)

    for k in model_dict.keys():
        if k[:module_prefix_len] != module_prefix:
            net.load_state_dict(model_dict)
            return 0

    del_keys = filter(lambda x: 'num_batches_tracked' in x, model_dict.keys())
    for k in del_keys:
        del model_dict[k]

    model_dict = OrderedDict([(k[module_prefix_len:], v) for k, v in model_dict.items()])
    net.load_state_dict(model_dict)
    return 0


def norm_v(x):
    x1 = x ** 2
    x2 = x1.sum()
    x3 = math.sqrt(x2)

    return x3
