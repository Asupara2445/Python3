import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from collections import deque, OrderedDict
import functools

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        if hasattr(m, 'weight'):
            m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1: 
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def branch_out(in_dim, out_dim=3):
    _layers = [ nn.ReflectionPad2d(1),
                nn.Conv2d(in_dim, out_dim, 
                kernel_size = 3, padding=0, bias=False)]    
    _layers += [nn.Tanh()]

    return nn.Sequential(*_layers)
    
def KL_loss(mu, log_sigma):
    loss = -log_sigma + .5 * (-1 + torch.exp(2. * log_sigma) + mu**2)
    loss = torch.mean(loss)
    return loss
 
def pad_conv_norm(dim_in, dim_out, norm_layer, kernel_size=3, use_activation=True, 
                  use_bias=False, activation=nn.ReLU(True)):
    # designed for generators
    seq = []
    if kernel_size != 1:
        seq += [nn.ReflectionPad2d(1)]
        
    seq += [nn.Conv2d(dim_in, dim_out, kernel_size=kernel_size, padding=0, bias=use_bias),
            norm_layer(dim_out)]
    
    if use_activation:
        seq += [activation]
    
    return nn.Sequential(*seq)

def conv_norm(dim_in, dim_out, norm_layer, kernel_size=3, stride=1, use_activation=True, 
              use_bias=False, activation=nn.ReLU(True), use_norm=True,padding=None):
    # designed for discriminator
    
    if kernel_size == 3:
        padding = 1 if padding is None else padding
    else:
        padding = 0 if padding is None else padding

    seq = [nn.Conv2d(dim_in, dim_out, kernel_size=kernel_size, padding=padding, bias=use_bias, stride=stride),
           ]
    if use_norm:
        seq += [norm_layer(dim_out)]
    if use_activation:
        seq += [activation]
    
    return nn.Sequential(*seq)
