# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 14:15:02 2023

@author: DELL
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils
from matplotlib import pyplot as plt
import imageio # to generate GIF
from BasicModel import BasicModel
from .network_unet import ResUNet, U_Net
from .skip import skip

class Shortcut(nn.Module):
    def __init__(self, net):
        super(Shortcut, self).__init__()
        self.net = net
    def forward(self, x):
        return self.net(x)
    
class DenseConnect(nn.Module):
    def __init__(self, net, n_feat):
        super(DenseConnect, self).__init__()
        self.net = nn.ModuleList(net)
        self.conv = nn.Conv2d(n_feat*len(net), n_feat, 1)
    def forward(self, x):
        output = []
        for i in range(len(self.net)):
            if i==0:
                temp = self.net(x)
            else:
                temp = self.net(temp)
            output.append(temp)
        output = self.conv(torch.cat(output, dim=1))
        return output

class DownsampleConv(nn.Module):
    def __init__(self, in_c, out_c, kernel_size=2, stride=2, padding=0, bias=True):
        super(DownsampleConv, self).__init__()
        self.conv = nn.Conv2d(in_c, out_c, kernel_size, stride=stride, padding=padding, bias=bias)
        
    def forward(self, x):
        return self.conv(x)


class PLRR_ResNet(nn.Module):
    def __init__(self, rank, 
                 in_c, 
                 n_feat, 
                 layerU = 5,
                 act = nn.LeakyReLU(1e-3), 
                 bias = True,
                 sigmoid_V = True,
                 softmax_U = True,
                ):
        super(PLRR_ResNet, self).__init__()
        
        self.device = 'cuda'
        self.aap = nn.AdaptiveAvgPool1d(rank)
        self.sigmoid_V = sigmoid_V
        self.softmax_U = softmax_U
        
        self.NetU = nn.Sequential(              
            nn.Conv2d(in_c, n_feat, 4, stride=2, padding=1, bias=bias), # 
            nn.BatchNorm2d(n_feat), act)
        for i in range(layerU-2):
            self.NetU.add_module('DConv'+str(i), DownsampleConv(n_feat, n_feat, 2, stride=2, padding=0, bias=bias))
            self.NetU.add_module('BN'+str(i), nn.BatchNorm2d(n_feat))
            self.NetU.add_module('ACT'+str(i), act)
        self.NetU.add_module('Tail', nn.Conv2d(n_feat, in_c, 2, stride=2, padding=0, bias=bias))

        self.NetV = nn.Sequential(
            nn.Conv2d(in_c,   n_feat, 3, padding=1, bias=bias),    nn.BatchNorm2d(n_feat), act, 
            Shortcut(nn.Sequential(nn.Conv2d(n_feat, n_feat//4, 3, padding=1, bias=bias), 
                                   nn.BatchNorm2d(n_feat//4), 
                                   act, 
                                   nn.Conv2d(n_feat//4, n_feat, 3, padding=1, bias=bias), 
                                   nn.BatchNorm2d(n_feat), 
                                   act)
                     ), 
            Shortcut(nn.Sequential(nn.Conv2d(n_feat, n_feat//4, 3, padding=1, bias=bias), 
                                   nn.BatchNorm2d(n_feat//4), 
                                   act, 
                                   nn.Conv2d(n_feat//4, n_feat, 3, padding=1, bias=bias), 
                                   nn.BatchNorm2d(n_feat), 
                                   act)
                     ), 
            Shortcut(nn.Sequential(nn.Conv2d(n_feat, n_feat//4, 3, padding=1, bias=bias), 
                                   nn.BatchNorm2d(n_feat//4), 
                                   act, 
                                   nn.Conv2d(n_feat//4, n_feat, 3, padding=1, bias=bias), 
                                   nn.BatchNorm2d(n_feat), 
                                   act)
                     ), 
            Shortcut(nn.Sequential(nn.Conv2d(n_feat, n_feat//4, 3, padding=1, bias=bias), 
                                   nn.BatchNorm2d(n_feat//4), 
                                   act, 
                                   nn.Conv2d(n_feat//4, n_feat, 3, padding=1, bias=bias), 
                                   nn.BatchNorm2d(n_feat), 
                                   act)
                     ), 
            Shortcut(nn.Sequential(nn.Conv2d(n_feat, n_feat//4, 3, padding=1, bias=bias), 
                                   nn.BatchNorm2d(n_feat//4), 
                                   act, 
                                   nn.Conv2d(n_feat//4, n_feat, 3, padding=1, bias=bias), 
                                   nn.BatchNorm2d(n_feat), 
                                   act)
                     ), 
            nn.Conv2d(n_feat, rank,   3, padding=1, bias=bias),
            )

    def preprocess(self, x):
        if type(x)==np.ndarray: # an array with shape (H,W,D) ranging from 0 to 1
            if x.ndim!=3:
                raise ValueError('only support the array with ndim=3, but got %d' % (x.ndim) )
            x = utils.array2tensor(x, self.device) # a tensor with shape (1,D,H,W)
        elif type(x)==torch.Tensor: # a tensor with shape (D,H,W) or (1,D,H,W)
            if x.ndim==4:
                x = x.float().to(self.device)
            elif x.ndim==3:
                x = x.float().to(self.device)[None,...]
            else:
                raise ValueError('only support the tensor with ndim=3 or 4, but got %d' % (x.ndim) )

        return x # Now, x is a float32 tensor with shape (1,D,H,W)
    
    def getU(self, x):
        _,DD,HH,WW = x.shape
        U = self.NetU(x)         # shape (1,D,H/8,W/8)
        U = U.reshape(1,DD,-1) # shape (1,D,HW/64)
        U = self.aap(U)        # shape (1,D,Rank)
        if self.softmax_U:
            U = torch.softmax(U, dim=-1)
        return U
    
    def getV(self, x):
        _,DD,HH,WW = x.shape
        V = self.NetV(x)
        if self.sigmoid_V:
            V = V.sigmoid()
        V = V.reshape(1,-1,HH*WW)
        return V
    
    def forward(self, x, mode='test'):
        x = self.preprocess(x) # shape (1,D,H,W)
        _,DD,HH,WW = x.shape
        U = self.getU(x)
        V = self.getV(x)
            
        y = torch.bmm(U,V)
        y = y.reshape(1,DD,HH,WW)

        if mode=='test':
            return y
        elif mode=='train':
            return y, U, V


class PLRR_DenseNet(nn.Module):
    def __init__(self, rank, 
                 in_c, 
                 n_feat, 
                 layerU = 5,
                 act = nn.LeakyReLU(1e-3), 
                 bias = True,
                 sigmoid_V = True,
                 softmax_U = True,
                ):
        super(PLRR_DenseNet, self).__init__()
        
        self.device = 'cuda'
        self.aap = nn.AdaptiveAvgPool1d(rank)
        self.sigmoid_V = sigmoid_V
        self.softmax_U = softmax_U
        
        self.NetU = nn.Sequential(              
            nn.Conv2d(in_c, n_feat, 4, stride=2, padding=1, bias=bias), # 
            nn.BatchNorm2d(n_feat), act)
        for i in range(layerU-2):
            self.NetU.add_module('DConv'+str(i), DownsampleConv(n_feat, n_feat, 2, stride=2, padding=0, bias=bias))
            self.NetU.add_module('BN'+str(i), nn.BatchNorm2d(n_feat))
            self.NetU.add_module('ACT'+str(i), act)
        self.NetU.add_module('Tail', nn.Conv2d(n_feat, in_c, 2, stride=2, padding=0, bias=bias))

        self.NetV = nn.Sequential(
            nn.Conv2d(in_c,   n_feat, 3, padding=1, bias=bias),    nn.BatchNorm2d(n_feat), act, 
            DenseConnect(
            [Shortcut(nn.Sequential(nn.Conv2d(n_feat, n_feat//4, 3, padding=1, bias=bias), 
                                   nn.BatchNorm2d(n_feat//4), 
                                   act, 
                                   nn.Conv2d(n_feat//4, n_feat, 3, padding=1, bias=bias), 
                                   nn.BatchNorm2d(n_feat), 
                                   act)
                     ), 
            Shortcut(nn.Sequential(nn.Conv2d(n_feat, n_feat//4, 3, padding=1, bias=bias), 
                                   nn.BatchNorm2d(n_feat//4), 
                                   act, 
                                   nn.Conv2d(n_feat//4, n_feat, 3, padding=1, bias=bias), 
                                   nn.BatchNorm2d(n_feat), 
                                   act)
                     ), 
            Shortcut(nn.Sequential(nn.Conv2d(n_feat, n_feat//4, 3, padding=1, bias=bias), 
                                   nn.BatchNorm2d(n_feat//4), 
                                   act, 
                                   nn.Conv2d(n_feat//4, n_feat, 3, padding=1, bias=bias), 
                                   nn.BatchNorm2d(n_feat), 
                                   act)
                     ), 
            Shortcut(nn.Sequential(nn.Conv2d(n_feat, n_feat//4, 3, padding=1, bias=bias), 
                                   nn.BatchNorm2d(n_feat//4), 
                                   act, 
                                   nn.Conv2d(n_feat//4, n_feat, 3, padding=1, bias=bias), 
                                   nn.BatchNorm2d(n_feat), 
                                   act)
                     ), 
            Shortcut(nn.Sequential(nn.Conv2d(n_feat, n_feat//4, 3, padding=1, bias=bias), 
                                   nn.BatchNorm2d(n_feat//4), 
                                   act, 
                                   nn.Conv2d(n_feat//4, n_feat, 3, padding=1, bias=bias), 
                                   nn.BatchNorm2d(n_feat), 
                                   act)
                     )]
            ),
            nn.Conv2d(n_feat, rank,   3, padding=1, bias=bias),
            )

    def preprocess(self, x):
        if type(x)==np.ndarray: # an array with shape (H,W,D) ranging from 0 to 1
            if x.ndim!=3:
                raise ValueError('only support the array with ndim=3, but got %d' % (x.ndim) )
            x = utils.array2tensor(x, self.device) # a tensor with shape (1,D,H,W)
        elif type(x)==torch.Tensor: # a tensor with shape (D,H,W) or (1,D,H,W)
            if x.ndim==4:
                x = x.float().to(self.device)
            elif x.ndim==3:
                x = x.float().to(self.device)[None,...]
            else:
                raise ValueError('only support the tensor with ndim=3 or 4, but got %d' % (x.ndim) )

        return x # Now, x is a float32 tensor with shape (1,D,H,W)
    
    def getU(self, x):
        _,DD,HH,WW = x.shape
        U = self.NetU(x)         # shape (1,D,H/8,W/8)
        U = U.reshape(1,DD,-1) # shape (1,D,HW/64)
        U = self.aap(U)        # shape (1,D,Rank)
        if self.softmax_U:
            U = torch.softmax(U, dim=-1)
        return U
    
    def getV(self, x):
        _,DD,HH,WW = x.shape
        V = self.NetV(x)
        if self.sigmoid_V:
            V = V.sigmoid()
        V = V.reshape(1,-1,HH*WW)
        return V
    
    def forward(self, x, mode='test'):
        x = self.preprocess(x) # shape (1,D,H,W)
        _,DD,HH,WW = x.shape
        U = self.getU(x)
        V = self.getV(x)
            
        y = torch.bmm(U,V)
        y = y.reshape(1,DD,HH,WW)

        if mode=='test':
            return y
        elif mode=='train':
            return y, U, V

class PLRR_UNet(nn.Module):
    def __init__(self, rank, 
                 in_c, 
                 n_feat, 
                 layerU = 5,
                 act = nn.LeakyReLU(1e-3), 
                 bias = True,
                 sigmoid_V = True,
                 softmax_U = True,
                ):
        super(PLRR_UNet, self).__init__()
        
        self.device = 'cuda'
        self.aap = nn.AdaptiveAvgPool1d(rank)
        self.sigmoid_V = sigmoid_V
        self.softmax_U = softmax_U
        
        self.NetU = nn.Sequential(              
            nn.Conv2d(in_c, n_feat, 4, stride=2, padding=1, bias=bias), # 
            nn.BatchNorm2d(n_feat), act)
        for i in range(layerU-2):
            self.NetU.add_module('DConv'+str(i), DownsampleConv(n_feat, n_feat, 2, stride=2, padding=0, bias=bias))
            self.NetU.add_module('BN'+str(i), nn.BatchNorm2d(n_feat))
            self.NetU.add_module('ACT'+str(i), act)
        self.NetU.add_module('Tail', nn.Conv2d(n_feat, in_c, 2, stride=2, padding=0, bias=bias))
        
        nlayer = 4
        self.NetV = skip(in_c, rank,  
               num_channels_down = [32,64,128,256], #[128]*nlayer,
               num_channels_up =   [32,64,128,256], #[128]*nlayer,
               num_channels_skip =    [4]*nlayer,  
               filter_size_up = 3, filter_size_down = 3,  filter_skip_size=1,
               upsample_mode='bilinear', 
               need1x1_up=False,
               need_sigmoid=False, need_bias=True, pad='reflection', act_fun='LeakyReLU').to(self.device)
        

    def preprocess(self, x):
        if type(x)==np.ndarray: # an array with shape (H,W,D) ranging from 0 to 1
            if x.ndim!=3:
                raise ValueError('only support the array with ndim=3, but got %d' % (x.ndim) )
            x = utils.array2tensor(x, self.device) # a tensor with shape (1,D,H,W)
        elif type(x)==torch.Tensor: # a tensor with shape (D,H,W) or (1,D,H,W)
            if x.ndim==4:
                x = x.float().to(self.device)
            elif x.ndim==3:
                x = x.float().to(self.device)[None,...]
            else:
                raise ValueError('only support the tensor with ndim=3 or 4, but got %d' % (x.ndim) )

        return x # Now, x is a float32 tensor with shape (1,D,H,W)
    
    def getU(self, x):
        _,DD,HH,WW = x.shape
        U = self.NetU(x)         # shape (1,D,H/8,W/8)
        U = U.reshape(1,DD,-1) # shape (1,D,HW/64)
        U = self.aap(U)        # shape (1,D,Rank)
        if self.softmax_U:
            U = torch.softmax(U, dim=-1)
        return U
    
    def getV(self, x):
        _,DD,HH,WW = x.shape
        V = self.NetV(x)
        if self.sigmoid_V:
            V = V.sigmoid()
        V = V.reshape(1,-1,HH*WW)
        return V
    
    def forward(self, x, mode='test'):
        x = self.preprocess(x) # shape (1,D,H,W)
        _,DD,HH,WW = x.shape
        U = self.getU(x)
        V = self.getV(x)
            
        y = torch.bmm(U,V)
        y = y.reshape(1,DD,HH,WW)

        if mode=='test':
            return y
        elif mode=='train':
            return y, U, V
