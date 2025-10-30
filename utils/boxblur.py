# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F

class BoxBlur1d(nn.Module):
    def __init__(self, win_size, mode):
        super(BoxBlur1d, self).__init__()
        self.mode = mode
        self.win_size = win_size
        self.kernel = torch.ones(1,1,win_size)/win_size
    def forward(self, x):
        _,n_channel,_ = x.shape
        kernel = self.kernel.to(x)
        kernel = torch.cat([kernel]*n_channel, dim=0)
        pad_size = self.win_size//2
        y = F.conv1d(F.pad(x, pad=[pad_size, pad_size], mode=self.mode), kernel, padding=0, groups=n_channel)
        return y
    
class BoxBlur2d(nn.Module):
    def __init__(self, win_size, mode):
        super(BoxBlur2d, self).__init__()
        self.mode = mode
        self.win_size = win_size
        self.kernel = torch.ones(1,1,win_size,win_size)/win_size/win_size
    def forward(self, x):
        _,n_channel,_,_ = x.shape
        kernel = self.kernel.to(x)
        kernel = torch.cat([kernel]*n_channel, dim=0)
        pad_size = self.win_size//2
        y = F.conv2d(F.pad(x, pad=[pad_size, pad_size, pad_size, pad_size], mode=self.mode), kernel, padding=0, groups=n_channel)
        return y