# -*- coding: utf-8 -*-
import torch
import torch.nn as nn


class OrthLoss(nn.Module):
    def __init__(self):
        super(OrthLoss, self).__init__()
        pass
    def forward(self, a):
        return torch.triu(a @ a.T, 1).mean()
    

class FFTL1_Loss(nn.Module):
    def __init__(self):
        super(FFTL1_Loss, self).__init__()

    def forward(self,x,y):
        _,c,h,w = x.shape
        return (torch.fft.rfft2(x)-torch.fft.rfft2(y)).abs().mean()

class SV_Loss(nn.Module):
    def __init__(self):
        super(SV_Loss, self).__init__()

    def forward(self,x,y,dim=2):
        _,c,h,w = x.shape
        x = x.reshape(_,c,h*w)
        y = y.reshape(_,c,h*w)
        torch.manual_seed(8888)
        torch.cuda.manual_seed_all(8888)
        _,sx,_ = torch.svd_lowrank(x,c)
        _,sy,_ = torch.svd_lowrank(y,c)
        return (sx-sy).abs().mean()

class TV_Loss(nn.Module):
    def __init__(self):
        super(TV_Loss, self).__init__()

    def forward(self,a):
        if len(a.shape)==4:
            gradient_a_x = torch.abs(a[  :, :, :, :-1] - a[  :, :, :, 1:])
            gradient_a_y = torch.abs(a[  :, :, :-1, :] - a[  :, :, 1:, :])
            return gradient_a_y.mean()+gradient_a_x.mean()
        elif len(a.shape)==3:
            gradient_a_x = torch.abs(a[ :, :, :-1] - a[ :, :, 1:])
            return gradient_a_x.mean()
        

class SSTV_Loss(nn.Module):
    def __init__(self):
        super(SSTV_Loss, self).__init__()

    def forward(self, a):
        gradient_a_z = torch.abs(a[ :, :-1, :, :] - a[ :, 1:, :, :])
        gradient_a_yz = torch.abs(gradient_a_z[ :, :, :-1, :] - gradient_a_z[ :, :, 1:, :])
        gradient_a_xz = torch.abs(gradient_a_z[ :, :, :, :-1] - gradient_a_z[ :, :, :, 1:])
        return gradient_a_yz.mean()+gradient_a_xz.mean()
    