# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 18:09:38 2022

@author: DELL
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils
# from matplotlib import pyplot as plt
# import imageio # to generate GIF
# from BasicModel import BasicModel
# from utils import NonLocalMeans

        
class DownsampleConv(nn.Module):
    def __init__(self, in_c, out_c, kernel_size=2, stride=2, padding=0, bias=True):
        super(DownsampleConv, self).__init__()
        self.conv = nn.Conv2d(in_c, out_c, kernel_size, stride=stride, padding=padding, bias=bias)
        
    def forward(self, x):
        return self.conv(x)

class Reshape(nn.Module):
    def __init__(self, new_shape):
        super(Reshape, self).__init__()
        self.new_shape = new_shape
    def forward(self, x):
        return x.reshape(self.new_shape)

class SumNorm(nn.Module):
    def __init__(self, ):
        super(SumNorm, self).__init__()
        pass
    def forward(self, x):
        return x/x.sum()

# class Kernel_Generator(nn.Module):
#     def __init__(self, kernel_size=31, scale_factor=3, shift='left'):
#         super(Kernel_Generator, self).__init__()
#         self.kernel_size=kernel_size
#         self.scale_factor=scale_factor
#         self.shift=shift

#     def forward(self, U):
#         '''
#         Generate Gaussian kernel according to cholesky decomposion.
#         \Sigma = M * M^T, M is a lower triangular matrix.
#         Input:
#             U: 2 x 2 torch tensor
#             sf: scale factor
#         Output:
#             kernel: 2 x 2 torch tensor
#         '''
#         kernel_size = self.kernel_size
#         sf = self.scale_factor
#         shift = self.shift
#         #  Mask
#         mask = torch.tensor([[1.0, 0.0],
#                              [0.0, 1.0]], dtype=torch.float32).to(U.device)
#         M = U * mask
    
#         # Set COV matrix using Lambdas and Theta
#         INV_SIGMA = torch.mm(M.t(), M)
    
#         # Set expectation position (shifting kernel for aligned image)
#         if shift.lower() == 'left':
#             MU = kernel_size // 2 - 0.5 * (sf - 1)
#         elif shift.lower() == 'center':
#             MU = float(kernel_size // 2)
#         elif shift.lower() == 'right':
#             MU = kernel_size // 2 + 0.5 * (sf - 1)
#         # else:
#         #     sys.exit('Please input corrected shift parameter: left , right or center!')
    
#         # Create meshgrid for Gaussian
#         X, Y = torch.meshgrid(torch.arange(kernel_size), torch.arange(kernel_size))
#         Z = torch.stack((X, Y), dim=2).unsqueeze(3).to(U.device)   # k x k x 2 x 1
    
#         # Calcualte Gaussian for every pixel of the kernel
#         ZZ = Z - MU
#         ZZ_t = ZZ.permute(0,1,3,2)                  # k x k x 1 x 2
#         raw_kernel = torch.exp(-0.5 * torch.squeeze(ZZ_t.matmul(INV_SIGMA).matmul(ZZ)))
        
    
#         # Normalize the kernel and return
#         kernel = raw_kernel / torch.sum(raw_kernel)   # k x k
#         return kernel.unsqueeze(0).unsqueeze(0)
    
class Kernel_Generator2(nn.Module):
    def __init__(self, kernel_size=31, scale_factor=3, shift='left'):
        super(Kernel_Generator2, self).__init__()
        self.kernel_size=kernel_size
        self.scale_factor=scale_factor
        self.shift=shift
        # self.U = nn.Parameter(torch.ones([2,2], dtype=torch.float), requires_grad=True)
        self.U = nn.Parameter(torch.tensor([[1,0],[0,1]], dtype=torch.float), requires_grad=True)
    def forward(self):
        '''
        Generate Gaussian kernel according to cholesky decomposion.
        \Sigma = M * M^T, M is a lower triangular matrix.
        Input:
            U: 2 x 2 torch tensor
            sf: scale factor
        Output:
            kernel: 2 x 2 torch tensor
        '''
        kernel_size = self.kernel_size
        sf = self.scale_factor
        shift = self.shift
        #  Mask
        mask = torch.tensor([[1.0, 0.0],
                             [1.0, 1.0]], dtype=torch.float32).to(self.U.device)
        M = self.U * mask
    
        # Set COV matrix using Lambdas and Theta
        INV_SIGMA = torch.mm(M.t(), M)
    
        # Set expectation position (shifting kernel for aligned image)
        if shift.lower() == 'left':
            MU = kernel_size // 2 - 0.5 * (sf - 1)
        elif shift.lower() == 'center':
            MU = float(kernel_size // 2)
        elif shift.lower() == 'right':
            MU = kernel_size // 2 + 0.5 * (sf - 1)
        # else:
        #     sys.exit('Please input corrected shift parameter: left , right or center!')
    
        # Create meshgrid for Gaussian
        X, Y = torch.meshgrid(torch.arange(kernel_size), torch.arange(kernel_size))
        Z = torch.stack((X, Y), dim=2).unsqueeze(3).to(self.U.device)   # k x k x 2 x 1
    
        # Calcualte Gaussian for every pixel of the kernel
        ZZ = Z - MU
        ZZ_t = ZZ.permute(0,1,3,2)                  # k x k x 1 x 2
        raw_kernel = torch.exp(-0.5 * torch.squeeze(ZZ_t.matmul(INV_SIGMA).matmul(ZZ)))
        
    
        # Normalize the kernel and return
        kernel = raw_kernel / torch.sum(raw_kernel)   # k x k
        return kernel.unsqueeze(0).unsqueeze(0)

class Downsampler(nn.Module):
    def __init__(self, kernel_size):
        super(Downsampler, self).__init__()
        self.kernel_size = kernel_size
        self.kernel_param = nn.Parameter(torch.ones(kernel_size**2, dtype=torch.float32)/kernel_size**2, requires_grad=True)
    
    def get_kernel(self, ):
        # return F.softmax(self.kernel, 0).reshape([1,1,self.kernel_size,self.kernel_size])
        self.kernel_tensor = F.relu(self.kernel_param, 0).reshape([1,1,self.kernel_size,self.kernel_size])
        self.kernel = np.squeeze(utils.tensor2array(self.kernel_tensor, False))
    
    def forward(self, x):
        nchannel = x.shape[1]
        self.get_kernel()
        kernel = torch.cat([self.kernel_tensor]*nchannel, dim=0).to(x)
        return F.conv2d(x, kernel, bias=None, stride=self.kernel_size, groups=nchannel)

class SRF_generator(nn.Module):
    def __init__(self, HSI_c, MSI_c):
        super(SRF_generator, self).__init__()
        self.HSI_c = HSI_c
        self.MSI_c = MSI_c
        self.srf_param = nn.Parameter(torch.ones([1,MSI_c,HSI_c], dtype=torch.float32), requires_grad=True)
        self.conv = utils.BoxBlur1d(int(2*(0.10*HSI_c)//2+1), 'replicate')
    def forward(self):
        return self.conv(F.softmax(self.srf_param, dim=-1))

class PLRR_Net_SISR(nn.Module):
    def __init__(self, rank, HSI_c, n_feat, n_blur=[200,1000], 
                 blur_size=31, scale_factor=4, 
                 act=nn.LeakyReLU(1e-3), bias=True):
        super(PLRR_Net_SISR, self).__init__()
        
        self.device = 'cuda'
        self.aap = nn.AdaptiveAvgPool1d(rank)
        
        self.blur_size = blur_size
        self.pad_size = blur_size//2
        
        self.scale_factor = scale_factor
        self.NetK = Kernel_Generator2(blur_size, scale_factor, 'center')
        self.NetS = Downsampler(scale_factor)

        self.NetU = nn.Sequential( # shape (1,D,H,W)                   
            nn.Conv2d(HSI_c, n_feat, 4, stride=2, padding=1, bias=bias), # shape (D,n_feat,H,W)
            nn.BatchNorm2d(n_feat), act,
            DownsampleConv(n_feat, n_feat, 2, stride=2, padding=0, bias=bias), 
            nn.BatchNorm2d(n_feat), act,
            DownsampleConv(n_feat, n_feat, 2, stride=2, padding=0, bias=bias), # shape (D,1,H/8,W/8)
            nn.BatchNorm2d(n_feat), act,
            nn.Conv2d(n_feat, HSI_c, 2, stride=2, padding=0, bias=bias),
            )
        self.NetV = nn.Sequential(
            nn.Conv2d(HSI_c,   n_feat, 3, padding=1, bias=bias),    nn.BatchNorm2d(n_feat), act, 
            nn.Conv2d(n_feat, n_feat//4, 3, padding=1, bias=bias), nn.BatchNorm2d(n_feat//4), act, 
            nn.Conv2d(n_feat//4, n_feat, 3, padding=1, bias=bias), nn.BatchNorm2d(n_feat), act, 
            nn.Conv2d(n_feat, n_feat//4, 3, padding=1, bias=bias), nn.BatchNorm2d(n_feat//4), act, 
            nn.Conv2d(n_feat//4, n_feat, 3, padding=1, bias=bias), nn.BatchNorm2d(n_feat), act, 
            nn.Conv2d(n_feat, n_feat//4, 3, padding=1, bias=bias), nn.BatchNorm2d(n_feat//4), act, 
            nn.Conv2d(n_feat//4, n_feat, 3, padding=1, bias=bias), nn.BatchNorm2d(n_feat), act,  
            nn.Conv2d(n_feat, n_feat//4, 3, padding=1, bias=bias), nn.BatchNorm2d(n_feat//4), act, 
            nn.Conv2d(n_feat//4, n_feat, 3, padding=1, bias=bias), nn.BatchNorm2d(n_feat), act, 
            nn.Conv2d(n_feat, n_feat//4, 3, padding=1, bias=bias), nn.BatchNorm2d(n_feat//4), act, 
            nn.Conv2d(n_feat//4, n_feat, 3, padding=1, bias=bias), nn.BatchNorm2d(n_feat), act, 
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
        # U = torch.softmax(U, dim=-1)
        U = U.abs()
        return U

    def getV(self, x):
        _,DD,HH,WW = x.shape
        V = self.NetV(x)
        # V = V.sigmoid()
        V = V.reshape(1,-1,HH*WW)
        V = V.abs()
        return V
    

    def forward(self, LRHS, mode='test'):
        
        # generate image
        LRHS = self.preprocess(LRHS) 
        _,HSI_c,hh,ww = LRHS.shape
        HH = int(self.scale_factor*hh)
        WW = int(self.scale_factor*ww)
        
        U = self.getU(LRHS) # [1,band,rank]
        V = self.getV(F.interpolate(LRHS, scale_factor=self.scale_factor, mode='bicubic', align_corners=True)) # [1,rank,HH*WW]
        HRHS = torch.bmm(U,V)  # [1,HS_C,HH*WW]
        

        if mode=='test':
            HRHS = HRHS.reshape(1,HSI_c,HH,WW)
            return HRHS
        elif mode=='train':
            # generate kernel and LRHS
            HRHS = HRHS.reshape(1,HSI_c,HH,WW)
            blur_kernel = self.NetK()
            self.blur_kernel = np.array(blur_kernel.detach().cpu().squeeze())
            blur_kernel = torch.cat([blur_kernel]*HSI_c, dim=0)
            LRHS_hat = F.conv2d(F.pad(HRHS, pad=[self.pad_size,self.pad_size,self.pad_size,self.pad_size], mode='circular'), 
                                blur_kernel, padding=0, groups=HSI_c, bias=None)
            LRHS_hat = self.NetS(LRHS_hat)

            return LRHS_hat, blur_kernel
        
class PLRR_Net_Fusion(nn.Module):
    def __init__(self, rank, HSI_c, MSI_c, n_feat, n_blur=[200,1000], n_srf=[200,1000], blur_size=31, scale_factor=4, act=nn.LeakyReLU(1e-3), bias=True):
        super(PLRR_Net_Fusion, self).__init__()
        
        self.device = 'cuda'
        self.aap = nn.AdaptiveAvgPool1d(rank)
        
        self.blur_size = blur_size
        self.pad_size = blur_size//2
        
        # self.kernel_input = torch.rand(1,20,device=self.device)/10.
        # self.NetK = nn.Sequential(
        #     nn.Linear(20, 100,bias=True),
        #     act,
        #     nn.Linear(100, 4),
        #     nn.Softplus(),
        #     Reshape([2,2]),
        #     Kernel_Generator(blur_size, scale_factor, 'center')
        #     )
        self.srf_input = torch.rand(1,n_blur[0],device=self.device)/10.
        self.NetF = nn.Sequential(
            nn.Linear(n_srf[0], n_srf[1],bias=True),
            act,
            nn.Linear(n_srf[1], HSI_c*MSI_c),
            Reshape([1,MSI_c,HSI_c]),
            nn.Softmax(dim=-1),
            utils.BoxBlur1d(int(2*(0.10*HSI_c)//2+1), 'replicate')
            )
        # self.NetS = Downsampler(scale_factor)
        
        self.NetK = Kernel_Generator2(blur_size, scale_factor, 'center')
        self.NetS = Downsampler(scale_factor)
        # self.NetF = SRF_generator(HSI_c,MSI_c)
        
        
        
        
        self.NetU = nn.Sequential( # shape (1,D,H,W)                   
            nn.Conv2d(HSI_c, n_feat, 4, stride=2, padding=1, bias=bias), # shape (D,n_feat,H,W)
            nn.BatchNorm2d(n_feat), act,
            DownsampleConv(n_feat, n_feat, 2, stride=2, padding=0, bias=bias), 
            nn.BatchNorm2d(n_feat), act,
            # Downsample(n_feat, n_feat, 2, stride=2, padding=0, bias=bias), 
            # nn.BatchNorm2d(n_feat), act,
            DownsampleConv(n_feat, n_feat, 2, stride=2, padding=0, bias=bias), # shape (D,1,H/8,W/8)
            nn.BatchNorm2d(n_feat), act,
            nn.Conv2d(n_feat, HSI_c, 2, stride=2, padding=0, bias=bias),
            )
        self.NetV = nn.Sequential(
            nn.Conv2d(MSI_c,   n_feat, 3, padding=1, bias=bias),    nn.BatchNorm2d(n_feat), act, 
            nn.Conv2d(n_feat, n_feat//4, 3, padding=1, bias=bias), nn.BatchNorm2d(n_feat//4), act, 
            nn.Conv2d(n_feat//4, n_feat, 3, padding=1, bias=bias), nn.BatchNorm2d(n_feat), act, 
            nn.Conv2d(n_feat, n_feat//4, 3, padding=1, bias=bias), nn.BatchNorm2d(n_feat//4), act, 
            nn.Conv2d(n_feat//4, n_feat, 3, padding=1, bias=bias), nn.BatchNorm2d(n_feat), act, 
            nn.Conv2d(n_feat, n_feat//4, 3, padding=1, bias=bias), nn.BatchNorm2d(n_feat//4), act, 
            nn.Conv2d(n_feat//4, n_feat, 3, padding=1, bias=bias), nn.BatchNorm2d(n_feat), act,  
            nn.Conv2d(n_feat, n_feat//4, 3, padding=1, bias=bias), nn.BatchNorm2d(n_feat//4), act, 
            nn.Conv2d(n_feat//4, n_feat, 3, padding=1, bias=bias), nn.BatchNorm2d(n_feat), act, 
            nn.Conv2d(n_feat, n_feat//4, 3, padding=1, bias=bias), nn.BatchNorm2d(n_feat//4), act, 
            nn.Conv2d(n_feat//4, n_feat, 3, padding=1, bias=bias), nn.BatchNorm2d(n_feat), act, 
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
        # U = torch.softmax(U, dim=-1)
        U = U.abs()
        return U

    def getV(self, x):
        _,DD,HH,WW = x.shape
        V = self.NetV(x)
        # V = V.sigmoid()
        V = V.reshape(1,-1,HH*WW)
        V = V.abs()
        return V
    

    
    def forward(self, LRHS, HRMS, mode='test'):
        
        # generate image
        LRHS = self.preprocess(LRHS) 
        HRMS = self.preprocess(HRMS) 
        _,HSI_c,hh,ww = LRHS.shape
        _,MSI_c,HH,WW = HRMS.shape
        ratio = int(HH/hh)
        
        U = self.getU(LRHS) # [1,band,rank]
        V = self.getV(HRMS) # [1,rank,HH*WW]
        # V = self.getV(torch.cat([HRMS,F.interpolate(LRHS, scale_factor=ratio, mode='bicubic', align_corners=True)], dim=1)) # [1,rank,HH*WW]
        HRHS = torch.bmm(U,V)  # [1,HS_C,HH*WW]
        

        if mode=='test':
            HRHS = HRHS.reshape(1,HSI_c,HH,WW)
            return HRHS
        elif mode=='train':
            # generate SRF and HRMS
            srf = self.NetF(self.srf_input)
            # srf = self.NetF()
            HRMS_hat = srf.matmul(HRHS).reshape(1,MSI_c,HH,WW)
            self.srf = np.array(srf.detach().cpu().squeeze())
            # generate kernel and LRHS
            HRHS = HRHS.reshape(1,HSI_c,HH,WW)
            # blur_kernel = self.NetK(self.kernel_input)
            blur_kernel = self.NetK()
            self.blur_kernel = np.array(blur_kernel.detach().cpu().squeeze())
            blur_kernel = torch.cat([blur_kernel]*HSI_c, dim=0)
            LRHS_hat = F.conv2d(F.pad(HRHS, pad=[self.pad_size,self.pad_size,self.pad_size,self.pad_size], mode='circular'), 
                                blur_kernel, padding=0, groups=HSI_c, bias=None)
            # LRHS_hat = LRHS_hat[:,:,::ratio,::ratio]
            LRHS_hat = self.NetS(LRHS_hat)
            # generate LRMS1 from LRHS
            # LRMS_hat = srf.matmul(LRHS.reshape(1,HSI_c,hh*ww)).reshape(1,MSI_c,hh,ww)
            return LRHS_hat, HRMS_hat, blur_kernel, srf

