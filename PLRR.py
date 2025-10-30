# -*- coding: utf-8 -*-
"""
Created on Thu Sep 15 18:47:51 2022

@author: DELL
"""

import numpy as np
import torch
import torch.nn as nn
import utils
from matplotlib import pyplot as plt
# import imageio # to generate GIF
from BasicModel import BasicModel
from networks.PLRR_Net import PLRR_Net
from networks.PLRR_Net_SR import PLRR_Net_Fusion
from networks.PLRR_Net_other_backbones import PLRR_ResNet, PLRR_DenseNet, PLRR_UNet
# from skimage.io import imshow

class PLRR(BasicModel):  
    def __init__(self, 
                 img_degraded, 
                 gt=None, 
                 mask=None, 
                 guidance=None, 
                 guidance_lr=None,
                 rank = 15, 
                 tol = 1e-6, 
                 n_feat = 256, 
                 init_weight=None, 
                 net_mode='conv', 
                 num_epoch=1000, 
                 lr=1e-3, 
                 reg_sigma=0.01, 
                 smooth_coef = 0.8, 
                 seed = 8888, 
                 input_mode = 'degraded_img',
                 loss_mode = 'l1', 
                 metric_mode = 'psnr3d', 
                 print_every = 100, 
                 rgb = [30,14,3], 
                 decloud_shape=None,
                 task = None,
                 gif_dict = {'save_root': None, 'frequency': 100, 'speed': 0.5, 'resize': [512,512]},
                 print_image = False):
        super().__init__(img_degraded=img_degraded, gt=gt, mask=mask, guidance=guidance, guidance_lr=guidance_lr,
                         net_mode=net_mode, rank = rank, n_feat = n_feat, init_weight=init_weight, 
                         num_epoch=num_epoch, lr=lr, tol = tol, loss_mode = loss_mode, 
                         input_mode = input_mode, reg_sigma=reg_sigma, 
                         seed = seed, task = task,
                         smooth_coef = smooth_coef, 
                         metric_mode = metric_mode, print_every = print_every, rgb = rgb, 
                         decloud_shape=decloud_shape,
                         gif_dict = gif_dict, print_image = print_image)
        
        if self.task == 'denoising':
            self.closure = self.closure_denoising
        elif self.task == 'inpainting':
            self.input_mode = 'PLRR_inpainting'
            self.reg_sigma = 0 # disenable the noise regularization trick
            self.smooth_coef = 0 # disenable the moving average trick
            self.closure = self.closure_inpainting
        elif self.task == 'decloud':
            self.input_mode = 'PLRR_inpainting'
            self.reg_sigma = 0 # disenable the noise regularization trick
            self.smooth_coef = 0 # disenable the moving average trick
            self.closure = self.closure_decloud
        elif self.task == 'sr':
            self.input_mode = 'degraded_img'
            self.reg_sigma = 0 # disenable the noise regularization trick
            self.smooth_coef = 0 # disenable the moving average trick
            self.closure = self.closure_sr
        elif self.task == 'hsmsf':
            self.input_mode = 'degraded_img'
            self.reg_sigma = 0 # disenable the noise regularization trick
            self.smooth_coef = 0 # disenable the moving average trick
            self.closure = self.closure_hsmsf

    def closure_denoising(self):
        # regularize input
        net_input = self.Input + torch.randn_like(self.Input) * self.reg_sigma
        self.model.train()
        self.optimizer.zero_grad()
        xhat, U, V = self.model(net_input, mode='train') 
        loss = self.loss_fn(xhat, self.label)
        loss.backward()
        self.optimizer.step()
        if self.device == 'cuda':
            torch.cuda.empty_cache()
        self.loss_list.append(loss.item())
        return xhat
    
    def closure_inpainting(self):
        if self.mask is None:
            raise Exception('mask must be provided for inpainting! （填充任务必须提供变量mask！）')

        self.model.train()
        self.optimizer.zero_grad()
        xhat = self.model(self.Input)
        loss = self.loss_fn(xhat*self.mask_tensor, self.label*self.mask_tensor)
        loss.backward()
        self.optimizer.step()
        if self.device == 'cuda':
            torch.cuda.empty_cache()
        self.loss_list.append(loss.item())
        return xhat

    def closure_decloud(self):
        if self.mask is None:
            raise Exception('mask must be provided for inpainting! （填充任务必须提供变量mask！）')
        
        self.model.train()
        self.optimizer.zero_grad()
        xhat = self.model(self.Input)
        loss = self.loss_fn(xhat*self.mask_tensor, self.label*self.mask_tensor)
        loss.backward()
        self.optimizer.step()
        if self.device == 'cuda':
            torch.cuda.empty_cache()
        self.loss_list.append(loss.item())
        
        if self.gt is not None:
            nband = self.decloud_shape[2]
            xhat = xhat[:,:nband,:,:]
        return xhat
    
    def closure_sr(self):
        self.model.train()
        self.optimizer.zero_grad()
        LRHS_hat, K = self.model(self.Input, 'train')
        loss = self.loss_fn(LRHS_hat, self.Input)

        loss.backward()
        self.optimizer.step()
        if self.device == 'cuda':
            torch.cuda.empty_cache()
        self.loss_list.append(loss.item())
        
        with torch.no_grad():
            self.LRHS_hat = utils.tensor2array(LRHS_hat, clip=True)
            self.model.eval()
            xhat = self.model(self.Input, 'test')
        return xhat
    
    def closure_hsmsf(self):
        self.model.train()
        self.optimizer.zero_grad()
        LRHS_hat, HRMS_hat, K, F = self.model(self.Input, self.guidance_tensor, 'train')
        loss = 0
        loss = self.loss_fn(LRHS_hat, self.Input)
        loss = loss + self.loss_fn(HRMS_hat, self.guidance_tensor)
       
        loss.backward()
        self.optimizer.step()
        if self.device == 'cuda':
            torch.cuda.empty_cache()
        self.loss_list.append(loss.item())
        
        with torch.no_grad():
            self.LRHS_hat = utils.tensor2array(LRHS_hat, clip=True)
            self.HRMS_hat = utils.tensor2array(HRMS_hat, clip=True)
            self.model.eval()
            xhat = self.model(self.Input, self.guidance_tensor, 'test')
        return xhat


    def set_net(self):
        print('Network structure: %s'%(self.net_mode))
        if self.task=='hsmsf':
            MSI_c = self.guidance.shape[-1]
            HSI_c = self.img_degraded.shape[-1]
            self.model = PLRR_Net_Fusion(rank=self.rank, HSI_c=HSI_c, MSI_c=MSI_c, 
                                         n_blur=[200,1000], n_srf=[200,1000], blur_size=31,
                                         n_feat=self.n_feat).to(self.device)
        else:
            in_c = self.img_degraded.shape[-1]
            softmax_U = True
            sigmoid_V = True
            layerU = 5
            if self.net_mode == 'conv':
                self.model = PLRR_Net(in_c=in_c, 
                                      rank=self.rank, 
                                      n_feat=self.n_feat,
                                      layerU = layerU,
                                      sigmoid_V = sigmoid_V,
                                      softmax_U = softmax_U).to(self.device)
            elif self.net_mode == 'resnet':
                self.model = PLRR_ResNet(in_c=in_c, 
                                      rank=self.rank, 
                                      n_feat=self.n_feat,
                                      layerU = layerU,
                                      sigmoid_V = sigmoid_V,
                                      softmax_U = softmax_U).to(self.device)
            elif self.net_mode == 'densenet':
                self.model = PLRR_DenseNet(in_c=in_c, 
                                      rank=self.rank, 
                                      n_feat=self.n_feat,
                                      layerU = layerU,
                                      sigmoid_V = sigmoid_V,
                                      softmax_U = softmax_U).to(self.device)
            elif self.net_mode == 'unet':
                self.model = PLRR_UNet(in_c=in_c, 
                                      rank=self.rank, 
                                      n_feat=self.n_feat,
                                      layerU = layerU,
                                      sigmoid_V = False,
                                      softmax_U = softmax_U).to(self.device)
        if self.init_weight is not None:
            self.model.load_state_dict(self.init_weight)
            if self.task == 'hsmsf':
                self.model.NetK.U = nn.Parameter(torch.tensor([[1,0],[0,1]], dtype=torch.float), requires_grad=True)
                

    def getV(self):
        with torch.no_grad():
            _,_,HH,WW = self.Input.shape
            V = self.model.cpu().getV(self.Input.cpu()) # [1,rank,HW]
            V = utils.tensor2array(V, clip=False) # [1,HW,rank]
            V = np.reshape(V, [HH, WW, -1])
            return V
    
    def getU(self):
        with torch.no_grad():
            U = self.model.cpu().getU(self.Input.cpu()) # [1,D,rank]
            U = utils.tensor2array(U, clip=False)
            return U
        
    # def init_x(self, img_noisy):
    #     rank = 3
    #     hh,ww,cc = img_noisy.shape
    #     uu,ss,vv = np.linalg.svd(img_noisy.reshape([hh*ww, cc]), False)
    #     X = np.reshape(uu[:,:rank] @ np.diag(ss[:rank]) @ vv[:rank,:], [hh,ww,cc])
    #     return np.clip(X, 0,1)
    
    def plotU(self, ):
        U = self.getU() # [rank, D]
        for i in range(U.shape[0]):
            plt.plot(U[i,:])
    
    def plotV(self, ):
        V = self.getV()
        rank = V.shape[-1]
        if rank==2:
            row = 2
            col = 1
        else:
            row = 3
            col = rank//row
            if rank%row!=0:
                col = col+1
        f, ax = plt.subplots(row, col, sharey=True, figsize=(15,15))
        index = 0
        for i in range(row):
            for j in range(col):
                index = index+1
                if index<=rank:
                    ax[i][j].imshow((255*V[:,:,index-1]).astype(np.uint8))
                    ax[i][j].set_title('The %dth Base'%(index))
                else:
                    ax[i][j].imshow(np.zeros(shape=(V.shape[:2]), dtype=np.uint8))
                    ax[i][j].set_title('Null Base')
        plt.tight_layout()
        plt.show()
       
        
