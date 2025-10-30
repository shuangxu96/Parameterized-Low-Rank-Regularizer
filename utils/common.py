# -*- coding: utf-8 -*-
"""
Created on Mon Aug 15 16:49:32 2022

@author: BSawa
"""
import numpy as np
import torch
import random
from skimage.io import imshow, imread, imsave

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
 
def tensor2array(x, clip=True):
    if x.ndim==4: # an image [1,C,H,W]
        x = np.array(x.detach().squeeze(0).permute(1,2,0).cpu()) # [H,W,C]
    elif x.ndim==3: # an signal [1,C,N]
        x = np.array(x.detach().squeeze(0).permute(1,0).cpu()) # [N,C]
    if clip:
        x = np.clip(x, 0., 1.)
    return x

def array2tensor(x, device):
    x = torch.tensor(x, dtype=torch.float32, device=device)
    x = x.permute(2,0,1)[None,...]
    return x

def np2torch(x):
    ''' array [H,W,C] -> tensor [1,C,H,W] ''' 
    x = torch.Tensor(x)
    x = x.permute(2,0,1)
    x = x[None,:,:,:]
    return x

def torch2np(x):
    ''' tensor [1,C,H,W] -> array [H,W,C] ''' 
    x = x.detach().squeeze(0).cpu() 
    x = np.array(x)
    return x

def load_img(fname, torch=False):
    ''' load image '''
    # load image as an array
    img = imread(fname)
    
    # unsqueeze the image if it is gray: [H,W] -> [H,W,1]
    if np.ndim(img)==2:
        img = img[:,:,None]
    
    # [0,255] -> [0,1]
    img = img.astype(np.float32) / 255.
    
    if torch:
        # array [H,W,C] -> tensor [1,C,H,W]
        img = np2torch(img)
    
    return img

def write_img(fname, arr):
    ''' write image '''
    # convert a tensor as an array: tensor [1,C,H,W] -> array [H,W,C]
    if torch.is_tensor(arr):
        arr = torch2np(arr)
    
    # convert np.float32 as np.uint8
    arr = 255.*np.clip(arr, 0., 1.)
    arr = arr.astype(np.uint8)
    
    # write image
    imsave(fname, arr)

def show_img(img):
    if isinstance(img, torch.Tensor):
        img = tensor2array(img, clip=False)
    imshow(img)

def im2double(x):
    x = x.astype(np.float32)
    x = x/255.
    return x


                    
                    
