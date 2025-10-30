# -*- coding: utf-8 -*-
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr_fn
from skimage.metrics import structural_similarity as ssim_fn


def PSNR3D(x,y):
    psnr_val = 0.
    x = (255.*x).astype(np.uint8)
    y = (255.*y).astype(np.uint8)
    for i in range(x.shape[-1]):
        psnr_val = psnr_val+psnr_fn(x[:,:,i], y[:,:,i])
    psnr_val = psnr_val/x.shape[-1]
    return psnr_val  

def SSIM3D(x,y):
    ssim_val = 0.
    x = (255.*x).astype(np.uint8)
    y = (255.*y).astype(np.uint8)
    for i in range(x.shape[-1]):
        ssim_val = ssim_val+ssim_fn(x[:,:,i], y[:,:,i])
    ssim_val = ssim_val/x.shape[-1]
    return ssim_val  

def SAM(x,y):
    HH,WW,CC = x.shape
    x = x.reshape(HH*WW,CC)
    y = y.reshape(HH*WW,CC)
    sam = np.sum(x*y,axis=-1) / np.sqrt( (np.sum(x*x,axis=-1)+1e-6) * (np.sum(y*y,axis=-1)+1e-6) )
    sam = np.clip(sam, 0., 1.)
    sam = np.arccos(sam)
    sam = np.mean(sam)
    sam = 180*sam/np.pi
    return sam

def ERGAS(img_fake, img_real, scale=1):
    """ERGAS for 2D (H, W) or 3D (H, W, C) image; uint or float [0, 1].
    scale = spatial resolution of HRMS / spatial resolution of MUL, default 4."""
    if not img_fake.shape == img_real.shape:
        raise ValueError('Input images must have the same dimensions.')
    img_fake_ = img_fake.astype(np.float64)
    img_real_ = img_real.astype(np.float64)
    if img_fake_.ndim == 2:
        mean_real = img_real_.mean()
        mse = np.mean((img_fake_ - img_real_)**2)
        return 100 / scale * np.sqrt(mse / (mean_real**2 + np.finfo(np.float64).eps))
    elif img_fake_.ndim == 3:
        means_real = img_real_.reshape(-1, img_real_.shape[2]).mean(axis=0)
        mses = ((img_fake_ - img_real_)**2).reshape(-1, img_fake_.shape[2]).mean(axis=0)
        return 100 / scale * np.sqrt((mses / (means_real**2 + np.finfo(np.float64).eps)).mean())
    else:
        raise ValueError('Wrong input image dimensions.')

def print_metrics(metrics, col_labels=['PSNR', 'SSIM', 'ERGAS']):
    num_row, num_col = metrics.shape
    first_row  = 'Metrics \t '
    for i in range(num_col):
        first_row += '%s \t    '%(col_labels[i])
    print(first_row)
    for j in range(num_row):
        temp_row = '%7d \t'%(j+1)
        for i in range(num_col):
            temp_row += '%2.3f \t'%(metrics[j,i])
        print(temp_row)


def Convert3D(x):
    data_shape = x.shape
    new_shape = [data_shape[0], data_shape[1], np.prod(data_shape[2:])]
    return np.reshape(x, new_shape)

def HSI_metrics(gt_img, test_img):
    # gt_img and test_img are float arrays ranging from 0 to 1.
    gt_img = Convert3D(gt_img)
    test_img = Convert3D(test_img)
    psnr_val = PSNR3D(gt_img, test_img)
    ssim_val = SSIM3D(gt_img, test_img)
    ergas_val  = ERGAS(gt_img, test_img)
    sam_val = SAM(gt_img, test_img)
    return [psnr_val, ssim_val, ergas_val, sam_val]
