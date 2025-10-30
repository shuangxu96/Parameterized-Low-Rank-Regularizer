# -*- coding: utf-8 -*-
import os
import numpy as np
from scipy import signal
from scipy.ndimage import convolve
from scipy.io import loadmat, savemat
from skimage.transform import resize as imresize
from skimage.io import imread, imsave, imshow

def matlab_style_gauss2D(shape=(3,3),sigma=0.5):
    # generate filter same with fspecial('gaussian') function
    m,n = [(ss-1.)/2. for ss in shape]
    y,x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
    h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h

def prepare_hsmsf(file_name):
    # ratio is the missing rate
    os.makedirs(r'.\data\test_hsmsf', exist_ok=True)
    save_path = r'.\data\test_hsmsf\%s.mat'%(file_name)
    if not os.path.exists(save_path):
        gt_path = r'.\data\%s.mat'%(file_name)
        HRMS = loadmat(gt_path)['HRMS'].copy().astype(np.float32)/255. # the uint8 array
        HRHS = loadmat(gt_path)['HRHS'].copy().astype(np.float32)/255. # the uint8 array
        kernel = matlab_style_gauss2D((31,31), sigma=5)
        scale = 4
        
        hh,ww,nband = HRHS.shape
        LRHS = np.zeros((hh,ww,nband))
        for i in range(nband):
            LRHS[:,:,i] = convolve(HRHS[:,:,i],kernel, mode='mirror')
        LRHS = LRHS[::scale, ::scale, :]
        LRHS = np.uint8(255*np.clip(LRHS, 0., 1.))
        
        HRMS = np.uint8(255*np.clip(HRMS, 0., 1.))
        HRHS = np.uint8(255*np.clip(HRHS, 0., 1.))
        savemat(save_path, {'LRHS': LRHS, 'GT': HRHS, 'HRMS': HRMS})

def prepare_decloud(file_name, mode, new_shape=[200,200]):
    # ratio is the missing rate
    os.makedirs(r'.\data\test_decloud', exist_ok=True)
    save_path = r'.\data\test_decloud\%s_mode%s.mat'%(file_name, mode.upper())
    if not os.path.exists(save_path):
        gt_path = r'.\data\%s.mat'%(file_name)
        gt = loadmat(gt_path)['data'].copy() # the uint8 array
        gt = gt.astype(np.float32)/255.
        hh,ww,nband,nframe = gt.shape
        gt = np.reshape(gt, [hh,ww,nband*nframe],'F')
        gt = imresize(gt, new_shape)
        gt = np.reshape(gt, [new_shape[0],new_shape[1],nband,nframe],'F')
        hh,ww,nband,nframe = gt.shape
        noisy = gt.copy()
        # hh,ww,nband = gt.shape
        
        if mode == 'L':
            mask = imread('data/cloudmask_large.jpg')
        elif mode == 'M':
            mask = imread('data/cloudmask_middle.jpg')
        elif mode == 'S':
            mask = imread('data/cloudmask_small.jpg')
        mask = mask.astype(np.float32)/255.
        
        if mask.ndim ==3:
            mask = np.mean(mask, axis=2)

        mask = imresize(mask, output_shape=new_shape)
        mask[mask>=0.5] = 1
        mask[mask<0.5] = 0
        mask = np.stack([mask]*nband, axis=-1)[...,None]
        mask = np.concatenate([mask,np.ones([hh,ww,nband,nframe-1])], axis=-1)
        # noisy[mask==0] = 1 # only the image of the first frame has cloud
        noisy = noisy*mask
        
        mask = mask.astype(np.bool)
        mask = np.reshape(mask, [hh,ww,nband*nframe],'F')
        gt = np.reshape(gt, [hh,ww,nband*nframe],'F')
        noisy = np.reshape(noisy, [hh,ww,nband*nframe],'F')
        
        noisy = np.clip(noisy, 0., 1.)
        gt    = np.uint8(255*gt)
        noisy = np.uint8(255*noisy)
        savemat(save_path, {'Input': noisy, 'GT': gt, 'Mask': mask, 'shape':[hh,ww,nband,nframe]})
        

def prepare_inpainting(file_name, ratio):
    # ratio is the missing rate
    os.makedirs(r'.\data\test_inpainting', exist_ok=True)
    save_path = r'.\data\test_inpainting\%s_random_inpainting_ratio%.3f.mat'%(file_name, ratio)
    if not os.path.exists(save_path):
        gt_path = r'.\data\%s.mat'%(file_name)
        gt = loadmat(gt_path)['data'].copy() # the uint8 array
        gt = gt.astype(np.float32)/255.
        gtshape = gt.shape
        if len(gtshape)==4:
            hh,ww,nband,nframe = gt.shape
            gt = np.reshape(gt, [hh,ww,nband*nframe])
        noisy = gt.copy()
        hh,ww,nband = gt.shape
        
        if ratio>0 and ratio<1: # randomly sample
            element_num = hh*ww*nband # number of elements
            obs_num = int(element_num*(1-ratio)) # number of observed elements 
            mask = np.zeros([hh,ww,nband])
            obs_ind = np.unravel_index(np.random.choice(element_num, obs_num, replace=False), shape=[hh,ww,nband], order='F')
            mask[obs_ind] = 1

        noisy = mask*noisy
        mask = mask.astype(bool)
        
        noisy = np.clip(noisy, 0., 1.)
        gt    = np.uint8(255*gt)
        noisy = np.uint8(255*noisy)
        # if len(gtshape)==4:
        #     gt = np.reshape(gt, gtshape)
        #     noisy = np.reshape(noisy, gtshape)
        #     mask = np.reshape(mask, gtshape)
        savemat(save_path, {'Input': noisy, 'GT': gt, 'Mask': mask})
        

def degrade_img(x, mode, par1):
    
    output = x.copy()
    
    if mode == 'gauss': # add gauss noise
        sig = par1/255.
        output = output + np.random.randn(*x.shape)*sig
        
    
    if mode == 's&p' or mode == 'impulse':
        ratio = par1
        mask = np.random.rand(*x.shape)
        output[mask<ratio/2] = 0.
        output[mask>1-ratio/2] = 1.
        
    
    if mode == 'stripe':
        ratio = par1
        HH,WW = x.shape
        num_stripe = int(ratio*HH)
        loc_stripe = np.random.choice(HH, num_stripe, replace=False)
        output[loc_stripe,:] = output[loc_stripe,:] + np.random.rand(num_stripe,1)*0.5-0.25
        
    if mode == 'deadline':
        ratio = par1
        HH,WW = x.shape
        num_deadline = int(ratio*HH)
        loc_deadline = np.random.choice(HH, num_deadline, replace=False)
        output[loc_deadline,:] = 0
        

    output = np.clip(output, a_min=0., a_max=1.)
    return output

def prepare_HSIDenoising(file_name, case):
    os.makedirs(r'.\data\test_denoising', exist_ok=True)
    save_path = r'.\data\test_denoising\%s_case%d.mat'%(file_name, case)
    if not os.path.exists(save_path):
        gt_path = r'.\data\%s.mat'%(file_name)
        gt = loadmat(gt_path)['data'].copy() # the uint8 array
        gt = gt.astype(np.float32)/255.
        noisy = gt.copy()
        nband = gt.shape[-1]
        if case == 1: # iid gauss
            noisy = degrade_img(gt, 'gauss', 25.)
        elif case == 2: # non-iid gauss
            for band in range(nband):
                std = 10. + np.random.rand()*60.
                noisy[:,:,band] = degrade_img(gt[:,:,band], 'gauss', std)
        elif case == 3: # non-iid gauss + impulse
            for band in range(nband):
                std = 10. + np.random.rand()*60.
                noisy[:,:,band] = degrade_img(gt[:,:,band], 'gauss', std)
                add_sp = np.random.rand()<1/3
                if add_sp:
                    ratio = 0.1 + np.random.rand()*0.2
                    noisy[:,:,band] = degrade_img(noisy[:,:,band], 's&p', ratio)
        elif case == 4: # non-iid gauss + stripe
            for band in range(nband):
                std = 10. + np.random.rand()*60.
                noisy[:,:,band] = degrade_img(gt[:,:,band], 'gauss', std)
                add_st = np.random.rand()<1/3
                if add_st:
                    ratio = 0.05 + np.random.rand()*0.1
                    noisy[:,:,band] = degrade_img(noisy[:,:,band], 'stripe', ratio)
        elif case == 5: # non-iid gauss + deadline
            for band in range(nband):
                std = 10. + np.random.rand()*60.
                noisy[:,:,band] = degrade_img(gt[:,:,band], 'gauss', std)
                add_ddl = np.random.rand()<1/3
                if add_ddl:
                    ratio = 0.05 + np.random.rand()*0.1
                    noisy[:,:,band] = degrade_img(noisy[:,:,band], 'deadline', ratio)
        elif case == 6: # non-iid gauss + impulse/stripe/deadline
            for band in range(nband):
                std = 10. + np.random.rand()*60.
                noisy[:,:,band] = degrade_img(gt[:,:,band], 'gauss', std)
                noise_type = np.random.randint(0,3)
                if noise_type==0: # impulse
                    ratio = 0.1 + np.random.rand()*0.2
                    noisy[:,:,band] = degrade_img(noisy[:,:,band], 's&p', ratio)
                elif noise_type==1: # stripe
                    ratio = 0.05 + np.random.rand()*0.1
                    noisy[:,:,band] = degrade_img(noisy[:,:,band], 'stripe', ratio)
                elif noise_type==2: # deadline
                    ratio = 0.05 + np.random.rand()*0.1
                    noisy[:,:,band] = degrade_img(noisy[:,:,band], 'deadline', ratio)
        noisy = np.clip(noisy, 0., 1.)
        gt    = np.uint8(255*gt)
        noisy = np.uint8(255*noisy)
        savemat(save_path, {'Input': noisy, 'GT': gt})