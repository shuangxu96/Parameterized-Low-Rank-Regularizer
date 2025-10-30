# -*- coding: utf-8 -*-

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import utils
import numpy as np
from PLRR import PLRR
from scipy.io import loadmat
from time import time

test_data = ['wdc256', 'Urban', 'BayArea13_256']
test_epoch = [500, 500, 500]
test_rank = [15, 15, 15]
reg_sigma = 0.03

method_name = 'PLRR'

num_cases = 6
num_metrics = 5

for test_index in range(0,len(test_data)):
    metrics = np.zeros((num_cases,num_metrics))
    save_path = os.path.join('output_denoising', test_data[test_index], method_name)
    os.makedirs(name=save_path, exist_ok=True)
    for case in range(num_cases):
        case += 1
        utils.prepare_HSIDenoising(test_data[test_index], case)
        load_path = 'data/test_denoising/%s_case%d.mat'%(test_data[test_index], case)
        data = loadmat(load_path)
        GT, Input = utils.im2double(data['GT']), utils.im2double(data['Input'])

        # gif_root = os.path.join(save_path, 'case_%d.gif'%(case))
        gif_root = None
        method = PLRR(img_degraded=Input, # degraded image
                      gt = GT, # GT image (for calculate metrics)
                      task = 'denoising', # set task
                      rank = test_rank[test_index], # rank
                      n_feat = 256, # number of features
                      num_epoch = test_epoch[test_index], # number of epochs
                      lr = 1e-3, # learning rate
                      smooth_coef=0.8, # smoothing coefficients
                      reg_sigma=reg_sigma, #
                      seed = 8888, # random seed 
                      loss_mode = 'l1', # L1 loss
                      metric_mode = 'psnr3d', #
                      print_every = 20, 
                      print_image = False
                      # rgb = [30,14,3],  
                      # gif_dict = {'save_root': gif_root, 'frequency': 'default', 'speed': 0.5, 'resize': [512,512]}
                      )
        start_time = time()
        method.optimize()
        end_time = time() - start_time
        print('Elapsed time is %.2f seconds.' % end_time)
    
        method.plot_loss(os.path.join(save_path, 'case_%d_loss.jpg'%(case)))
        method.plot_metric(os.path.join(save_path, 'case_%d_metric.jpg'%(case)))
        
        method.save_info(os.path.join(save_path, 'case_%d_loss.csv'%(case)), mode='loss')
        method.save_info(os.path.join(save_path, 'case_%d_metric.csv'%(case)), mode='metric')
        method.save_result(os.path.join(save_path, 'case_%d.mat'%(case)), mode='best')
        
        metrics[case-1, :4] = utils.HSI_metrics(GT, method.img_recon_best)
        metrics[case-1, -1] = end_time
        np.savetxt(os.path.join(save_path, 'metrics.txt'), metrics, fmt='%.6f')

        utils.print_metrics(metrics, col_labels=['PSNR', 'SSIM', 'ERGAS', 'SAM', 'Time'])
        
        method.plotV()


