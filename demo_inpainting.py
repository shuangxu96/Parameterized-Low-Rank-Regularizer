# -*- coding: utf-8 -*-
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import utils
import numpy as np
from PLRR import PLRR
from scipy.io import loadmat
from time import time

test_data = ['wdc256', 'Urban', 'BayArea13_256']
test_epoch = [3000, 3000, 3000]
test_rank = [15, 15, 15]

method_name = 'PLRR'

num_cases = 3
num_metrics = 5

for test_index in range(3):
    metrics = np.zeros((num_cases,num_metrics))
    save_path = os.path.join('output_inpainting', test_data[test_index], method_name)
    os.makedirs(name=save_path, exist_ok=True)
    case = 0
    
    ratio_list = [0.90, 0.925, 0.95] 
    
    for j in range(len(ratio_list)):
        ratio = ratio_list[j]
        case += 1
        
        utils.prepare_inpainting(test_data[test_index], ratio)
        load_path = 'data/test_inpainting/%s_random_inpainting_ratio%.3f.mat'%(test_data[test_index],  ratio)
        data = loadmat(load_path)
        GT, Input, Mask = utils.im2double(data['GT']), utils.im2double(data['Input']), data['Mask']
        
        gif_root = os.path.join(save_path, 'case_%d.gif'%(case))
        method = PLRR(img_degraded=Input, 
                      gt = GT, 
                      mask=Mask, 
                      task = 'inpainting',
                      rank = test_rank[test_index], 
                      n_feat = 256, 
                      num_epoch = test_epoch[test_index], 
                      lr = 1e-3, 
                      seed = 8888, 
                      loss_mode = 'l1', 
                      metric_mode = 'psnr3d', 
                      print_every = 100, 
                      print_image = False
                      # rgb = [30,15,4],  
                      # gif_dict = {'save_root': gif_root, 'frequency': 'default', 'speed': 0.5, 'resize': [512,512]}
                      )
        start_time = time()
        method.optimize()
        end_time = time() - start_time
        print('Elapsed time is %.2f seconds.' % end_time)
    
        method.plot_loss(os.path.join(save_path, 'case_%d_loss.jpg'%(case)))
        method.plot_metric(os.path.join(save_path, 'case_%d_psnr.jpg'%(case)))
        
        method.save_info(os.path.join(save_path, 'case_%d_loss.csv'%(case)), mode='loss')
        method.save_info(os.path.join(save_path, 'case_%d_metric.csv'%(case)), mode='metric')
        method.save_result(os.path.join(save_path, 'case_%d.mat'%(case)), mode='best')
        
        
        metrics[case-1, :4] = utils.HSI_metrics(GT, method.img_recon_best)
        metrics[case-1, -1] = end_time
        np.savetxt(os.path.join(save_path, 'metrics.txt'), metrics, fmt='%.6f')

        utils.print_metrics(metrics, col_labels=['PSNR', 'SSIM', 'ERGAS', 'SAM', 'Time'])
        



    