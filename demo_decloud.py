# -*- coding: utf-8 -*-

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import utils
import numpy as np
from PLRR import PLRR
from scipy.io import loadmat
from time import time

test_data = [ 'MS_munich', 'Uzbekistan17', 'Uzbekistan49']
test_epoch = [500, 1000, 500]
test_rank = [12, 20, 12]
method_name = 'PLRR' 

mode_list = ['S', 'M', 'L']

num_cases = len(mode_list)
num_metrics = 5

for test_index in range(len(test_data)):
    metrics = np.zeros((num_cases,num_metrics))
    save_path = os.path.join('output_decloud', test_data[test_index], method_name)
    os.makedirs(name=save_path, exist_ok=True)
    case = 0
    for j in range(len(mode_list)):
        mode = mode_list[j]
        case += 1
        
        utils.prepare_decloud(test_data[test_index], mode, [256,256])
        load_path = 'data/test_decloud/%s_mode%s.mat'%(test_data[test_index],  mode)
        data = loadmat(load_path)
        GT, Input, Mask = utils.im2double(data['GT']), utils.im2double(data['Input']), data['Mask']
        shape = data['shape'][0]

        gif_root = os.path.join(save_path, 'case_%d.gif'%(case))
        method = PLRR(img_degraded=Input, 
                      gt = GT[:,:,:shape[2]], 
                      mask=Mask, 
                      task = 'decloud',
                      rank = test_rank[test_index], 
                      n_feat = 256, 
                      decloud_shape=shape,
                      num_epoch = test_epoch[test_index], 
                      lr = 1e-3, 
                      seed = 8888, 
                      loss_mode = 'l1', 
                      metric_mode = 'psnr3d', 
                      print_every = 50, 
                      print_image = False
                      # rgb = [0,1,2],  
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
        
        
        metrics[case-1, :4] = utils.HSI_metrics(GT[:,:,:shape[2]], method.img_recon_best[:,:,:shape[2]])
        metrics[case-1, -1] = end_time
        np.savetxt(os.path.join(save_path, 'metrics.txt'), metrics, fmt='%.6f')

        utils.print_metrics(metrics, col_labels=['PSNR', 'SSIM', 'SAM', 'SAM', 'Time'])
        
        # method.plotV()

    
    