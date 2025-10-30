# -*- coding: utf-8 -*-

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import utils
import numpy as np
from PLRR import PLRR
from scipy.io import loadmat, savemat
from time import time

test_data = ['Houston18_A_CUCaNet_sigma3', 'Houston18_A_CUCaNet_sigma5', 'Houston18_A_CUCaNet_sigma10',
             'Houston18_B_CUCaNet_sigma3', 'Houston18_B_CUCaNet_sigma5', 'Houston18_B_CUCaNet_sigma10', 
             'Houston18_C_CUCaNet_sigma3', 'Houston18_C_CUCaNet_sigma5', 'Houston18_C_CUCaNet_sigma10']
# test_epoch = [2000, 2000, 2000, 2000, 2000, 2000, 2000, 2000, 2000]
test_epoch = [1500, 1500, 1500, 1500, 1500, 1500, 1500, 1500, 1500]
test_rank = [15,15,15,15,15,15,15,15,15]


method_name = 'PLRR_UV1500_'
num_metrics = 5

metrics = np.zeros((len(test_data),num_metrics))
for test_index in range(len(test_data)):
    
    save_path = os.path.join('output_hsmsf', method_name)
    os.makedirs(name=save_path, exist_ok=True)

    utils.prepare_hsmsf(test_data[test_index])
    load_path = 'data/test_hsmsf/%s.mat'%(test_data[test_index])
    data = loadmat(load_path)
    GT = utils.im2double(data['GT'])*255
    HRMS = utils.im2double(data['HRMS'])*255
    LRHS = utils.im2double(data['LRHS'])*255

    

    gif_root = os.path.join(save_path, '%s.gif'%(test_data[test_index]))
    method = PLRR(img_degraded=LRHS, 
                  guidance=HRMS, 
                  gt = GT, 
                  task = 'hsmsf',
                  rank = test_rank[test_index], 
                  n_feat = 256, 
                  num_epoch = test_epoch[test_index], 
                  lr = [1e-4,1e-4,1e-2,1e-4], 
                  seed = 8888, 
                  loss_mode = 'l1', 
                  metric_mode = 'psnr3d', 
                  print_every = 100, 
                  # rgb = [30,14,3],  
                  # gif_dict = {'save_root': gif_root, 'frequency': 'default', 'speed': 0.5, 'resize': [512,512]}
                  )
    start_time = time()
    method.optimize()
    end_time = time() - start_time
    print('Elapsed time is %.2f seconds.' % end_time)
    # for i in range(GT.shape[-1]):
    #     GT[:,:,i] *= GT_max[i]
    #     method.img_recon_best[:,:,i] *= LRHS_max[i]

    method.plot_loss(os.path.join(save_path, '%s_loss.jpg'%(test_data[test_index])))
    method.plot_metric(os.path.join(save_path, '%s_psnr.jpg'%(test_data[test_index])))
    method.plot_srf(os.path.join(save_path, '%s_srf.jpg'%(test_data[test_index])))
    method.plot_blur_kernel(os.path.join(save_path, '%s_blur_kernel.jpg'%(test_data[test_index])))
    method.plot_downsampler(os.path.join(save_path, '%s_downsampler.jpg'%(test_data[test_index])))
    
    method.save_result(os.path.join(save_path, '%s.mat'%(test_data[test_index])), mode='best')
    savemat(os.path.join(save_path, '%s_blur_kernel.mat'%(test_data[test_index])), {'blur_kernel': method.model.blur_kernel})
    savemat(os.path.join(save_path, '%s_srf.mat'%(test_data[test_index])), {'srf': method.model.srf})
    savemat(os.path.join(save_path, '%s_downsampler.mat'%(test_data[test_index])), {'downsampler': method.model.NetS.kernel})

    
    
    metrics[test_index, :4] = utils.HSI_metrics(GT, method.img_recon_best)
    metrics[test_index, -1] = end_time
    np.savetxt(os.path.join(save_path, 'metrics.txt'), metrics, fmt='%.6f')

    utils.print_metrics(metrics, col_labels=['PSNR', 'SSIM', 'ERGAS', 'SAM', 'Time'])

    

    
    