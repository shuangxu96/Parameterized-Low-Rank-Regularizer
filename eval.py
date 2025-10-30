# -*- coding: utf-8 -*-
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import utils
import numpy as np
from glob import glob
from scipy.io import loadmat
from os.path import join

task = 'inpainting'
print(task)
if task == 'denoising':
    num_case = 6
    data = ['BayArea13_256', 'paviac256', 'wdc256']
    method = ['PLRR', 'DIP2D', 'S2DIP', 'NGMeet', 'E3DTV', 'ETPTV', 'CTV', 
              'FastHyMix', 'QRNN3D', 'D2Net', 'TRQ3D', 'SERT']
elif task == 'inpainting':
    num_case = 3
    data = ['BayArea13_256', 'paviac256', 'wdc256']
    # method = ['PLRR', 'DIP2D', 'S2DIP', 'HLRTF', 'HaLRTC', 'SNN_TV', 'SPC_TV', 
    #           'TNN_FFT', 'FTNN', 'tCTV']
    method = [ 'FTNN']


tab = np.zeros((len(data), len(method), num_case, 4))

for i in range(len(data)):
    gt = utils.im2double(loadmat('data/%s.mat'%(data[i]))['data']) # [h,w,c]
    print(data[i])
    for j in range(len(method)):
        print(method[j])
        path = 'output_%s/%s/%s'%(task, data[i], method[j])
        filelist = glob(join(path, '*.mat'))
        results = np.zeros((len(filelist),4))
        for k in range(len(filelist)):
            filepath = filelist[k]
            recon = utils.im2double(loadmat(filepath)['img_recon'])  # [h,w,c]
            results[k,:] = utils.HSI_metrics(gt, recon)
        np.savetxt(join(path, 'eval.txt'), results.T, fmt='%.6f', delimiter='\0')
        tab[i,j,:,:] = results

a = tab.transpose(1,3,0,2)
b = a.reshape(len(method)*4, -1)
np.savetxt('output_%s/result.csv'%(task), b, fmt='%.4f', delimiter=',')