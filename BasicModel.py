# -*- coding: utf-8 -*-
"""
Created on Tue Oct  4 21:59:21 2022

@author: DELL
"""

from matplotlib import pyplot as plt
from scipy.io import loadmat, savemat

from PIL import ImageDraw, Image, ImageFont
import numpy as np
import utils 
import torch
import torch.nn as nn
from skimage.metrics import peak_signal_noise_ratio as psnr_fn
from skimage.metrics import structural_similarity as ssim_fn
import imageio 
from skimage.io import imshow
from skimage.transform import resize as imresize

class BasicModel:
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
                 print_image = False,
                 rgb = [30,14,3], 
                 decloud_shape=None,
                 task = None,
                 gif_dict = {'save_root': None, 'frequency': 100, 'speed': 0.5, 'resize': [512,512]}):
        
        # common configurations
        self.seed = seed
        self.task = task
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.data_type = torch.float32
        
        # data 
        self.img_degraded = img_degraded.copy() # degrade image
        self.gt = gt # ground truth
        self.output = None # reconstructed 
        self.img_recon_best = None
        
        # network settings 
        self.net_mode = net_mode
        self.n_feat = n_feat
        self.rank = rank
        self.init_weight = init_weight
        
        # input
        self.input_mode = input_mode
        self.reg_sigma = reg_sigma # regularize input (often used for denoising task)
        
        # post-process (moving average trick)
        self.smooth_flag = False if smooth_coef==0 else True
        self.smooth_coef = smooth_coef
        
        # train
        self.lr = lr
        self.loss_mode = loss_mode
        self.num_epoch = num_epoch
        self.tol = tol
        self.auto_stop = False
        
        # print/display info
        self.print_every = print_every 
        self.rgb = rgb
        self.print_image = print_image
        
        # print metric
        self.metric_mode = metric_mode
        self.set_metric(metric_mode)
        metric_name = metric_mode.upper()
        self.metric_name = metric_name
        
        # set gif 
        self.set_gif_dict(gif_dict)
        
        # set seed
        utils.setup_seed(self.seed)
        
        # set input and label 
        self.label = utils.array2tensor(self.img_degraded, self.device) # label
        
        # set mask (for inpainting and decloud tasks)
        self.mask = mask
        if self.mask is not None:
            self.mask_tensor = utils.array2tensor(self.mask, self.device) 
        
        # set guidance image (i.e. HS-MSI) (for HSMSF task)
        self.guidance = guidance
        if self.guidance is not None:
            self.guidance_tensor = utils.array2tensor(self.guidance, self.device) 
            
        # decloud shape (for decloud task)
        self.decloud_shape = decloud_shape # [Height,Width,Channel,Timestamp]
        
        # set network, loss, optimizer
        self.set_net() # network 
        self.set_loss(self.loss_mode) # loss
        self.set_optimizer() # optimizer 
        self.set_loop_state()
        
    
    def set_optimizer(self):
        if self.task == 'hsmsf':
            self.optimizer = torch.optim.Adam([{'params': self.model.NetU.parameters(), 'lr': self.lr[0]},
                                               {'params': self.model.NetV.parameters(), 'lr': self.lr[1]},
                                               {'params': self.model.NetS.parameters(), 'lr': 1e-2},
                                               {'params': self.model.NetK.parameters(), 'lr': self.lr[2]},
                                               {'params': self.model.NetF.parameters(), 'lr': self.lr[3]}]
                                              )
        elif self.task == 'sr':
            self.optimizer = torch.optim.Adam([{'params': self.model.NetU.parameters(), 'lr': self.lr[0]},
                                               {'params': self.model.NetV.parameters(), 'lr': self.lr[1]},
                                               {'params': self.model.NetS.parameters(), 'lr': 1e-2},
                                               {'params': self.model.NetK.parameters(), 'lr': self.lr[2]}]
                                              )
        else:
            self.optimizer = torch.optim.Adam(self.model.parameters(), self.lr)
        
    def optimize(self):
        xhat_old = self.img_degraded.copy()
        
        self.set_input(mode=self.input_mode) # network input

        n_params = self.get_parameter_number()
        if self.print_every<=self.num_epoch:
            print('Network parameters = %d'%(n_params['Total']))
            print('Trainable network parameters = %d'%(n_params['Trainable']))

        # loop
        
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark =True
        for epoch in range(self.num_epoch):
            self.epoch = epoch

            # train  network
            xhat = self.closure()

            # convert the torch tensor as a numpy array 
            xhat_np = utils.tensor2array(xhat, clip=True) # if clip==True, xhat_np is clipped into [0,1]
            
            # moving average trick (if smooth_coef==0, this trick will be disenabled)
            self.postprocess(xhat_np) # incorporate xhat_np into self.output
            
            # converge or not
            xhat_old = xhat_old[:,:,:self.output.shape[-1]]
            if xhat_old.shape[0]==self.output.shape[0]:
                rel_err = np.linalg.norm(self.output.flatten() - xhat_old.flatten(), ord=2) / np.linalg.norm(xhat_old.flatten(), ord=2)
            else:
                rel_err = 1.
            if rel_err!=0 and rel_err<self.tol:
                print('Converge!')
                break
            else:
                xhat_old = self.output.copy()
            
            # adjust lr
            if self.task == 'hsmsf' and epoch==2500:
                self.optimizer = torch.optim.Adam([{'params': self.model.NetU.parameters(), 'lr': self.lr[0]*0.5},
                                                   {'params': self.model.NetV.parameters(), 'lr': self.lr[1]*0.5},
                                                   {'params': self.model.NetS.parameters(), 'lr': 1e-2},
                                                   {'params': self.model.NetK.parameters(), 'lr': self.lr[2]},
                                                   {'params': self.model.NetF.parameters(), 'lr': self.lr[3]*0.5}]
                                                  )
            
            # calculate metrics, prepare gif, print information
            self.calculate_metric() # works only if gt is provided
            self.set_gif_frame(epoch) # works only if gt is provided
            if self.print_every<=self.num_epoch:
                self.print_info(epoch, rel_err)
            

        self.img_recon = self.output
        self.model.eval()
        if self.gt is not None:
            self.metric_last = self.metric_fn(self.img_recon, self.gt)
            print('%s* = %2.2f, %s# = %2.2f' %(
                self.metric_name,
                max(self.metric_list),
                self.metric_name,
                self.metric_last,
                )
            )
            print('%s* is the best metric value over the training steps.'%(self.metric_name)  )
            print('%s# is the last metric value.'%(self.metric_name))
        if self.gif_dict['save_root'] is not None:
            imageio.mimsave(self.gif_dict['save_root'], self.frames, 'GIF', duration=self.gif_dict['speed'])
    
    # def train_template(self, optimize_process):
    #     xhat_old = self.img_degraded.copy()

    #     # set random seed and other constants
    #     utils.setup_seed(self.seed)
        
    #     # set data 
    #     self.set_input(mode=self.input_mode) # network input
    #     if self.task=='hsmsf':
    #         self.guidance_tensor = utils.array2tensor(self.guidance, self.device) 
    #     self.label = utils.array2tensor(self.img_degraded, self.device) # label
        
    #     # set mask (works for inpainting only)
    #     if self.mask is not None:
    #         self.mask_tensor = utils.array2tensor(self.mask, self.device) 

        
    #     # set net, loss function, optimizer
    #     self.set_net()
    #     if self.init_weight is not None:
    #         self.model.load_state_dict(self.init_weight)
    #         if self.task == 'hsmsf':
    #             self.model.NetK.U = nn.Parameter(torch.tensor([[1,0],[0,1]], dtype=torch.float), requires_grad=True)
    #         self.model.to('cuda')
    #     n_params = self.get_parameter_number()
    #     if self.print_every<=self.num_epoch:
    #         print('Network parameters = %d'%(n_params['Total']))
    #         print('Trainable network parameters = %d'%(n_params['Trainable']))
    #     self.set_loss(self.loss_mode)
    #     if self.task == 'hsmsf':
    #         self.optimizer = torch.optim.Adam([{'params': self.model.NetU.parameters(), 'lr': self.lr[0]},
    #                                            {'params': self.model.NetV.parameters(), 'lr': self.lr[1]},
    #                                            {'params': self.model.NetS.parameters(), 'lr': 1e-2},
    #                                            {'params': self.model.NetK.parameters(), 'lr': self.lr[2]},
    #                                            {'params': self.model.NetF.parameters(), 'lr': self.lr[3]}]
    #                                           )
    #         # niter1 = int(self.num_epoch*0.2)
    #         # niter2 = self.num_epoch - niter1
    #         # def lambda_rule(epoch):
    #         #     lr_l = 1.0 - max(0, epoch + 2 - niter1) / float(niter2 + 1) / 2
    #         #     return lr_l
    #         # self.schedule = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda_rule)
    #         # niter1 = int(self.num_epoch*0.1)
    #         # niter2 = int(self.num_epoch*0.9) - niter1
    #         # niter3 = self.num_epoch - niter2
    #         # def lambda_rule(epoch):
    #         #     if epoch<=niter1:
    #         #         lr_l = (epoch*(1e-4-1e-8)/niter1+1e-8)/1e-4
    #         #     elif epoch<=niter2:
    #         #         lr_l = 1.0
    #         #     else:
    #         #         lr_l = 1.0 - max(0, epoch + 2 - niter2) / float(niter3 + 1) / 2
    #         #     return lr_l
    #         # self.schedule = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda_rule)
    #         # self.schedule = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[int(self.num_epoch*0.9)], gamma=0.1)
    #     else:
    #         self.optimizer = torch.optim.Adam(self.model.parameters(), self.lr)

    #     # loop
    #     self.set_loop_state()
    #     torch.backends.cudnn.enabled = True
    #     torch.backends.cudnn.benchmark =True
    #     for epoch in range(self.num_epoch):
    #         self.epoch = epoch

    #         # train DIP network
    #         xhat = optimize_process()

    #         # convert the torch tensor as a numpy array and smooth it
    #         xhat_np = utils.tensor2array(xhat, clip=True)
    #         self.postprocess(xhat_np) # incorporate xhat_np into self.output
            
    #         # converge or not
    #         xhat_old = xhat_old[:,:,:self.output.shape[-1]]
    #         if xhat_old.shape[0]==self.output.shape[0]:
    #             rel_err = np.linalg.norm(self.output.flatten() - xhat_old.flatten(), ord=2) / np.linalg.norm(xhat_old.flatten(), ord=2)
    #         else:
    #             rel_err = 1.
    #         if rel_err!=0 and rel_err<self.tol:
    #             print('Converge!')
    #             break
    #         else:
    #             xhat_old = self.output.copy()
            
    #         # if self.task == 'hsmsf' and epoch==100:
    #         #     self.optimizer.param_groups[2]['lr'] = self.optimizer.param_groups[2]['lr']*0.1
    #             # self.optimizer.param_groups[3]['lr'] = self.optimizer.param_groups[3]['lr']*0.1
    #         if self.task == 'hsmsf' and epoch==2500:
    #             self.optimizer = torch.optim.Adam([{'params': self.model.NetU.parameters(), 'lr': self.lr[0]*0.5},
    #                                                {'params': self.model.NetV.parameters(), 'lr': self.lr[1]*0.5},
    #                                                {'params': self.model.NetS.parameters(), 'lr': 1e-2},
    #                                                {'params': self.model.NetK.parameters(), 'lr': self.lr[2]},
    #                                                {'params': self.model.NetF.parameters(), 'lr': self.lr[3]*0.5}]
    #                                               )
    #         # calculate metrics, prepare gif, print information
    #         self.calculate_metric() # works only if gt is provided
    #         self.set_gif_frame(epoch) # works only if gt is provided
    #         if self.print_every<=self.num_epoch:
    #             self.print_info(epoch, rel_err)
            

    #     self.img_recon = self.output
    #     self.model.eval()
    #     if self.gt is not None:
    #         self.metric_last = self.metric_fn(self.img_recon, self.gt)
    #         print('%s* = %2.2f, %s# = %2.2f' %(
    #             self.metric_name,
    #             max(self.metric_list),
    #             self.metric_name,
    #             self.metric_last,
    #             )
    #         )
    #         print('%s* is the best metric value over the training steps.'%(self.metric_name)  )
    #         print('%s# is the last metric value.'%(self.metric_name))
    #     if self.gif_dict['save_root'] is not None:
    #         imageio.mimsave(self.gif_dict['save_root'], self.frames, 'GIF', duration=self.gif_dict['speed'])
        
    
    def get_parameter_number(self):
        total_num = sum(p.numel() for p in self.model.parameters())
        trainable_num = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        return {'Total': total_num, 'Trainable': trainable_num}
    
    def plot_curve(self, x_val, y_val, args, path=None):
        plt.figure()
        plt.rcParams['font.family'] = 'Arial'
        plt.plot(x_val, y_val)
        
        ax = plt.gca()
        ax.set_title(args['title'], fontdict={'style':'normal','size':16,'color':'black'}) if 'title' in args else None
        ax.set_xlabel(args['xlabel'], fontdict={'style':'normal','size':16,'color':'black'}) if 'xlabel' in args else None
        ax.set_ylabel(args['ylabel'], fontdict={'style':'normal','size':16,'color':'black'}) if 'ylabel' in args else None
        ax.set_xscale(args['xscale'])
        ax.set_yscale(args['yscale'])
        plt.setp(ax.get_xticklabels(), fontsize=12)
        plt.setp(ax.get_yticklabels(), fontsize=12)
        
        plt.tight_layout()
        
        if path is not None:
            plt.savefig(path, bbox_inches='tight', dpi=600)
    
    def plot_blur_kernel(self, path=None):
        plt.figure()
        imshow(self.model.blur_kernel)
        plt.title('Blur Kernel')
        if path is not None:
            plt.savefig(path, bbox_inches='tight', dpi=600)
            
    def plot_downsampler(self, path=None):
        plt.figure()
        imshow(self.model.NetS.kernel)
        plt.title('Downsampler')
        if path is not None:
            plt.savefig(path, bbox_inches='tight', dpi=600)
        
        
    def plot_srf(self, path=None):
        y_val  = self.model.srf.T
        x_val  = list(range(len(y_val)))
        args   = {'xlabel': 'Band Index', 'ylabel': 'SRF',
                  'xscale': 'linear',     'yscale': 'linear'}
        self.plot_curve(x_val, y_val, args, path)
    
    def plot_loss(self, path=None):
        y_val  = self.loss_list
        x_val  = list(range(len(y_val)))
        args   = {'xlabel': 'Iteration', 'ylabel': 'Loss',
                  'xscale': 'linear',    'yscale': 'log'}
        self.plot_curve(x_val, y_val, args, path)

    def plot_metric(self, path=None):        
        y_val  = self.metric_list
        x_val  = list(range(len(y_val)))
        args   = {'xlabel': 'Iteration', 'ylabel': self.metric_name,
                  'xscale': 'linear',    'yscale': 'linear'}
        self.plot_curve(x_val, y_val, args, path)
    
    def save_info(self, path, mode):
        if mode == 'loss':
            np.savetxt(path, self.loss_list, fmt='%.18e', delimiter=',')
        elif mode == 'metric':
            np.savetxt(path, self.metric_list, fmt='%.18e', delimiter=',')
    
    def save_result(self, path, mode):
        if mode == 'last' or self.img_recon_best is None:
            savemat(path, {'img_recon': np.uint8(self.img_recon*255.) } )
        elif mode == 'best' and self.img_recon_best is not None:
            savemat(path, {'img_recon': np.uint8(self.img_recon*255.) } )
    
    def set_input(self, mode='uniform'): 
        self.data_shape = self.img_degraded.shape
        if mode in ['uniform' , 'random' , 'rand']:
            height, width, n_channel = self.img_degraded.shape
            self.Input = torch.rand([1,n_channel,height,width], dtype=self.data_type, device=self.device)/10.
            print('Input is set by uniform distribution')
        elif mode == 'degraded_img':
            self.Input = utils.array2tensor(self.img_degraded, self.device) 
            print('Input is set by degraded img')
        elif mode == 'PLRR_inpainting':
            img_degraded = utils.set_input_plrr_inpainting(self.img_degraded.copy(), self.mask.copy())
            self.Input = utils.array2tensor(img_degraded, self.device)
            print('Input is set by PLRR_inpainting')
            # frame_size = int(2*((self.img_degraded.shape[-1]*0.1)//2)+1)
            # img_degraded = []
            # for i in range(self.img_degraded.shape[-1]//frame_size+1):
            #     s1 = i*frame_size
            #     s2 = min((i+1)*frame_size, self.img_degraded.shape[-1])
            #     img_degraded.append(utils.set_input_dlrp_inpainting(self.img_degraded.copy()[...,s1:s2], self.mask.copy()[...,s1:s2]))
            # img_degraded = np.concatenate(img_degraded, -1)
            # self.Input = utils.array2tensor(img_degraded, self.device)
        else:
            raise ValueError('Cannot initialize input with mode: %s'%(mode))
        
        if self.print_every<=self.num_epoch:
            print('Initialized network input with mode: %s'%(mode))
            
    def set_loss(self, mode='l1'):
        if mode=='l1':
            loss_fn = nn.L1Loss() 
        elif mode=='l2':
            loss_fn = nn.MSELoss()
        self.loss_fn = loss_fn
        if self.print_every<=self.num_epoch:
            print('Set loss function with mode: %s'%(mode))
    
    def set_metric(self, mode):
        if mode=='psnr':
            self.metric_fn = psnr_fn
        elif mode=='ssim':
            self.metric_fn = ssim_fn
        elif mode=='psnr3d':
            self.metric_fn = utils.PSNR3D
        elif mode=='ssim3d':
            self.metric_fn = utils.SSIM3D
        # print(self.metric_fn)

    def set_loop_state(self, ):
        self.loss_list = []
        if self.gt is not None:
            self.metric_list = []
        if self.gif_dict['save_root'] is not None:
            self.frames = []
    
    def postprocess(self, xhat_np):
        smooth_flag = False if self.smooth_coef==0 else True
        # denoising
        if self.task == 'denoising': 
            if smooth_flag: # smooth it if smooth_coef is not 0
                if self.output is None: # when output is None, output equals to xhat_np
                    self.output = xhat_np.copy()
                else:
                    self.output =  self.smooth_coef*self.output + (1-self.smooth_coef)*xhat_np
            else: # if do not smooth it, output equals to xhat_np
                self.output = xhat_np.copy()
        # inpainting 
        elif self.task == 'inpainting':
            self.output = (1-self.mask)*xhat_np + self.mask*self.img_degraded
        elif self.task == 'hsmsf':
            self.output = xhat_np.copy()
        elif self.task == 'decloud':
            nband = xhat_np.shape[-1]
            mask = self.mask[:,:,:nband]
            img_degraded = self.img_degraded[:,:,:nband]
            self.output = (1-mask)*xhat_np + mask*img_degraded
        else:
            raise Exception('No post-process is defined for %s task'%self.task)
            
    
    def calculate_metric(self, ):
        # print(self.gt.shape)
        # print(self.output.shape)
        if self.gt is not None:
            if self.output.shape[-1]!= self.gt.shape[-1]: # works for decloud
                self.output = self.output[:,:,:self.gt.shape[-1]]
            if self.output.shape[0]!=self.gt.shape[0] or self.output.shape[1]!=self.gt.shape[1]:
                metric_value = self.metric_fn(imresize(self.output,  output_shape=[self.gt.shape[0],self.gt.shape[1]]), self.gt)
            else:
                metric_value = self.metric_fn(self.output, self.gt)
            
            self.metric_list.append(metric_value)
            if metric_value==max(self.metric_list):
                self.img_recon_best = self.output.copy()
    
    def print_info(self, epoch, rel_err):
        if self.task=='hsmsf':
            if self.print_every!=0:
                if (epoch%self.print_every==0 or epoch==self.num_epoch-1) and epoch!=0:
                    # print(self.model.NetS.kernel)
                    if self.gt is not None:
                        lr = self.optimizer.state_dict()['param_groups'][0]['lr']
                        print('Epoch [%04d/%04d]: Loss=%.2e, Metric=%2.2f/%2.2f, %1.1e, RelErr=%.2e' % 
                              (epoch, self.num_epoch, self.loss_list[-1], self.metric_list[-1], max(self.metric_list), lr, rel_err))
                        
                        if self.print_image:
                            f, ([ax1, ax2, ax3, ax7, ax9], [ax4, ax5, ax6, ax8, ax10]) = plt.subplots(2, 5, figsize=(18,7))
                            ax1.imshow(imresize(self.img_degraded[:,:,self.rgb], output_shape=[self.gt.shape[0],self.gt.shape[1]]))
                            ax1.set_title('Input LRHS')
                            ax4.imshow(imresize(self.LRHS_hat[:,:,self.rgb], output_shape=[self.gt.shape[0],self.gt.shape[1]]))  
                            ax4.set_title('Output LRHS [epoch=%d]'%(epoch))
                            
                            ax2.imshow(self.guidance[:,:,[2,1,0]])  if self.guidance.shape[-1]!=1 else ax2.imshow(self.guidance[:,:,0])
                            ax2.set_title('HRMS')
                            ax5.imshow(self.HRMS_hat[:,:,[2,1,0]])  if self.guidance.shape[-1]!=1 else ax5.imshow(self.HRMS_hat[:,:,0])
                            ax5.set_title('Output HRMS [epoch=%d]'%(epoch))
                            
                            ax3.imshow(self.gt[:,:,self.rgb])  
                            ax3.set_title('HRHS (Ground Truth)')
                            ax6.imshow(self.output[:,:,self.rgb])  
                            ax6.set_title('Output HRHS [epoch=%d]'%(epoch))
                            
                            ax7.imshow(imresize(self.model.blur_kernel, output_shape=[self.output.shape[0],self.output.shape[1]]))
                            ax7.set_title('Blur Kernel')
                            ax8.plot(self.model.srf.T) 
                            ax8.set_title('SRF')
                            
                            ax9.plot(self.loss_list)
                            ax9.set_title('Loss Curve')
                            ax10.plot(self.metric_list)
                            ax10.set_title('Metrics Curve')
    
                            plt.show()
                        
                    else:
                        print('Epoch [%04d/%04d]: Loss=%.4e, RelErr=%.2e' % (epoch, self.num_epoch, self.loss_list[-1], rel_err))
                        
                        if self.print_image:
                            f, ([ax1, ax2], [ax4, ax5]) = plt.subplots(2, 2, sharey=True, figsize=(15,15))
                            ax1.imshow(imresize(self.img_degraded[:,:,self.rgb], output_shape=[self.gt.shape[0],self.gt.shape[1]]))
                            ax1.set_title('Input LRHS')
                            ax4.imshow(imresize(self.LRHS_hat[:,:,self.rgb], output_shape=[self.gt.shape[0],self.gt.shape[1]]))  
                            ax4.set_title('Output LRHS [epoch=%d]'%(epoch))
                            
                            ax2.imshow(self.guidance[:,:,[0,1,2]])  
                            ax2.set_title('HRMS')
                            ax5.imshow(self.HRMS_hat[:,:,[0,1,2]])  
                            ax5.set_title('Output HRMS [epoch=%d]'%(epoch))
                            
                            plt.show()
                
        else:
            if self.print_every!=0:
                if epoch%self.print_every==0 or epoch==self.num_epoch-1:
                    if self.gt is not None:
                        print('Epoch [%04d/%04d]: Loss=%.4e, Metric=%2.2f/%2.2f, RelErr=%.2e' % 
                              (epoch, self.num_epoch, self.loss_list[-1], self.metric_list[-1], max(self.metric_list), rel_err))
                        if self.print_image:
                            f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True, figsize=(15,15))
                            ax1.imshow(imresize(self.img_degraded[:,:,self.rgb], output_shape=[self.gt.shape[0],self.gt.shape[1]]))
                            ax1.set_title('Degraded Image')
                            ax2.imshow(self.output[:,:,self.rgb])  
                            ax2.set_title('Reconstructed Image [epoch=%d]'%(epoch))
                            ax3.imshow(self.gt[:,:,self.rgb])  
                            ax3.set_title('Ground Truth Image')
                            plt.show()
                    else:
                        print('Epoch [%04d/%04d]: Loss=%.4e, RelErr=%.2e' % (epoch, self.num_epoch, self.loss_list[-1], rel_err))
                        if self.print_image:
                            f, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(15,15))
                            ax1.imshow(self.img_degraded[:,:,self.rgb])
                            ax1.set_title('Degraded Image')
                            ax2.imshow(self.output[:,:,self.rgb])  
                            ax2.set_title('Reconstructed Image [epoch=%d]'%(epoch))
                            plt.show()
    
    def set_gif_dict(self, gif_dict):
        if gif_dict['save_root'] is not None: # in this case the code will generate a gif file
            # the frequency to add a frame into the gif file
            if gif_dict['frequency']=='default' or 'frequency' not in gif_dict:
                gif_dict['frequency'] = int(self.num_epoch/10) 
            # the speed to play the gif file
            if 'speed' not in gif_dict:
                gif_dict['speed'] = 0.5
            # resize the frame 
            if 'resize' not in gif_dict:
                gif_dict['resize'] = [512,512]
        self.gif_dict = gif_dict

    def set_gif_frame(self, epoch):
        if self.gif_dict['save_root'] is not None:
            if epoch%self.gif_dict['frequency'] == 0 or epoch==self.num_epoch-1:
                temp_frame = (255*self.output[:,:,self.rgb]).astype(np.uint8)
                temp_frame = Image.fromarray(temp_frame)
                if self.gif_dict['resize'] is not None:
                    temp_frame = temp_frame.resize(self.gif_dict['resize'])
                
                # set text content, font family & font size 
                font = ImageFont.truetype(font='arial.ttf', size=np.floor(3e-2*1000+0.5).astype('int32'))
                if self.gt is not None:
                    text_content = 'Epoch=%05d, PSNR=%2.2f'% (epoch, self.metric_fn(self.output, self.gt))
                else:
                    text_content = 'Epoch=%05d'% (epoch)
                draw = ImageDraw.Draw(temp_frame)
                text_size = draw.textsize(text_content, font)
                text_content = text_content.encode('utf-8')
                
                # draw rectangle and text
                draw.rectangle([0,0,0+text_size[0],0+text_size[1]], fill=(0,0,0), outline=(0,0,0))
                draw.text([0,0], text=text_content.decode(), fill=(255,255,255), font=font)
                
                # add to frames
                self.frames.append(temp_frame)

    