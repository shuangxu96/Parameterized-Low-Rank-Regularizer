# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F


class BoxFilter(nn.Module):
    def __init__(self, window_size, reduction='mean'):
        # :param window_size: Int or Tuple(Int, Int) in (win_width, win_height) order
        # :param reduction: 'mean' | 'sum'
        super().__init__()
        wx, wy = window_size if isinstance(window_size, (list, tuple)) else (window_size, window_size)
        assert wx % 2 == 1 and wy % 2 == 1, 'window size must be odd'
        self.rx, self.ry = wx // 2, wy // 2
        self.area = wx * wy
        self.reduction = reduction
    
    def forward(self, tensor):
        # :param tensor: torch.Tensor(N, C, H, W, ...)
        # :return: torch.Tensor(N, C, H, W, ...)
        local_sum = torch.zeros_like(tensor)
        for x_shift in range(-self.rx, self.rx + 1):
            for y_shift in range(-self.ry, self.ry + 1):
                local_sum += torch.roll(tensor, shifts=(y_shift, x_shift), dims=(2, 3))
    
        return local_sum if self.reduction == 'sum' else local_sum / self.area

class NonLocalMeans(nn.Module):
    def __init__(self, h=1, search_window_size=11, patch_size=5):
        super().__init__()
        # self.h = nn.Parameter(torch.tensor([float(1)]), requires_grad=True)
        self.h = h

        self.box_sum = BoxFilter(window_size=patch_size, reduction='sum')
        self.r = search_window_size // 2

    def forward(self, rgb):
        batch_size, _, height, width = rgb.shape
        weights = torch.zeros((batch_size, 1, height, width)).float().to(rgb.device)  # (N, 1, H, W)
        denoised_rgb = torch.zeros_like(rgb)  # (N, 3, H, W)

        y = rgb.mean(1, keepdim=True)  # (N, 1, H, W)
        # y = rgb.clone()

        for x_shift in range(-self.r, self.r + 1):
            for y_shift in range(-self.r, self.r + 1):
                shifted_rgb = torch.roll(rgb, shifts=(y_shift, x_shift), dims=(2, 3))  # (N, 3, H, W)
                shifted_y = torch.roll(y, shifts=(y_shift, x_shift), dims=(2, 3))  # (N, 1, H, W)
                with torch.no_grad():

                    distance = torch.sqrt(self.box_sum((y - shifted_y) ** 2))  # (N, 1, H, W)
                # weight = torch.exp(-distance / (torch.relu(self.h) + 1e-6))  # (N, 1, H, W)
                    weight = torch.exp(-distance / (self.h + 1e-6))  # (N, 1, H, W)

                denoised_rgb += shifted_rgb * weight  # (N, 3, H, W)
                weights += weight  # (N, 1, H, W)

        # return torch.clamp(denoised_rgb / weights, 0, 1)  # (N, 3, H, W)
        return denoised_rgb / weights
