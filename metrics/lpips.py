"""
StarGAN v2
Copyright (c) 2020-present NAVER Corp.

This work is licensed under the Creative Commons Attribution-NonCommercial
4.0 International License. To view a copy of this license, visit
http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
"""

import paddle_torch as torch
import paddle_torch.nn as nn
from paddle_torch.vision.models.lpips import  LPIPS
# from torchvision import models




@torch.no_grad()
def calculate_lpips_given_images(group_of_images):
    # group_of_images = [torch.randn(N, C, H, W) for _ in range(10)]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    lpips = LPIPS(pretrained_weights_fn="./metrics/LPIPS_pretrained.pdparams")
    lpips.eval()
    lpips_values = []
    num_rand_outputs = len(group_of_images)

    # calculate the average of pairwise distances among all random outputs
    for i in range(num_rand_outputs-1):
        for j in range(i+1, num_rand_outputs):
            lpips_values.append(lpips(group_of_images[i], group_of_images[j]))
    lpips_value = torch.mean(torch.stack(lpips_values, dim=0))
    return lpips_value.numpy()