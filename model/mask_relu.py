#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 22:21:43 2017

@author: wayne

先将activations在通道维度求和，得到二维mask，其中低于均值的位置扔掉。
然后按照这个mask将每一个feature map中不要的位置设为0，只让保留的位置通过。

https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
https://github.com/KaimingHe/deep-residual-networks

https://discuss.pytorch.org/t/indexing-a-variable-with-a-mask-generated-from-another-variable/326
"""

import torch
from torch import nn


class Mask_relu(nn.Module):
    def __init__(self):
        super(Mask_relu, self).__init__()
        
        print('using mask relu')
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
#        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        self.mask = torch.sum(x,1, keepdim=True) # nchw
        self.mask = (self.mask > torch.mean(torch.mean(self.mask, 1, keepdim=True),2, keepdim=True)).float()#.int()
#        print(self.mask)
        x = x * self.mask
#        x = x[self.mask.expand(x.size())]
#        x = torch.masked_select(x, self.mask)
        x = torch.cat((self.avg_pool(x), self.max_pool(x)), 1)

        return x

if __name__ == "__main__":
    mask_relu = Mask_relu()
    x = torch.randn(1,1,3,4)
    y = mask_relu(x)
    print(x)
    print(y)