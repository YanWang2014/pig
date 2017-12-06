#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 19:17:57 2017

@author: wayne
"""

from PIL import Image
import os
import os.path
import torch.utils.data as data
import json
import torch
from torchvision import transforms

x = torch.randn(1,2,2,2) # nchw
print(x)
#y = x.mean(dim=0, keepdim=True).mean(dim=2, keepdim=True).mean(dim=3, keepdim=True)
#print(y)
#print(y+1)
#print(y.squeeze())
#print((y+1).squeeze().sqrt())
#print((y+1).squeeze()**2)
#print('start here')
dataloader = [x]

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


#calculate mean and variance
mean_meter = AverageMeter()
for i, image in enumerate(dataloader):  # nchw
    if i%10 ==0:
        print(i)
    mean_meter.update(image.mean(dim=0, keepdim=True).mean(dim=2, keepdim=True).mean(dim=3, keepdim=True), image.size(0))  

mean = mean_meter.avg
print(mean.squeeze())
std_meter =  AverageMeter()
for i, image in enumerate(dataloader):  # nchw
    if i%10 ==0:
        print(i)
    std_meter.update(((image-mean)**2).mean(dim=0, keepdim=True).mean(dim=2, keepdim=True).mean(dim=3, keepdim=True), image.size(0))  
print(std_meter.avg.squeeze().sqrt())