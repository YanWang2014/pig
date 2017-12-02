#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  1 22:10:50 2017

@author: wayne
"""

import torch
from torch import nn
import torchvision


class Mask_relu(nn.Module):
    def __init__(self):
        super(Mask_relu, self).__init__()
        
    def forward(self, x):
        self.mask = torch.sum(x,1, keepdim=True) # nchw
        self.mask = (self.mask > torch.mean(torch.mean(self.mask, 1, keepdim=True),2, keepdim=True)).float()#.int()
        x = x * self.mask

        return x

class Mask_resnet(nn.Module):
    def __init__(self, arch, num_classes):
        super(Mask_resnet, self).__init__()
        
        print('using mask resnet')
        self.model = torchvision.models.__dict__[arch](pretrained=True)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.layer4 = self.model.layer4
        self.fc = nn.Linear(self.model.fc.in_features, num_classes)

        self.layer5 = Mask_relu()
        
    def forward(self, x):
        
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.layer4(x)

        x = self.layer5(x)
        
#        x = torch.cat((self.avg_pool(x), self.max_pool(x)), 1)
        x = self.avg_pool(x)
        
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        


        return x
