#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 22:15:16 2017

@author: wayne
"""

import torchvision
import torch.nn as nn

model = torchvision.models.__dict__['resnet152'](pretrained=True)

print(model)
#  )
#  (avgpool): AvgPool2d (size=7, stride=7, padding=0, ceil_mode=False, count_include_pad=True)
#  (fc): Linear (2048 -> 1000)
#)


model.avgpool = nn.Sequential(model.avgpool)

