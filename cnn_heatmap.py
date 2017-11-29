#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
adapted from https://github.com/CSAILVision/places365/blob/master/run_placesCNN_unified.py
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import time
import json
from model import load_model
from config import data_transforms
import pickle
import csv
from params import *
import torchvision.datasets as td
import cv2
import numpy as np
from scipy.misc import imresize as imresize

plot_num = 20

phases = ['test_A']
batch_size = BATCH_SIZE
os.mkdir('heat_map')

if phases[0] == 'test_A':
    test_root = 'data/test_A'
elif phases[0] == 'test_B':
    test_root = 'data/test_B'
elif phases[0] == 'val':
    test_root = 'data/validation_folder_full'
    

use_gpu = torch.cuda.is_available()
checkpoint_filename = arch + '_' + pretrained
best_check = 'checkpoint/' + checkpoint_filename + '_best.pth.tar' #tar


model_conv = load_model(arch, pretrained, use_gpu=use_gpu, num_classes=30,  AdaptiveAvgPool=AdaptiveAvgPool, SPP=SPP, num_levels=num_levels, pool_type=pool_type, bilinear=bilinear, stage=stage, SENet=SENet,se_stage=se_stage,se_layers=se_layers)
for param in model_conv.parameters():
    param.requires_grad = False #节省显存

best_checkpoint = torch.load(best_check)
# hook the feature extractor
features_blobs = []
def hook_feature(module, input, output):
    features_blobs.append(np.squeeze(output.data.cpu().numpy()))
features_names = ['layer4','avgpool'] # this is the last conv layer of the resnet
for name in features_names:
    model_conv._modules.get(name).register_forward_hook(hook_feature)

tf = data_transforms('validation')

def returnCAM(feature_conv, weight_softmax, class_idx):
    # generate the class activation maps upsample to 256x256
    size_upsample = (256, 256)
    nc, h, w = feature_conv.shape
    output_cam = []
    for idx in class_idx:
        cam = weight_softmax[class_idx].dot(feature_conv.reshape((nc, h*w)))
        cam = cam.reshape(h, w)
        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)
        cam_img = np.uint8(255 * cam_img)
        output_cam.append(imresize(cam_img, size_upsample))
    return output_cam

# get the softmax weight
params = list(model_conv.parameters())
weight_softmax = params[-2].data.numpy()
weight_softmax[weight_softmax<0] = 0
    
images = os.listdir(test_root)
for i in range(0, plot_num):
    if images[i].split('.')[1] in ['jpg', 'JPG']:
        name = test_root + '/' + images[i]
        img = Image.open(name)
        input_img = Variable(tf(img).unsqueeze(0), volatile=True)
        
        # forward pass
        logit = model_conv(input_img)
        h_x = F.softmax(logit).data.squeeze()
        probs, idx = h_x.sort(0, True)
        
        # generate class activation mapping
        CAMs = returnCAM(features_blobs[0], weight_softmax, [idx[0]])
        
        img = cv2.imread(name)
        height, width, _ = img.shape
        heatmap = cv2.applyColorMap(cv2.resize(CAMs[0],(width, height)), cv2.COLORMAP_JET)
        result = heatmap * 0.4 + img * 0.5
        cv2.imwrite('heat_map/'+images[i], result)