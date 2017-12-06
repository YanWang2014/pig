#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 17:26:25 2017

@author: wayne


test脚本用的就是这个类，直接从一个文件夹中读取图片

https://discuss.pytorch.org/t/normalization-in-the-mnist-example/457/13
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # rerange them to [-1, +1]

 0.3139
 0.2796
 0.2770
 
 0.2941
 0.2688
 0.2680

"""

from PIL import Image
import os
import os.path
import torch.utils.data as data
import json
import torch
from torchvision import transforms

class PigDataset(data.Dataset):

    def __init__(self, json_labels, root_dir, transform=None):
        self.label_raw = json_labels
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.label_raw)

    def __getitem__(self, idx):
#        if phases[0] == 'val':
#            img_name = self.root_dir+ '/' + str(self.label_raw[idx]['label_id']+1) + '/'+ self.label_raw[idx]['image_id']
#        else:
        img_name = os.path.join(self.root_dir, self.label_raw[idx]['image_id'])
        img_name_raw = self.label_raw[idx]['image_id']
        image = Image.open(img_name)
        label = self.label_raw[idx]['label_id']

        if self.transform:
            image = self.transform(image)

        return image #, label, img_name_raw



if __name__ == "__main__":
    x = torch.randn(5,2,3,4) # nchw
    y = x.mean(dim=0, keepdim=True).mean(dim=2, keepdim=True).mean(dim=3, keepdim=True)
    print(y)
    print(y+1)
    print(y.squeeze())
    print((y+1).squeeze().sqrt())
    print((y+1).squeeze()**2)
    print('start here')

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
    
    
    phases = ['train']
    batch_size  = 256
    INPUT_WORKERS = 8
    
    if phases[0] == 'test_A':
        test_root = '../data/pig_test_resize'
    elif phases[0] == 'test_B':
        test_root = '../data/test_B'
    elif phases[0] == 'val':
        test_root = '../data/validation_folder_det_resize'
    elif phases[0] == 'train':
        test_root = '../data/train_folder_det_resize'
    
    with open(test_root+'/pig_test_annotations.json', 'r') as f: #label文件, 测试的是我自己生成的
        label_raw_test = json.load(f)
    
    transformed_dataset_test = PigDataset(json_labels=label_raw_test,
                                            root_dir=test_root,
                                               transform=transforms.ToTensor()
                                               )           
    dataloader = data.DataLoader(transformed_dataset_test, batch_size=batch_size,shuffle=False, num_workers=INPUT_WORKERS)

    
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