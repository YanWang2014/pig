#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 14:30:13 2017

@author: wayne
"""

import json
import os

'''
#由于triplt不使用imagefolder，所以不需要对sorted(os.listdir())引起的label错位进行修正。所以要使得test.py中的val正确计算，可能也需要修改相应的代码
'''
triplet = True
phases = ['val']

if not triplet:  
    if phases[0] == 'test_A':
        test_root ='data/pig_test_resize'
    elif phases[0] == 'test_B':
        test_root = 'data/test_B'
    elif phases[0] == 'val':
        test_root = 'data/validation_folder_det_resize1'
    elif phases[0] == 'train':
        test_root = 'data/train_folder_det_resize1'
else:
    if phases[0] == 'test_A':
        test_root ='data/pig_test_resize'
    elif phases[0] == 'test_B':
        test_root = 'data/test_B'
    elif phases[0] == 'val':
        test_root = 'data/validation_folder_det_resize'
    elif phases[0] == 'train':
        test_root = 'data/train_folder_det_resize'  

label_raw = []

def file_name2(file_dir):   #特定类型的文件
    L=[]   
    image = []
    if phases[0] == 'val' or phases[0] == 'train':
        if not triplet:
            sorted_dir = sorted(os.listdir(test_root))
            for dir_ in os.listdir(file_dir):
                if dir_.endswith('.json'):
                    pass
                else:
                    for file in os.listdir(test_root+'/'+dir_):
                        if os.path.splitext(file)[1] in ['.jpg', '.JPG']:   
                            L.append(test_root+'/'+dir_+'/'+file)
                            image.append(file)
                            label_raw.append({'image_id':file, 'label_id':sorted_dir.index(dir_)})
        else:
            for root, dirs, files in os.walk(file_dir):  
                for file in files:  
                    if os.path.splitext(file)[1] in ['.jpg', '.JPG']:   
                        L.append(os.path.join(root, file))
                        image.append(file)
                        label_raw.append({'image_id':file, 'label_id':int(file.split('-')[0][-2:])-1})  # 程序中都是从0开始计算类别的
    else:
        for root, dirs, files in os.walk(file_dir):  
            for file in files:  
                if os.path.splitext(file)[1] in ['.jpg', '.JPG']:   
                    L.append(os.path.join(root, file))
                    image.append(file)
                    label_raw.append({'image_id':file, 'label_id':1})
    return L, image

path, image_id = file_name2(test_root) #图片目录


with open(test_root+'/pig_test_annotations.json', 'w') as f:
    json.dump(label_raw, f)
    

with open(test_root+'/pig_test_annotations.json', 'r') as f: 
    label_raw_test = json.load(f)