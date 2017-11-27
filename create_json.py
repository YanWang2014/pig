#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 14:30:13 2017

@author: wayne
"""

import json
import os


phases = ['val']

if phases[0] == 'test_A':
    test_root = 'data/test_A'
elif phases[0] == 'test_B':
    test_root = 'data/test_B'
elif phases[0] == 'val':
    test_root = 'data/validation_folder'

label_raw = []

def file_name2(file_dir):   #特定类型的文件
    L=[]   
    image = []
    if phases[0] == 'val':
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
                    label_raw.append({'image_id':file, 'label_id':1})
    return L, image

path, image_id = file_name2(test_root) #图片目录


with open(test_root+'/pig_test_annotations.json', 'w') as f:
    json.dump(label_raw, f)
    

with open(test_root+'/pig_test_annotations.json', 'r') as f: 
    label_raw_test = json.load(f)