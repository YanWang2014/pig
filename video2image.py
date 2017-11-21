#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 11:50:42 2017

@author: wayne
"""

from subprocess import call
import os
import shutil

'''
video to videos_folder
'''
#for ids in range(12,31):
#    os.system('ffmpeg -i videos_folder/{}/{}.mp4 -vf fps=10  videos_folder/{}/image{}-%4d.jpg'.format(ids,ids,ids,ids))



'''
videos_folder to train_folder and validation_folder. 
'''
TRAIN_ROOT = 'data/train_folder/'
VALIDATION_ROOT = 'data/validation_folder/'
videos_folder = 'data/videos_folder/'
total_frames = 592
split_ratio = 4
split_point = 500
recreate = False

if recreate:
    shutil.rmtree(TRAIN_ROOT) 
    shutil.rmtree(VALIDATION_ROOT) 
    os.makedirs('data/train_folder') 
    os.makedirs('data/validation_folder') 

def im_path(pig_class, im_id, folder):
    if im_id == None:
        return folder + str(pig_class)
    else:
        return folder + str(pig_class) + '/' + ('image%d-%04d'%(pig_class, im_id)) + '.jpg'
    
    
for pig_class in range(1,31):
    for im_id in range(1,total_frames):
        if im_id >= split_point: #% split_ratio == 0:
            try:
                shutil.copy(im_path(pig_class, im_id, videos_folder), im_path(pig_class, im_id, VALIDATION_ROOT))
            except:
#            FileNotFoundError as e:
#                print(e)
#                raise
                os.makedirs(im_path(pig_class, None, VALIDATION_ROOT))
                shutil.copy(im_path(pig_class, im_id, videos_folder), im_path(pig_class, im_id, VALIDATION_ROOT))
        else:
            try:
                shutil.copy(im_path(pig_class, im_id, videos_folder), im_path(pig_class, im_id,TRAIN_ROOT))
            except:
                os.makedirs(im_path(pig_class, None, TRAIN_ROOT))
                shutil.copy(im_path(pig_class, im_id, videos_folder), im_path(pig_class, im_id,TRAIN_ROOT))
        


