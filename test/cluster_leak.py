# -*- coding: utf-8 -*-
"""
Created on Sun Nov 26 11:30:56 2017

@author: Beyond
"""
import os

file = 'result/tf/A_1.csv'
test_root = 'data/test_A/'

a = os.listdir(test_root)

stats = {}
for image in a:
    if image.endswith('.JPG'): # there is a json file --
        stats[int(image.split('.')[0])] = os.stat(test_root + image)[-2]

sorted_images = sorted(stats, key=stats.get)

# should be neighbours
print(stats[2122])
print(stats[1901])
print(stats[4777])
print(time.localtime(stats[2122]))
print(time.localtime(stats[1901]))
print(time.localtime(stats[4777]))
print(sorted_images.index(2122))
print(sorted_images.index(1901))
print(sorted_images.index(4777))


#####################average the softmax########################

