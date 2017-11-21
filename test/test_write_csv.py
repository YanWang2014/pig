#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 15:08:14 2017

@author: wayne
"""

import csv

aug_softmax = {'213': [1.2332, 3324, 3424]}

def write_to_csv(aug_softmax): #aug_softmax[img_name_raw[item]] = temp[item,:]
    with open('test.csv', 'w') as csvfile:
        spamwriter = csv.writer(csvfile,dialect='excel')
        for item in aug_softmax.keys():
            for c in range(0,3):
                spamwriter.writerow([item, c+1, aug_softmax[item][c]])


write_to_csv(aug_softmax)