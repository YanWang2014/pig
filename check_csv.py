#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 15:44:29 2017

@author: wayne
"""

import pandas as pd

file = 'result/first/test_A_1.csv'
#file = 'result/tf/A_1.csv'

df = pd.read_csv(file, header = None)

#(0,1)
print(df[2].max())
print(df[2].min())

df2 = df.groupby(df[0]).sum()

#465
print(df2[1].max())
print(df2[1].min())

#1
print(df2[2].max())
print(df2[2].min())


def post_process(p):
    if p < 1/30.0:
        return 1/30.0 +(p - 1/30.0)*0.9
    else:
        return p

df[2] = df[2].apply(post_process) # 0是id, 1是类别, 2是概率
temp2 = df.groupby(0)[2].sum()
temp3 = df[0].map(df.groupby(0)[2].sum())
df3 = df.assign(normalized=df[2].div(df[0].map(df.groupby(0)[2].sum())))
df[2] = df3['normalized']
df.to_csv(file.split('.')[0]+'_post.csv', header = None, index = None)
print('=================after post-processing========================')


#(0,1)
print(df[2].max())
print(df[2].min())

df2 = df.groupby(df[0]).sum()

#465
print(df2[1].max())
print(df2[1].min())

#1
print(df2[2].max())
print(df2[2].min())
