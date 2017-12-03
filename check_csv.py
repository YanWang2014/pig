#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 15:44:29 2017

@author: wayne
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#file = 'result/torch/test_A_1.csv'
#file = 'result/tf6/A_1.csv'
#file = 'result/val_1.csv'
#file = 'result/test_A_1.csv'
#file = 'result/me/test_A_1.csv'
file = 'result/me2/152test_A_1.csv'

#KK = '30'  
#name = 'test'#val, test
#file = ('result/tf6_%s/A_1_%s.csv' %(name,KK))

df = pd.read_csv(file, header = None)

df_max = df.groupby(df[0]).max()
df_max.to_csv(file.split('.')[0]+'_max.csv', header = None)
plt.hist(df_max[2], bins='auto')  # arguments are passed to np.histogram
plt.show()
df_max[3] = np.log(df_max[2])
print(-df_max[3].sum()/len(df_max))

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


df_max = df.groupby(df[0]).max()
df_max.to_csv(file.split('.')[0]+'_max.csv', header = None)
plt.hist(df_max[2], bins='auto')  # arguments are passed to np.histogram
plt.show()
df_max[3] = np.log(df_max[2])
print(-df_max[3].sum()/len(df_max))

# 计算val的loss
#if 'val' in file:
#    df

# 统计预测类别的分布， 与txt2csv类似
freq = df.sort_values(2, ascending=False).drop_duplicates([0])
plt.hist(freq[1], bins=30)  # arguments are passed to np.histogram
