# -*- coding: utf-8 -*-
"""
Created on Sun Nov 26 09:49:34 2017

@author: Beyond
"""

import pandas as pd
import csv
import matplotlib.pyplot as plt

#file = 'result/tf6/A_1.csv'
#file1 = 'result/tf6/test_image_name.csv'
#file2 = 'result/tf6/test_image_result.csv'

KK = '100'
MODE = 'valid'
file = ('result/tf6_val/A_1_%s.csv' %KK)
file1 = ('result/tf6_val/%s_image_name.csv'%MODE)
file2 = ('result/tf6_val/%s_k_%s_image_result.csv' %(MODE,KK))

def write_to_csv(aug_softmax):
    with open(file, 'w', encoding='utf-8') as csvfile:
        spamwriter = csv.writer(csvfile,dialect='excel')
        for item in aug_softmax.keys():
            the_sum = sum(aug_softmax[item])
            for c in range(0,30):
                spamwriter.writerow([item, c+1, aug_softmax[item][c]/the_sum])


df1 = pd.read_csv(file1, header = None)
df2 = pd.read_csv(file2, header = None, sep=' ')

print(max(df2.sum(axis=1)))
#print(min(df2.min(axis=1))


aug_softmax = {}
for index in df1.index:
    aug_softmax[df1[0][index]] = list(df2.loc[index])
    
write_to_csv(aug_softmax)



print('=========================================')
df = pd.read_csv(file, header = None)

#(0,1)
print(df[2].max())
print(df[2].min())

df3 = df.groupby(df[0]).sum()

#465
print(df3[1].max())
print(df3[1].min())

#1
print(df3[2].max())
print(df3[2].min())


# 统计每一类出现的次数，没有考虑属于多个类概率相同的情况
freq = df2.idxmax(axis=1)
plt.hist(freq, bins=30)  # arguments are passed to np.histogram
