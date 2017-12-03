#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  2 10:00:13 2017

@author: wayne
"""
import pandas as pd

files = ['result/me2/18test_A_1.csv', 'result/me2/50test_A_1.csv', 'result/me2/152test_A_1.csv']

dfs = []
for file in files:
    dfs.append(pd.read_csv(file, header = None))

#temps = [df.groupby([0,1])[2].sum() for df in dfs]
#means = sum(temps)/len(temps)

#df = pd.DataFrame(columns=(0, 1, 2))
#for i in range(len(means)):
#    if i%1000 == 0:
#        print (i)
#    df.loc[i] = [means.index[i][0], means.index[i][1], means[means.index[i][0], means.index[i][1]]]
#
#    

pivots = []
for i in range(len(dfs)):
    pivots.append(pd.pivot_table(dfs[i], index=[0,1]))

mean_pivot = sum(pivots)/len(pivots)


df = mean_pivot.reset_index()  
df.to_csv('result/me2/ensemble.csv', header = None, index = None)
