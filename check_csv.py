#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 15:44:29 2017

@author: wayne
"""

import pandas as pd

df = pd.read_csv('result/test_A_1.csv', header = None)

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