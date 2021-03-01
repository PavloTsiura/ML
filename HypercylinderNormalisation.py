#!/usr/bin/env python
# coding: utf-8




import pandas as pd
import numpy as np
import math as mt



df = pd.read_csv('C:\\Desktop\\train.txt', sep = '\t').to_numpy()


print(df)


sumallrovs = sum(df)
rsd = np.sqrt(sum(df**2))
print(rsd.shape)
result = sumallrovs / np.array(list(map(zeroExc,rsd)))

print(result)


def zeroExc (x):
    return x if x != 0 else 0.00001
   


pd.DataFrame(result).to_csv('C:/Users/Desktop/TrainNormalize.csv')





