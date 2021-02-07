#!/usr/bin/env python
# coding: utf-8

# In[101]:


import pandas as pd
import numpy as np
import math as mt


# In[189]:


df = pd.read_csv('C:\\Desktop\\train.txt', sep = '\t').to_numpy()


print(df)


# In[212]:


sumallrovs = sum(df)
kvad = np.sqrt(sum(df**2))
print(kvad.shape)
result = sumallrovs / np.array(list(map(fas,kvad)))

print(result)


# In[196]:


def fas (kvad):
    return kvad if kvad != 0 else 0.00001
    


# In[213]:


pd.DataFrame(result).to_csv('C:/Users/Desktop/TrainNormalize.csv')


# In[ ]:




