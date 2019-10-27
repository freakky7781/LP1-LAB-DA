# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 13:45:39 2019

@author: Freakky7781_VRikk
"""

from sklearn.naive_bayes import GaussianNB
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

data=pd.read_csv('diabetes.csv')
print(data.head(5))
print(data.shape)
print(data['x2'].describe())


train=np.array(data.iloc[0:600])
test=np.array(data.iloc[600:768])
test.shape
print(train.shape)
model=GaussianNB()
model.fit(train[ : ,0:8],train[ : ,8])
predicted=model.predict(test[ : ,0:8])
print(test[ : ,8])
print(predicted)

count=0

for i in range(168):
    if(predicted[i]==test[i,8]):
        count+=1
print(count)
print(count/168)