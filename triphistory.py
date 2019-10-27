# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 14:08:21 2019

@author: Freakky7781_VRikk
"""

from sklearn.naive_bayes import GaussianNB
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
data=pd.read_csv('tripdata.csv')

print(data.head(5))
data=data.drop('Start date',axis=1)
data=data.drop('End date',axis=1)
data=data.drop('Start station',axis=1)
data=data.drop('End station',axis=1)
print(data.shape)

model=GaussianNB()
le=LabelEncoder()
le.fit(data['Member type'])
data['Member type']=le.transform(data['Member type'])

le=LabelEncoder()
le.fit(data['Bike number'])
data['Bike number']=le.transform(data['Bike number'])

train=np.array(data.iloc[0:85000])
test=np.array(data.iloc[85000:])
print(train.shape)
print(test.shape)
model.fit(train[ : ,0:4],train[ : ,4])
predicted=model.predict(test[ : ,0:4])
print(test[ : ,4])
print(predicted)
count=0
for i in range(30597):
    if(predicted[i]==test[i,4]):
        count+=1
print(count)
print(count/30597)