# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 13:16:12 2019

@author: Freakky7781_VRikk
"""

from sklearn.naive_bayes import GaussianNB
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


data=pd.read_csv('Iris.csv')
data=data.drop('Id',axis=1)
print(data.head(5))
print(data.shape)

plt.hist(data['x1'],bins=30)
plt.xlabel("Dimensions")
plt.ylabel("no of times")
plt.show()

sns.boxplot(y=data['x1'])