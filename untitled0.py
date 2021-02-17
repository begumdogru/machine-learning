# -*- coding: utf-8 -*-
"""
Created on Tue Feb  9 11:16:59 2021

@author: Begum Dogru
"""


#1.kutuphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#2.veri onisleme
#2.1.veri yukleme
veriler = pd.read_csv(r'C:\Users\Lenovo\Desktop\Machine Learning\tenis.txt')
#pd.read_csv("veriler.csv")
#test
print(veriler)

#encoder: Kategorik -> Numeric

from sklearn import preprocessing

veriler2 = veriler.apply(preprocessing.LabelEncoder().fit_transform) #bütün
#verileri encode eder ben buradan sadece işime gelenleri kullanacağım.
outlook = veriler2.iloc[:,:1]


from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(categorical_features='all')
outlook = ohe.fit_transform(outlook).toarray()
print(outlook)


'''
#numpy dizileri dataframe donusumu
sonuc = pd.DataFrame(data=outlook, index = range(14), columns = ['r','o','s'])
print(sonuc)
sonuc2 = pd.DataFrame(data=windy, index = range(14), columns = ['t','f'])
print(sonuc2)


#dataframe birlestirme islemi
s=pd.concat([sonuc,veriler.iloc[:,1:3]], axis=1)
print(s)
s2 = pd.concat([s,sonuc2],axis =1)


#verilerin egitim ve test icin bolunmesi
from sklearn.model_selection import train_test_split

x_train, x_test,y_train,y_test = train_test_split(s,play,test_size=0.33, random_state=0)
x_train = x_train.sort_index()
x_test = x_test.sort_index()
y_train = y_train.sort_index()
y_test = y_test.sort_index()

from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr = lr.fit(x_train,y_train)
predict_y = lr.predict(x_test)

import statsmodels.api as sm
X = np.append(arr=np.ones((14,1)).astype(int),values=veriler, axis=1)
Xlist = veriler.iloc[:,[0,1,2,3,4]].values
Xlist = np.array(Xlist,dtype=float())
model = sm.OLS(play, Xlist).fit()
print(model.summary())



'''