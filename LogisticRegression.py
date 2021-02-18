# -*- coding: utf-8 -*-
"""
Created on Thu Feb 18 12:52:25 2021

@author: Begum Dogru
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

veriler = pd.read_csv(r'C:\Users\Lenovo\Desktop\Machine Learning\veriler.txt')

x = veriler.iloc[:,1:4]
y = veriler.iloc[:, 4:]

X = x.values
Y = y.values
#ülke sayısal olmadığı için onu encodeladık
from sklearn import preprocessing
le =  preprocessing.LabelEncoder()
ulke =  veriler.iloc[:,:1].values
print(ulke)
ulke[:,0] = le.fit_transform(veriler.iloc[:,0])
ohe = preprocessing.OneHotEncoder()
ulke = ohe.fit_transform(ulke).toarray()

#encodeladığımız ülke yi verilerle birleştiriyoruz
sonuc = pd.DataFrame(data= ulke, index=range(22),columns=(['fr','tr','us']))
sonuc2 = pd.DataFrame(data = X, index=range(22),columns=(['boy','kilo','yas']))
sonuc3 = pd.DataFrame(data = Y, index=range(22),columns=(['cinsiyet']))

s= pd.concat([sonuc,sonuc2], axis=1)
s2 = pd.concat([s,sonuc3], axis=1)



#train ediyoruz
from sklearn.model_selection import train_test_split
x_train, x_test,y_train, y_test = train_test_split(sonuc2, sonuc3,test_size=0.33,random_state= 0)


#verileri birbirine göre ölçeklendirelim yani aynı oranlarda olsunlar
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test)

from sklearn.linear_model import LogisticRegression
logis_reg = LogisticRegression(random_state=0)
logis_reg.fit(X_train, y_train)

y_predict = logis_reg.predict(X_test)
print(y_predict)