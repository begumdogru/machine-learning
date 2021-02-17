# -*- coding: utf-8 -*-
"""
Created on Sun Feb  7 12:35:42 2021

@author: Begum Dogru
"""

# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
#kutuphaneler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


#data import - veri yükleme
veriler = pd.read_csv(r'C:\Users\Lenovo\Desktop\Machine Learning\eksikveriler.txt')
print(veriler)


#eksik veriler
#eksik verilerin bulunduğu sayısal haneye o kolonun ortalamasını alarak
#doldur.
#makine öğrenme sürecini fit ile, uygulanma
#sürecini ttansform ile çağırdık
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values= np.nan,strategy='mean')
Yas = veriler.iloc[:,1:4].values
print(Yas)
imputer = imputer.fit(Yas[:,1:4])
Yas[:,1:4]= imputer.transform(Yas[:,1:4])
print(Yas)

ulke = veriler.iloc[:,0:1].values
print(ulke)
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
ulke[:,0] =le.fit_transform(veriler.iloc[:,0])
print(ulke)
ohe = preprocessing.OneHotEncoder()
ulke= ohe.fit_transform(ulke).toarray()
print(ulke)
#verilerin birleştirilmesi ve data frame oluşturulması

sonuc = pd.DataFrame(data = ulke, index = range(22), columns=['fr','tr','us'])
sonuc2 = pd.DataFrame(data = Yas,index = range(22), columns=['boy','kilo','yas'])
cinsiyet = veriler.iloc[:,-1].values
print(cinsiyet)
sonuc3 = pd.DataFrame(data= cinsiyet, index = range(22), columns=['cinsiyet'])
print(sonuc)
print(sonuc2)
print(sonuc3)

s=pd.concat([sonuc,sonuc2],axis = 1)
print(s)
s2 = pd.concat([s,sonuc3],axis=1)
print(s2)

#dataset i bölmek ve test etmek:
from sklearn.model_selection import train_test_split
#test size'ı elimizdeki verilerin %33ü olacak şekilde bölüyoruz

x_train,x_test,y_train,y_test = train_test_split(s,sonuc3,test_size=0.33,random_state=0)

#verilerin birbirine yakın değerler ile ölçeklendirilmesi
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)
