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

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values= np.nan,strategy='mean')
Yas = veriler.iloc[:,1:4].values
print(Yas)
imputer = imputer.fit(Yas[:,1:4])
Yas[:,1:4]= imputer.transform(Yas[:,1:4])
print(Yas)