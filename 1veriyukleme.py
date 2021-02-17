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
veriler = pd.read_csv(r'C:\Users\Lenovo\Desktop\Machine Learning\veriler.txt')
#print(veriler)
boy = veriler[['boy']]
#print(boy)
boykilo = veriler[['boy','kilo']]
print(boykilo)