# -*- coding: utf-8 -*-
"""
Created on Fri Jun 15 21:35:05 2018

@author: Romildo Alves
"""

import numpy as np
import matplotlib as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:, 3].values

# Taking care of the missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'Nan', strategy = 'mean', axis = 0)
imputer.fit(X)