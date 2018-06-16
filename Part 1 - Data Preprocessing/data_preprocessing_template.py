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
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:, 3].values