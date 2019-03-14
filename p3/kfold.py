#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 17 22:19:48 2018

@author: francisco
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold # import KFold


X = np.array([[1,1], [1,2], [1,3], [1,4], 
			  [1,5], [1,6], [1,7], [1,8], 
			  [1,9]]) # create an array

y = np.array([1, 1, 2, 2, 3, 3, 4, 4, 1]) # Create another array
#kf = KFold(n_splits=2) # Define the split - into 2 folds 
kf = StratifiedKFold(n_splits=2)
kf.get_n_splits(X, y)# returns the number of splitting iterations in the cross-validator

for train_index, test_index in kf.split(X,y):
	print('TRAIN:', train_index, 'TEST:', test_index)
	print(y[train_index])
	X_train, X_test = X[train_index], X[test_index]
	y_train, y_test = y[train_index], y[test_index]