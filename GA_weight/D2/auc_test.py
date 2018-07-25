# -*- coding: utf-8 -*-
"""
Created on Wed Jul 25 16:04:33 2018

@author: zhang_yu
"""

import numpy as np
from sklearn import metrics
from sklearn.metrics import roc_auc_score
y = np.array([0, 0, 1, 1])
scores = np.array([0.1, 0.4, 0.35, 0.8])
auc=roc_auc_score(y, scores)
print(auc)
