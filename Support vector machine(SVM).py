#!/usr/bin/env python
# coding: utf-8

import pandas as pd
from sklearn.datasets import load_digits

digits = load_digits()

dir(digits)

digits.data

df = pd.DataFrame(digits.data,digits.target)
df.head()

df['target'] = digits.target
df.head()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df.drop('target',axis='columns'), df.target, test_size=0.3)

from sklearn.svm import SVC


model = SVC(kernel='linear')

model.fit(X_train,y_train)
model.score(X_test,y_test)







