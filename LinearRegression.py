#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 30 21:40:40 2020

@author: josephsmith
"""

#import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.linear_model import LinearRegression


#read_tsv = pd.read_csv("/Users/josephsmith/data.tsv", delimiter="\t")

#read_tsv.head()

#x_axis = np.arange(0., 10., 1)
#y_axis = np. arange(0., 20., 0.5)

#plt.plot(x_axis, 'ro', y_axis, 'r--')
#plt.show()

#x = np.arange(5)
 
reviews = pd.read_csv("/Users/josephsmith/winemag-data-130k-v2.csv", index_col=0)
#print(reviews.head())
sort = reviews.groupby(['country', 'province']).price.agg([len, min, max])
#print(sort.iloc[1,-1])
#print(reviews.head())
#print("\n")

points_price = reviews.loc[:, ['points', 'price']]


points_var = reviews.points.value_counts(ascending=True)
#print(points_var)



x = reviews.loc[:, ['points']].fillna(0)
y = reviews.loc[:, ['price']].fillna(0)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
regressor = LinearRegression()
regressor.fit(x_train, y_train)
print(regressor.intercept_)

y_pred = regressor.predict(x_test)
y_pred_df = pd.DataFrame({"price": y_pred[:,0]})

data = {'Actual': y_test.reset_index().drop(['index'], axis=1), 'Predicted': y_pred_df}
df = pd.DataFrame([data])
print(df)
print("\n")

plt.plot(x, y, 'o')
plt.plot(x_test, y_pred, 'r--')
plt.show()










