#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 11 15:25:02 2020

@author: josephsmith
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

reviews = pd.read_csv("/Users/josephsmith/winemag-data-130k-v2.csv", index_col=0)

plt.scatter(reviews['points'], reviews['price'], color='green', marker='*')
plt.show()