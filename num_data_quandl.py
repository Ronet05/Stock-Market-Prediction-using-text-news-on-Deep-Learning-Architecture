# -*- coding: utf-8 -*-
"""
Created on Sun Feb  9 19:03:21 2020

@author: ronet
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import glob
import os
import csv
import json
import requests
from bs4 import BeautifulSoup
import datetime
import pandas_datareader.data as web
from pandas import Series, DataFrame

start=datetime.datetime(2013,1,2)
end=datetime.datetime(2019,12,31)
api_key="mYnpdhz7HnGbCoT5qz6-"

with open('snp500_formatted.txt','r') as f:
    data=f.read()
data=data.split('\n')
all_filenames=data

for name in all_filenames:
    df=web.DataReader(name, 'quandl', start, end, access_key=api_key)
    df=df.drop(columns=['ExDividend', 'SplitRatio', 'AdjOpen', 'AdjHigh', 'AdjLow', 'AdjVolume'])
    df.to_csv(name+".csv")
    

