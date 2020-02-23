import datetime as datetime
import os
import pandas_datareader.data as pdr
import fix_yahoo_finance as yf
from pandas_datareader._utils import RemoteDataError

yf.pdr_override

start = datetime.datetime(2017, 2, 6)
end = datetime.datetime(2020, 2, 19)

with open('snp500_formatted.txt', 'r') as f:
    data = f.read()
data = data.split('\n')
all_filenames = data

os.chdir("D:/Ronet's Corner/VIT/Capstone Project/Code/Stock Prediction using news articles on a deep learning architecture/2019 data")

for name in all_filenames:
    try:
        df = pdr.DataReader(name, 'yahoo', start, end)
        df=df.iloc[::-1]
        df.to_csv(name + ".csv")
    except (RemoteDataError, KeyError):
        continue
