import numpy as no
import csv
import sentiment
import datetime
import pandas as pd


def getStockData(symbol, date):
    try:
        csv_file = pd.read_csv('2019 data/' + symbol + '.csv')
    except FileNotFoundError:
        return -1
    # getstock data for the next date
    date += datetime.timedelta(days=1)
    data = []
    print("Getting stock data for {} for date {}".format(symbol, date.strftime('%Y-%m-%d')))

    if csv_file.loc[csv_file['Date'] == date.strftime('%Y-%m-%d')]:
        data.append(csv_file.loc[csv_file['Date'] == date.strftime('%Y-%m-%d'), 'Open'])
        data.append(csv_file.loc[csv_file['Date'] == date.strftime('%Y-%m-%d'), 'Close'])

        return data

    return -1


alphas = [1, 1.5, 3, 5, 8, 10, 15]

for alpha in alphas:
    file_to_process = open('process_avg/process_file_avg' + str(alpha) + '.csv', 'a', newline="", encoding='utf-8')
    writer = csv.writer(file_to_process)

    date = datetime.date(2018, 2, 6)
    endDate = datetime.date(2020, 2, 19)

    while date < endDate:
        print('Checking for date: ' + date.strftime('%Y-%m-%d'))
        day = date.weekday()
        if day == 5 or day == 6:
            date += datetime.timedelta(days=1)
            continue

        filename = date.strftime('%Y-%m-%d')
        try:
            news_df = pd.read_csv('2019 data/news/' + filename + '.csv', encoding='utf-8')
        except FileNotFoundError:
            date += datetime.timedelta(days=1)
            continue



