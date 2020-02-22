import csv
import sentiment
import os
from textblob import TextBlob
import datetime


def getStockData(symbol, date):
    file = open('2019 data/' + symbol + '.csv', 'r')
    csv_file = csv.reader(file)

    date += datetime.timedelta(days=1)
    data = []

    print("Getting stock data for {} for date {}".format(symbol, date.strftime('%Y-%m-%d')))

    for row in csv_file:
        if (row[0] == date.strftime('%Y-%m-%d')):
            data.append(float(row[1]))
            data.append(float(row[4]))

            return data

    return -1


file_to_process = open('2019 data/process_file.csv', 'a')
writer = csv.writer(file_to_process)

# date (start date needs to be fed (pickle original)
endDate = datetime.date(2020, 2, 19)

while date < endDate:
    print('Checking for date' + date.strftime('%Y-%m-%d'))
    day = date.weekday()
    if day == 4 or day == 5:
        date += datetime.timedelta(days=1)
        continue

    filename = date.strftime('%Y-%m-%d')
    file = open('2019 data/news/' + fname + '.csv')
    csv_file = csv.reader(file)

    for row in csv_file:
        comp = row.split(',', 1)[0]
        comp_news = row.split(',', 1)[1]
        stockData = getStockData(comp, date)
        if stockData == -1:
            continue

        sentdata = TextBlob(comp_news)

        data = []
        data.append((comp, date.timetuple().tm_yday))
        data.extend((sentiment.sent_score(comp_news), sentiment.sent_magnitude(comp_news)))
        data.extend(stockData)
        writer.writerow(data)

    date += datetime.timedelta(days=1)
