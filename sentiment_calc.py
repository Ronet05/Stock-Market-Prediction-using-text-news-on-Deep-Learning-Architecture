import csv
import sentiment
import datetime


# needs to be optimized using pandas to vectorize the working of the getStockData function

def getStockData(symbol, date):
    try:
        file = open('2019 data/' + symbol + '.csv', 'r')
    except FileNotFoundError:
        return -1
    csv_file = csv.reader(file)
    date += datetime.timedelta(days=1)
    data = []
    print("Getting stock data for {} for date {}".format(symbol, date.strftime('%Y-%m-%d')))

    for row in csv_file:
        if row[0] == date.strftime('%Y-%m-%d'):
            data.append(float(row[3]))
            data.append(float(row[4]))

            return data

    return -1

alphas = [0.5, 1.5, 3, 5, 8, 10, 15]  # 15 default as per VADER

for alpha in alphas:
    file_to_process = open('process/process_file_' + str(alpha) + '.csv', 'a', newline="", encoding='utf-8')
    writer = csv.writer(file_to_process)

    date = datetime.date(2017, 2, 6)
    endDate = datetime.date(2020, 2, 19)

    while date < endDate:
        print('Checking for date: ' + date.strftime('%Y-%m-%d'))
        day = date.weekday()
        if day == 5 or day == 6:
            date += datetime.timedelta(days=1)
            continue

        filename = date.strftime('%Y-%m-%d')
        try:
            file = open('2019 data/news/' + filename + '.csv', encoding='utf-8')
        except FileNotFoundError:
            date += datetime.timedelta(days=1)
            continue

        for row in file:
            comp = row.split(',', 1)[0]
            comp_news = row.split(',', 1)[1]
            stockData = getStockData(comp, date)
            if stockData == -1:
                continue

            data = []
            data.extend((comp, date.timetuple().tm_yday))
            data.extend((sentiment.sent_score(comp_news), sentiment.sent_magnitude(comp_news, alpha)))
            data.extend(stockData)
            writer.writerow(data)

        date += datetime.timedelta(days=1)