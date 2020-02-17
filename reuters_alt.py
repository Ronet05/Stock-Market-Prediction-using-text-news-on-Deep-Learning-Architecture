import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import datetime as dt
import json
import os
import requests
import csv

token_dict = {'AAPL': 'AAPL.OQ', 'AMZN': 'AMZN.OQ', 'GOOGL': 'GOOGL.O'}

for tok in token_dict.keys():
    token = tok
    ticker = token_dict[tok]
    use_date = None
    time = 1581379199000000000
    n_data = []
    while time >= 1475884800000000000:

        r1 = requests.get("https://wireapi.reuters.com/v8/feed/rcom/us/marketnews/ric:{}?until={}".format(ticker, time))
        sample = json.loads(r1.text)

        try:
            for item in sample["wireitems"]:
                if 'ad-' in (item['templates'][0]['cid']):
                    continue
                else:
                    # print(item['wireitem_id'])
                    ts = int(item['wireitem_id']) / (10 ** 9)
                    date_curr = dt.datetime.utcfromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')

                    if use_date is None:
                        use_date = date_curr[0:10]
                        n_data.append(item['templates'][0]['story']['hed'])
                        continue

                    elif use_date == date_curr[0:10]:
                        n_data.append(item['templates'][0]['story']['hed'])

                    else:
                        n_data.insert(0, token)
                        update = [n_data]
                        with open('2019 data/news/' + use_date + '.csv', 'a', newline="",
                                  encoding="utf-8") as news_file:
                            writer = csv.writer(news_file)
                            for row in update:
                                writer.writerow(row)
                        use_date = date_curr[0:10]
                        n_data.clear()
                        n_data.append(item['templates'][0]['story']['hed'])

            time = int(item['wireitem_id'])

        except KeyError:
            break
