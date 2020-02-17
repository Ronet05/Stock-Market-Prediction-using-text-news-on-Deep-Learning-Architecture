import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import datetime as dt
import json
import os
import requests
import csv

time = 1581379199000000000
while time >= 1475884800000000000:
    token = 'AAPL'
    r1 = requests.get("https://wireapi.reuters.com/v8/feed/rcom/us/marketnews/ric:AAPL.OQ?until={}".format(time))
    sample = json.loads(r1.text)
    try:
        for item in sample["wireitems"]:
            if 'ad-' in (item['templates'][0]['cid']):
                continue
            else:
                print(item['wireitem_id'])
                ts = int(item['wireitem_id']) / (10 ** 9)
                date_curr = dt.datetime.utcfromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
                print(date_curr[0:10])
                data=[token, item['templates'][0]['story']['hed']]
                with open('2019 data/news/' + date_curr + '.csv', 'a') as news_file:
                    writer = csv.writer(news_file)
                    if token not in writer[0]:
                        writer.writerow(data)
                    else:
                        writer.


                file.write(token)

                print(item['templates'][0]['story']['hed'])
                print('\n')

        time = int(item['wireitem_id'])



    except KeyError:
        pass
