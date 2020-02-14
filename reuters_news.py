import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import datetime as datetime
import json
import os
import requests

time = 1581255730000000000
while time >= 1475884800000000000:
    r1 = requests.get("https://wireapi.reuters.com/v8/feed/rcom/us/marketnews/ric:AAPL.OQ?until={}".format(time))
    sample = json.loads(r1.text)
    try:
        for item in sample["wireitems"]:
            if 'ad-' in (item['templates'][0]['cid']):
                continue
            else:
                print(item['wireitem_id'])
                ts = int(item['wireitem_id']) / (10 ** 9)
                print(datetime.utcfromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S'))
                print(item['templates'][0]['story']['hed'])
                print('\n')

        time = int(item['wireitem_id'])

    except KeyError:
        pass

