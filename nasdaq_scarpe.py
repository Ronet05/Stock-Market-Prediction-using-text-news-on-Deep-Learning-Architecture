import requests
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import os
import datetime as dt

r = requests.get("https://www.nasdaq.com/api/v1/loadmore/content_feed_2_3?content_types%5B0%5D=article&title=Latest%20News&load_more_text=See%20More&filter_by=symbol&filter_value=81&limit=50&total_limit=40&press_releases_tab_enabled=1&offset=250")
soup=BeautifulSoup(r.text, 'html.parser')
print(soup.prettify())
for node in soup.find_all(class_='content-feed__card-timestamp'):
    if dt.datetime.strptime(str(node.find(text=True)), "%b %d, %Y") > dt.datetime(2018, 10, 8):
        print("Exclude")
    else:




