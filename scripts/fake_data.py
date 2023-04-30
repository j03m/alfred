#!/usr/bin/env python3
import pandas as pd

start_date = pd.Timestamp('2013-12-18 00:00:00-05:00')
data = []
close = 19

for i in range(180):
    row = [start_date.strftime('%Y-%m-%d %H:%M:%S%z'), close, close, close, close, close, 5457200]
    data.append(row)
    start_date += pd.Timedelta(days=1)
    close += 10

df = pd.DataFrame(data, columns=['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume'])
df.to_csv('./data/fake.csv', index=False)