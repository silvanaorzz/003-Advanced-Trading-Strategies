## 1. Data

import yfinance as yf
import pandas as pd
import numpy as np

def descargar_datos(tickers, start, end):
    data = yf.download(tickers, start=start, end=end)["Close"]
    if isinstance(data, pd.Series):
        data = data.to_frame()
    return data

def handle_missing(data):
    data = data.ffill().bfill().dropna()
    return data

def split_data(data):
    n = len(data)
    train = data.iloc[:int(0.6*n)]
    test = data.iloc[int(0.6*n):int(0.8*n)]
    val = data.iloc[int(0.8*n):]
    return train, test, val
