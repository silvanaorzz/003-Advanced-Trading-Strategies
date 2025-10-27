## 2. pairs

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller

def select_pairs(data, corr_threshold=0.7):
    tickers = data.columns
    corr = data.corr()
    pairs = []
    for i in range(len(tickers)):
        for j in range(i+1, len(tickers)):
            if corr.iloc[i, j] > corr_threshold:
                x, y = data[tickers[i]], data[tickers[j]]
                model = sm.OLS(y, sm.add_constant(x)).fit()
                residuals = model.resid
                adf_p = adfuller(residuals)[1]
                if adf_p < 0.05:
                    pairs.append((tickers[i], tickers[j]))
    return pairs
