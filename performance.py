## 7. Performance

import numpy as np

def evaluate_performance(portfolio):
    portfolio = np.array(portfolio)
    returns = np.diff(portfolio) / portfolio[:-1]
    cagr = (portfolio[-1]/portfolio[0]) ** (1/15) - 1
    sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252)
    drawdown = np.min(portfolio/np.maximum.accumulate(portfolio) - 1)
    return {
        "Final Value": portfolio[-1],
        "CAGR (%)": cagr * 100,
        "Sharpe": sharpe,
        "Max Drawdown (%)": drawdown * 100
    }
