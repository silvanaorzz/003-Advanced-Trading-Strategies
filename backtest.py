## 6. Backtesting

import numpy as np

def backtesting(data, tickers, kf, commission, borrow_rate, invest_ratio):
    capital = 1_000_000
    portfolio = [capital]
    a, b = tickers
    spreads = []

    for t in range(1, len(data)):
        x, y = data[a].iloc[t-1], data[b].iloc[t-1]
        kf.predict()
        kf.update(x, y)

        spread = y - (kf.x[0] + kf.x[1]*x)
        spreads.append(spread)

        if len(spreads) < 20:
            portfolio.append(capital)
            continue

        spread_normalized = (np.array(spreads) - np.mean(spreads)) / np.std(spreads)
        z = spread_normalized[-1]
        hedge_ratio = kf.x[1]
        position_value = capital * invest_ratio

        pnl = 0
        if z > 1.5:  # short spread
            capital *= (1 - commission)
            pnl = position_value * (data[a].iloc[t] - hedge_ratio * data[b].iloc[t]) / position_value
        elif z < -1.5:  # long spread
            capital *= (1 - commission)
            pnl = -position_value * (data[a].iloc[t] - hedge_ratio * data[b].iloc[t]) / position_value

        # Apply borrow cost daily
        capital -= position_value * (borrow_rate / 252)
        capital += pnl
        portfolio.append(capital)

    spread_normalized = (np.array(spreads) - np.mean(spreads)) / np.std(spreads)
    return portfolio, spread_normalized

