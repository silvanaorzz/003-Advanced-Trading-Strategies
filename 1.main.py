# main.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from data import descargar_datos, split_data, handle_missing
from pairs import select_pairs
from KALMAN import KalmanFilterReg
from backtest import backtesting
from performance import evaluate_performance

# ---------------- CONFIG ----------------
TICKERS = ["AMZN", "MSFT","AMD", "GOOGL", "AAPL", "TSLA", "NVDA", "META", "INTC", "IBM"]
START, END = "2010-01-01", "2025-01-01"
COMMISSION = 0.00125       # 0.125%
BORROW_RATE = 0.0025       # 0.25% annualized
INVEST_RATIO = 0.8         # 80% capital
# ----------------------------------------

def run_analysis():
    # Step 1: Load and prepare data
    data = descargar_datos(TICKERS, START, END)
    data = handle_missing(data)
    train, test, val = split_data(data)

    # Step 2: Select cointegrated pairs
    selected_pairs = select_pairs(train)

    if not selected_pairs or not all(len(p) == 2 for p in selected_pairs):
        print("⚠️ No valid cointegrated pairs found. Using default tickers for analysis.")
        selected_pairs = [(TICKERS[0], TICKERS[1])]  # fallback

    print("✅ Selected pairs:")
    for pair in selected_pairs:
        print(f" - {pair[0]} & {pair[1]}")

    # Step 3: Analyze each pair
    for idx, (a, b) in enumerate(selected_pairs):
        print(f"\n🔹 Analyzing pair {idx+1}: {a} & {b}")

        df_pair = data[[a, b]].dropna()
        if df_pair.empty:
            print(f"⚠️ No overlapping data for {a} & {b}, skipping this pair.")
            continue

        try:
            kf = KalmanFilterReg()
            portfolio_value, spread_normalized = backtesting(
                df_pair, [a, b], kf,
                commission=COMMISSION,
                borrow_rate=BORROW_RATE,
                invest_ratio=INVEST_RATIO
            )

            # Ensure spread_normalized length consistency
            if len(spread_normalized) != len(df_pair):
                spread_normalized = df_pair[b] - df_pair[a]
                spread_normalized = (spread_normalized - spread_normalized.mean()) / spread_normalized.std()

            # Step 4: Performance metrics
            stats = evaluate_performance(portfolio_value)

            # ---- Print Results ----
            initial_value = float(portfolio_value[0])
            final_value = float(portfolio_value[-1])
            returns = (final_value - initial_value) / initial_value * 100

            print("\n📊 Portfolio Statistics:")
            print(f"Initial Value: ${initial_value:,.2f}")
            print(f"Final Value: ${final_value:,.2f}")
            print(f"Return: {returns:.2f}%")
            print(f"Maximum Value: ${max(portfolio_value):,.2f}")

            print("\n📈 Performance Summary (15-year backtest):")
            for k, v in stats.items():
                print(f"{k}: {v:.2f}")

            # Step 5: Plot results
            plt.figure(figsize=(15, 10))

            plt.subplot(311)
            plt.plot(df_pair.index, df_pair[a], label=a)
            plt.plot(df_pair.index, df_pair[b], label=b)
            plt.title(f"Stock Prices: {a} & {b}")
            plt.legend()
            plt.grid(True)

            plt.subplot(312)
            plt.plot(df_pair.index, spread_normalized, label="Spread (Normalized)", color='b')
            plt.axhline(y=0, color='k', linestyle='--')
            plt.axhline(y=1.5, color='r', linestyle='--', label='+1.5σ')
            plt.axhline(y=-1.5, color='g', linestyle='--', label='-1.5σ')
            plt.title("Normalized Spread")
            plt.legend()
            plt.grid(True)

            plt.subplot(313)
            portfolio_index = df_pair.index[:len(portfolio_value)-1]
            plt.plot(portfolio_index, portfolio_value[:-1], label="Portfolio Value", color='g')
            plt.title("Portfolio Performance")
            plt.legend()
            plt.grid(True)

            plt.tight_layout()
            plt.show()

        except Exception as e:
            print(f"❌ Error analyzing pair {a} & {b}: {e}")

if __name__ == "__main__":
    run_analysis()
