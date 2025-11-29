import pandas as pd
import ta

def add_indicators(df):
    df = df.copy()

    df["rsi"] = ta.momentum.rsi(df["Close"], window=14)
    df["macd"] = ta.trend.macd_diff(df["Close"])
    df["volatility"] = df["Close"].pct_change().rolling(20).std()
    df["return"] = df["Close"].pct_change()

    df = df.dropna()
    return df
