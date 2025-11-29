import yfinance as yf
import pandas as pd

def load_stock(ticker, years=10):
    data = yf.download(ticker, period=f"{years}y")
    data = data.dropna()
    return data
