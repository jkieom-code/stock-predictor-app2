import numpy as np
from sklearn.preprocessing import MinMaxScaler

def create_supervised(data, lookback=60, horizon=7):
    X, y = [], []
    for i in range(len(data) - lookback - horizon):
        X.append(data[i:i+lookback])
        y.append(data[i+lookback:i+lookback+horizon, 0])
    return np.array(X), np.array(y)

def scale_features(df):
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df)
    return scaled, scaler
